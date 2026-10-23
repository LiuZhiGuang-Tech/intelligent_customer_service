"""
action - 动作执行节点模块

核心功能：实现 LangGraph 消息处理图的动作节点 action_node——执行
          policy 节点预测出的动作：按动作名查注册表实例化 Action、
          合并元数据后调用 run()、把响应累积进 final_responses 并
          写入 tracker；含闲聊动作的 LLM 配置注入与 utter 动作的
          fallback 回退机制。

执行流程（图中的位置：policy 条件边之后，执行完固定进入 guard）：
    1. 读取状态：tracker/prediction/final_responses/动作计数等；
       无预测或预测无动作时防御性短路返回；
    2. 查动作注册表（get_action）——未注册动作以失败结果返回；
    3. 合并元数据（消息级 metadata 为底座、预测级 metadata 覆盖）；
       闲聊动作特判：从命令生成器配置提取 LLM 连接参数注入；
    4. 异步执行 Action.run——utter 动作空响应时回退到预测预设的
       fallback_action（FlowPolicy 的 collect 机制）；
    5. 响应累积：每条响应追加进 final_responses 并封装 BotMessage
       写入 tracker；更新 tracker.latest_action_name；
    6. 等待判定：预测携带 slot_to_collect（且非 flow_completed）
       时标记本轮完成——问完槽位等待用户输入；
    7. 异常兜底：捕获全部异常以失败结果 + error 字段返回。

副作用：
    - tracker 修改：add_bot_message 写入机器人消息、
      latest_action_name 更新为当前动作名；
    - final_responses 通道累积机器人响应（agent.py 终态提取）；
    - 动作自身的运行时副作用（LLM 调用/domain 模板读取等）发生
      在 Action.run 内部；本节点还经 logger 写流程日志。

主要依赖：
    - logging：流程日志；
    - app.ics_ai.agent.actions：get_action（动作注册表查找）与
      ActionResult（执行结果 DTO：responses/success）；
    - app.ics_ai.core.tracker.BotMessage：机器人消息封装；
    - 组件（命令生成器）经 _command_generator 通道注入——仅闲聊
      动作需要其 LLM 配置。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，PEP 604 联合
# 类型注解（X | None）不受运行时版本限制。
from __future__ import annotations

# 导入日志器工厂：执行过程记录 info/warning/error 日志。
import logging

# 导入类型注解工具：Any 类型通配符；TYPE_CHECKING 隔离仅注解导入
# （Dict 已按 PEP 585 新式写法移除）。
from typing import Any, TYPE_CHECKING

# 导入动作注册表查找与执行结果 DTO：get_action 按名实例化动作，
# ActionResult 携带响应列表与成功标记。
from app.ics_ai.agent.actions import get_action, ActionResult

# 导入机器人消息封装：响应写入 tracker 的载体。
from app.ics_ai.core.tracker import BotMessage

# 类型检查期专属导入：图状态只出现在注解中，运行时不导入。
if TYPE_CHECKING:
    from app.ics_ai.agent.graph.state import MessageProcessingState

# 模块级日志器：以模块路径为名。
logger = logging.getLogger(__name__)


async def action_node(state: "MessageProcessingState") -> dict[str, Any]:
    """动作节点：执行预测的动作。

    业务流程：
        1. 读取状态字段；无预测或预测无动作时防御性短路（正常
           路由下 policy 已过滤此类情况，此处兜底）；
        2. 查动作注册表：未注册的动作名以失败结果返回（计数 +1）；
        3. 合并元数据并执行动作；闲聊动作注入 LLM 连接配置；
        4. utter 动作空响应时回退到预测预设的 fallback_action
           （支持 FlowPolicy collect 步骤中 action_ask_xxx 优先于
           utter_ask_xxx 的机制——utter 模板缺失时用问询动作兜底）；
        5. 累积响应到 final_responses 并写入 tracker；
        6. 预测携带 slot_to_collect 且非 flow_completed 时标记
           本轮完成（等待用户输入槽位值）。

    副作用：见文件头（tracker 修改/final_responses 累积/日志）。

    参数：
        state: 当前图状态（tracker/prediction/metadata/组件引用
               的读取源）。

    返回：
        状态更新字典（LangGraph 按通道合并）：
        - 正常路径：tracker、current_action_result（ActionResult）、
          final_responses（累积后全量列表）、action_count（+1）、
          is_finished（slot_to_collect 等待输入时 True，否则
          False）、node_history（追加 "action"）；
        - 短路路径：current_action_result=None、action_count 原值、
          node_history（不触碰 tracker/final_responses）；
        - 失败/异常路径：current_action_result=ActionResult(False)、
          action_count（+1）、error（异常路径独有）、node_history。
    """
    # 从状态读取核心字段：tracker 按图契约必传（写入端
    # create_initial_state 显式构造全部通道），下标访问安全——
    # total=False 使类型检查器视所有键为 NotRequired，此处按运行时
    # 事实压制；其余字段可选，用 get 缺省读取。
    tracker = state["tracker"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    domain = state.get("domain")
    metadata = state.get("metadata", {})
    # 策略预测结果：动作名与执行参数的来源（policy 节点写入）。
    current_prediction = state.get("current_prediction")
    # 累积响应列表：浅拷贝出独立列表做追加（返回时整体替换通道值，
    # 不原地修改 LangGraph 状态中的列表）。
    final_responses = list(state.get("final_responses", []))
    # 已执行动作计数：guard 节点据此判断是否超过 max_actions 上限。
    action_count = state.get("action_count", 0)

    # 获取命令生成器引用：仅闲聊动作需要其 LLM 配置。
    command_generator = state.get("_command_generator")

    # 防御性短路：无预测或预测无动作名——正常路由下 edges 已把
    # 此类状态引向 response，不应触达本节点；触达则原样返回
    # （计数不增、不动 tracker/final_responses）。
    if not current_prediction or not current_prediction.action:
        return {
            "current_action_result": None,
            "action_count": action_count,
            "node_history": state.get("node_history", []) + ["action"],
        }

    # 取动作名：查注册表、执行、回退判定都用它。
    action_name = current_prediction.action
    logger.info(f"执行动作: {action_name}")

    try:
        # 查动作注册表：名字 → Action 实例（未注册得 None）。
        action = get_action(action_name)

        if action is None:
            # 未注册动作：以失败结果返回（计数 +1 与异常路径一致，
            # 让 guard 的上限判断覆盖失败重试场景）。
            logger.warning(f"动作未找到: {action_name}")
            return {
                "current_action_result": ActionResult(success=False),
                "action_count": action_count + 1,
                "node_history": state.get("node_history", []) + ["action"],
            }

        # 合并元数据：消息级 metadata 为底座，预测级 metadata 覆盖
        # （预测元数据更具体——如 fallback_action/slot_to_collect）。
        kwargs = dict(metadata)
        kwargs.update(current_prediction.metadata or {})

        # 闲聊动作特判：action_chitchat_response 内部用 LLM 生成
        # 回复，从命令生成器的配置提取连接参数注入 kwargs（动作
        # 内部据此构造 LLM 客户端，不再依赖全局配置）。
        if action_name == "action_chitchat_response" and command_generator:
            config = command_generator.config
            kwargs["llm_config"] = {
                "type": config.type,
                "model": config.model,
                "api_key": config.api_key,
                "api_base": config.api_base,
                "enable_thinking": config.enable_thinking,
            }

        # 异步执行动作：run 签名为 (tracker, domain, **kwargs)，
        # 返回 ActionResult（responses + success）。
        result = await action.run(tracker, domain, **kwargs)

        # utter 动作 fallback 机制：FlowPolicy 的 collect 步骤预测
        # utter_ask_{slot}（从 domain 模板取话术）并预设
        # fallback_action=action_ask_{slot}——模板缺失导致空响应时
        # 回退到问询动作（其话术可动态生成）。
        prediction_metadata = current_prediction.metadata or {}
        fallback_action_name = prediction_metadata.get("fallback_action")

        # 回退条件三缺一不可：响应为空 + utter_ 前缀 + 预设了回退。
        if (
            not result.responses
            and action_name.startswith("utter_")
            and fallback_action_name
        ):
            logger.info(
                f"[action_node] utter 动作无响应，尝试 fallback: {fallback_action_name}"
            )
            # 查回退动作并执行：kwargs 复用（含合并后的元数据）。
            fallback_action = get_action(fallback_action_name)
            if fallback_action:
                result = await fallback_action.run(tracker, domain, **kwargs)
                # 更新动作名：后续 latest_action_name 记录实际生效
                # 的动作（而非失败的 utter_ask_xxx）。
                action_name = fallback_action_name

        # 累积响应：逐条追加进 final_responses，并同步封装为
        # BotMessage 写入 tracker（text 取响应的 text 键，data 存
        # 完整响应字典——含按钮/附加数据等）。
        for resp in result.responses:
            final_responses.append(resp)
            # 添加机器人消息到 tracker：对话历史记录机器人侧输出。
            bot_message = BotMessage(
                text=resp.get("text", ""),
                data=resp,
            )
            tracker.add_bot_message(bot_message)

        # 更新 tracker 的最新动作名：下一轮 FlowPolicy 据此判断
        # 上轮动作（如是否刚问完槽位/是否刚执行收尾）。
        tracker.latest_action_name = action_name

        logger.info(f"[action_node] 动作执行完成, 产生 {len(result.responses)} 个响应")

        # 等待用户输入判定：FlowPolicy collect 步骤的预测携带
        # slot_to_collect——执行完问槽位话术后本轮应停止（等用户
        # 回答）；例外是 flow_completed（Flow 收尾动作仍需继续
        # 循环执行，不能停在问句后）。
        prediction_metadata = current_prediction.metadata or {}
        flow_completed = prediction_metadata.get("flow_completed", False)
        wait_for_input = "slot_to_collect" in prediction_metadata and not flow_completed

        if wait_for_input:
            logger.info(
                f"[action_node] 检测到 slot_to_collect={prediction_metadata.get('slot_to_collect')}, "
                f"设置 is_finished=True"
            )

        # 正常返回：结果 + 累积响应 + 计数 + 等待标记 + 轨迹。
        return {
            "tracker": tracker,
            "current_action_result": result,
            "final_responses": final_responses,
            "action_count": action_count + 1,
            "is_finished": wait_for_input,  # 收集槽位后等待用户输入（flow_completed 时除外）
            "node_history": state.get("node_history", []) + ["action"],
        }

    except Exception as e:
        # 异常兜底：以失败结果返回（计数 +1），error 字段供调试，
        # 不重抛——guard/edges 据此继续路由，图执行不断链。
        logger.error(f"[action_node] 动作执行失败: {e}")
        return {
            "current_action_result": ActionResult(success=False),
            "action_count": action_count + 1,
            "error": str(e),
            "node_history": state.get("node_history", []) + ["action"],
        }


# 导出：模块对外暴露的公共 API。
__all__ = ["action_node"]
