"""
policy - 策略预测节点模块

核心功能：实现 LangGraph 消息处理图的策略节点 policy_node——决定
          系统下一步执行什么动作。两条决策路径：
          1. 直通路径：第一轮且 CommandProcessor 已明确 next_action
             时直接采信（置信度 1.0），跳过策略集成器；
          2. 集成路径：其余情况调用 PolicyEnsemble 收集各策略预测
             并择优（FlowPolicy 流程推进 / EnterpriseSearchPolicy
             知识库检索兜底）。

执行流程（图中的位置：understand 之后，见 builder.py 的固定边）：
    1. 读取状态：tracker/domain/flows/策略集成器引用/命令处理
       结果/动作计数；
    2. 直通判定：仅第一轮（action_count == 0）且 process_result
       携带 next_action 且非流程推进类动作（action_run_flow_*
       前缀）时，构造 PolicyPrediction 提前返回——流程推进类动作
       必须交由 FlowPolicy 预测（含栈帧检查等流程逻辑）；
    3. 集成预测：policy_ensemble.predict 异步产出 PolicyPrediction，
       action_listen（等待用户输入）或 None（无动作）视为本轮结束；
    4. 异常兜底：预测失败时以 error 字段返回并标记完成，路由到
       response 结束本轮（保证图执行不断链）。

副作用：无直接状态修改——产出写入状态通道（current_prediction/
        is_finished），供 edges.py 路由（should_execute_action）与
        action 节点执行消费；不读写文件、不调用网络（LLM/检索
        调用发生在各策略的 predict 内部）。

主要依赖：
    - logging：预测过程日志；
    - app.ics_ai.policies.base_policy.PolicyPrediction：预测结果
      DTO（构造直通预测）；
    - app.ics_ai.shared.constants.ACTION_LISTEN：等待用户输入动作
      常量（值 "action_listen"），本轮结束的判定标记之一；
    - 策略集成器由 agent.py 装配期经 _policy_ensemble 通道注入，
      本模块不构造。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，PEP 604 联合
# 类型注解（X | None）不受运行时版本限制。
from __future__ import annotations

# 导入日志器工厂：预测过程记录 debug/info/warning/error 日志。
import logging

# 导入类型注解工具：Any 类型通配符；TYPE_CHECKING 隔离仅注解导入
# （Dict 已按 PEP 585 新式写法移除）。
from typing import Any, TYPE_CHECKING

# 导入预测结果 DTO：直通路径构造 CommandProcessor 专属预测用。
from app.ics_ai.policies.base_policy import PolicyPrediction

# 导入系统动作常量：action_listen 表示"等待用户输入"，是本轮
# 结束的判定标记。
from app.ics_ai.shared.constants import ACTION_LISTEN

# 类型检查期专属导入：图状态只出现在注解中，运行时不导入。
if TYPE_CHECKING:
    from app.ics_ai.agent.graph.state import MessageProcessingState

# 模块级日志器：以模块路径为名。
logger = logging.getLogger(__name__)


async def policy_node(state: "MessageProcessingState") -> dict[str, Any]:
    """策略节点：预测下一个动作。

    业务流程：
        1. 从状态读取字段：tracker 必传，domain/flows/集成器/
           process_result/action_count 可选（get 缺省）；
        2. 直通判定：第一轮（action_count == 0）且 CommandProcessor
           已确定 next_action 且非 action_run_flow_* 前缀（流程推进
           类动作交由 FlowPolicy）时，以置信度 1.0 构造预测并提前
           返回；next_action 为 action_listen 时标记本轮结束；
        3. 集成预测：调用 PolicyEnsemble.predict（异步），预测动作
           为 action_listen 或 None 时标记本轮结束；
        4. 异常兜底：捕获全部异常，返回 error 字段并标记完成
           （is_finished=True），edges 据此路由到 response。

    副作用：无直接状态修改（产出写入状态通道供路由与执行消费）。

    参数：
        state: 当前图状态（tracker/组件引用/process_result/
               action_count 的读取源）。

    返回：
        状态更新字典（LangGraph 按通道合并）：
        - 直通路径：current_prediction（CommandProcessor 专属预测）、
          is_finished、node_history（追加 "policy"）；
        - 集成路径：current_prediction（集成器择优预测）、
          is_finished、node_history；
        - 异常路径：current_prediction=None、is_finished=True、
          error（异常描述）、node_history。
    """
    # 从状态读取核心字段：tracker 按图契约必传（写入端
    # create_initial_state 显式构造全部通道），下标访问安全——
    # total=False 使类型检查器视所有键为 NotRequired，此处按运行时
    # 事实压制；其余字段才是可选的，用 get 缺省读取。
    tracker = state["tracker"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    domain = state.get("domain")
    flows = state.get("flows")
    # 策略集成器引用：装配期注入（下划线通道），未配置时为 None。
    policy_ensemble = state.get("_policy_ensemble")
    # 命令处理结果：understand 节点写入（经 state.py 的 process_result
    # 通道），第一轮直通判定的依据。
    process_result = state.get("process_result")
    # 已执行动作计数：first-round 判定依据（0 表示本轮首次预测）。
    action_count = state.get("action_count", 0)

    logger.debug("[policy_node] 开始策略预测")

    # 默认预测结果：无直通且无集成器可用时保持 None（edges 会把
    # 无预测路由到 response）。
    current_prediction = None
    # 默认标记完成：安全兜底——没有任何策略可用时必须结束本轮，
    # 避免图在无动作可执行的情况下继续循环。
    is_finished = True

    try:
        # 直通分支：仅第一轮（action_count == 0）且 CommandProcessor
        # 已明确下一步动作时采信（后续轮次动作执行会更新
        # action_count，转由集成器预测）。
        if action_count == 0 and process_result and process_result.next_action:
            # 取 CommandProcessor 推导的下一步动作（如
            # action_listen/action_give_reply/action_run_flow_<id>）。
            next_action = process_result.next_action
            # 流程推进类动作豁免：action_run_flow_<id> 由
            # CommandProcessor 对 start flow 命令推导，但 Flow 的
            # 推进逻辑（栈帧检查/槽位收集/话术播报）属于
            # FlowPolicy 的职责——直通会绕过这些逻辑，故此类动作
            # 落到下方集成路径由 FlowPolicy 预测。
            if not next_action.startswith("action_run_flow_"):
                # 构造直通预测：置信度 1.0（命令处理阶段已确定的
                # 确定性动作），policy_name 标记来源便于调试追踪；
                # metadata 透传命令处理阶段的附加信息。
                current_prediction = PolicyPrediction(
                    action=next_action,
                    confidence=1.0,
                    policy_name="CommandProcessor",
                    metadata=process_result.metadata,
                )
                # 结束判定：下一步是"等待用户输入"即本轮完成
                # （edges 据此路由到 response 而非 action）。
                is_finished = next_action == ACTION_LISTEN

                logger.info(
                    f"[policy_node] 使用 CommandProcessor 动作: {next_action}, "
                    f"是否结束: {is_finished}"
                )

                # 提前返回：直通结果已确定，不再询问集成器。
                return {
                    "current_prediction": current_prediction,
                    "is_finished": is_finished,
                    "node_history": state.get("node_history", []) + ["policy"],
                }

        # 集成分支：无直通命中（后续轮次/无命令结果/流程推进类
        # 动作）时，由策略集成器收集各策略预测并择优。
        if policy_ensemble:
            # 异步预测：集成器内部遍历策略（FlowPolicy/企业搜索
            # 兜底），按优先级与置信度裁决出唯一预测。
            prediction = await policy_ensemble.predict(tracker, domain, flows)
            current_prediction = prediction

            # 结束判定：action_listen（等待用户输入）或 None（无
            # 动作可执行）都视为本轮完成。
            is_finished = (
                prediction.action == ACTION_LISTEN or prediction.action is None
            )

            logger.info(
                f"[policy_node] 预测动作: {prediction.action}, "
                f"置信度: {prediction.confidence:.2f}, "
                f"是否结束: {is_finished}"
            )
        else:
            # 未配置集成器：告警后保持默认（无预测 + 已完成），
            # edges 会把本轮路由到 response。
            logger.warning("[policy_node] 未配置策略集成器，跳过预测")

    except Exception as e:
        # 异常兜底：预测失败时标记完成并携带错误信息（不重抛），
        # edges 据此路由到 response 结束本轮，保证图执行不断链。
        logger.error(f"[policy_node] 预测失败: {e}")
        return {
            "current_prediction": None,
            "is_finished": True,
            "error": str(e),
            "node_history": state.get("node_history", []) + ["policy"],
        }

    # 正常返回：预测结果 + 完成标记 + 轨迹（无直通命中的出口）。
    return {
        "current_prediction": current_prediction,
        "is_finished": is_finished,
        "node_history": state.get("node_history", []) + ["policy"],
    }


# 导出：模块对外暴露的公共 API。
__all__ = ["policy_node"]
