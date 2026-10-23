"""
understand - 消息理解节点模块

核心功能：实现 LangGraph 消息处理图的第一个节点 understand_node——
          调用 LLMCommandGenerator 把用户输入翻译为命令流，再调用
          CommandProcessor 执行命令更新对话状态；另提供
          parse_set_slots_payload 供按钮点击的 /SetSlots(...) 格式
          payload 绕过 LLM 直接解析为 SetSlotCommand 列表。

执行流程（每轮图执行从本节点开始，见 builder.py 的 START 边）：
    1. 读取状态：从 state 取 tracker/输入消息/组件引用
       （_command_generator/_command_processor 经状态通道注入）；
    2. payload 直构分支：输入以 /SetSlots( 开头时走
       parse_set_slots_payload 直接解析——解析出命令且有处理器则
       执行后提前返回（跳过 LLM）；解析为空或无处理器则回落到
       正常 LLM 流程（回退语义，payload 不报错）；
    3. 正常流程：封装 UserMessage 更新 tracker → 生成器异步产出
       GenerationResult → 处理器执行命令产出 ProcessResult；
    4. 异常兜底：任何异常被捕获后以 error 字段返回，不中断图执行。

副作用：
    - tracker 写入用户消息（update_with_message）；
    - 命令执行经 CommandProcessor 修改对话状态（槽位写入/压栈帧）；
    - 经 logger 写多条流程日志（生成/处理/payload 解析结果）；
    - 不直接读写文件、不调用网络（LLM 调用发生在生成器内部）。

主要依赖：
    - logging：流程日志；
    - app.ics_ai.core.tracker.UserMessage：用户消息封装；
    - app.ics_ai.dialogue_understanding.commands.slot_commands：
      SetSlotCommand（payload 解析产物）；
    - 组件（生成器/处理器）由 agent.py 装配期经状态通道注入，
      本模块不构造。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，PEP 604 联合
# 类型注解（X | None）不受运行时版本限制。
from __future__ import annotations

# 导入日志器工厂：节点全程记录流程日志（处理消息/生成/处理结果）。
import logging

# 导入正则模块：payload 前缀检测与 key=value 解析。
import re

# 导入类型注解工具：Any 类型通配符；TYPE_CHECKING 隔离仅注解导入
# （Dict/List/Optional 已按 PEP 585/604 新式写法移除）。
from typing import Any, TYPE_CHECKING

# 导入用户消息封装：payload 直构与正常流程都要包装后写入 tracker。
from app.ics_ai.core.tracker import UserMessage

# 导入槽位设置命令：payload 解析的产物类型（按钮点击直接设槽位）。
from app.ics_ai.dialogue_understanding.commands.slot_commands import SetSlotCommand

# 类型检查期专属导入：这些类型只出现在注解中，运行时不导入。
if TYPE_CHECKING:
    # 图状态：understand_node 的入参与返回键的契约来源。
    from app.ics_ai.agent.graph.state import MessageProcessingState

    # 命令基类：payload 解析产物的元素类型。
    from app.ics_ai.dialogue_understanding.commands.base import Command

# 模块级日志器：以模块路径为名。
logger = logging.getLogger(__name__)


def parse_set_slots_payload(payload: str) -> list["Command"]:
    """解析 /SetSlots(slot=value) 格式的 payload。

    业务流程：1. 用正则提取括号内的全部内容（不匹配则告警返回
              空列表）；2. 在内容中迭代扫描 key=value 对——槽位名
              按模式捕获，值按双引号/单引号/裸值三分支择一，再经
              字符串归一化（数字/布尔）构造 SetSlotCommand。

    副作用：经 logger 记录解析告警与每个槽位的解析结果（无状态
            修改——真正的槽位写入发生在命令执行阶段）。

    参数：
        payload: 以 /SetSlots( 开头的字符串（前缀由调用方
                 understand_node 检测后才进入本函数）。

    返回：
        SetSlotCommand 列表——按 payload 中的书写顺序排列；格式
        不合法（无右括号收尾）时返回空列表，不抛异常。
    """
    # 初始化命令列表：逐对 key=value 解析后追加。
    commands: list["Command"] = []

    # 提取括号内的内容——正则逐段拆解：
    # /SetSlots\(      —— 字面量前缀（re.match 隐式行首锚定）；
    # (.+)             —— 捕获组1：括号内全部内容（贪婪——匹配到
    #                     最后一个右括号，值本身含右括号也安全）；
    # \)$              —— 收尾右括号 + 行尾锚定。
    match = re.match(r"/SetSlots\((.+)\)$", payload.strip())
    if not match:
        # 括号结构不合法（缺右括号收尾等）：告警并返回空列表。
        logger.warning(f"[parse_set_slots_payload] 无法解析 payload: {payload}")
        return commands

    # 取组1：括号内的 key=value 串（如 order_id=123, name="订单"）。
    content = match.group(1)

    # key=value 解析模式——正则逐段拆解：
    # (\w+)            —— 组1：槽位名（字母/数字/下划线）；
    # \s*=\s*          —— 等号，两侧容忍空白；
    # (?: ... )        —— 非捕获分组，包裹值的三分支择一：
    #   "([^"]*)"      ——   双引号值 → 组2（引号本身不捕获）；
    #   | '([^']*)'    ——   单引号值 → 组3；
    #   | ([^,\s]+)    ——   裸值 → 组4（不含逗号/空白，到逗号或
    #                        内容结尾止——支撑逗号分隔多对）。
    pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^,\s]+))'
    for m in re.finditer(pattern, content):
        # 组1：槽位名。
        slot_name = m.group(1)
        # 取第一个非空的值（三分支只有一组命中，其余为 None）。
        slot_value = m.group(2) or m.group(3) or m.group(4)

        # 值类型归一化：纯数字转 int，true/false 转布尔（lower
        # 兼容 True/FALSE 等大小写变体），其余保持字符串。
        if slot_value.isdigit():
            slot_value = int(slot_value)
        elif slot_value.lower() == "true":
            slot_value = True
        elif slot_value.lower() == "false":
            slot_value = False

        # 构造槽位设置命令并追加（extractor 走默认值 "llm"）。
        commands.append(SetSlotCommand(name=slot_name, value=slot_value))
        logger.info(f"[parse_set_slots_payload] 解析槽位: {slot_name}={slot_value}")

    return commands


async def understand_node(state: "MessageProcessingState") -> dict[str, Any]:
    """理解节点：生成命令并处理。

    业务流程：
        1. 从状态读取字段与组件引用（生成器/处理器经状态通道
           注入，未装配时相应环节跳过）；
        2. payload 直构分支：输入以 /SetSlots( 开头时直接解析为
           SetSlotCommand——解析出命令且有处理器则更新 tracker、
           执行命令后提前返回（绕过 LLM）；否则回落到正常流程；
        3. 正常流程：封装 UserMessage 更新 tracker → 生成器异步
           产出 GenerationResult → 处理器执行命令产出 ProcessResult；
        4. 异常兜底：捕获全部异常，以 error 字段返回（不重抛，
           保证图执行不断链）。

    副作用：见文件头（tracker 写入/命令执行状态修改/日志）。

    参数：
        state: 当前图状态（tracker/输入消息/metadata/domain/
               flows/组件引用的读取源）。

    返回：
        状态更新字典（LangGraph 按通道合并）：
        - 正常/payload 路径：tracker、current_commands
          （GenerationResult 或 None）、process_result（经
          state.py 声明的通道传递，policy 节点第一轮优先采用其
          next_action）、node_history（追加 "understand"）；
        - 异常路径：tracker、current_commands=None、error
          （异常描述）、node_history。
    """
    # 从状态读取核心字段：tracker/input_message 按图契约必传（写入端
    # create_initial_state 显式构造全部通道），下标访问安全——total=False
    # 使类型检查器视所有键为 NotRequired，此处按运行时事实压制；
    # metadata/domain/flows 才是真正可选的字段，用 get 缺省读取。
    tracker = state["tracker"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    input_message = state["input_message"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    metadata = state.get("metadata", {})
    domain = state.get("domain")
    flows = state.get("flows")

    # 从状态读取组件引用：装配期由 agent.py 注入（下划线通道），
    # 未装配时为 None，相应环节跳过。
    command_generator = state.get("_command_generator")
    command_processor = state.get("_command_processor")

    logger.info(f"[understand_node] 处理消息: {input_message[:50]}...")

    # 1. payload 直构分支：按钮点击绕过 LLM（前缀检测）。
    if input_message.strip().startswith("/SetSlots("):
        logger.info("[understand_node] 检测到 /SetSlots payload，绕过 LLM 直接解析")
        # 直接解析为 SetSlotCommand 列表（不合法格式得空列表）。
        commands = parse_set_slots_payload(input_message)

        # 回退语义：解析出命令且有处理器才走直构路径；否则落到
        # 下方正常 LLM 流程（payload 内容被当作普通文本处理）。
        if commands and command_processor:
            # 创建用户消息（记录原始输入）：payload 标记置 True
            # 且用户 metadata 兜底展开（payload 标记优先）。
            user_message = UserMessage(
                text=input_message,
                sender_id=tracker.sender_id,
                metadata={"payload": True, **metadata},
            )
            # 写入 tracker：消息历史与 payload 分支的入口记录。
            tracker.update_with_message(user_message)

            # 直接处理解析出的命令：不经过生成器。
            process_result = command_processor.process(commands, tracker)

            logger.info(
                f"[understand_node] payload 解析了 {len(commands)} 个命令, "
                f"处理了 {process_result.commands_executed} 个"
            )

            # 提前返回：payload 路径不产生 GenerationResult
            # （current_commands 置 None），node_history 追加本节点名。
            return {
                "tracker": tracker,
                "current_commands": None,
                "process_result": process_result,
                "node_history": state.get("node_history", []) + ["understand"],
            }

    # 2. 正常流程入口：创建用户消息并更新 tracker（无 payload 标记）。
    user_message = UserMessage(
        text=input_message,
        sender_id=tracker.sender_id,
        metadata=metadata,
    )
    tracker.update_with_message(user_message)

    # 初始化结果容器：生成/处理失败或跳过时保持 None/空。
    current_commands = None
    # 注意：events 在本函数中赋值后从未读取（事件已随
    # process_result 返回），为历史遗留变量——保留赋值不动，
    # 仅供阅读时知悉。
    events = []
    process_result = None

    try:
        # 3. 使用命令生成器生成命令：LLM 把用户输入翻译为命令流。
        if command_generator:
            # FlowsList 解包为 list 传入（生成器按序列遍历 Flow）。
            flows_list = flows.flows if flows else []
            # 异步调用生成器：内部构建提示词并调用 LLM。
            generation_result = await command_generator.generate(
                tracker, domain, flows_list
            )
            # 记录生成结果：整体对象写入状态（含命令与解析诊断）。
            current_commands = generation_result

            # 以 warning 级别输出生成明细（调试痕迹：正常流程
            # 信息但需要醒目观察 LLM 的命令产出）。
            logger.warning(
                f"[understand_node] 生成了 {len(generation_result.commands)} 个命令: "
                f"{[str(c) for c in generation_result.commands]}"
            )

            # 4. 使用命令处理器执行命令：有命令且有处理器才处理。
            if generation_result.commands and command_processor:
                process_result = command_processor.process(
                    generation_result.commands, tracker
                )
                events = process_result.events

                logger.info(
                    f"[understand_node] 处理了 {process_result.commands_executed} 个命令, "
                    f"产生 {len(events)} 个事件, "
                    f"下一动作: {process_result.next_action}"
                )
        else:
            # 未配置生成器：告警后按空结果返回（图继续走 policy）。
            logger.warning("[understand_node] 未配置命令生成器，跳过命令生成")

    except Exception as e:
        # 异常兜底：任何环节失败都以 error 字段返回（不重抛），
        # 保证图执行不断链、错误信息进入状态供调试。
        logger.error(f"[understand_node] 处理失败: {e}")
        return {
            "tracker": tracker,
            "current_commands": None,
            "error": str(e),
            "node_history": state.get("node_history", []) + ["understand"],
        }

    # 正常返回：三个结果 + 轨迹（process_result 经 state.py 声明的
    # 同名通道传递，policy 节点第一轮优先采用其 next_action）。
    return {
        "tracker": tracker,
        "current_commands": current_commands,
        "process_result": process_result,
        "node_history": state.get("node_history", []) + ["understand"],
    }


# 导出：模块对外暴露的公共 API（parse_set_slots_payload 仅模块内
# 使用，不纳入导出）。
__all__ = ["understand_node"]
