"""
command_processor - 命令处理器模块

核心功能：对话理解链路的末端执行环节——接收命令生成阶段产出的
          Command 列表，逐个执行并把结果落地为对话状态变更
          （Tracker 槽位/命令历史 + DialogueStack 栈帧），归类响应
          类型并推断下一步动作提示，供 policy 节点第一轮优先采用。

执行流程：
    1. 装配期：message_processor 构造 CommandProcessor（配置/
       Domain/Flows），构造与 set_flows 时预建 Flow ID 集合备用；
    2. 每轮消息（graph understand 节点触发）：process 四步走——
       a. force_slot_filling 过滤：处于 collect 步骤（正在收集
          槽位）时丢弃非当前槽位的 SetSlotCommand 与全部
          StartFlowCommand，防 LLM 误设槽位/误启流程；
       b. 逐命令执行：command.run 产出事件、命令字典记入 Tracker，
          单命令失败记入 errors 不中断整批；
       c. 响应类型归类：按命令类型写 result.response_type；
       d. 下一步动作推断：_determine_next_action 按响应类型与栈
          状态设置 next_action——已在 run() 中压栈帧的类型不设置，
          交给 Policy 检测栈帧决定（policy 节点仅第一轮优先采用
          本处的显式动作）。
    3. 校验通道：validate_command/filter_valid_commands 供外部
       预校验命令合法性（流程存在性/槽位存在性）。

副作用：
    - 执行命令会修改 Tracker 状态（槽位值/对话栈帧/命令历史）；
    - 使用 logging 记录调试/警告/错误日志；
    - 无文件/数据库读写、无网络请求。

主要依赖：
    - commands.base：Command 基类（类型标注与统一接口）；
    - commands.flow_commands / slot_commands / answer_commands /
      session_commands：各命令族——isinstance 归类与校验分支需要
      具体命令类型（运行时依赖）；
    - core.tracker / core.domain：状态追踪器与业务域（仅注解用，
      经 TYPE_CHECKING 隔离避免循环导入）。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，TYPE_CHECKING 分支
# 中的 tracker/domain 前向引用无需引号即可书写，PEP 604 联合类型注解
# （X | None）也不受运行时版本限制。
from __future__ import annotations

# 导入 logging：模块级日志（命令处理进度/过滤告警/执行失败错误）。
import logging

# 导入 dataclass 装饰器与 field：配置与结果的数据类定义；
# field(default_factory=...) 为可变默认值（列表/字典字段）指定
# 每次实例化时的工厂函数，避免可变默认值在实例间共享。
from dataclasses import dataclass, field

# 导入类型注解工具：Any 类型通配符；TYPE_CHECKING 隔离仅注解导入
# （注解已按 PEP 585/604 新式写法，Dict/List/Optional 不再需要）。
from typing import Any, TYPE_CHECKING

# Command 基类：全部命令的公共父类，process 的入参类型与
# command_name/run/as_dict 调用契约的来源。
from app.ics_ai.dialogue_understanding.commands.base import Command

# Flow 命令族：启动/取消/切换流程——响应类型归类与流程校验分支使用。
from app.ics_ai.dialogue_understanding.commands.flow_commands import (
    StartFlowCommand,
    CancelFlowCommand,
    ChangeFlowCommand,
)

# 槽位命令：设置槽位值——校验与 collect 过滤分支使用。
from app.ics_ai.dialogue_understanding.commands.slot_commands import SetSlotCommand

# 回答命令族：闲聊/知识问答/无法处理——响应类型归类分支使用。
from app.ics_ai.dialogue_understanding.commands.answer_commands import (
    ChitChatAnswerCommand,
    CannotHandleCommand,
    KnowledgeAnswerCommand,
)

# 会话命令族：澄清追问/人工转接/会话开始/会话重启——响应类型
# 归类分支使用。
from app.ics_ai.dialogue_understanding.commands.session_commands import (
    ClarifyCommand,
    HumanHandoffCommand,
    SessionStartCommand,
    RestartCommand,
)

# 类型检查期专属导入：这些类型只出现在方法注解中，运行时不导入
# （避免与 core 包形成循环依赖或引入不必要的运行时开销）。
if TYPE_CHECKING:
    # 对话状态追踪器：process 的读写目标（槽位/栈/命令历史）。
    from app.ics_ai.core.tracker import DialogueStateTracker

    # 业务域定义：槽位校验的依据（slots 字典）。
    from app.ics_ai.core.domain import Domain

# 模块级日志记录器：日志名即本模块完整导入路径。
logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    """处理器配置。

    职责：承载命令处理器的行为开关——是否允许并行执行、是否校验
          Flow 存在性、是否校验槽位存在性。
    设计模式：数据类（参数对象，避免构造函数散装传参）。
    生命周期：随处理器构造创建并长期只读复用（可整体替换）。

    Attributes:
        allow_parallel_commands: 是否允许并行执行多个命令
        validate_flows: 是否验证Flow存在性
        validate_slots: 是否验证槽位存在性
    """

    # 是否允许并行执行多个命令：已声明但当前实现未读取（process
    # 始终按列表顺序逐条执行），为未来并行执行策略预留的开关。
    allow_parallel_commands: bool = True
    # 是否校验 Flow 存在性：开启时 StartFlowCommand 的目标流程
    # 必须在已知 Flow ID 集合内（见 validate_command）。
    validate_flows: bool = True
    # 是否校验槽位存在性：开启时 SetSlotCommand 的槽位名必须在
    # Domain 定义中（见 validate_command）。
    validate_slots: bool = True


@dataclass
class ProcessResult:
    """处理结果。

    职责：聚合一次 process 调用的全部产出——事件、执行计数、错误、
          响应类型、下一步动作提示与附加元数据，作为处理器的
          统一返回信封。
    设计模式：数据类（结果 DTO/信封对象）。
    生命周期：每次 process 调用创建一个新实例，随返回值交出，
              不跨轮复用。

    Attributes:
        events: 产生的事件列表
        commands_executed: 成功执行的命令数
        errors: 错误列表
        next_action: 下一步应该执行的动作
        response_type: 响应类型（flow, chitchat, knowledge, cannot_handle等）
    """

    # 事件列表：每个命令 run 产出的事件字典（结构由各命令的 run
    # 实现定义），按执行顺序追加。
    events: list[dict[str, Any]] = field(default_factory=list)
    # 成功执行的命令数：每条命令 run 成功 +1；success 判定的依据之一。
    commands_executed: int = 0
    # 错误列表：单命令执行失败的错误信息（格式
    # "Failed to execute <命令名>: <异常>"），一条失败一条记录。
    errors: list[str] = field(default_factory=list)
    # 下一步动作提示：形如 action_listen/action_run_flow_<id> 的
    # 动作名；None 表示不显式指定——policy 节点仅在第一轮优先采用
    # 该值，未设置时由 Policy 检测栈帧决定动作。
    next_action: str | None = None
    # 响应类型：按最后一条可归类命令的类型覆盖写入
    # （flow/cancel_flow/change_flow/session_start/restart/chitchat/
    # knowledge/cannot_handle/clarify/human_handoff）；
    # 初始 "none" 表示尚无可归类命令。
    response_type: str = "none"
    # 附加元数据：归类过程中的补充信息（如 change_flow 写入
    # target_flow 目标流程 ID）。
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """是否成功处理了命令。

        返回：
            至少执行了一条命令且错误列表为空时 True；
            未执行任何命令（含空列表/全部被过滤）或存在错误时 False。
        """
        # 双条件与：执行数大于 0 且无错误记录。
        return self.commands_executed > 0 and len(self.errors) == 0


class CommandProcessor:
    """命令处理器。

    职责：执行命令并更新对话状态——将命令生成阶段产出的 Command
          列表落地为 Tracker/Stack 状态变更与事件，归类响应类型，
          推断下一步动作提示。这是对话理解模块的核心组件之一。
    设计模式：命令模式消费者（命令自身封装执行意图 run()，处理器
              只负责调度、过滤、归类与状态回填）。
    生命周期：随 message_processor 构造并跨多轮消息复用；
              domain/flows 可经 set_domain/set_flows 热更新。

    处理流程：
    1. 接收命令列表
    2. 按顺序执行每个命令
    3. 更新对话状态（Tracker, Stack）
    4. 返回产生的事件和下一步动作
    """

    def __init__(
        self,
        config: ProcessorConfig | None = None,
        domain: Domain | None = None,
        flows: list[Any] | None = None,
    ):
        """初始化处理器。

        业务流程：1. 配置缺省时创建默认 ProcessorConfig；2. 保存
                  Domain/Flows 引用（均允许空，后续可热更新）；
                  3. 预建 Flow ID 集合供流程校验使用。

        副作用：创建 config/domain/flows/_flow_ids 四个实例属性；
                不做任何校验或 IO。

        参数：
            config: 处理器配置，允许 None（用默认配置）。
            domain: Domain定义，允许 None（槽位校验将放行）。
            flows: 可用的Flow列表，允许 None（按空列表处理）。
        """
        # 配置兜底：未显式传入时使用默认配置（允许并行 + 开启两类校验）。
        self.config = config or ProcessorConfig()
        # 保存 Domain 引用：槽位校验依据；允许 None（此时校验放行）。
        self.domain = domain
        # 保存 Flow 列表：命令执行的上下文（command.run 第二参数）；
        # None 归一化为空列表。
        self.flows = flows or []
        # 预建 Flow ID 集合：从 flows 逐个提取 id 属性（元素无 id
        # 属性时退化为 str(f)，兼容裸字符串列表），供 validate_command
        # 校验流程存在性；flows 为空时置空集合（校验被短路放行）。
        self._flow_ids = (
            set(getattr(f, "id", str(f)) for f in self.flows) if self.flows else set()
        )

    def process(
        self,
        commands: list[Command],
        tracker: DialogueStateTracker,
    ) -> ProcessResult:
        """处理命令列表。

        业务流程：
            0. 空命令列表直接返回空结果；
            1. force_slot_filling 过滤：collect 步骤中丢弃非法命令，
               过滤后为空则提前返回；
            2. 按序逐条执行：_execute_command 产出事件并累计执行数，
               _update_response_type 归类响应类型；单命令异常记入
               errors 后继续后续命令（可用性优先，不中断整批）；
            3. _determine_next_action 依据响应类型与栈状态推断
               next_action，填充进结果。

        副作用：经命令 run 修改 Tracker 状态（槽位/对话栈/命令
                历史）；写日志；只读 self.config/domain/flows。

        参数：
            commands: 要处理的命令列表，允许空列表。
            tracker: 对话状态追踪器（执行命令时的读写对象）。

        返回：
            ProcessResult：events/errors/commands_executed/
            response_type/next_action 全量产出；任何单命令失败
            都不会抛出（错误记入 errors）。
        """
        # 初始化结果信封：各字段随执行逐步填充。
        result = ProcessResult()

        # 空命令列表：无内容可处理，直接返回空结果。
        if not commands:
            logger.debug("No commands to process")
            return result

        logger.debug(f"Processing {len(commands)} commands")

        # force_slot_filling 过滤：collect 步骤（正在收集某槽位）中
        # 仅保留设置当前槽位的命令与启动类之外的命令。
        commands = self._filter_commands_during_collect(commands, tracker)

        # 过滤后为空：全部被丢弃，返回空结果（不执行任何命令）。
        if not commands:
            logger.debug("All commands filtered out during collect step")
            return result

        # 按列表顺序逐条执行命令（allow_parallel_commands 当前未生效）。
        for command in commands:
            try:
                # 执行单命令：返回其产生的事件列表。
                events = self._execute_command(command, tracker)
                # 事件按序并入结果。
                result.events.extend(events)
                # 成功执行计数 +1。
                result.commands_executed += 1

                # 按命令类型归类响应类型（写入 result）。
                self._update_response_type(command, result)

            except Exception as e:
                # 单命令失败兜底：记录错误并继续后续命令——一条坏命令
                # 不应拖垮整批处理（错误格式含命令名便于定位）。
                error_msg = f"Failed to execute {command.command_name()}: {e}"
                result.errors.append(error_msg)
                logger.error(error_msg)

        # 全部命令执行完毕：依据最终状态推断下一步动作提示。
        self._determine_next_action(tracker, result)

        return result

    def _execute_command(
        self,
        command: Command,
        tracker: DialogueStateTracker,
    ) -> list[dict[str, Any]]:
        """执行单个命令。

        业务流程：1. 委托命令自身的 run 方法执行（命令模式——执行
                  逻辑封装在命令内，处理器只调度）；2. 把命令字典
                  记录进 Tracker 的命令历史。

        副作用：run 内部对 Tracker/对话栈的状态变更（如压栈帧、
                设槽位）；tracker.add_commands 追加命令记录。

        参数：
            command: 要执行的命令对象。
            tracker: 对话状态追踪器（透传给 run）。

        返回：
            命令执行产生的事件列表（结构由各命令 run 定义）；
            run 抛异常时向上传播（由 process 统一捕获记入 errors）。
        """
        # 使用命令自身的run方法：flows 作为执行上下文透传
        # （如 StartFlowCommand 据此解析目标流程定义）。
        events = command.run(tracker, self.flows)

        # 记录命令到tracker：以字典形式追加进命令历史，供持久化与调试回放。
        tracker.add_commands([command.as_dict()])

        return events

    def _update_response_type(
        self,
        command: Command,
        result: ProcessResult,
    ) -> None:
        """更新响应类型。

        业务流程：按命令类型 isinstance 匹配，把对应类别写入
                  result.response_type；change_flow 额外把目标流程
                  ID 写入 result.metadata["target_flow"]。

        副作用：修改 result.response_type 与 result.metadata。

        参数：
            command: 刚执行成功的命令。
            result: 当前处理结果（就地修改）。

        说明：
            多命令场景下每个成功命令都会调用本方法，后执行的
            可归类命令覆盖先执行的（最终以最后一条可归类命令为准）；
            SetSlotCommand 无匹配分支，不改变响应类型。
        """
        # 类型归类链：依序匹配命令类型，命中即写入类别。
        if isinstance(command, StartFlowCommand):
            # 启动流程：响应类型 flow。
            result.response_type = "flow"
        elif isinstance(command, CancelFlowCommand):
            # 取消流程：响应类型 cancel_flow。
            result.response_type = "cancel_flow"
        elif isinstance(command, ChangeFlowCommand):
            # 切换流程：响应类型 change_flow，并记录目标流程 ID。
            result.response_type = "change_flow"
            result.metadata["target_flow"] = command.flow
        elif isinstance(command, SessionStartCommand):
            # 会话开始：响应类型 session_start。
            result.response_type = "session_start"
        elif isinstance(command, RestartCommand):
            # 会话重启：响应类型 restart。
            result.response_type = "restart"
        elif isinstance(command, ChitChatAnswerCommand):
            # 闲聊回答：响应类型 chitchat。
            result.response_type = "chitchat"
        elif isinstance(command, KnowledgeAnswerCommand):
            # 知识问答：响应类型 knowledge。
            result.response_type = "knowledge"
        elif isinstance(command, CannotHandleCommand):
            # 无法处理：响应类型 cannot_handle。
            result.response_type = "cannot_handle"
        elif isinstance(command, ClarifyCommand):
            # 澄清追问：响应类型 clarify。
            result.response_type = "clarify"
        elif isinstance(command, HumanHandoffCommand):
            # 人工转接：响应类型 human_handoff。
            result.response_type = "human_handoff"
        # SetSlotCommand 不改变响应类型（无匹配分支，保持原值）。

    def _determine_next_action(
        self,
        tracker: DialogueStateTracker,
        result: ProcessResult,
    ) -> None:
        """确定下一步动作。

        业务流程：按 result.response_type 分派——flow 类读活动
                  Flow 推导 action_run_flow_<id>；cancel/change/
                  session_start/restart/clarify 各映射固定动作；
                  chitchat/knowledge/cannot_handle/human_handoff
                  不设置（命令已在 run() 中压入栈帧，交给 Policy
                  检测栈帧决定）；默认分支看活动 Flow——有则交给
                  FlowPolicy（不设置），无则 action_listen。

        副作用：修改 result.next_action（就地修改，仅赋值不抛出）。

        参数：
            tracker: 对话状态追踪器（读 active_flow）。
            result: 当前处理结果（就地修改）。

        设计说明：
            对于已经在 Command.run() 中压入栈帧的 Command 类型
            （chitchat, knowledge, cannot_handle, human_handoff），
            不设置 next_action，让 Policy 通过检测栈帧来决定动作；
            policy 节点仅在第一轮优先采用本处设置的显式动作。
        """
        # 根据响应类型决定下一步。
        if result.response_type == "flow":
            # Flow已启动，执行Flow的下一步：读活动 Flow 拼接动作名。
            if tracker.active_flow:
                # 有活动 Flow：下一步动作即运行该 Flow。
                result.next_action = f"action_run_flow_{tracker.active_flow}"
            else:
                # 无活动 Flow（启动即失败等）：退回监听等待输入。
                result.next_action = "action_listen"

        elif result.response_type == "cancel_flow":
            # 取消流程：映射取消动作。
            result.next_action = "action_cancel_flow"

        elif result.response_type == "change_flow":
            # 切换流程：映射切换动作。
            result.next_action = "action_change_flow"

        elif result.response_type == "session_start":
            # 会话开始：映射会话启动动作。
            result.next_action = "action_session_start"

        elif result.response_type == "restart":
            # 会话重启：映射重启动作。
            result.next_action = "action_restart"

        elif result.response_type == "clarify":
            # ClarifyCommand 没有压入栈帧，需要显式设置 next_action。
            result.next_action = "action_clarify"

        # 以下类型的 Command 已经在 run() 中压入了栈帧，
        # 不设置 next_action，让 Policy 检测栈帧来决定动作：
        # - chitchat → ChitChatStackFrame → Policy 返回 action_send_text
        # - knowledge → SearchStackFrame → Policy 返回 action_send_text
        # - cannot_handle → CannotHandleStackFrame → Policy 返回 action_send_text
        # - human_handoff → HumanHandoffStackFrame → Policy 返回 action_send_text

        elif result.response_type in (
            "chitchat",
            "knowledge",
            "cannot_handle",
            "human_handoff",
        ):
            # 不设置 next_action，让 Policy 通过栈帧检测来决定。
            pass

        else:
            # 默认情况处理（如仅 SetSlotCommand、response_type 仍为 none）。
            # 如果存在活跃 flow，不设置 next_action，让 FlowPolicy 决定下一步
            # 这对于按钮点击 SetSlots 等情况尤为重要。
            if tracker.active_flow:
                # 不设置 next_action，让 FlowPolicy 处理 flow 的下一步。
                pass
            else:
                # 没有活跃 flow，等待用户输入。
                result.next_action = "action_listen"

    def validate_command(self, command: Command) -> bool:
        """验证命令是否有效。

        业务流程：StartFlowCommand 校验目标流程在已知 Flow ID 集合
                  内；SetSlotCommand 校验槽位在 Domain 定义中；其余
                  命令默认有效。开关关闭或依据缺失时放行（宽松策略）。

        副作用：无（只读 config/_flow_ids/domain）。

        参数：
            command: 要验证的命令。

        返回：
            是否有效——校验开启且依据存在时按规则判定；否则 True。
        """
        # 验证StartFlowCommand的flow是否存在。
        if isinstance(command, StartFlowCommand):
            # 校验开关开启且已知流程集合非空时才判定
            # （集合为空说明未提供流程清单，放行避免误杀）。
            if self.config.validate_flows and self._flow_ids:
                return command.flow in self._flow_ids

        # 验证SetSlotCommand的槽位是否存在。
        if isinstance(command, SetSlotCommand):
            # 校验开关开启且已提供 Domain 时才判定
            # （无 Domain 无法校验，放行）。
            if self.config.validate_slots and self.domain:
                return command.name in self.domain.slots

        # 其余命令类型无校验规则，默认有效。
        return True

    def filter_valid_commands(self, commands: list[Command]) -> list[Command]:
        """过滤出有效的命令。

        业务流程：对命令列表逐个调用 validate_command，保留全部
                  有效命令（顺序不变）。

        副作用：无（纯过滤，不修改入参与状态）。

        参数：
            commands: 命令列表，允许空列表（返回空列表）。

        返回：
            有效的命令列表（validate_command 为 True 的子集）。
        """
        # 列表推导：逐个校验，仅保留有效命令。
        return [cmd for cmd in commands if self.validate_command(cmd)]

    def _get_current_slot_to_collect(
        self,
        tracker: DialogueStateTracker,
    ) -> str | None:
        """获取当前正在收集的槽位名称。

        业务流程：从 Tracker 的对话栈取栈顶 Flow 帧，读其
                  slot_to_collect 字段。

        副作用：无（只读 tracker.dialogue_stack）。

        参数：
            tracker: 对话状态追踪器。

        返回：
            当前正在收集的槽位名；不在 collect 步骤（无 Flow 帧/
            帧无该字段）或 Tracker 无对话栈属性时返回 None。
        """
        # 防御：Tracker 无 dialogue_stack 属性时（畸形/简化对象）
        # 直接返回 None，避免 AttributeError。
        if not hasattr(tracker, "dialogue_stack"):
            return None

        # 取栈顶 Flow 帧：当前活跃流程的执行上下文（无活跃流程为 None）。
        flow_frame = tracker.dialogue_stack.top_flow_frame()
        # 帧存在且带 slot_to_collect 字段时返回当前收集的槽位名。
        if flow_frame and hasattr(flow_frame, "slot_to_collect"):
            return flow_frame.slot_to_collect

        # 不在 collect 步骤（无 Flow 帧或帧未声明收集槽位）。
        return None

    def _filter_commands_during_collect(
        self,
        commands: list[Command],
        tracker: DialogueStateTracker,
    ) -> list[Command]:
        """基于 force_slot_filling 机制，过滤 collect 步骤中的无效命令。

        业务流程：1. 读取当前正在收集的槽位名；2. 不在 collect
                  步骤时原样返回；3. 在 collect 步骤时逐命令筛选——
                  SetSlotCommand 仅保留设置当前槽位的（丢弃其他
                  槽位，防 LLM 同时设置多个槽位）；StartFlowCommand
                  全部丢弃（防 LLM 错误触发新流程）；其余命令保留。

        副作用：无（纯过滤，不改状态）；写调试/警告日志。

        参数：
            commands: 原始命令列表。
            tracker: 对话状态追踪器（读当前收集槽位）。

        返回：
            过滤后的命令列表（保序）；不在 collect 步骤时为原列表。
        """
        # 读取当前正在收集的槽位名（None 表示不在收集步骤）。
        slot_to_collect = self._get_current_slot_to_collect(tracker)

        if not slot_to_collect:
            # 不在 collect 步骤，不过滤（原样返回）。
            return commands

        logger.debug(f"[force_slot_filling] 当前正在收集槽位: {slot_to_collect}")

        # 过滤结果收集列表：按序追加保留的命令。
        filtered = []
        for command in commands:
            if isinstance(command, SetSlotCommand):
                if command.name == slot_to_collect:
                    # 设置的正是当前收集的槽位：保留（force_slot_filling
                    # 的主路径——用户回答当前问题）。
                    filtered.append(command)
                    logger.debug(
                        f"[force_slot_filling] 保留 SetSlotCommand: {command.name}={command.value}"
                    )
                else:
                    # 设置其他槽位：丢弃并告警（防 LLM 一轮设置多个槽位）。
                    logger.warning(
                        f"[force_slot_filling] 忽略非当前槽位的设置: {command.name}, "
                        f"当前正在收集: {slot_to_collect}"
                    )
            elif isinstance(command, StartFlowCommand):
                # collect 步骤中禁止启动新流程（防止 LLM 误判用户意图），
                # 丢弃并告警。
                logger.warning(
                    f"[force_slot_filling] 忽略 collect 步骤中的 StartFlowCommand: {command.flow}"
                )
            else:
                # 保留其他命令（如 CancelFlowCommand、ChitChatAnswerCommand 等）。
                filtered.append(command)

        return filtered

    def set_domain(self, domain: Domain) -> None:
        """设置Domain。

        业务流程：替换实例的 Domain 引用，供槽位校验等后续使用
                  （与 __init__ 的 domain 参数等效的热更新入口）。

        副作用：修改 self.domain 实例属性。

        参数：
            domain: Domain定义（槽位校验依赖其 slots 字典）。
        """
        # 直接替换引用：旧 Domain 不再被本处理器持有。
        self.domain = domain

    def set_flows(self, flows: list[Any]) -> None:
        """设置可用的Flows。

        业务流程：替换 flows 列表并重建 Flow ID 集合（提取逻辑与
                  __init__ 一致），保证流程校验基于最新清单。

        副作用：修改 self.flows 与 self._flow_ids 两个实例属性。

        参数：
            flows: Flow列表，允许空列表（集合置空后校验放行）。
        """
        # 替换 Flow 列表：命令执行的上下文随之更新。
        self.flows = flows
        # 重建 Flow ID 集合：逐个提取 id（无 id 属性退化 str(f)），
        # 空列表时置空集合。
        self._flow_ids = (
            set(getattr(f, "id", str(f)) for f in flows) if flows else set()
        )


# 便捷函数：一次性处理场景的模块级入口（跨轮复用请自行持有处理器实例）。


def process_commands(
    commands: list[Command],
    tracker: DialogueStateTracker,
    domain: Domain | None = None,
    flows: list[Any] | None = None,
) -> ProcessResult:
    """便捷函数：处理命令。

    业务流程：以默认配置临时构造 CommandProcessor（注入 domain/
              flows）并执行 process——适合一次性调用场景；每轮
              消息都处理命令时应持有处理器实例复用。

    副作用：同 CommandProcessor.process（经命令 run 修改 Tracker
            状态、写日志）。

    参数：
        commands: 命令列表，允许空列表。
        tracker: 对话状态追踪器。
        domain: Domain定义，允许 None（槽位校验放行）。
        flows: Flow列表，允许 None（按空列表处理）。

    返回：
        ProcessResult：处理结果信封（同 process 返回值）。
    """
    # 临时构造处理器：默认配置（不启用自定义开关），注入业务定义。
    processor = CommandProcessor(domain=domain, flows=flows)
    # 执行命令处理并返回结果。
    return processor.process(commands, tracker)


# 导出：模块对外暴露的公共 API（处理器/配置/结果/便捷函数，
# 按字母序，便于 IDE 提示与 diff 整洁）。
__all__ = [
    "CommandProcessor",
    "ProcessorConfig",
    "ProcessResult",
    "process_commands",
]
