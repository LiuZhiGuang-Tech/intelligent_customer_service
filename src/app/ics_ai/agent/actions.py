"""
actions - Action 系统模块

核心功能：定义对话系统"动作（Action）"的统一契约与全部内置动作——
          Action 抽象基类约定 name 属性 + 异步 run 方法；ActionResult
          数据类承载响应/事件；15 个内置动作覆盖监听、重启、会话管理、
          降级、闲聊、流程取消/切换/完成、槽位提取、模板响应等场景；
          注册表 + get_action 工厂按名称解析动作实例。

执行流程：
    1. 装配期：内置动作类登记进 _BUILTIN_ACTIONS 注册表；自定义动作
       经 register_action 写入 _CUSTOM_ACTIONS（覆盖优先级更高）；
    2. 运行期：策略层选定动作名 → get_action 解析实例（自定义 >
       内置 > utter_ 模板约定）→ await run(tracker, domain) 得到
       ActionResult → 上层消费 responses/events 并推进对话状态。

副作用：
    - 各动作 run 可能修改 tracker（槽位/对话栈/重启），或经 LLM
      提取槽位（ActionExtractSlots，网络调用）；模块自身无 IO；
    - 模块级注册表 _BUILTIN_ACTIONS/_CUSTOM_ACTIONS 为全局可变状态，
      register_action 会跨会话持续生效。

主要依赖：
    - abc：ABC/abstractmethod 定义动作契约。
    - dataclasses：ActionResult 数据类。
    - typing：Any + TYPE_CHECKING（tracker/domain 仅注解用）。
    - app.ics_ai.core.tracker / core.domain：动作操作的状态与配置。
    - app.ics_ai.dialogue_understanding.stack.stack_frame：降级/闲聊/
      人工转接/搜索/完成等栈帧（run 内局部导入）。
    - app.ics_ai.shared.llm：槽位提取的 LLM 客户端（局部导入）。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，支持前向引用
# （如 TYPE_CHECKING 分支中的 tracker/domain 类型）。
from __future__ import annotations

# 导入抽象基类工具：ABC 标记动作基类，abstractmethod 强制子类实现
# name 与 run。
from abc import ABC, abstractmethod

# 导入 dataclass 工具：ActionResult 自动生成 __init__/__repr__；
# field(default_factory=...) 为可变默认值（list/dict）提供独立工厂。
from dataclasses import dataclass, field

# 导入类型注解工具：Any 类型通配符；TYPE_CHECKING 隔离仅注解导入。
from typing import TYPE_CHECKING, Any

# 类型检查期专属导入：这两个类型只出现在方法注解中，运行时不导入
# （避免与 core 包形成循环依赖）。
if TYPE_CHECKING:
    # 对话状态追踪器：动作读写状态的数据来源（仅注解用）。
    from app.ics_ai.core.tracker import DialogueStateTracker

    # 领域定义：ActionUtter 读取响应模板时使用（仅注解用）。
    from app.ics_ai.core.domain import Domain


@dataclass
class ActionResult:
    """动作执行结果。

    职责：承载单次动作运行的输出——待发送响应列表、产生的事件列表、
          成功标志与扩展元数据；提供 add_response/add_event 两个
          便捷构建方法。
    设计模式：数据传输对象（DTO）+ 建造者风格的增量填充方法。
    生命周期：每次动作 run 由动作内部创建并返回，被上层（Agent/
              消息处理器）消费后即可丢弃。

    Attributes:
        responses: 待发送响应列表；每项为 {"text": ...} 及附加键
                   （buttons/image/custom 等）的字典。
        events: 产生的事件列表；每项为 {"event": 事件类型} 及附加键
                的字典，供事件总线/日志消费。
        success: 是否执行成功；False 时上层可走错误处理路径。
        metadata: 额外元数据（自由扩展键值对）。
        reject_action_listen: 拒绝监听标志；True 表示动作要求打断当前
                              轮次监听（预留约定，上层暂未消费）。
    """

    # 响应列表：可变默认值经 field 工厂独立创建，避免实例间共享。
    responses: list[dict[str, Any]] = field(default_factory=list)
    # 事件列表：同上，独立工厂创建。
    events: list[dict[str, Any]] = field(default_factory=list)
    # 成功标志：默认 True（动作正常完成）。
    success: bool = True
    # 元数据字典：可变默认值经 field 工厂独立创建。
    metadata: dict[str, Any] = field(default_factory=dict)
    # 拒绝监听标志：True 表示动作要求打断当前轮次监听（预留约定，
    # 上层暂未消费）；带默认值追加在末尾，不影响既有构造调用。
    reject_action_listen: bool = False

    def add_response(self, text: str, **kwargs: Any) -> None:
        """添加文本响应。

        业务流程：以 {"text": text} 为基底，用 kwargs 合并附加字段
                  （如 buttons/image），追加到 responses 列表。

        副作用：修改本实例的 responses 字段。

        参数：
            text: 响应文本内容。
            **kwargs: 附加字段（buttons/image/custom 等），与 text
                      合并为单个响应字典。
        """
        # 构建基底响应字典。
        response = {"text": text}
        # 合并附加字段（kwargs 键覆盖同名的 "text" 需调用方自查）。
        response.update(kwargs)
        # 追加到响应列表。
        self.responses.append(response)

    def add_event(self, event_type: str, **kwargs: Any) -> None:
        """添加事件。

        业务流程：以 {"event": event_type} 为基底，用 kwargs 合并
                  附加字段（如 reason/flow_id），追加到 events 列表。

        副作用：修改本实例的 events 字段。

        参数：
            event_type: 事件类型名（如 "session_started"）。
            **kwargs: 事件附加字段。
        """
        # 构建基底事件字典。
        event = {"event": event_type}
        # 合并附加字段。
        event.update(kwargs)
        # 追加到事件列表。
        self.events.append(event)


class Action(ABC):
    """动作抽象基类。

    职责：定义动作的统一契约——name 抽象 property（动作名，即
          domain/flow 中引用的标识）与 run 抽象方法（执行具体逻辑、
          返回 ActionResult）；run_sync 提供同步适配层。
    设计模式：命令模式——每个动作封装一次对话操作，由策略层按名称
              选择执行；抽象基类 + 工厂（get_action）配合。
    生命周期：内置动作每次 get_action 时新建实例（无状态，创建
              成本低）；自定义动作经 register_action 注册单例复用。

    说明：动作是对话系统执行具体操作的单元——策略（Policy）负责
          "决定做什么"，动作（Action）负责"真正去做"。
    """

    # property + abstractmethod 组合：子类必须以 property 形式提供名称。
    @property
    @abstractmethod
    def name(self) -> str:
        """动作名称。

        返回：
            动作名字符串（如 "action_listen"）；由子类实现。
        """
        # 兜底保护：绕过抽象检查时给出明确报错。
        raise NotImplementedError()

    @abstractmethod
    async def run(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """执行动作。

        业务流程：由子类实现——读写 tracker 状态、产生响应与事件；
                  本基类不提供默认行为。

        副作用：由子类决定（可能修改槽位/对话栈，或调用外部服务）。

        参数：
            tracker: 对话状态追踪器，必需。
            domain: Domain 定义，允许 None（ActionUtter 等需要）。
            **kwargs: 额外参数（如降级原因、目标流程名）。

        返回：
            ActionResult 执行结果；子类未覆盖时抛 NotImplementedError。
        """
        # 兜底保护：子类未覆盖时抛出明确错误。
        raise NotImplementedError()

    def run_sync(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """同步版本的 run 方法。

        业务流程：获取/复用当前线程事件循环 → run_until_complete
                  阻塞驱动异步 run。

        副作用：事件循环缺失时新建并安装到当前线程；驱动的 run
                实现可能产生实际业务副作用。

        参数/返回：同 run。

        说明：仅适合同步上下文调用；已有运行中循环时会抛 RuntimeError。
        """
        # 局部导入：仅同步适配路径需要 asyncio。
        import asyncio

        try:
            # 优先复用当前线程已有的事件循环。
            loop = asyncio.get_event_loop()
        # 当前线程无循环 → 新建并安装。
        except RuntimeError:
            # 新建事件循环。
            loop = asyncio.new_event_loop()
            # 安装为当前线程循环，供后续复用。
            asyncio.set_event_loop(loop)

        # 阻塞驱动异步 run 至完成并取回结果。
        return loop.run_until_complete(self.run(tracker, domain, **kwargs))


# =============================================================================
# 内置动作
# =============================================================================


class ActionListen(Action):
    """监听动作。

    职责：等待用户输入——不执行任何操作，仅表示"控制权交还用户"。
    设计模式：空对象（Null Object）变体——run 返回空 ActionResult。
    生命周期：get_action 每次新建实例（无状态）。
    """

    # 命名 property：动作名与注册表键一致。
    @property
    def name(self) -> str:
        """动作名称。

        返回：
            固定字符串 "action_listen"。
        """
        # 固定名称。
        return "action_listen"

    async def run(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """等待用户输入。

        业务流程：直接返回空结果——无响应、无事件、无状态修改。

        副作用：无。

        返回：
            空的 ActionResult。
        """
        # 空结果：仅表示"等待用户"这一状态。
        return ActionResult()


class ActionRestart(Action):
    """重启动作。

    职责：重置对话状态（槽位/对话栈经 tracker.restart() 清空）。
    设计模式：命令模式的具体动作；状态修改委托给 tracker。
    生命周期：get_action 每次新建实例。
    """

    # 命名 property：动作名与注册表键一致。
    @property
    def name(self) -> str:
        """动作名称。

        返回：
            固定字符串 "action_restart"。
        """
        # 固定名称。
        return "action_restart"

    async def run(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """重启对话。

        业务流程：1. 调用 tracker.restart() 重置全部状态；
                  2. 记录 conversation_restarted 事件。

        副作用：清空 tracker 的对话状态（槽位/对话栈等）。

        返回：
            含重启事件的 ActionResult。
        """
        # 重置 tracker：槽位恢复初始值、对话栈清空。
        tracker.restart()

        # 构建结果并记录重启事件。
        result = ActionResult()
        # 事件：供日志/前端感知会话已重启。
        result.add_event("conversation_restarted")
        return result


class ActionSessionStart(Action):
    """会话开始动作。

    职责：初始化新会话——重置对话状态并记录会话启动事件。
    设计模式：命令模式的具体动作；与 ActionRestart 类似但语义
              为"新会话起点"而非"用户主动重启"。
    生命周期：get_action 每次新建实例。
    """

    # 命名 property：动作名与注册表键一致。
    @property
    def name(self) -> str:
        """动作名称。

        返回：
            固定字符串 "action_session_start"。
        """
        # 固定名称。
        return "action_session_start"

    async def run(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """开始新会话。

        业务流程：1. 构建 ActionResult；2. 调用 tracker.restart()
                  重置状态；3. 记录 session_started 事件。

        副作用：清空 tracker 的对话状态。

        返回：
            含会话启动事件的 ActionResult。
        """
        # 构建结果对象。
        result = ActionResult()

        # 重置 tracker 状态（新会话从干净状态开始）。
        tracker.restart()

        # 事件：标记会话启动。
        result.add_event("session_started")
        return result


class ActionDefaultFallback(Action):
    """默认降级动作。

    职责：压入 CannotHandleStackFrame，标记当前无法处理；
          实际降级响应由 EnterpriseSearchPolicy 检测该栈帧后生成。
    设计模式：命令模式 + 栈帧驱动的关注点分离——动作只负责
              "标记状态"，响应内容由策略层生成。
    生命周期：get_action 每次新建实例。
    """

    # 命名 property：动作名与注册表键一致。
    @property
    def name(self) -> str:
        """动作名称。

        返回：
            固定字符串 "action_default_fallback"。
        """
        # 固定名称。
        return "action_default_fallback"

    async def run(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """压入 CannotHandleStackFrame 触发降级。

        业务流程：1. 从 kwargs 取降级原因（默认"无法理解用户意图"）；
                  2. 局部导入栈帧类并压入对话栈；3. 记录降级事件。

        副作用：向 tracker.dialogue_stack 压入新帧（修改对话栈）。

        参数：
            **kwargs: reason——降级原因字符串，可选。

        返回：
            含 cannot_handle_triggered 事件（附 reason）的 ActionResult。
        """
        # 构建结果对象。
        result = ActionResult()

        # 获取降级原因：kwargs 可选传入，缺省用通用文案。
        reason = kwargs.get("reason", "无法理解用户意图")

        # 局部导入栈帧类：避免模块级循环依赖。
        from app.ics_ai.dialogue_understanding.stack.stack_frame import (
            CannotHandleStackFrame,
        )

        # 压入"无法处理"帧：策略层据此生成降级响应。
        tracker.dialogue_stack.push(CannotHandleStackFrame(reason=reason))
        # 事件：携带原因，便于前端/日志追溯。
        result.add_event("cannot_handle_triggered", reason=reason)

        return result


class ActionChitChatResponse(Action):
    """闲聊响应动作。

    职责：压入 ChitChatStackFrame，标记当前处于闲聊模式；
          实际闲聊回复由 EnterpriseSearchPolicy 生成。
    设计模式：同 ActionDefaultFallback——动作标记状态，策略生成内容。
    生命周期：get_action 每次新建实例。
    """

    # 命名 property：动作名与注册表键一致。
    @property
    def name(self) -> str:
        """动作名称。

        返回：
            固定字符串 "action_chitchat_response"。
        """
        # 固定名称。
        return "action_chitchat_response"

    async def run(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """压入 ChitChatStackFrame 触发闲聊。

        业务流程：1. 构建 ActionResult；2. 局部导入并压入闲聊帧；
                  3. 记录 chitchat_triggered 事件。

        副作用：向对话栈压入闲聊帧。

        返回：
            含 chitchat_triggered 事件的 ActionResult。
        """
        # 构建结果对象。
        result = ActionResult()

        # 局部导入闲聊栈帧类：避免模块级循环依赖。
        from app.ics_ai.dialogue_understanding.stack.stack_frame import (
            ChitChatStackFrame,
        )

        # 压入闲聊帧：标记闲聊模式。
        tracker.dialogue_stack.push(ChitChatStackFrame())
        # 事件：标记闲聊触发。
        result.add_event("chitchat_triggered")

        return result


class ActionCancelFlow(Action):
    """取消流程动作。

    职责：结束当前活动的 Flow（经 tracker.end_flow() 操作对话栈），
          并向用户回执取消结果。
    设计模式：命令模式的具体动作；流程状态修改委托给 tracker。
    生命周期：get_action 每次新建实例。
    """

    # 命名 property：动作名与注册表键一致。
    @property
    def name(self) -> str:
        """动作名称。

        返回：
            固定字符串 "action_cancel_flow"。
        """
        # 固定名称。
        return "action_cancel_flow"

    async def run(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """取消当前流程。

        业务流程：1. 构建 ActionResult；2. 有活动 Flow 时记录 ID、
                  结束流程、回执确认文案；3. 无活动 Flow 时回执
                  提示文案。

        副作用：有活动 Flow 时修改对话栈（弹出当前 Flow 帧）。

        返回：
            含取消事件与用户回执文案的 ActionResult。
        """
        # 构建结果对象。
        result = ActionResult()

        # 检查是否有活动的 Flow 并结束。
        if tracker.active_flow:
            # 先记录流程 ID（end_flow 后不可再取）。
            flow_id = tracker.active_flow
            # 结束当前 Flow（弹出栈顶 Flow 帧）。
            tracker.end_flow()
            # 事件：携带被取消的流程 ID。
            result.add_event("flow_cancelled", flow_id=flow_id)
            # 用户回执：确认已取消。
            result.add_response("好的，已取消当前操作。")
        else:
            # 无活动流程：仅回执提示，不发事件。
            result.add_response("当前没有进行中的流程。")

        return result


class ActionChangeFlow(Action):
    """切换流程动作。

    职责：结束当前 Flow 并启动新 Flow——经 tracker 方法操作对话栈。
    设计模式：命令模式的具体动作；组合"取消 + 启动"两步状态变更。
    生命周期：get_action 每次新建实例。
    """

    # 命名 property：动作名与注册表键一致。
    @property
    def name(self) -> str:
        """动作名称。

        返回：
            固定字符串 "action_change_flow"。
        """
        # 固定名称。
        return "action_change_flow"

    async def run(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """切换流程：结束旧流程，启动新流程。

        业务流程：1. 从 kwargs 取 target_flow；2. 未指定时回执错误
                  文案并返回；3. 有活动 Flow 时先结束并记录事件；
                  4. 启动新 Flow 并记录事件。

        副作用：修改对话栈（弹出旧 Flow 帧、压入新 Flow 帧）。

        参数：
            **kwargs: target_flow——目标流程 ID，必需。

        返回：
            含切换事件（或错误回执）的 ActionResult。
        """
        # 构建结果对象。
        result = ActionResult()

        # 目标流程 ID：kwargs 必需项，缺省空串。
        target_flow = kwargs.get("target_flow", "")

        # 参数校验：未指定目标流程 → 回执错误并终止（无状态变更）。
        if not target_flow:
            # 提示调用方补全参数。
            result.add_response("无法切换流程：未指定目标流程。")
            # 提前返回。
            return result

        # 结束当前 Flow（如果有）：切换前先清理旧流程。
        if tracker.active_flow:
            # 记录旧流程 ID 供事件使用。
            old_flow_id = tracker.active_flow
            # 结束旧流程。
            tracker.end_flow()
            # 事件：旧流程被取消。
            result.add_event("flow_cancelled", flow_id=old_flow_id)

        # 启动新 Flow：压入新流程帧。
        tracker.start_flow(target_flow)
        # 事件：新流程已启动。
        result.add_event("flow_started", flow_id=target_flow)

        return result


class ActionCleanStack(Action):
    """清理对话栈动作。

    职责：清空 dialogue_stack 中的所有帧（经 tracker.cancel_flow()
          内部调用 dialogue_stack.clear()）。
    设计模式：命令模式的具体动作；清栈逻辑委托给 tracker。
    生命周期：get_action 每次新建实例。
    """

    # 命名 property：动作名与注册表键一致。
    @property
    def name(self) -> str:
        """动作名称。

        返回：
            固定字符串 "action_clean_stack"。
        """
        # 固定名称。
        return "action_clean_stack"

    async def run(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """清理对话栈。

        业务流程：1. 清空前统计帧数（供事件报告）；2. 调用
                  tracker.cancel_flow() 清空对话栈；3. 记录清理事件。

        副作用：清空对话栈（全部帧被移除）。

        返回：
            含 stack_cleaned 事件（附清理帧数）的 ActionResult。
        """
        # 构建结果对象。
        result = ActionResult()

        # 清空前统计当前帧数：事件中报告清理规模。
        cleared_count = len(tracker.dialogue_stack.frames)

        # 清空对话栈（cancel_flow 内部调用 dialogue_stack.clear()）。
        tracker.cancel_flow()

        # 事件：携带被清理的帧数。
        result.add_event("stack_cleaned", frames_cleared=cleared_count)
        return result


class ActionExtractSlots(Action):
    """提取槽位动作。

    职责：从用户最新消息中提取指定槽位值（简化版：经 LLM 提取），
          提取成功则写入 tracker 槽位并记录 slot_set 事件。
    设计模式：命令模式的具体动作；LLM 调用失败时静默跳过
              （容错降级——槽位留空，后续 collect 步骤可再问）。
    生命周期：get_action 每次新建实例。
    """

    # 命名 property：动作名与注册表键一致。
    @property
    def name(self) -> str:
        """动作名称。

        返回：
            固定字符串 "action_extract_slots"。
        """
        # 固定名称。
        return "action_extract_slots"

    async def run(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """从消息中提取槽位。

        业务流程：1. 从 kwargs 取待提取槽位列表、从 tracker 取用户
                  消息；2. 任一缺失直接返回空结果；3. 局部导入 LLM
                  工厂并按 llm_config 创建客户端（temperature=0 保证
                  提取稳定）；4. 构建提取提示词并调用 LLM；5. 解析
                  JSON（兼容 ```json 代码块包裹）；6. 非 None 值写入
                  tracker 槽位并记录事件；解析/调用失败静默跳过。

        副作用：写入 tracker 槽位；调用外部 LLM 服务（网络请求）。

        参数：
            **kwargs: slots_to_extract——槽位名列表，必需；
                      llm_config——LLM 配置字典（type/model/api_key/
                      api_base），api_key 缺失时不提取。

        返回：
            含 slot_set 事件（每成功槽位一条）的 ActionResult；
            失败/参数缺失时返回空结果（不抛异常）。
        """
        # 构建结果对象。
        result = ActionResult()

        # 获取需要提取的槽位：kwargs 传入的槽位名列表。
        slots_to_extract = kwargs.get("slots_to_extract", [])
        # 用户消息：默认空串，从 tracker 最新消息取。
        user_message = ""
        # latest_message 存在时才取文本（新会话可能尚无消息）。
        if tracker.latest_message:
            # 取最新消息文本作为提取源。
            user_message = tracker.latest_message.text

        # 参数守卫：无消息或无待提取槽位 → 直接返回空结果。
        if not user_message or not slots_to_extract:
            # 空结果：不做任何提取。
            return result

        # 尝试使用 LLM 提取槽位：整体 try 包裹，任何失败静默跳过。
        try:
            # 局部导入 LLM 工厂：避免模块级循环依赖。
            from app.ics_ai.shared.llm import create_llm_client

            # LLM 配置：kwargs 可选传入。
            llm_config = kwargs.get("llm_config", {})
            # api_key 缺失时无法调用 → 跳过提取。
            if llm_config.get("api_key"):
                # 创建客户端：temperature=0 保证提取结果确定性。
                client = create_llm_client(
                    # 提供商类型：默认 openai（兼容协议）。
                    type=llm_config.get("type", "openai"),
                    # 模型名：默认 gpt-4o-mini（轻量提取够用）。
                    model=llm_config.get("model", "gpt-4o-mini"),
                    # 密钥：上面已确认存在。
                    api_key=llm_config["api_key"],
                    # 自定义端点：可选（兼容服务等）。
                    api_base=llm_config.get("api_base"),
                    # 温度 0：提取任务要求稳定输出。
                    temperature=0.0,
                )

                # 构建提取提示词：逐槽位列出字段名。
                slot_descriptions = []
                # 遍历槽位名生成描述行。
                for slot_name in slots_to_extract:
                    # 每行格式 "- 槽位名"。
                    slot_descriptions.append(f"- {slot_name}")

                # 提取提示词：要求 JSON 输出、无法提取置 null。
                prompt = f"""从以下用户消息中提取指定的信息。

用户消息: {user_message}

需要提取的字段:
{chr(10).join(slot_descriptions)}

请以JSON格式返回提取结果，如果无法提取某个字段，值设为null。
只返回JSON，不要其他内容。"""

                # 组装消息列表：单条 user 消息。
                messages = [{"role": "user", "content": prompt}]
                # 调用 LLM：await 异步完成。
                response = await client.complete(messages)

                # 响应非空且含内容时才解析。
                if response and response.content:
                    # 局部导入 json：仅本分支需要。
                    import json

                    try:
                        # 尝试解析 JSON（兼容 ```json 代码块包裹）。
                        # 去除首尾空白。
                        content = response.content.strip()
                        # 以 ``` 开头 → 剥离代码块围栏。
                        if content.startswith("```"):
                            # 取围栏内内容（split 第二段）。
                            content = content.split("```")[1]
                            # 剥离语言标记 "json" 前缀。
                            if content.startswith("json"):
                                # 去掉前 4 个字符。
                                content = content[4:]

                        # 解析 JSON 为字典。
                        extracted = json.loads(content)

                        # 设置提取到的槽位：仅写非 None 值。
                        for slot_name, slot_value in extracted.items():
                            # None 表示 LLM 判定无法提取 → 跳过。
                            if slot_value is not None:
                                # 写入 tracker 槽位。
                                tracker.set_slot(slot_name, slot_value)
                                # 事件：记录槽位名与新值。
                                result.add_event(
                                    "slot_set",
                                    name=slot_name,
                                    value=slot_value,
                                )
                    # JSON 解析失败：静默跳过（槽位留空）。
                    except json.JSONDecodeError:
                        # 不抛异常，保持链路继续。
                        pass
        # LLM 调用/导入等任何异常：静默跳过提取。
        except Exception:
            # 容错降级：槽位留空，后续 collect 步骤可再询问用户。
            pass

        return result


class ActionSendText(Action):
    """发送文本动作。

    职责：直接发送调用方指定的文本——文本经 kwargs 传入而非读取
          模板，适合作运行时动态拼装的响应出口。
    设计模式：命令模式的具体动作；无状态、纯转发。
    生命周期：get_action 每次新建实例（无状态）。
    """

    # 命名 property：动作名与注册表键一致。
    @property
    def name(self) -> str:
        """动作名称。

        返回：
            固定字符串 "action_send_text"。
        """
        # 固定名称。
        return "action_send_text"

    async def run(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """发送指定文本。

        业务流程：1. 构建 ActionResult；2. 从 kwargs 取 text；
                  3. text 非空时追加为响应。

        副作用：无（不修改 tracker 状态）。

        参数：
            tracker: 对话状态追踪器（本动作不使用）。
            domain: Domain 定义，允许 None（本动作不使用）。
            **kwargs: text——待发送文本，缺省空串（空串时不发送）。

        返回：
            ActionResult；text 为空时 responses 为空列表。
        """
        # 构建结果对象。
        result = ActionResult()

        # 从 kwargs 取待发送文本；缺省空串。
        text = kwargs.get("text", "")
        # 空文本跳过：避免发送空消息。
        if text:
            # 追加文本响应。
            result.add_response(text)

        return result


class ActionHandleHelp(Action):
    """处理帮助请求动作。

    职责：响应用户的帮助请求，回复固定格式的通用帮助信息
          （列出系统能力清单并引导用户继续说明需求）。
    设计模式：命令模式的具体动作；文案硬编码、无状态。
    生命周期：get_action 每次新建实例（无状态）。
    """

    # 命名 property：动作名与注册表键一致。
    @property
    def name(self) -> str:
        """动作名称。

        返回：
            固定字符串 "action_handle_help"。
        """
        # 固定名称。
        return "action_handle_help"

    async def run(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """回复通用帮助信息。

        业务流程：1. 构建 ActionResult；2. 逐行拼接能力清单文案；
                  3. 追加为响应。

        副作用：无（不修改 tracker 状态）。

        参数：
            tracker: 对话状态追踪器（本动作不使用）。
            domain: Domain 定义，允许 None（本动作不使用）。
            **kwargs: 额外参数（本动作不使用）。

        返回：
            含一条帮助文案响应的 ActionResult。
        """
        # 构建结果对象。
        result = ActionResult()

        # 拼接帮助文案：标题行 + 能力清单（固定内容，运行时不读取
        # 用户消息做个性化——原注释中"针对性帮助"并未实现）。
        help_text = "我可以帮助您完成以下任务：\n"
        # 能力清单第 1 项：回答问题。
        help_text += "1. 回答问题\n"
        # 能力清单第 2 项：信息查询。
        help_text += "2. 提供信息查询\n"
        # 能力清单第 3 项：日常对话。
        help_text += "3. 进行日常对话\n"
        # 尾部引导语：促使用户继续输入具体需求。
        help_text += "\n请告诉我您具体需要什么帮助？"

        # 追加帮助文案为响应。
        result.add_response(help_text)
        return result


class ActionClarify(Action):
    """澄清动作。

    职责：系统无法理解用户意图或信息不足时，向用户发出澄清提问；
          支持自定义问题文案与编号选项列表，并记录澄清事件。
    设计模式：命令模式的具体动作；文案可经 kwargs 定制，缺省走
              兜底文案。
    生命周期：get_action 每次新建实例（无状态）。
    """

    # 命名 property：动作名与注册表键一致。
    @property
    def name(self) -> str:
        """动作名称。

        返回：
            固定字符串 "action_clarify"。
        """
        # 固定名称。
        return "action_clarify"

    async def run(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """请求用户澄清。

        业务流程：1. 构建 ActionResult；2. 从 kwargs 取 question 与
                  options；3. 拼接澄清文案（有选项时追加编号列表）；
                  4. 追加响应与 clarification_requested 事件。

        副作用：无（不修改 tracker 状态；事件由上层消费）。

        参数：
            tracker: 对话状态追踪器（本动作不使用）。
            domain: Domain 定义，允许 None（本动作不使用）。
            **kwargs: question——自定义澄清问题，缺省空串走默认文案；
                      options——选项列表（可迭代，缺省空列表）。

        返回：
            含澄清文案响应与澄清事件的 ActionResult。
        """
        # 构建结果对象。
        result = ActionResult()

        # 取自定义澄清问题；缺省空串。
        question = kwargs.get("question", "")
        # 取选项列表；缺省空列表。
        options = kwargs.get("options", [])

        # 有自定义问题 → 用之；否则走兜底默认文案。
        if question:
            # 使用调用方提供的问题。
            clarify_text = question
        else:
            # 兜底文案：礼貌请求用户重新表述。
            clarify_text = "抱歉，我不太确定您的意思。能否请您再说明一下？"

        # 有选项时：追加编号选项提示，降低用户输入成本。
        if options:
            # 选项区标题（与问题文案间空一行）。
            clarify_text += "\n\n您可以选择："
            # enumerate 从 1 起编号，逐项追加"序号. 选项"行。
            for i, opt in enumerate(options, 1):
                # 追加单个选项行。
                clarify_text += f"\n{i}. {opt}"

        # 追加澄清文案为响应。
        result.add_response(clarify_text)
        # 事件：记录本次澄清的问题与选项，供日志/策略参考。
        result.add_event("clarification_requested", question=question, options=options)

        return result


class ActionHumanHandoff(Action):
    """人工转接动作。

    职责：系统无法处理用户请求或用户明确要求人工时，压入
          HumanHandoffStackFrame 标记转接意图——本动作只负责压栈，
          转接话术由 EnterpriseSearchPolicy 检测栈帧后生成
          （栈帧驱动关注点分离：动作管状态变更，策略管响应文案）。
    设计模式：命令模式的具体动作；压栈型，与 Search/Completed 同族。
    生命周期：get_action 每次新建实例（无状态）。
    """

    # 命名 property：动作名与注册表键一致。
    @property
    def name(self) -> str:
        """动作名称。

        返回：
            固定字符串 "action_human_handoff"。
        """
        # 固定名称。
        return "action_human_handoff"

    async def run(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """压入 HumanHandoffStackFrame 触发人工转接。

        业务流程：1. 构建 ActionResult；2. 从 kwargs 取转接原因；
                  3. 局部导入栈帧类型并压入对话栈；4. 记录转接事件。

        副作用：修改 tracker 的 dialogue_stack（压入转接栈帧）。

        参数：
            tracker: 对话状态追踪器（读写其对话栈）。
            domain: Domain 定义，允许 None（本动作不使用）。
            **kwargs: reason——转接原因描述，缺省空串。

        返回：
            含转接事件的 ActionResult。
        """
        # 构建结果对象。
        result = ActionResult()

        # 取转接原因；缺省空串（策略侧可对空原因做兜底文案）。
        reason = kwargs.get("reason", "")

        # 局部导入栈帧类型：与类型检查期导入隔离，避免循环依赖。
        from app.ics_ai.dialogue_understanding.stack.stack_frame import (
            HumanHandoffStackFrame,
        )

        # 压入转接栈帧：EnterpriseSearchPolicy 据此生成转接响应。
        tracker.dialogue_stack.push(HumanHandoffStackFrame(reason=reason))
        # 事件：记录转接触发及原因，供日志/前端感知。
        result.add_event("human_handoff_triggered", reason=reason)

        return result


class ActionTriggerSearch(Action):
    """触发搜索动作。

    职责：压入空的 SearchStackFrame 标记当前进入搜索模式——
          不在栈帧内存储 query，实际检索时由策略从
          tracker.latest_message 取最新用户消息；检索执行同样由
          EnterpriseSearchPolicy 完成（动作管状态，策略管响应）。
    设计模式：命令模式的具体动作；空栈帧 + 前置校验的轻量设计。
    生命周期：get_action 每次新建实例（无状态）。
    """

    # 命名 property：动作名与注册表键一致。
    @property
    def name(self) -> str:
        """动作名称。

        返回：
            固定字符串 "action_trigger_search"。
        """
        # 固定名称。
        return "action_trigger_search"

    async def run(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """压入 SearchStackFrame 触发搜索。

        业务流程：1. 构建 ActionResult；2. 校验 tracker 有用户消息，
                  缺失时回复引导文案并中止；3. 局部导入栈帧类型并
                  压入空 SearchStackFrame；4. 记录搜索触发事件。

        副作用：校验通过时修改 tracker 的 dialogue_stack（压入
                搜索栈帧）。

        参数：
            tracker: 对话状态追踪器（读最新消息、写对话栈）。
            domain: Domain 定义，允许 None（本动作不使用）。
            **kwargs: 额外参数（本动作不使用）。

        返回：
            含搜索事件的 ActionResult；无用户消息时为含引导文案、
            无事件的结果。
        """
        # 构建结果对象。
        result = ActionResult()

        # 前置校验：latest_message 及其 text 均需存在——两个条件
        # 用 or 短路连接，任一缺失即视为"没有可检索的问题"。
        if not tracker.latest_message or not tracker.latest_message.text:
            # 引导用户先输入问题，不压栈（避免空搜索）。
            result.add_response("请提供您的问题。")
            return result

        # 局部导入栈帧类型：与类型检查期导入隔离，避免循环依赖。
        from app.ics_ai.dialogue_understanding.stack.stack_frame import SearchStackFrame

        # 压入空栈帧：query 不入库，检索时从 latest_message 现取。
        tracker.dialogue_stack.push(SearchStackFrame())
        # 事件：标记搜索模式已触发。
        result.add_event("search_triggered")

        return result


class ActionFlowCompleted(Action):
    """Flow 完成动作。

    职责：Flow 执行完毕后压入 CompletedStackFrame（记录刚完成的
          流程名）——策略检测到该栈帧后生成"是否还有其他需求"的
          续接话术；本动作只压栈，不产出响应（动作管状态，
          策略管响应）。
    设计模式：命令模式的具体动作；压栈型，与 HumanHandoff/Search
              同族。
    生命周期：get_action 每次新建实例（无状态）。
    """

    # 命名 property：动作名与注册表键一致。
    @property
    def name(self) -> str:
        """动作名称。

        返回：
            固定字符串 "action_flow_completed"。
        """
        # 固定名称。
        return "action_flow_completed"

    async def run(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """压入 CompletedStackFrame 标记 Flow 完成。

        业务流程：1. 构建 ActionResult；2. 从 kwargs 取刚完成的
                  流程名；3. 局部导入栈帧类型并压入完成栈帧；
                  4. 记录完成事件。

        副作用：修改 tracker 的 dialogue_stack（压入完成栈帧）。

        参数：
            tracker: 对话状态追踪器（写对话栈）。
            domain: Domain 定义，允许 None（本动作不使用）。
            **kwargs: completed_flow——刚完成流程的名称/ID，缺省空串。

        返回：
            含流程完成事件的 ActionResult。
        """
        # 构建结果对象。
        result = ActionResult()

        # 取刚完成的流程名；缺省空串（栈帧侧允许空值）。
        completed_flow = kwargs.get("completed_flow", "")

        # 局部导入栈帧类型：与类型检查期导入隔离，避免循环依赖。
        from app.ics_ai.dialogue_understanding.stack.stack_frame import (
            CompletedStackFrame,
        )

        # 压入完成栈帧：携带 previous_flow_name 供策略生成续接话术。
        tracker.dialogue_stack.push(
            CompletedStackFrame(previous_flow_name=completed_flow)
        )
        # 事件：记录流程完成（flow_id 即刚完成的流程名）。
        result.add_event("flow_completed_handled", flow_id=completed_flow)

        return result


class ActionUtter(Action):
    """模板响应动作。

    职责：按模板名从 domain 读取响应模板组，随机挑选一条并发送；
          文本中的 {slot_name} 占位符用 tracker 当前槽位值替换，
          模板的 buttons/image/custom 一并透传。
    设计模式：命令模式 + 工厂约定——不以类注册，get_action 对
              "utter_" 前缀动作名按需构造本类实例。
    生命周期：get_action 解析 utter_ 动作时按模板名新建实例
              （实例有状态：_utter_name）。
    """

    def __init__(self, utter_name: str):
        """初始化模板响应动作。

        业务流程：仅保存模板名，供 name 与 run 使用。

        副作用：仅写入实例属性，无 IO 操作。

        参数：
            utter_name: 响应模板名称（约定以 "utter_" 开头，
                        如 "utter_greet"），不能为空。
        """
        # 保存模板名：name property 直接返回它（动作名=模板名）。
        self._utter_name = utter_name

    # 命名 property：utter 动作名即模板名。
    @property
    def name(self) -> str:
        """动作名称。

        返回：
            构造时传入的响应模板名（如 "utter_greet"）。
        """
        # 直接返回模板名：动作名与注册/引用名保持一致。
        return self._utter_name

    async def run(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """执行模板响应。

        业务流程：1. 构建 ActionResult；2. domain 存在时按模板名取
                  响应模板组；3. 有模板则 random.choice 随机选一条；
                  4. 用 tracker 槽位值替换文本占位符；5. 连同
                  buttons/image/custom 追加为响应。

        副作用：无（只读 tracker 槽位与 domain 模板）。

        参数：
            tracker: 对话状态追踪器（读全部槽位做占位符替换）。
            domain: Domain 定义；None 或取不到模板时不产出响应。
            **kwargs: 额外参数（本动作不使用）。

        返回：
            ActionResult；domain 缺失/无模板时 responses 为空列表。
        """
        # 构建结果对象。
        result = ActionResult()

        # 前置条件：domain 缺失则无法查询模板，直接返回空结果。
        if domain:
            # 按模板名取响应模板组（返回 ResponseTemplate 列表）。
            responses = domain.get_response(self._utter_name)
            # 模板组非空才继续（模板名可能未在 domain 中配置）。
            if responses:
                # 局部导入 random：仅模板随机选择路径需要。
                import random

                # 随机选一条：同模板多变体（多话术）场景下的随机化。
                response = random.choice(responses)

                # 取模板文本，准备做槽位占位符替换。
                text = response.text
                # 文本非空才替换（空文本直接透传兜底值）。
                if text:
                    # 取 tracker 全部槽位（名称 → 值的字典）。
                    all_slots = tracker.get_all_slots()
                    # 逐槽位替换 "{槽位名}" 占位符；None/空值替换为
                    # 空串，避免文本中出现 "None" 字样。
                    for slot_name, slot_value in all_slots.items():
                        text = text.replace(f"{{{slot_name}}}", str(slot_value or ""))

                # 追加响应：文本兜底空串，buttons 兜底空列表，
                # image/custom 原样透传（允许 None）。
                result.add_response(
                    text or "",
                    buttons=response.buttons or [],
                    image=response.image,
                    custom=response.custom,
                )

        return result


# =============================================================================
# Action 注册和工厂
# =============================================================================

# 内置动作注册表：动作名 → 动作类。值为类而非实例，get_action 每次
# 调用按需实例化（内置动作均无状态，实例化成本可忽略）。模块级全局
# 可变状态：装配期构建后仅读取，正常运行期不被修改。
# 注意：注册表未包含 ActionUtter（它按 utter_ 前缀约定动态构造）。
_BUILTIN_ACTIONS: dict[str, type] = {
    # 列表项 1：监听动作——控制权交还用户。
    "action_listen": ActionListen,
    # 列表项 2：重启动作——重置全部对话状态。
    "action_restart": ActionRestart,
    # 列表项 3：会话开始动作——新会话的初始化。
    "action_session_start": ActionSessionStart,
    # 列表项 4：默认降级动作——无法处理时压降级栈帧。
    "action_default_fallback": ActionDefaultFallback,
    # 列表项 5：闲聊响应动作——压闲聊栈帧由策略出话术。
    "action_chitchat_response": ActionChitChatResponse,
    # 列表项 6：取消流程动作——压 CancelFlow 栈帧。
    "action_cancel_flow": ActionCancelFlow,
    # 列表项 7：切换流程动作——压 ChangeFlow 栈帧。
    "action_change_flow": ActionChangeFlow,
    # 列表项 8：清栈动作——清空对话栈。
    "action_clean_stack": ActionCleanStack,
    # 列表项 9：槽位提取动作——经 LLM 抽取槽位。
    "action_extract_slots": ActionExtractSlots,
    # 列表项 10：发送文本动作——直接发送 kwargs 中的文本。
    "action_send_text": ActionSendText,
    # 列表项 11：帮助响应动作——回复固定能力清单。
    "action_handle_help": ActionHandleHelp,
    # 列表项 12：澄清动作——请求用户澄清意图。
    "action_clarify": ActionClarify,
    # 列表项 13：人工转接动作——压人工转接栈帧。
    "action_human_handoff": ActionHumanHandoff,
    # 列表项 14：流程完成动作——压完成栈帧。
    "action_flow_completed": ActionFlowCompleted,
    # 列表项 15：搜索触发动作——压空搜索栈帧。
    "action_trigger_search": ActionTriggerSearch,
}

# 自定义动作注册表：动作名 → 动作实例（与内置表不同，值为实例——
# 自定义动作可能携带状态，注册什么实例就复用什么实例）。查询优先级
# 高于内置表，可借此覆盖内置动作的行为。
_CUSTOM_ACTIONS: dict[str, Action] = {}


def register_action(action: Action) -> None:
    """注册自定义动作。

    业务流程：以动作实例的 name 为键写入 _CUSTOM_ACTIONS；
              同名重复注册时后者覆盖前者。

    副作用：修改模块级注册表 _CUSTOM_ACTIONS（全局可变状态，
            跨会话持续生效；键与内置表重复时覆盖内置动作）。

    参数：
        action: 已实例化的动作对象；注册键直接取 action.name，
                与 get_action 的查询名保持一致。
    """
    # 以动作自身 name 为键注册：保证注册名与查询名天然一致。
    _CUSTOM_ACTIONS[action.name] = action


def get_action(name: str) -> Action | None:
    """按名称解析动作实例。

    业务流程：1. 先查自定义注册表（覆盖优先级最高）；2. 再查内置
              注册表并即时实例化；3. 都未命中且名称以 "utter_" 开头
              时按约定构造 ActionUtter；4. 全部未命中返回 None。

    副作用：内置动作路径会新建实例（无其他副作用）。

    参数：
        name: 动作名称（domain/flow 中引用的标识），如
              "action_listen"、"utter_greet"。

    返回：
        Action 实例；未知名称返回 None（调用方需处理空值）。
    """
    # 第一步：自定义动作优先——允许业务侧覆盖内置行为。
    if name in _CUSTOM_ACTIONS:
        # 直接返回注册的实例（不新建，保留自定义状态）。
        return _CUSTOM_ACTIONS[name]

    # 第二步：内置动作——按类即时实例化（内置动作均无状态）。
    if name in _BUILTIN_ACTIONS:
        # 类调用产生新实例。
        return _BUILTIN_ACTIONS[name]()

    # 第三步：utter_ 前缀约定——模板响应动作不入注册表，按名构造。
    if name.startswith("utter_"):
        # 以动作名作为模板名构造 ActionUtter。
        return ActionUtter(name)

    # 兜底：无法解析时返回 None，由调用方决定降级路径。
    return None


def get_all_action_names() -> list[str]:
    """获取所有已注册的动作名称。

    业务流程：内置表键列表在前、自定义表键列表在后合并返回；
              不含 utter_ 约定动作（它们不在任何注册表中）。

    副作用：无。

    返回：
        动作名字符串列表（自定义与内置同名时会出现重复项）。
    """
    # 取内置表全部键（列表化，得到独立副本）。
    names = list(_BUILTIN_ACTIONS.keys())
    # 追加自定义表全部键（同名时靠后，但查询时优先级更高）。
    names.extend(_CUSTOM_ACTIONS.keys())
    # 返回合并后的名称列表。
    return names


# 导出清单：本模块对外暴露的公共 API（动作类 + 结果对象 + 工厂）。
__all__ = [
    # 动作基类与结果对象。
    "Action",
    "ActionResult",
    # 内置动作类（与 _BUILTIN_ACTIONS 注册表一一对应）。
    "ActionListen",
    "ActionRestart",
    "ActionSessionStart",
    "ActionDefaultFallback",
    "ActionChitChatResponse",
    "ActionCancelFlow",
    "ActionChangeFlow",
    "ActionCleanStack",
    "ActionExtractSlots",
    "ActionSendText",
    "ActionHandleHelp",
    "ActionClarify",
    "ActionHumanHandoff",
    "ActionFlowCompleted",
    "ActionTriggerSearch",
    # 模板响应动作（按 utter_ 前缀约定动态构造）。
    "ActionUtter",
    # 注册与工厂函数。
    "register_action",
    "get_action",
    "get_all_action_names",
]
