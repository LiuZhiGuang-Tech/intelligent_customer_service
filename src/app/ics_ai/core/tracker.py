"""
tracker - 对话状态追踪器（Tracker）

核心功能：定义单个会话的完整对话状态容器——消息数据类（UserMessage/
          BotMessage）、对话轮次（DialogueTurn）与对话状态追踪器
          DialogueStateTracker。Tracker 记录槽位值、对话历史、对话栈
          （活跃 Flow 等上下文的唯一状态源）与 Flow 执行历史，是策略
          决策、LLM 提示词组装与持久化恢复的数据基础。

执行流程：本模块被导入时定义三个数据类与追踪器类；运行时每条用户消息
          经 update_with_message 开启新轮次，命令/回复经 add_commands/
          add_bot_message 累积进当前轮次，轮次结束 finalize_turn 归档；
          FlowPolicy 经 start_flow/end_flow 操作对话栈；会话结束由
          TrackerStore 经 to_dict/from_dict 序列化往返。

副作用：仅操作内存对象（列表/字典/对话栈），时间戳取自系统时钟
        time.time()；不读写文件/数据库、不发起网络请求、不修改全局
        状态（持久化由 TrackerStore 完成）。

主要依赖：app.ics_ai.core.slots（Slot 槽位与工厂）、
          app.ics_ai.dialogue_understanding.stack.dialogue_stack
          （DialogueStack 对话栈）、app.ics_ai.shared.constants
          （ACTION_LISTEN/DEFAULT_SENDER_ID 常量）。
"""

# 导入 time：生成消息/轮次/Tracker 的时间戳（Unix 秒，浮点）。
import time

# 导入 deepcopy：from_dict 恢复槽位时深拷贝 Domain 模板，
# 使各会话槽位实例互不共享状态。
from copy import deepcopy

# 导入 dataclass 与 field：数据类定义；field(default_factory=...)
# 为可变默认值（字典/列表/时间戳）指定每次实例化时的新建函数，
# 避免默认值在实例间共享。
from dataclasses import dataclass, field

# 导入 Any：任意类型注解（typing 中未弃用的部分，无内置替代）；
# Dict/List/Optional 已按 PEP 585/604 迁移为 dict/list、X | None 写法。
from typing import Any

# 导入槽位类型与工厂：Slot 供类型注解与 from_dict 反序列化，
# create_slot 供 set_slot 动态建槽。
from app.ics_ai.core.slots import Slot, create_slot

# 导入对话栈：Tracker 内唯一的上下文状态源（Flow/搜索/闲聊等帧）。
from app.ics_ai.dialogue_understanding.stack.dialogue_stack import DialogueStack

# 导入常量：ACTION_LISTEN 监听动作（默认动作名）、
# DEFAULT_SENDER_ID 默认会话 ID（未指定发送者时的兜底）。
from app.ics_ai.shared.constants import (
    ACTION_LISTEN,
    DEFAULT_SENDER_ID,
)


@dataclass
class UserMessage:
    """用户消息。

    职责：封装用户发送的一条消息及其元数据（发送者/时间/通道）。
    设计模式：数据类（不可变语义，仅序列化往返）。
    生命周期：随用户消息到达创建，存入 DialogueTurn 后随 Tracker 持久化。

    属性：
        text: 消息文本
        sender_id: 发送者ID
        timestamp: 时间戳
        input_channel: 输入通道名称
        metadata: 额外元数据
    """

    # 消息正文：字符串，原样传给 LLM 与命令生成，无默认值（必填）。
    text: str
    # 发送者ID：字符串，默认 "default"（单机调试场景的兜底会话标识）。
    sender_id: str = DEFAULT_SENDER_ID
    # 时间戳：Unix 秒（浮点），default_factory 使每条消息取创建时刻。
    timestamp: float = field(default_factory=time.time)
    # 输入通道名（rest/socketio/console）；None 表示未记录来源。
    input_channel: str | None = None
    # 额外元数据：通道透传的附加信息（如网页端用户名），默认空字典。
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式。

        业务流程：将 5 个字段平铺进字典，供持久化与跨进程传输。

        副作用：无（只读序列化）。

        返回：
            包含 text/sender_id/timestamp/input_channel/metadata 的字典。
        """
        return {
            # 消息正文原样写入。
            "text": self.text,
            # 发送者ID原样写入。
            "sender_id": self.sender_id,
            # 时间戳为浮点秒，JSON 兼容。
            "timestamp": self.timestamp,
            # 通道名可能为 None，序列化时保留 None 语义。
            "input_channel": self.input_channel,
            # 元数据字典浅拷贝引用（数据类字段不被外部修改时安全）。
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserMessage":
        """从字典创建用户消息。

        业务流程：逐键取值并给默认值，缺键不报错（容忍精简数据）。

        参数：
            data: to_dict 产出的字典；各键允许缺失。

        返回：
            UserMessage 实例（text 缺省为空串、timestamp 缺省取当前时刻）。
        """
        return cls(
            # 正文缺省空串：避免 KeyError，空串语义为"无内容"。
            text=data.get("text", ""),
            # 发送者缺省回到默认会话ID。
            sender_id=data.get("sender_id", DEFAULT_SENDER_ID),
            # 时间戳缺省取当前时刻（手工构造数据的兜底）。
            timestamp=data.get("timestamp", time.time()),
            # 通道缺省 None（未记录）。
            input_channel=data.get("input_channel"),
            # 元数据缺省空字典。
            metadata=data.get("metadata", {}),
        )


@dataclass
class BotMessage:
    """Bot响应消息。

    职责：封装 Bot 发出的一条响应（文本、结构化数据与元数据）。
    设计模式：数据类；一轮可含多条（如先按钮后文本）。
    生命周期：随 NLG/Action 产出创建，存入 DialogueTurn 后随 Tracker 持久化。

    属性：
        text: 消息文本
        data: 结构化数据(按钮、图片等)
        timestamp: 时间戳
        metadata: 额外元数据
    """

    # 响应文本：None 表示纯结构化消息（如仅按钮附件）。
    text: str | None = None
    # 结构化附件：按钮/图片/自定义载荷，键为元素类型名。
    data: dict[str, Any] = field(default_factory=dict)
    # 时间戳：Unix 秒（浮点），每条消息取创建时刻。
    timestamp: float = field(default_factory=time.time)
    # 额外元数据：生成来源等调试信息，默认空字典。
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式。

        返回：
            包含 text/data/timestamp/metadata 的字典。
        """
        return {
            # 文本可为 None，保留语义。
            "text": self.text,
            # 结构化附件原样写入。
            "data": self.data,
            # 时间戳浮点秒。
            "timestamp": self.timestamp,
            # 元数据原样写入。
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BotMessage":
        """从字典创建Bot消息。

        参数：
            data: to_dict 产出的字典；各键允许缺失。

        返回：
            BotMessage 实例（全字段均有缺省值）。
        """
        return cls(
            # 文本缺省 None（纯结构化消息）。
            text=data.get("text"),
            # 附件缺省空字典。
            data=data.get("data", {}),
            # 时间戳缺省取当前时刻。
            timestamp=data.get("timestamp", time.time()),
            # 元数据缺省空字典。
            metadata=data.get("metadata", {}),
        )


@dataclass
class DialogueTurn:
    """对话轮次。

    职责：聚合一次完整交互——一条用户消息、若干 Bot 响应、
          本轮产生的命令列表与执行的动作名。
    设计模式：数据类（组合模式：内嵌消息对象）。
    生命周期：update_with_message 时创建为 _current_turn，
              finalize_turn/下一轮开始时归档进 dialogue_turns。

    属性：
        user_message: 用户消息
        bot_messages: Bot响应消息列表
        commands: 生成的命令列表
        action_name: 执行的动作名称
        timestamp: 轮次时间戳
    """

    # 本轮用户消息：None 表示无用户输入的轮次（如纯主动播报）。
    user_message: UserMessage | None = None
    # Bot 响应列表：一轮可多条，default_factory 每实例新建空列表。
    bot_messages: list[BotMessage] = field(default_factory=list)
    # 本轮命令列表（LLM 产出的结构化命令字典），供调试与追溯。
    commands: list[dict[str, Any]] = field(default_factory=list)
    # 本轮最终执行的动作名；None 表示尚未执行。
    action_name: str | None = None
    # 轮次创建时间戳：Unix 秒（浮点）。
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式。

        业务流程：用户消息序列化（无则为 None）、Bot 消息逐条序列化，
        命令与动作名原样写入。

        返回：
            描述本轮全部内容的字典。
        """
        return {
            # 条件序列化：user_message 存在才转字典，否则写 None。
            "user_message": self.user_message.to_dict() if self.user_message else None,
            # 列表推导：逐条 Bot 消息序列化。
            "bot_messages": [m.to_dict() for m in self.bot_messages],
            # 命令列表本身即字典列表，直接引用。
            "commands": self.commands,
            # 动作名可写 None。
            "action_name": self.action_name,
            # 轮次时间戳。
            "timestamp": self.timestamp,
        }


class DialogueStateTracker:
    """对话状态追踪器。

    职责：管理单个用户会话的完整状态——对话历史（轮次归档）、槽位
          读写与重置、对话栈（Flow/搜索/闲聊等上下文的唯一状态源）、
          Flow 执行历史，以及整套序列化/反序列化能力。
    设计模式：门面（聚合消息/轮次/槽位/栈多个子结构）+
              计算属性（active_flow 从栈派生，不独立存储）。
    生命周期：随会话创建（TrackerStore 按需新建或 from_dict 恢复），
              会话结束随 Store 销毁；可经 copy 深拷贝做隔离实验。

    核心功能：
    - 记录对话历史(用户消息和Bot响应)
    - 管理槽位状态
    - 跟踪活跃的Flow（通过dialogue_stack）
    - 支持状态序列化和反序列化

    状态管理：
        使用统一的 dialogue_stack 管理所有对话上下文（Flow、搜索、闲聊等）。
        active_flow 是从 dialogue_stack 派生的计算属性。

    属性：
        sender_id: 会话ID(通常是用户ID)
        slots: 槽位状态字典
        dialogue_turns: 对话轮次历史
        dialogue_stack: 对话栈（唯一状态源）
        active_flow: 当前活跃的Flow名称（从dialogue_stack计算）
        latest_message: 最新的用户消息
        latest_action_name: 最新执行的动作
    """

    def __init__(
        self,
        sender_id: str = DEFAULT_SENDER_ID,
        slots: dict[str, Slot] | None = None,
        max_turns: int = 100,
    ) -> None:
        """初始化对话状态追踪器。

        业务流程：1. 保存会话标识与槽位字典；2. 初始化空的历史/
        当前轮次/对话栈/Flow 历史；3. 置最新消息为空、动作为监听；
        4. 记录创建与更新时间戳。

        副作用：仅写实例属性；取两次系统时钟。

        参数：
            sender_id: 会话ID，字符串，缺省 "default"。
            slots: 初始槽位字典（名称->Slot），None 按空字典处理。
            max_turns: 历史保留上限，正整数，默认 100（超出裁剪最旧轮次）。
        """
        # 会话标识：TrackerStore 按此键存取会话。
        self.sender_id = sender_id
        # 槽位表：None 兜底为空字典，避免共享可变默认值。
        self.slots: dict[str, Slot] = slots or {}
        # 历史轮次上限：归档时超出则裁掉最旧轮次。
        self.max_turns = max_turns

        # 对话历史：已归档的轮次列表（按时间正序）。
        self.dialogue_turns: list[DialogueTurn] = []

        # 当前轮次(正在进行中)：消息/命令/回复先累积于此，归档后置 None。
        self._current_turn: DialogueTurn | None = None

        # 统一的对话栈 - 唯一的状态管理：活跃 Flow 等上下文由此派生。
        self.dialogue_stack: DialogueStack = DialogueStack()

        # Flow历史记录（用于追溯）：Flow 启动/结束与内置 Pattern 的执行记录。
        self.flow_history: list[dict[str, Any]] = []

        # 最新状态：最后一条用户消息（供策略/LLM 读取）。
        self.latest_message: UserMessage | None = None
        # 最新动作名：初始为监听（等待用户输入）。
        self.latest_action_name: str = ACTION_LISTEN

        # 元数据：下一轮强制执行的动作名（None 表示无）。
        self.followup_action: str | None = None
        # 暂停标记：True 时暂缓自动决策（如等待人工接管）。
        self.paused: bool = False
        # 创建时间戳：Unix 秒。
        self.created_at: float = time.time()
        # 最后更新时间戳：每次状态变更刷新。
        self.updated_at: float = time.time()

    @property
    def active_flow(self) -> str | None:
        """当前活跃的Flow名称（从dialogue_stack计算）。

        返回：
            栈中最靠近栈顶的活动 Flow 帧的流程名；无活动 Flow 返回 None。
        """
        # 取栈中第一个活动状态的 Flow 帧。
        frame = self.dialogue_stack.active_flow_frame()
        # 有帧返回其流程ID，否则 None（计算属性，不落存储）。
        return frame.flow_id if frame else None

    def update_with_message(self, message: UserMessage) -> None:
        """使用新的用户消息更新状态。

        业务流程：1. 上一轮未归档则先归档；2. 以新消息创建当前轮次；
        3. 更新最新消息引用；4. 动作名复位为监听；5. 刷新更新时间。

        副作用：修改 _current_turn/latest_message/latest_action_name/
                updated_at；可能向 dialogue_turns 追加归档。

        参数：
            message: 用户消息，不允许为 None。
        """
        # 保存之前的轮次：上一轮没 finalize 就来了新消息，先归档防丢。
        if self._current_turn is not None:
            self._save_current_turn()

        # 开始新轮次：当前轮次仅含用户消息，回复/命令后续追加。
        self._current_turn = DialogueTurn(user_message=message)
        # 更新最新消息引用：策略层经 latest_message 读取本次输入。
        self.latest_message = message
        # 重置 latest_action_name，表示等待新的动作。
        self.latest_action_name = ACTION_LISTEN
        # 刷新更新时间戳。
        self.updated_at = time.time()

    def add_bot_message(self, message: BotMessage) -> None:
        """添加Bot响应消息。

        业务流程：当前轮次不存在（如主动播报无用户输入）时先建空轮次，
        再把消息追加进本轮响应列表。

        副作用：修改 _current_turn.bot_messages 与 updated_at。

        参数：
            message: Bot消息，不允许为 None。
        """
        # 惰性建轮次：保证响应总有归属的轮次容器。
        if self._current_turn is None:
            self._current_turn = DialogueTurn()

        # 追加到当前轮次的响应列表末尾。
        self._current_turn.bot_messages.append(message)
        # 刷新更新时间戳。
        self.updated_at = time.time()

    def set_slot(
        self,
        slot_name: str,
        value: Any,
        create_if_missing: bool = True,
    ) -> None:
        """设置槽位值。

        业务流程：1. 槽位已存在则经 value.setter 赋值（触发类型验证）；
        2. 不存在且允许创建时先建任意类型槽位再赋值；3. 刷新时间戳。

        副作用：修改 slots 字典（可能新增键）与 updated_at；
                验证失败抛 InvalidSlotValueError。

        参数：
            slot_name: 槽位名称，字符串。
            value: 槽位值，任意类型；None 语义为清空。
            create_if_missing: 槽位不存在时是否创建，默认 True。
        """
        if slot_name in self.slots:
            # 已有槽位：经 setter 赋值，按其类型规则验证。
            self.slots[slot_name].value = value
        elif create_if_missing:
            # 创建新槽位时 initial_value 应该是 None，这样 reset() 才能正确重置
            # （动态槽无 Domain 模板，初始值只能是"未设置"）。
            self.slots[slot_name] = create_slot(name=slot_name, initial_value=None)
            # 再经 setter 写入目标值（任意类型槽，验证恒通过）。
            self.slots[slot_name].value = value

        # 刷新更新时间戳（槽位不存在且不创建时也刷新，语义为一次尝试）。
        self.updated_at = time.time()

    def get_slot(self, slot_name: str) -> Any:
        """获取槽位值。

        参数：
            slot_name: 槽位名称。

        返回：
            槽位值；槽位不存在返回 None（与"已设置值为 None"不可区分，
            需区分时先查 slots 字典）。
        """
        # 字典.get 天然兜底缺失键。
        slot = self.slots.get(slot_name)
        # 有槽位返回其当前值，无槽位返回 None。
        return slot.value if slot else None

    def get_all_slots(self) -> dict[str, Any]:
        """获取所有槽位的值。

        返回：
            槽位名称到值的映射字典（新建字典，修改它不影响原槽位）。
        """
        # 字典推导：逐槽取当前值。
        return {name: slot.value for name, slot in self.slots.items()}

    def reset_slots(self) -> None:
        """重置所有槽位为初始值。

        副作用：逐槽调用 reset()（恢复 initial_value）并刷新 updated_at。
        """
        # 遍历全部槽位逐一重置。
        for slot in self.slots.values():
            slot.reset()
        # 刷新更新时间戳。
        self.updated_at = time.time()

    def set_latest_action(self, action_name: str) -> None:
        """设置最新执行的动作。

        业务流程：更新最新动作名；当前轮次存在时同步写入轮次的
        action_name 字段（保证归档后可追溯）。

        副作用：修改 latest_action_name/_current_turn.action_name/
                updated_at。

        参数：
            action_name: 动作名称，字符串。
        """
        # 更新追踪器级最新动作。
        self.latest_action_name = action_name
        # 轮次级同步：仅当当前轮次存在时写入。
        if self._current_turn:
            self._current_turn.action_name = action_name
        # 刷新更新时间戳。
        self.updated_at = time.time()

    def add_commands(self, commands: list[dict[str, Any]]) -> None:
        """添加生成的命令到当前轮次。

        业务流程：当前轮次不存在时先建空轮次，再将命令批量并入
        （extend 而非 append，入参即列表）。

        副作用：修改 _current_turn.commands 与 updated_at。

        参数：
            commands: 命令字典列表，允许空列表（无操作效果）。
        """
        # 惰性建轮次：命令到达早于用户消息时也有容器可存。
        if self._current_turn is None:
            self._current_turn = DialogueTurn()

        # 批量并入本轮命令列表。
        self._current_turn.commands.extend(commands)
        # 刷新更新时间戳。
        self.updated_at = time.time()

    def start_flow(self, flow_name: str, step_id: str = "START") -> None:
        """开始执行Flow。

        业务流程：1. 向对话栈压入 Flow 帧（成为当前活动上下文）；
        2. 在 flow_history 追加一条进行中记录；3. 刷新时间戳。

        副作用：修改 dialogue_stack（压栈）、flow_history（追加）与
                updated_at。

        参数：
            flow_name: Flow名称，对应 flows/ 目录中的流程定义名。
            step_id: 起始步骤ID，默认入口步骤 START。
        """
        # 压入FlowStackFrame到dialogue_stack：帧成为栈顶活动上下文。
        self.dialogue_stack.push_flow(flow_name, step_id)

        # 记录到历史：ended_at/completed 留空，由 end_flow 回填。
        self.flow_history.append(
            {
                # 流程名：与栈帧的 flow_id 一致。
                "flow_name": flow_name,
                # 启动时间：Unix 秒。
                "started_at": time.time(),
                # 结束时间：None 表示进行中。
                "ended_at": None,
                # 完成标记：False 表示进行中。
                "completed": False,
            }
        )
        # 刷新更新时间戳。
        self.updated_at = time.time()

    def end_flow(self) -> str | None:
        """结束当前Flow。

        业务流程：1. 定位栈顶 Flow 帧（无则直接返回 None）；2. 先弹出
        该 Flow 之上的全部帧（如嵌套/中断帧），再弹出 Flow 本身；
        3. 回填 flow_history 中对应记录的结束信息；4. 刷新时间戳。

        副作用：修改 dialogue_stack（出栈若干帧）、flow_history 与
                updated_at。

        返回：
            结束的Flow名称；栈中无 Flow 帧时返回 None。
        """
        # 获取栈顶Flow帧：end_flow 语义只针对最内层（最近）Flow。
        flow_frame = self.dialogue_stack.top_flow_frame()
        if flow_frame is None:
            # 无 Flow 可结束：返回 None 由调用方分支处理。
            return None

        # 记录流程名，供返回值与历史回填使用。
        flow_name = flow_frame.flow_id

        # 弹出到该Flow（包括它上面的所有帧）：目标 Flow 升至栈顶。
        self.dialogue_stack.pop_to_flow(flow_name)
        # 再弹出Flow本身：两步保证嵌套上下文一并清理。
        self.dialogue_stack.pop()

        # 更新flow_history中对应项的结束信息：倒序找最近一条同名且未结束的记录。
        for hist in reversed(self.flow_history):
            # 双条件：流程名匹配 + 尚未回填结束时间。
            if hist["flow_name"] == flow_name and hist["ended_at"] is None:
                # 回填结束时间。
                hist["ended_at"] = time.time()
                # 标记为正常完成。
                hist["completed"] = True
                # 只回填一条（同名 Flow 可能多次执行），立即退出。
                break

        # 刷新更新时间戳。
        self.updated_at = time.time()
        # 返回被结束的流程名。
        return flow_name

    def cancel_flow(self) -> None:
        """取消所有活跃的Flow。

        业务流程：直接清空对话栈（不逐帧改状态，栈清空即全部取消）。

        副作用：修改 dialogue_stack（清空）与 updated_at。
        """
        # 清栈：所有 Flow 帧（含嵌套/中断）一并移除。
        self.dialogue_stack.clear()
        # 刷新更新时间戳。
        self.updated_at = time.time()

    def record_pattern(self, pattern_type: str, completed: bool = True) -> None:
        """记录内置 Pattern 的执行历史。

        业务流程：以 "pattern_<类型>" 为名向 flow_history 追加一条
        记录（与 Flow 记录同构），供 Inspect 页面统一展示。

        副作用：修改 flow_history（追加）与 updated_at。

        参数：
            pattern_type: Pattern 类型，如 "chitchat"、"search"、"cannot_handle"、
                         "completed"、"human_handoff"。
            completed: 是否执行完成，默认为 True（False 表示进行中，
                       由调用方后续回填）。
        """
        # 统一取当前时刻，保证 started/ended 一致。
        current_time = time.time()
        # 追加与 Flow 记录同构的字典。
        self.flow_history.append(
            {
                # 统一加 pattern_ 前缀：与真实 Flow 名区分。
                "flow_name": f"pattern_{pattern_type}",
                # 启动时间。
                "started_at": current_time,
                # 已完成则起止同刻；未完成留 None 待回填。
                "ended_at": current_time if completed else None,
                # 完成标记。
                "completed": completed,
            }
        )
        # 刷新更新时间戳。
        self.updated_at = current_time

    def get_conversation_history(
        self,
        max_turns: int | None = None,
    ) -> list[dict[str, Any]]:
        """获取对话历史。

        业务流程：1. 拷贝已归档轮次（浅拷贝防外泄内部列表）；2. 当前
        进行中轮次存在则附在末尾；3. 按 max_turns 截取最近 N 轮；
        4. 逐轮序列化。

        副作用：无（只读）。

        参数：
            max_turns: 最大返回轮次数；None/0 表示不限制。

        返回：
            对话轮次字典列表（时间正序，最多 max_turns 条）。
        """
        # 浅拷贝归档列表：后续切片与追加不影响 self.dialogue_turns。
        turns = self.dialogue_turns[:]
        # 进行中轮次一并展示（含未归档的输入与命令）。
        if self._current_turn:
            turns.append(self._current_turn)

        # 截取最近 N 轮：max_turns 为 0/None 时不截取。
        if max_turns:
            turns = turns[-max_turns:]

        # 逐轮序列化为字典。
        return [turn.to_dict() for turn in turns]

    def get_messages_for_llm(
        self,
        max_turns: int = 10,
    ) -> list[dict[str, str]]:
        """获取用于LLM的消息历史。

        业务流程：1. 取最近 N 轮归档；2. 每轮先输出 user 消息、再依次
        输出 assistant 消息（只取有文本的）；3. 追加当前进行中轮次的
        用户消息。

        副作用：无（只读）。

        参数：
            max_turns: 最大轮次数；0/None 表示全部历史。

        返回：
            LLM 消息列表，元素形如 {"role": "user"|"assistant",
            "content": str}，时间正序，可直接拼入提示词。
        """
        # 累积消息容器。
        messages = []
        # 归档轮次切片：max_turns 为 0/None 时取全部。
        turns = self.dialogue_turns[-max_turns:] if max_turns else self.dialogue_turns

        # 逐轮展开为 LLM 消息序列。
        for turn in turns:
            # 本轮用户消息存在才输出（主动播报轮无用户消息）。
            if turn.user_message:
                messages.append(
                    {
                        # 角色标记：用户。
                        "role": "user",
                        # 消息正文。
                        "content": turn.user_message.text,
                    }
                )

            # 本轮所有 Bot 消息依次输出。
            for bot_msg in turn.bot_messages:
                # 纯结构化消息（text 为 None）跳过，LLM 只吃文本。
                if bot_msg.text:
                    messages.append(
                        {
                            # 角色标记：助手。
                            "role": "assistant",
                            # 响应正文。
                            "content": bot_msg.text,
                        }
                    )

        # 添加当前轮次的用户消息：尚未归档的本次输入也要进提示词。
        if self._current_turn and self._current_turn.user_message:
            messages.append(
                {
                    # 角色标记：用户。
                    "role": "user",
                    # 当前输入正文。
                    "content": self._current_turn.user_message.text,
                }
            )

        # 返回按时间正序的消息列表。
        return messages

    def _save_current_turn(self) -> None:
        """保存当前轮次到历史。

        业务流程：1. 当前轮次存在则追加进归档列表；2. 超出 max_turns
        时裁剪最旧轮次；3. 置空当前轮次。

        副作用：修改 dialogue_turns（追加/裁剪）与 _current_turn。
        """
        # 无进行中轮次则无事可做。
        if self._current_turn:
            # 归档：追加到历史末尾（时间正序）。
            self.dialogue_turns.append(self._current_turn)

            # 限制历史长度：超出上限时仅保留最近 max_turns 轮。
            if len(self.dialogue_turns) > self.max_turns:
                self.dialogue_turns = self.dialogue_turns[-self.max_turns :]

            # 置空：轮次生命周期结束，下次消息到来时新建。
            self._current_turn = None

    def finalize_turn(self) -> None:
        """完成当前轮次。

        业务流程：归档当前轮次并刷新更新时间。

        副作用：修改 dialogue_turns/_current_turn/updated_at。
        """
        # 归档当前轮次（内部自行判空）。
        self._save_current_turn()
        # 刷新更新时间戳。
        self.updated_at = time.time()

    def restart(self) -> None:
        """重启会话，清除所有状态。

        业务流程：清空对话历史与当前轮次、重置槽位、清空对话栈、
        复位最新消息/动作/元数据标记；保留 flow_history 与
        created_at（追溯与审计不随重启丢失）。

        副作用：重置上述全部内存状态并刷新 updated_at。
        """
        # 清空已归档轮次（原地清空）。
        self.dialogue_turns.clear()
        # 丢弃进行中轮次。
        self._current_turn = None
        # 所有槽位恢复初始值。
        self.reset_slots()
        # 清空对话栈：所有 Flow 上下文一并取消。
        self.dialogue_stack.clear()
        # 清空最新消息引用。
        self.latest_message = None
        # 动作复位为监听。
        self.latest_action_name = ACTION_LISTEN
        # 清除 followup 标记。
        self.followup_action = None
        # 解除暂停。
        self.paused = False
        # 刷新更新时间戳。
        self.updated_at = time.time()

    def current_state(self) -> dict[str, Any]:
        """获取当前完整状态。

        业务流程：聚合会话标识、槽位值快照、活跃 Flow、对话栈、
        最新消息/动作与各项元数据为一份只读视图。

        副作用：无（只读；用于调试/Inspect 展示，不用于持久化）。

        返回：
            状态字典（槽位只含当前值，不含槽位定义）。
        """
        return {
            # 会话标识。
            "sender_id": self.sender_id,
            # 槽位名称->当前值 映射。
            "slots": self.get_all_slots(),
            # 活跃 Flow 名（计算属性，可能为 None）。
            "active_flow": self.active_flow,
            # 对话栈整体序列化（含各帧明细）。
            "dialogue_stack": self.dialogue_stack.as_dict(),
            # 最新用户消息（无则 None）。
            "latest_message": self.latest_message.to_dict()
            if self.latest_message
            else None,
            # 最新执行的动作名。
            "latest_action_name": self.latest_action_name,
            # 暂停标记。
            "paused": self.paused,
            # 已归档轮次数（不含进行中轮次）。
            "dialogue_turns_count": len(self.dialogue_turns),
            # 创建时间戳。
            "created_at": self.created_at,
            # 最后更新时间戳。
            "updated_at": self.updated_at,
        }

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。

        业务流程：1. 先归档当前轮次（保证不丢进行中内容）；2. 聚合
        全部状态字段——槽位含完整定义、轮次逐条序列化、对话栈、
        Flow 历史与元数据。

        副作用：归档动作会修改 dialogue_turns/_current_turn。

        返回：
            完整状态字典（与 from_dict 配对，供持久化存储）。
        """
        # 确保当前轮次被保存：序列化前先归档，避免漏掉进行中内容。
        self._save_current_turn()

        return {
            # 会话标识。
            "sender_id": self.sender_id,
            # 槽位逐个完整序列化（含类型定义与初始值，供恢复）。
            "slots": {name: slot.to_dict() for name, slot in self.slots.items()},
            # 对话轮次逐条序列化。
            "dialogue_turns": [turn.to_dict() for turn in self.dialogue_turns],
            # 对话栈序列化（帧按 type 路由键存储）。
            "dialogue_stack": self.dialogue_stack.as_dict(),
            # Flow 执行历史原样写入。
            "flow_history": self.flow_history,
            # 最新动作名。
            "latest_action_name": self.latest_action_name,
            # 最新用户消息（无则 None）。
            "latest_message": self.latest_message.to_dict()
            if self.latest_message
            else None,
            # 暂停标记。
            "paused": self.paused,
            # 创建时间戳。
            "created_at": self.created_at,
            # 最后更新时间戳。
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        domain_slots: dict[str, Slot] | None = None,
    ) -> "DialogueStateTracker":
        """从字典反序列化。

        业务流程：1. 逐槽恢复——优先用 Domain 模板深拷贝以保留槽位
        类型，仅回填值；无模板时经 Slot.from_dict 从数据重建；
        2. 构造 Tracker；3. 恢复对话栈与其余状态字段；4. 恢复最新
        消息与对话历史。

        副作用：创建新对象；不修改入参字典。

        参数：
            data: to_dict 产出的状态字典；各键允许缺失（按默认值处理）。
            domain_slots: Domain定义的槽位(用于恢复槽位类型)；
                          None 时全部槽位按持久化数据重建。

        返回：
            DialogueStateTracker实例。
        """
        # 恢复槽位
        slots = {}
        # 逐个槽位还原。
        for slot_name, slot_data in data.get("slots", {}).items():
            if domain_slots and slot_name in domain_slots:
                # 使用Domain定义的槽位类型：深拷贝模板，会话间互不影响。
                slot = deepcopy(domain_slots[slot_name])
                # 仅回填持久化的当前值（直写 _value，绕过验证——
                # 与 Slot.from_dict 同样的兼容性考虑）。
                slot._value = slot_data.get("value")
            else:
                # 从数据恢复：持久化数据自带完整槽位定义。
                slot = Slot.from_dict(slot_data)
            # 存入槽位表。
            slots[slot_name] = slot

        # 用恢复的标识与槽位构造 Tracker（其余字段先取默认值）。
        tracker = cls(
            sender_id=data.get("sender_id", DEFAULT_SENDER_ID),
            slots=slots,
        )

        # 恢复dialogue_stack：经注册表按 type 还原各帧。
        if "dialogue_stack" in data:
            tracker.dialogue_stack = DialogueStack.from_dict(data["dialogue_stack"])

        # 恢复其他状态：Flow 历史（缺省空列表）。
        tracker.flow_history = data.get("flow_history", [])
        # 最新动作名（缺省监听）。
        tracker.latest_action_name = data.get("latest_action_name", ACTION_LISTEN)
        # 暂停标记（缺省 False）。
        tracker.paused = data.get("paused", False)
        # 创建时间（缺省取当前时刻）。
        tracker.created_at = data.get("created_at", time.time())
        # 更新时间（缺省取当前时刻）。
        tracker.updated_at = data.get("updated_at", time.time())

        # 恢复 latest_message：存在才重建。
        if data.get("latest_message"):
            tracker.latest_message = UserMessage.from_dict(data["latest_message"])

        # 恢复对话历史(简化处理，只恢复基本信息)
        for turn_data in data.get("dialogue_turns", []):
            # 先按标量字段构造轮次骨架。
            turn = DialogueTurn(
                # 轮次时间戳（缺省当前时刻）。
                timestamp=turn_data.get("timestamp", time.time()),
                # 动作名（可缺省 None）。
                action_name=turn_data.get("action_name"),
                # 命令列表（缺省空列表）。
                commands=turn_data.get("commands", []),
            )

            # 恢复本轮用户消息（存在才重建）。
            if turn_data.get("user_message"):
                turn.user_message = UserMessage.from_dict(turn_data["user_message"])

            # 逐条恢复本轮 Bot 消息。
            for bot_msg_data in turn_data.get("bot_messages", []):
                turn.bot_messages.append(BotMessage.from_dict(bot_msg_data))

            # 追加进恢复后的历史列表。
            tracker.dialogue_turns.append(turn)

        # 返回恢复完成的 Tracker。
        return tracker

    def copy(self) -> "DialogueStateTracker":
        """创建Tracker的深拷贝。

        业务流程：经 to_dict -> from_dict 序列化往返实现隔离拷贝，
        槽位/轮次/栈均被重建，副本与原实例互不影响。

        返回：
            新的DialogueStateTracker实例。
        """
        # 序列化往返：比 deepcopy 更可控（只拷贝声明的状态字段）。
        return DialogueStateTracker.from_dict(self.to_dict())

    def __repr__(self) -> str:
        """调试用字符串表示：会话ID + 槽位数 + 轮次数 + 活跃Flow。"""
        # 拼接关键字段，便于日志快速定位会话状态。
        return (
            f"DialogueStateTracker(sender_id={self.sender_id}, "
            f"slots={len(self.slots)}, turns={len(self.dialogue_turns)}, "
            f"active_flow={self.active_flow})"
        )
