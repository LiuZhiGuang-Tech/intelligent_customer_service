"""
栈帧定义

功能：
定义对话栈（DialogueStack）中的各种栈帧类型——抽象基类 StackFrame、六种具体帧（Flow/搜索/闲聊/无法处理/已完成/人工转接）、
帧状态枚举 FrameState、Flow 帧类型枚举 FlowFrameType、类型注册装饰器与反序列化入口 create_frame_from_dict。

每个栈帧记录对话上下文的一个状态点，压栈/出栈即对话流程的推进与回退。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import dataclasses
from enum import Enum
from typing import Any
import uuid


def generate_frame_id() -> str:
    """
    生成唯一的帧 ID
    """
    # 8 位十六进制短 ID 字符串
    # .hex 去掉连字符
    return uuid.uuid4().hex[:8]


# 帧类型注册表：帧类型标识字符串 -> 帧类
# 由下方装饰器等级
_FRAME_TYPE_REGISTRY: dict[str, type["StackFrame"]] = {}


# 泛型函数（generic function）：判断依据是函数名后紧跟中括号形式的“类型参数列表”（PEP 695 语法，Python 3.12+ 支持）
# [C: type["StackFrame"]]：类型参数声明
def register_frame_type[C: type["StackFrame"]](cls: C) -> C:
    """
    栈帧类型注册装饰器

    :param cls: StackFrame 子类实例
    :return: 原样返回入参类（类型不变，便于装饰器叠加与正常使用）
    """
    _FRAME_TYPE_REGISTRY[cls.frame_type()] = cls
    # 原样返回类，保持装饰器透明
    return cls


class FrameState(str, Enum):
    """
    帧状态枚举
    """

    ACTIVE = "active"  # 活动状态：帧正在处理中（默认初始状态）
    COMPLETED = "completed"  # 已完成状态：帧对应的任务/流程已正常结束
    INTERRUPTED = "interrupted"  # 已中断状态：被更高优先级流程打断（可恢复）
    CANCELLED = "cancelled"  # 已取消状态：被用户或系统主动取消


class FlowFrameType(str, Enum):
    """
    Flow 帧类型枚举
    """

    # 常规帧：正常执行的业务流程
    REGULAR = "regular"
    # 中断帧：打断当前流程的临时流程，用于处理紧急情况或切换到其他流程
    INTERRUPT = "interrupt"
    # 链接帧：流程跳转产生的衔接帧
    LINK = "link"


@dataclass
class StackFrame(ABC):
    """
    栈帧抽象基类
    """

    # 帧唯一标识：default_factory 使每个实例创建时自动生成新 ID
    # 同时避免可变默认值在实例间共享。
    frame_id: str = field(default_factory=generate_frame_id)
    # 当前状态：新帧默认为活动状态
    state: FrameState = FrameState.ACTIVE

    @classmethod
    @abstractmethod
    def frame_type(cls) -> str:
        """
        返回当前帧的类型标识字符串
        """
        # 抽象方法：由子类实现，基类不应该被调用
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "StackFrame":
        """
        从字典创建帧实例

        :param data: 包含帧数据的字典
        :return: 帧实例
        """
        # 抽象方法：各子类按自身字段实现反序列化
        raise NotImplementedError()

    def as_dict(self) -> dict[str, Any]:
        """
        将帧实例转换为字典表示

        :return: 包含帧数据的字典
        """
        # 初始化结果字典
        data = {}

        # 遍历本数据类声明的所有字段（包含子类新增字段）
        for f in dataclasses.fields(self):
            # 取当前字段值
            value = getattr(self, f.name)
            # 枚举类型转为其字符串值，保证可 JSON 序列化
            if isinstance(value, Enum):
                data[f.name] = value.value
            else:
                # 非枚举字段原样写入
                data[f.name] = value

        # 追加类型字段：反序列化的路由依据
        data["type"] = self.frame_type()
        return data

    def is_active(self) -> bool:
        """
        判断帧是否为活动状态

        :return: 是否为活动状态
        """
        return self.state == FrameState.ACTIVE

    def is_completed(self) -> bool:
        """
        判断帧是否为已完成状态

        :return: 是否为已完成状态
        """
        return self.state == FrameState.COMPLETED

    def complete(self) -> None:
        """
        标记帧为已完成状态
        """
        # 状态迁移：正常结束
        self.state = FrameState.COMPLETED

    def interrupt(self) -> None:
        """
        标记帧为已中断状态
        """
        # 状态迁移：被其他流程打断，可恢复
        self.state = FrameState.INTERRUPTED

    def cancel(self) -> None:
        """
        标记帧为已取消状态
        """
        # 状态迁移：主动取消，终态
        self.state = FrameState.CANCELLED


@register_frame_type
@dataclass
class FlowStackFrame(StackFrame):
    """
    Flow 栈帧
    """

    # 正在执行的 Flow 标识，对应 flows/ 目录中的流程定义
    flow_id: str = ""
    # 当前执行到的步骤 ID ；START 为入口步骤
    step_id: str = "START"
    # 帧角色：常规/中断/链接，默认为常规帧
    flow_frame_type: FlowFrameType = FlowFrameType.REGULAR
    # 当前正在收集的槽位名
    slot_to_collect: str | None = None
    # 完成中标记：True 表示 Flow 已走完，正在等收尾 Action 执行
    completing: bool = False

    @classmethod
    def frame_type(cls) -> str:
        """
        返回当前帧的类型标识字符串
        """
        return "flow"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlowStackFrame":
        """
        从字典创建帧实例

        :param data: 包含帧数据的字典
        :return: 帧实例
        """
        # 取状态字符串，缺省为 ACTIVE 的字符串值
        state = data.get("state", FrameState.ACTIVE.value)

        # 字符串转枚举：持久化数据中状态以字符串存储
        if isinstance(state, str):
            state = FrameState(state)

        # 支持flow_frame_type 和旧的 frame_type 字段
        # 优先取新字段 flow_frame_type，为空时回退旧字段 frame_type
        # 两者皆无时兜底为常规帧
        flow_frame_type = data.get("flow_frame_type") or data.get(
            "frame_type", FlowFrameType.REGULAR.value
        )

        # 字符串转枚举
        if isinstance(flow_frame_type, str):
            flow_frame_type = FlowFrameType(flow_frame_type)

        # 按字典字段构造帧实例，缺省值与类字段默认值一致
        return FlowStackFrame(
            # 帧 ID：缺省重新生成
            frame_id=data.get("frame_id", generate_frame_id()),
            # 状态：已转枚举
            state=state,
            # 流程ID：缺省空串
            flow_id=data.get("flow_id", ""),
            # 当前执行到的步骤 ID ；START 为入口步骤
            step_id=data.get("step_id", "START"),
            # 帧角色：常规/中断/链接，默认为常规帧
            flow_frame_type=flow_frame_type,
            # 待收集槽位：缺省 None
            slot_to_collect=data.get("slot_to_collect", None),
            # 完成中标记：缺省 False
            completing=data.get("completing", False),
        )

    def advance_to_step(self, step_id: str) -> None:
        """
        前进到指定步骤

        :param step_id: 目标步骤 ID
        :return: None
        """
        # 状态迁移：正常执行
        self.step_id = step_id

    def is_interrupted(self) -> bool:
        """
        判断帧是否为已中断状态

        :return: 是否为已中断状态
        """
        return self.flow_frame_type == FlowFrameType.INTERRUPT


@register_frame_type
@dataclass
class SearchStackFrame(StackFrame):
    """搜索栈帧。

    职责：表示正在执行知识库搜索（RAG）。
          不存储query，从latest_message获取。
    生命周期：EnterpriseSearchPolicy 触发检索时压栈，回答生成后出栈。
    """

    @classmethod
    def frame_type(cls) -> str:
        """返回帧类型标识。"""
        # 注册表键：search。
        return "search"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchStackFrame":
        """从字典创建帧实例。

        业务流程：恢复 state 后构造（本帧无业务字段）。

        参数：
            data: as_dict 产出的字典。

        返回：
            SearchStackFrame 实例。
        """
        # 取状态字符串，缺省 ACTIVE。
        state = data.get("state", FrameState.ACTIVE.value)
        # 字符串转枚举。
        if isinstance(state, str):
            state = FrameState(state)

        # 构造实例：仅公共字段。
        return SearchStackFrame(
            # 帧ID：缺省重新生成。
            frame_id=data.get("frame_id", generate_frame_id()),
            # 状态：已转枚举。
            state=state,
        )


@register_frame_type
@dataclass
class ChitChatStackFrame(StackFrame):
    """闲聊栈帧。

    职责：表示正在处理闲聊（与业务无关的寒暄对话）。
    生命周期：闲聊命令触发时压栈，回复生成后出栈。
    """

    @classmethod
    def frame_type(cls) -> str:
        """返回帧类型标识。"""
        # 注册表键：chitchat。
        return "chitchat"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChitChatStackFrame":
        """从字典创建帧实例。

        业务流程：恢复 state 后构造（本帧无业务字段）。

        参数：
            data: as_dict 产出的字典。

        返回：
            ChitChatStackFrame 实例。
        """
        # 取状态字符串，缺省 ACTIVE。
        state = data.get("state", FrameState.ACTIVE.value)
        # 字符串转枚举。
        if isinstance(state, str):
            state = FrameState(state)

        # 构造实例：仅公共字段。
        return ChitChatStackFrame(
            # 帧ID：缺省重新生成。
            frame_id=data.get("frame_id", generate_frame_id()),
            # 状态：已转枚举。
            state=state,
        )


@register_frame_type
@dataclass
class CannotHandleStackFrame(StackFrame):
    """无法处理栈帧。

    职责：表示系统无法处理用户请求，记录降级原因。
    生命周期：降级链走到底（CANNOT_HANDLE）时压栈，兜底回复后出栈。

    Attributes:
        reason: 无法处理的原因
    """

    # 无法处理的原因（对应 DegradationReason 常量取值），缺省空串。
    reason: str = ""

    @classmethod
    def frame_type(cls) -> str:
        """返回帧类型标识。"""
        # 注册表键：cannot_handle。
        return "cannot_handle"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CannotHandleStackFrame":
        """从字典创建帧实例。

        业务流程：恢复 state 与 reason 后构造。

        参数：
            data: as_dict 产出的字典。

        返回：
            CannotHandleStackFrame 实例。
        """
        # 取状态字符串，缺省 ACTIVE。
        state = data.get("state", FrameState.ACTIVE.value)
        # 字符串转枚举。
        if isinstance(state, str):
            state = FrameState(state)

        # 构造实例：公共字段 + 原因。
        return CannotHandleStackFrame(
            # 帧ID：缺省重新生成。
            frame_id=data.get("frame_id", generate_frame_id()),
            # 状态：已转枚举。
            state=state,
            # 原因：缺省空串。
            reason=data.get("reason", ""),
        )


@register_frame_type
@dataclass
class CompletedStackFrame(StackFrame):
    """完成栈帧。

    职责：表示所有Flow已完成，系统处于空闲状态。
          由 FlowPolicy 在 Flow 完成后自动压入。
          由 EnterpriseSearchPolicy 处理，生成询问用户是否还有其他需求的响应。
    生命周期：Flow 完成时压栈，用户新请求到来或会话重置时出栈。

    Attributes:
        previous_flow_name: 刚完成的Flow名称
    """

    # 刚完成的 Flow 名称，缺省空串；用于生成"还有其他需要吗"类回复。
    previous_flow_name: str = ""

    @classmethod
    def frame_type(cls) -> str:
        """返回帧类型标识。"""
        # 注册表键：completed。
        return "completed"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompletedStackFrame":
        """从字典创建帧实例。

        业务流程：恢复 state 与 previous_flow_name 后构造。

        参数：
            data: as_dict 产出的字典。

        返回：
            CompletedStackFrame 实例。
        """
        # 取状态字符串，缺省 ACTIVE。
        state = data.get("state", FrameState.ACTIVE.value)
        # 字符串转枚举。
        if isinstance(state, str):
            state = FrameState(state)

        # 构造实例：公共字段 + 刚完成的流程名。
        return CompletedStackFrame(
            # 帧ID：缺省重新生成。
            frame_id=data.get("frame_id", generate_frame_id()),
            # 状态：已转枚举。
            state=state,
            # 刚完成的 Flow 名称：缺省空串。
            previous_flow_name=data.get("previous_flow_name", ""),
        )


@register_frame_type
@dataclass
class HumanHandoffStackFrame(StackFrame):
    """人工转接栈帧。

    职责：表示需要将对话转接给人工客服。
          由 ActionHumanHandoff 压入，由 EnterpriseSearchPolicy 处理。
    生命周期：转接 Action 执行时压栈，转接提示回复后出栈。

    Attributes:
        reason: 转接原因
    """

    # 转接原因（如"用户要求人工"、"多次无法解决"），缺省空串。
    reason: str = ""

    @classmethod
    def frame_type(cls) -> str:
        """返回帧类型标识。"""
        # 注册表键：human_handoff。
        return "human_handoff"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HumanHandoffStackFrame":
        """从字典创建帧实例。

        业务流程：恢复 state 与 reason 后构造。

        参数：
            data: as_dict 产出的字典。

        返回：
            HumanHandoffStackFrame 实例。
        """
        # 取状态字符串，缺省 ACTIVE。
        state = data.get("state", FrameState.ACTIVE.value)
        # 字符串转枚举。
        if isinstance(state, str):
            state = FrameState(state)

        # 构造实例：公共字段 + 转接原因。
        return HumanHandoffStackFrame(
            # 帧ID：缺省重新生成。
            frame_id=data.get("frame_id", generate_frame_id()),
            # 状态：已转枚举。
            state=state,
            # 转接原因：缺省空串。
            reason=data.get("reason", ""),
        )


def create_frame_from_dict(data: dict[str, Any]) -> StackFrame:
    """从字典创建栈帧。

    业务流程：根据type字段确定帧类型，从注册表查到对应类后
    委托其 from_dict 完成构造。

    副作用：创建新对象，不修改入参字典。

    Args:
        data: 包含帧数据的字典

    Returns:
        栈帧实例

    Raises:
        ValueError: 如果帧类型未知或缺失 type 字段
    """
    # 读取类型字段：反序列化路由的唯一依据。
    frame_type = data.get("type")
    # 缺失 type：无法路由，直接报错（数据不完整，不猜类型）。
    if not frame_type:
        raise ValueError("Missing 'type' field in frame data")

    # 从注册表查找对应帧类。
    frame_cls = _FRAME_TYPE_REGISTRY.get(frame_type)
    # 未知类型：注册表无此键，报错（可能是数据来自更新后的版本）。
    if frame_cls is None:
        raise ValueError(f"Unknown frame type: {frame_type}")

    # 委托具体帧类的 from_dict 完成构造。
    return frame_cls.from_dict(data)


# 导出
__all__ = [
    # 列表项1：栈帧抽象基类。
    "StackFrame",
    # 列表项2：Flow栈帧。
    "FlowStackFrame",
    # 列表项3：搜索栈帧。
    "SearchStackFrame",
    # 列表项4：闲聊栈帧。
    "ChitChatStackFrame",
    # 列表项5：无法处理栈帧。
    "CannotHandleStackFrame",
    # 列表项6：完成栈帧。
    "CompletedStackFrame",
    # 列表项7：人工转接栈帧。
    "HumanHandoffStackFrame",
    # 列表项8：帧状态枚举。
    "FrameState",
    # 列表项9：Flow帧类型枚举。
    "FlowFrameType",
    # 列表项10：帧ID生成函数。
    "generate_frame_id",
    # 列表项11：类型注册装饰器。
    "register_frame_type",
    # 列表项12：反序列化入口。
    "create_frame_from_dict",
]
