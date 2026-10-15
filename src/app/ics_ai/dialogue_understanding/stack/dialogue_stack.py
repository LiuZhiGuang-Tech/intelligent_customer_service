"""
对话栈（dialogue_stack）

核心功能：定义对话上下文栈 DialogueStack——以后进先出（LIFO）结构管理
          栈帧（StackFrame），栈顶即当前活动上下文。在基础栈操作（压栈/
          弹栈/查顶/迭代）之上，提供 Flow 导航（定位活动 Flow 帧、按流程
          ID 查找与回退弹栈）、帧查找/移除与序列化能力，支撑 Flow 嵌套、
          中断和恢复。

执行流程：本模块被导入时定义 DialogueStack 数据类；运行时由策略层
          （FlowPolicy/EnterpriseSearchPolicy）经 push/pop 驱动帧进出，
          FlowPolicy 经 top_flow_frame/active_flow_frame 读取当前流程
          上下文并推进步骤，Tracker 持久化时经 as_dict/from_dict 序列化
          往返（from_dict 经帧类型注册表按 type 字段还原具体帧）。

副作用：仅操作内存中的帧列表，不读写文件/数据库、不发起网络请求、
        不修改全局状态。

主要依赖：app.ics_ai.dialogue_understanding.stack.stack_frame
          （StackFrame/FlowStackFrame 帧类型与反序列化入口）。
"""

# 延迟注解求值：类型注解以字符串形式保存，解决前向引用
# （如方法返回 "DialogueStack"）。
from __future__ import annotations

# 从 collections.abc 导入 Iterator：typing.Iterator 自 3.9 起已弃用
# （PEP 585），标准替代为 collections.abc.Iterator。
from collections.abc import Iterator

# 导入 dataclass 与 field：数据类定义；field(default_factory=list)
# 为帧列表指定每次实例化新建的空列表，避免可变默认值在实例间共享。
from dataclasses import dataclass, field

# 导入 Any：任意类型注解（typing 中未弃用的部分，无内置替代）；
# Dict/List/Optional/Type 已按 PEP 585/604 迁移为 dict/list、
# X | None、type[X] 写法。
from typing import Any

# 从栈帧模块导入核心类型：帧基类与 Flow 帧（导航目标）、
# 反序列化入口（from_dict 按注册表还原具体帧）。
from app.ics_ai.dialogue_understanding.stack.stack_frame import (
    FlowStackFrame,
    StackFrame,
    create_frame_from_dict,
)


@dataclass
class DialogueStack:
    """对话栈。

    职责：以 LIFO 栈管理对话上下文帧——基础栈操作（压/弹/查顶/清空/
          迭代）、Flow 导航（定位活动 Flow 帧、按 ID 查找与回退弹栈）、
          帧查找/移除、序列化/反序列化与隔离拷贝。
    设计模式：数据类 + 门面（聚合栈帧通用操作）+ 迭代器协议
              （__iter__ 默认自栈顶向栈底遍历，匹配"先看当前上下文"
              的决策顺序）。
    生命周期：随会话（Tracker）创建，会话结束或重置时销毁；
              可经 copy/序列化往返跨进程恢复。

    Attributes:
        frames: 栈帧列表，最后一个是栈顶
    """

    # 栈体：按压栈顺序存放帧；末尾元素即栈顶（当前活动上下文）。
    frames: list[StackFrame] = field(default_factory=list)

    def push(self, frame: StackFrame) -> None:
        """将帧压入栈顶。

        业务流程：追加到 frames 末尾，使其成为新的栈顶（当前活动帧）。

        副作用：修改 frames 列表（长度 +1）。

        参数：
            frame: 要压入的栈帧，不允许为 None。
        """
        # 追加到末尾：本栈约定"列表末尾 = 栈顶"。
        self.frames.append(frame)

    def pop(self) -> StackFrame | None:
        """弹出栈顶帧。

        业务流程：栈非空时移除并返回末尾元素；空栈返回 None
        （不抛异常，便于调用方直接走分支）。

        副作用：修改 frames 列表（长度 -1）。

        返回：
            栈顶帧；栈为空时返回 None。
        """
        # 空栈兜底：返回 None 而非抛 IndexError，语义为"无上下文可弹"。
        if self.frames:
            # pop() 移除并返回末尾元素（栈顶）。
            return self.frames.pop()
        # 空栈：无帧可弹。
        return None

    def top(self) -> StackFrame | None:
        """获取栈顶帧（不弹出）。

        返回：
            栈顶帧；栈为空时返回 None。
        """
        # 只读访问末尾元素，不改变栈内容。
        if self.frames:
            return self.frames[-1]
        # 空栈：无帧可查。
        return None

    def is_empty(self) -> bool:
        """检查栈是否为空。

        返回：
            无任何帧时返回 True。
        """
        # 与 len(self.frames) == 0 等价，语义更直白。
        return len(self.frames) == 0

    def size(self) -> int:
        """返回栈的大小。

        返回：
            栈内帧数量（与 len(stack) 等价，供偏好显式方法的调用方使用）。
        """
        # 直接取列表长度。
        return len(self.frames)

    def clear(self) -> None:
        """清空栈。

        副作用：移除全部帧（会话重置/流程重启时调用）。
        """
        # 原地清空，保持列表对象引用不变。
        self.frames.clear()

    def __len__(self) -> int:
        """返回栈的大小（支持 len(stack) 语法）。"""
        # 魔法方法委托：与 size() 行为一致。
        return len(self.frames)

    def __iter__(self) -> Iterator[StackFrame]:
        """从栈顶到栈底迭代（默认迭代方向）。

        业务流程：返回列表的反向迭代器——最内层（最新压入）的上下文
        优先被处理，符合对话"先看当前活动上下文"的决策顺序。

        返回：
            StackFrame 迭代器（栈顶 -> 栈底）。
        """
        # reversed 产生反向迭代器，经 iter() 显式转换以满足注解。
        return iter(reversed(self.frames))

    def bottom_up(self) -> Iterator[StackFrame]:
        """从栈底到栈顶迭代。

        返回：
            StackFrame 迭代器（栈底 -> 栈顶），用于按压栈顺序回溯。
        """
        # 列表原生迭代顺序即压栈顺序（栈底 -> 栈顶）。
        return iter(self.frames)

    # ========== Flow相关操作 ==========

    def top_flow_frame(self) -> FlowStackFrame | None:
        """获取栈顶的Flow帧。

        业务流程：自栈顶向栈底查找第一个 FlowStackFrame（不要求活动
        状态，被中断的 Flow 也算）。

        返回：
            最近的 Flow 帧；栈中无 Flow 帧时返回 None。
        """
        # 复用默认迭代（栈顶优先），短路返回第一个匹配。
        for frame in self:
            # isinstance 区分具体帧类型（栈内混有搜索/闲聊等帧）。
            if isinstance(frame, FlowStackFrame):
                return frame
        # 遍历结束无匹配。
        return None

    def active_flow_frame(self) -> FlowStackFrame | None:
        """获取当前活动的Flow帧。

        业务流程：自栈顶向栈底查找第一个"处于活动状态"的 Flow 帧；
        被中断的帧被跳过，继续向栈底寻找。

        返回：
            活动 Flow 帧；不存在时返回 None。
        """
        for frame in self:
            # 双条件：Flow 帧类型 + 活动状态。
            if isinstance(frame, FlowStackFrame) and frame.is_active():
                return frame
        # 遍历结束无匹配。
        return None

    def push_flow(self, flow_id: str, step_id: str = "START") -> FlowStackFrame:
        """压入新的Flow帧。

        业务流程：1. 构造 FlowStackFrame（起点 START 或指定步骤）；
        2. 压栈；3. 返回该帧供调用方继续配置（如 slot_to_collect）。

        副作用：修改 frames 列表。

        参数：
            flow_id: Flow ID，对应 flows/ 目录中的流程定义名。
            step_id: 起始步骤 ID，默认入口步骤 START。

        返回：
            新创建并入栈的 FlowStackFrame 实例。
        """
        # 构造帧：仅传流程与步骤，其余字段用默认值。
        frame = FlowStackFrame(flow_id=flow_id, step_id=step_id)
        # 压栈使其成为当前活动上下文。
        self.push(frame)
        # 返回引用，便于调用方继续配置。
        return frame

    def find_flow_frame(self, flow_id: str) -> FlowStackFrame | None:
        """查找指定Flow的帧。

        业务流程：自栈顶向栈底查找 flow_id 匹配的 Flow 帧。

        返回：
            匹配的帧（同名 Flow 取最靠近栈顶者）；不存在时返回 None。
        """
        for frame in self:
            # 类型匹配 + 流程 ID 匹配，双条件同时满足。
            if isinstance(frame, FlowStackFrame) and frame.flow_id == flow_id:
                return frame
        # 遍历结束无匹配。
        return None

    def has_flow(self, flow_id: str) -> bool:
        """检查栈中是否包含指定Flow。

        返回：
            存在返回 True（含被中断的帧）。
        """
        # 委托 find_flow_frame，仅判存在性。
        return self.find_flow_frame(flow_id) is not None

    def get_all_flow_ids(self) -> list[str]:
        """获取栈中所有Flow的ID列表。

        返回：
            Flow ID 列表，从栈顶到栈底排序（含重复，反映嵌套层次）。
        """
        # 列表推导：沿默认迭代方向收集全部 Flow 帧的 ID。
        return [frame.flow_id for frame in self if isinstance(frame, FlowStackFrame)]

    def pop_to_flow(self, flow_id: str) -> list[StackFrame]:
        """弹出栈帧直到到达指定Flow。

        业务流程：逐个弹出栈顶帧，直到栈顶为目标 Flow（目标本身不弹出）；
        若目标 Flow 不在栈中，会弹空整个栈（调用方应先 has_flow 确认）。

        副作用：修改 frames 列表（可能被清空）。

        参数：
            flow_id: 目标 Flow ID。

        返回：
            被弹出的帧列表（按弹出顺序：原栈顶在前）。
        """
        # 收集被弹出的帧。
        popped = []
        # 循环条件兼做空栈保护：每轮都保证有帧可弹。
        while self.frames:
            top = self.top()
            # 到达目标 Flow：停止弹出（目标帧保留在栈顶）。
            if isinstance(top, FlowStackFrame) and top.flow_id == flow_id:
                break
            # 未到目标：直接移除栈顶——栈必非空，结果非 None，
            # 比经 pop() 包装更精确（避免 Optional 混入列表）。
            popped.append(self.frames.pop())
        # 返回弹出序列，供调用方记录或回滚。
        return popped

    def interrupt_top_flow(self) -> FlowStackFrame | None:
        """中断栈顶的Flow。

        业务流程：定位最近的 Flow 帧并置为 INTERRUPTED 状态
        （帧仍留在栈中，供后续恢复）。

        副作用：修改目标帧的 state。

        返回：
            被中断的 Flow 帧；栈中无 Flow 帧时返回 None。
        """
        # 定位最近的 Flow 帧（不要求活动状态）。
        flow_frame = self.top_flow_frame()
        # 仅在找到时迁移状态，避免空引用。
        if flow_frame:
            flow_frame.interrupt()
        # 返回被中断的帧（或 None）。
        return flow_frame

    # ========== 帧查找操作 ==========

    def find_frame(self, frame_id: str) -> StackFrame | None:
        """根据帧ID查找帧。

        业务流程：自栈顶向栈底按 frame_id 精确匹配。

        返回：
            匹配的帧；不存在时返回 None。
        """
        for frame in self:
            # 帧ID由 generate_frame_id 生成，进程内几乎不重复。
            if frame.frame_id == frame_id:
                return frame
        # 遍历结束无匹配。
        return None

    def find_frames_of_type(self, frame_type: type[StackFrame]) -> list[StackFrame]:
        """查找指定类型的所有帧。

        参数：
            frame_type: 帧类（如 SearchStackFrame），按 isinstance 匹配。

        返回：
            匹配的帧列表，从栈顶到栈底排序；无匹配返回空列表。
        """
        # 列表推导 + isinstance：传入父类可匹配其全部子类实例。
        return [frame for frame in self if isinstance(frame, frame_type)]

    def remove_frame(self, frame_id: str) -> StackFrame | None:
        """移除指定ID的帧。

        业务流程：沿压栈顺序（栈底 -> 栈顶）查找首个匹配帧并从栈中移除。

        副作用：修改 frames 列表（找到并移除时）。

        参数：
            frame_id: 目标帧 ID。

        返回：
            被移除的帧；不存在时返回 None。
        """
        # enumerate 同时获得下标（移除用）与帧（匹配用）。
        for i, frame in enumerate(self.frames):
            if frame.frame_id == frame_id:
                # pop(i) 按下标移除并返回该帧。
                return self.frames.pop(i)
        # 遍历结束无匹配。
        return None

    # ========== 序列化 ==========

    def as_dict(self) -> dict[str, Any]:
        """将栈转换为字典。

        业务流程：按压栈顺序（栈底 -> 栈顶）逐帧序列化，
        帧内部经各自的 as_dict 完成（含 type 路由键）。

        副作用：无（只读序列化，供持久化/调试）。

        返回：
            形如 {"frames": [...]} 的字典。
        """
        return {
            # 原始列表顺序 = 栈底 -> 栈顶，恢复时保持同一物理顺序。
            "frames": [frame.as_dict() for frame in self.frames]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DialogueStack":
        """从字典创建栈。

        业务流程：1. 逐条帧数据经注册表路由还原为具体帧实例；
        2. 无法解析的帧（缺失/未知 type）跳过以容忍脏数据；
        3. 以还原序列重建栈。

        副作用：创建新对象，不修改入参字典。

        参数：
            data: as_dict 产出的字典；frames 键允许缺失（按空栈处理）。

        返回：
            DialogueStack 实例。
        """
        # 收集成功还原的帧。
        frames = []
        # 逐条还原；frames 缺省空列表 => 空栈。
        for frame_data in data.get("frames", []):
            try:
                # 经注册表按 type 字段路由到具体帧类。
                frame = create_frame_from_dict(frame_data)
                frames.append(frame)
            except ValueError:
                # 容错设计：单帧损坏不拖垮整个会话恢复，仅跳过该帧。
                continue

        # 先构造空栈再整体赋值，保持帧的原始顺序。
        stack = cls()
        stack.frames = frames
        return stack

    def copy(self) -> "DialogueStack":
        """创建栈的副本。

        业务流程：经 as_dict -> from_dict 序列化往返实现隔离拷贝：
        帧对象与帧内可变字段均被重建，副本与原栈互不影响。

        副作用：创建新对象。

        返回：
            内容相同但完全隔离的新 DialogueStack 实例。
        """
        # 序列化往返：比 copy.deepcopy 更可控（只拷贝帧的声明字段）。
        return DialogueStack.from_dict(self.as_dict())

    def __repr__(self) -> str:
        """返回栈的字符串表示（调试用）。

        返回：
            形如 DialogueStack([Flow(refund@START) > search]) 的字符串
            （左栈顶、右栈底）；空栈返回 DialogueStack(empty)。
        """
        # 空栈特判，输出更直观。
        if self.is_empty():
            return "DialogueStack(empty)"

        # 逐帧生成简述：Flow 帧带流程与步骤，其余帧仅类型名。
        frame_strs = []
        for frame in self:
            if isinstance(frame, FlowStackFrame):
                # Flow 帧：flow_id@step_id 展示当前执行位置。
                frame_strs.append(f"Flow({frame.flow_id}@{frame.step_id})")
            else:
                # 其他帧：仅展示帧类型标识。
                frame_strs.append(f"{frame.frame_type()}")

        # " > " 连接：左侧栈顶、右侧栈底，直观展示嵌套层次。
        return f"DialogueStack([{' > '.join(frame_strs)}])"


# 导出白名单：仅暴露对话栈类；栈帧相关类型经 stack_frame 模块导入。
__all__ = ["DialogueStack"]
