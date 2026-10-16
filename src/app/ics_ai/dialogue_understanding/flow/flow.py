"""
flow - 对话流程数据结构定义

核心功能：定义对话流程的数据结构——步骤类型枚举 StepType、单个流程步骤
          FlowStep、完整流程 Flow、流程集合容器 FlowsList。本模块是 flows
          YAML 配置在内存中的对象化表示：从配置字典（YAML 解析结果）构建
          流程对象图，供策略层按步骤 ID 顺序/跳转执行。

执行流程：加载 flows 配置后，逐层调用 FlowsList/Flow/FlowStep 的 from_dict
          构建对象图（Flow.from_dict 会为未显式指定 next 的步骤自动串联
          后续步骤，并建立步骤 ID 索引）；运行时经 get_step/get_next_step
          等导航方法遍历流程图驱动对话。

副作用：仅定义类与内存对象，不读写文件/数据库、不发起网络请求、
        不修改全局状态。

主要依赖：仅标准库——collections.abc（Iterator）、dataclasses（dataclass/field）、
          enum（Enum）、typing（Any）。
"""

# 从 __future__ 导入 annotations：启用 PEP 563 延迟注解求值，
# 类内注解以字符串形式保存（支持前向引用，降低运行时求值开销）。
from __future__ import annotations

# 导入 Iterator：迭代器类型注解（PEP 585 起标准位置为 collections.abc，
# typing.Iterator 已弃用）。
from collections.abc import Iterator

# 导入 dataclass 与 field：dataclass 自动生成 __init__/__repr__ 等方法，
# field(default_factory=...) 为可变默认值（dict/list）提供独立工厂，
# 避免所有实例共享同一个默认对象。
from dataclasses import dataclass, field

# 导入 Enum：用于定义步骤类型枚举 StepType。
from enum import Enum

# 导入类型注解工具：仅保留 Any（任意类型，typing 中未弃用、无内置替代）；
# Dict/List/Optional 已按 PEP 604/585 迁移为 dict/list、X | None 写法，
# 原 Union 因全文件未使用一并删除。
from typing import Any


class StepType(str, Enum):
    """步骤类型枚举。

    职责：定义 Flow 步骤的七种类型；FlowStep.from_dict 依据步骤配置
          字典中的特征键判定类型，执行侧按类型分发处理逻辑。
    设计模式：继承 str 的混入枚举——成员可直接当字符串参与比较与序列化。
    生命周期：随模块导入创建，进程内全局唯一，全程序只读复用。

    成员：
    - ACTION: 执行动作（自定义 Action 或 utter 响应）
    - COLLECT: 收集槽位（询问用户并填充槽位）
    - LINK: 链接到另一个 Flow
    - SET_SLOT: 预设槽位值（不经用户确认，直接写入）
    - CONDITION: 条件分支（if-then-else）
    - END: 流程结束
    - CALL: 调用另一个 Flow（结束后返回）
    """

    # 列表项1：action 键对应的步骤——执行动作。
    ACTION = "action"
    # 列表项2：collect 键对应的步骤——收集槽位。
    COLLECT = "collect"
    # 列表项3：link 键对应的步骤——链接其他流程。
    LINK = "link"
    # 列表项4：set_slot/set_slots 键对应的步骤——预设槽位；
    # 枚举值沿用单数 "set_slot"，YAML 的复数键 set_slots 在
    # from_dict 中归一处理。
    SET_SLOT = "set_slot"
    # 列表项5：if/condition 键对应的步骤——条件分支。
    CONDITION = "condition"
    # 列表项6：显式结束步骤。
    END = "end"
    # 列表项7：call 键对应的步骤——调用其他流程。
    CALL = "call"


@dataclass
class FlowStep:
    """Flow步骤。

    职责：表示 Flow 中的一个执行步骤，承载该步骤的全部配置
          （类型/动作/槽位/条件跳转/描述等），提供与配置字典互转
          （from_dict/as_dict）与结束判断（is_end）能力。
    设计模式：数据传输对象（DTO）+ 工厂方法（from_dict）。
    生命周期：由 Flow.from_dict 解析 YAML 步骤配置时批量创建，
              随所属 Flow 存续，随 FlowsList/Domain 销毁。

    Attributes:
        id: 步骤ID
        step_type: 步骤类型
        action: 要执行的动作
            - 当step_type为ACTION时：要执行的动作
            - 当step_type为COLLECT时：可选，显式指定询问动作（utter_xxx或action_xxx）
        collect: 要收集的槽位（当step_type为COLLECT时）
        ask_before_filling: 是否在LLM预填充后仍询问用户确认（collect步骤可选）
        reset_after_flow_ends: flow结束时是否重置该槽位（collect步骤，默认True）
        next: 下一个步骤ID（YAML 中也可能写成条件分支列表，本类原样
              保存不做类型区分，由执行侧解释）
        condition: 条件表达式（当step_type为CONDITION时）
        then: 条件为真时的下一步
        else_: 条件为假时的下一步
        slots: 多槽位字典（当step_type为SET_SLOT时）
        flow_id: 要调用的Flow ID（当step_type为CALL或LINK时）
        description: 步骤描述
        metadata: 额外元数据
    """

    # 步骤 ID：流程内唯一标识，next/then/else 跳转的目标即此 ID；
    # 未显式指定时由 Flow.from_dict 自动编号为 step_{i}。
    id: str
    # 步骤类型：缺省 ACTION（from_dict 中无特征键命中时兜底）。
    step_type: StepType = StepType.ACTION
    # 动作名：ACTION 步骤为必填动作；COLLECT 步骤可选（显式指定询问动作）；
    # 无值时为 None。
    action: str | None = None
    # 要收集的槽位名：仅 COLLECT 步骤使用；无值时为 None。
    collect: str | None = None
    # 是否填充前强制询问：True 时每次进入都重新询问（避免槽位残留），
    # 缺省 False。
    ask_before_filling: bool = False
    # 流程结束后是否重置该槽位：缺省 True（临时性槽位用完即清）。
    reset_after_flow_ends: bool = True
    # 下一步骤 ID：None 表示未指定（由 Flow.from_dict 自动串联；
    # 条件分支步骤保存的是列表）。
    next: str | None = None
    # 条件表达式：仅 CONDITION 步骤使用（YAML 的 if 或 condition 键）。
    condition: str | None = None
    # 条件为真时的下一步骤 ID；无值时为 None。
    then: str | None = None
    # 条件为假时的下一步骤 ID；属性名带下划线（else 是 Python 关键字），
    # 对应 YAML 键名 else。
    else_: str | None = None
    # 预设槽位字典：仅 SET_SLOT 步骤使用，键为槽位名、值为要写入的值；
    # 可变默认值经 field 工厂独立创建。
    slots: dict[str, Any] = field(default_factory=dict)
    # 目标流程 ID：仅 LINK/CALL 步骤使用（link/call 键归一而来）。
    flow_id: str | None = None
    # 步骤描述：供生成询问话术与 LLM 理解用途；无值时为 None。
    description: str | None = None
    # 额外元数据：扩展信息字典，框架不直接消费；可变默认值经 field 工厂
    # 独立创建。
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, step_id: str, data: dict[str, Any]) -> "FlowStep":
        """从字典创建步骤。

        业务流程：1. 按特征键判定步骤类型（collect→COLLECT、link→LINK、
        set_slot/set_slots→SET_SLOT、if/condition→CONDITION、call→CALL、
        end→END，均未命中兜底 ACTION）；2. 逐字段提取配置（YAML 键
        if/else 映射到属性 condition/else_）；3. 将 set_slots 列表或
        set_slot 字典统一解析为 slots 字典；4. 提取 link/call 目标为
        flow_id；5. 构造 FlowStep 返回。

        副作用：创建新对象，不修改入参字典。

        参数：
            step_id: 步骤ID，字符串，流程内唯一。
            data: 步骤配置字典（YAML 解析结果），允许为空字典
                  （此时得到缺省 ACTION 步骤）。

        返回：
            FlowStep 实例；配置缺键按默认值兜底，不抛异常。
        """
        # 确定步骤类型：优先按"特征键"识别——键名是类型最直接的证据，
        # 均未命中时兜底为 ACTION。
        step_type = StepType.ACTION
        # 含 collect 键：收集槽位步骤。
        if "collect" in data:
            step_type = StepType.COLLECT
        # 含 link 键：链接其他流程步骤。
        elif "link" in data:
            step_type = StepType.LINK
        # 含 set_slot 或 set_slots 键：预设槽位步骤（两种 YAML 写法兼容）。
        elif "set_slot" in data or "set_slots" in data:
            step_type = StepType.SET_SLOT
        # 含 if 或 condition 键：条件分支步骤。
        elif "if" in data or "condition" in data:
            step_type = StepType.CONDITION
        # 含 call 键：调用其他流程步骤。
        elif "call" in data:
            step_type = StepType.CALL
        # id 或 action 为 "end"：显式结束步骤（命名约定识别）。
        elif data.get("id") == "end" or data.get("action") == "end":
            step_type = StepType.END

        # 提取字段：YAML 键名与属性名基本一致，仅 if/else 需映射到
        # condition/else_（else 是 Python 关键字，属性名加下划线）。
        action = data.get("action")
        collect = data.get("collect")
        ask_before_filling = data.get("ask_before_filling", False)
        reset_after_flow_ends = data.get("reset_after_flow_ends", True)
        next_step = data.get("next")
        # 条件表达式：if 键优先，缺省时退而取 condition 键（or 短路）。
        condition = data.get("if") or data.get("condition")
        then = data.get("then")
        else_ = data.get("else")

        # set_slot/set_slots 处理：两种 YAML 写法统一合并为 slots 字典。
        slots: dict[str, Any] = {}

        if "set_slots" in data:
            # set_slots 是列表格式：[{slot1: value1}, {slot2: value2}, ...]
            set_slots = data["set_slots"]
            # 仅当确实是列表时逐项合并（防御配置写错类型）。
            if isinstance(set_slots, list):
                for slot_item in set_slots:
                    # 仅合并字典项（跳过非法元素，保证健壮性）。
                    if isinstance(slot_item, dict):
                        slots.update(slot_item)
        elif "set_slot" in data:
            # set_slot 是字典格式：{slot: value}
            set_slot = data["set_slot"]
            # 仅当确实是字典时拷贝（防御配置写错类型）。
            if isinstance(set_slot, dict):
                slots = set_slot.copy()

        # link/call 处理：两者语义都是"指向另一个流程"，
        # link 优先，or 短路归一为一个 flow_id 属性。
        flow_id = data.get("link") or data.get("call")

        # 用提取的字段构造步骤实例：各实参与上方提取的变量一一对应。
        return cls(
            # id：使用入参 step_id（而非 data 内的 id 键），保证 ID 来源唯一。
            id=step_id,
            # step_type：上方判定的步骤类型。
            step_type=step_type,
            # action：动作名（可能为 None）。
            action=action,
            # collect：要收集的槽位名（可能为 None）。
            collect=collect,
            # ask_before_filling：是否填充前强制询问（缺省 False）。
            ask_before_filling=ask_before_filling,
            # reset_after_flow_ends：流程结束后是否重置槽位（缺省 True）。
            reset_after_flow_ends=reset_after_flow_ends,
            # next：下一步骤指向（可能为 None，由 Flow.from_dict 自动补全）。
            next=next_step,
            # condition：条件表达式（if/condition 键归一）。
            condition=condition,
            # then：条件成立时的跳转目标。
            then=then,
            # else_：条件不成立时的跳转目标（YAML 键名 else）。
            else_=else_,
            # slots：预设槽位字典（set_slots/set_slot 归一后的结果）。
            slots=slots,
            # flow_id：link/call 指向的目标流程 ID。
            flow_id=flow_id,
            # description：步骤描述（缺省 None）。
            description=data.get("description"),
            # metadata：额外元数据（缺省空字典）。
            metadata=data.get("metadata", {}),
        )

    def as_dict(self) -> dict[str, Any]:
        """转换为字典。

        业务流程：1. 以 id 起始组装字典；2. 各字段仅在非空/非默认值时
        写入（保证序列化结果精简）；3. slots 还原为 set_slots 列表格式，
        else_ 还原为 YAML 键名 else，flow_id 还原为 link 键。

        副作用：无（只读序列化，供持久化或调试输出）。

        返回：
            步骤配置字典，结构与 from_dict 入参 data 对应，
            可再次经 from_dict 重建等价步骤。
        """
        # 起始字典：id 恒写入（步骤的必需标识）；
        # 显式标注 dict[str, Any]——否则类型检查器会按字面量把 data 推断为
        # dict[str, str]，后续写入 list/bool 等异构值时报类型不匹配。
        data: dict[str, Any] = {"id": self.id}

        # 以下各字段按需序列化：仅在有意义（非空/非默认值）时写入键。
        # 有动作才写 action 键。
        if self.action:
            data["action"] = self.action
        # 有收集槽位才写 collect 键。
        if self.collect:
            data["collect"] = self.collect
        # ask_before_filling 仅在 True（非默认 False）时序列化。
        if self.ask_before_filling:
            data["ask_before_filling"] = self.ask_before_filling
        # reset_after_flow_ends 仅在非默认值（False）时序列化，
        # 默认 True 不写入。
        if not self.reset_after_flow_ends:
            data["reset_after_flow_ends"] = self.reset_after_flow_ends
        # 有下一步指向才写 next 键。
        if self.next:
            data["next"] = self.next
        # 有条件表达式才写 condition 键。
        if self.condition:
            data["condition"] = self.condition
        # 有条件成立分支才写 then 键。
        if self.then:
            data["then"] = self.then
        # 有条件不成立分支才写 else 键（属性名 else_ 还原为 YAML 键名）。
        if self.else_:
            data["else"] = self.else_
        # 有预设槽位才写 set_slots 键：还原为列表格式，  
        # 每个槽位单独包一层字典 {槽位名: 值}。
        if self.slots:
            data["set_slots"] = [{k: v} for k, v in self.slots.items()]
        # 有目标流程才写 link 键（flow_id 还原为 YAML 键名 link）。
        if self.flow_id:
            data["link"] = self.flow_id
        # 有描述才写 description 键。
        if self.description:
            data["description"] = self.description
        # 有元数据才写 metadata 键。
        if self.metadata:
            data["metadata"] = self.metadata

        return data

    def is_end(self) -> bool:
        """检查是否是结束步骤。

        返回：
            步骤类型为 END 或步骤 ID 为 "end" 时返回 True，
            否则返回 False（双重判据：显式类型 + 命名约定）。
        """
        # 类型为 END，或 id 命中 "end" 约定，都视为结束步骤。
        return self.step_type == StepType.END or self.id == "end"


@dataclass
class Flow:
    """对话流程。

    职责：定义一个完整的对话流程——承载流程元信息（ID/名称/描述）、
          步骤列表、槽位初始值与持久化槽位，并维护步骤 ID 索引，
          提供按 ID 取步、取首步、取下一步、取收集步骤等导航能力。
    设计模式：数据传输对象（DTO）+ 工厂方法（from_dict）+
              索引表（_step_index 加速按 ID 查找）。
    生命周期：由 Domain 解析 flows 配置时经 from_dict 创建，
              随 FlowsList/Domain 存续。

    Attributes:
        id: Flow ID
        description: Flow描述
        steps: 步骤列表
        name: Flow显示名称
        slot_initial_values: 槽位初始值
        persisted_slots: flow结束后仍保留的槽位列表
        metadata: 额外元数据
    """

    # 流程 ID：唯一标识，FlowsList 按 ID 索引，运行时据此查找流程。
    id: str
    # 流程描述：LLM 匹配流程的依据；缺省空串。
    description: str = ""
    # 步骤列表：流程的主体执行序列；可变默认值经 field 工厂独立创建。
    steps: list[FlowStep] = field(default_factory=list)
    # 流程显示名：Inspect 页面与日志展示；缺省 None（无显示名）。
    name: str | None = None
    # 槽位初始值：流程开始时预设的槽位默认值字典；可变默认值经
    # field 工厂独立创建。
    slot_initial_values: dict[str, Any] = field(default_factory=dict)
    # 持久化槽位：流程结束后不重置的槽位名列表（如 user_id）；
    # 可变默认值经 field 工厂独立创建。
    persisted_slots: list[str] = field(default_factory=list)
    # 额外元数据：扩展信息字典，框架不直接消费；可变默认值经
    # field 工厂独立创建。
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后处理。

        业务流程：dataclass 生成的 __init__ 在末尾自动调用本方法；
        遍历 steps 建立"步骤ID → 步骤对象"索引，供 get_step O(1) 查找。

        副作用：写入实例属性 _step_index（私有索引表）。
        """
        # 初始化步骤索引：键为步骤 ID，值为步骤对象。
        self._step_index: dict[str, FlowStep] = {}
        # 遍历步骤列表，逐个登记到索引。
        for step in self.steps:
            self._step_index[step.id] = step

    @classmethod
    def from_dict(cls, flow_id: str, data: dict[str, Any]) -> "Flow":
        """从字典创建Flow。

        业务流程：1. 遍历 steps 列表逐个解析：字典项经 FlowStep.from_dict
        构建（缺 id 键时自动编号 step_{i}），非字典项（纯动作名字符串）
        构造为 ACTION 步骤；2. 为未显式指定 next 的步骤自动指向下一个
        步骤（末步不设 next，运行至末尾自然结束）；3. 组装 Flow 实例。

        副作用：创建新对象，不修改入参字典。

        参数：
            flow_id: Flow ID，字符串，全局唯一。
            data: 流程配置字典（YAML 中单个 flow 的内容），
                  steps 允许缺失（按空步骤列表处理）。

        返回：
            Flow 实例，内部已建好步骤索引与 next 串联链。
        """
        # 步骤列表：逐个解析后收集到这里。
        steps = []
        # 取出 steps 配置：缺失时按空列表处理（无步骤流程）。
        steps_data = data.get("steps", [])

        # 带序号遍历步骤配置：序号 i 用于自动编号与定位下一项。
        for i, step_data in enumerate(steps_data):
            # 字典项：按常规路径解析（含 id 键时用之，否则自动编号）。
            if isinstance(step_data, dict):
                # 确定步骤 ID：显式 id 优先，缺失时按序号生成 step_{i}。
                step_id = step_data.get("id", f"step_{i}")
                # 委托 FlowStep.from_dict 完成单步解析。
                step = FlowStep.from_dict(step_id, step_data)
            else:
                # 非字典项：视为"简单动作名"，直接构造 ACTION 步骤。
                step = FlowStep(
                    id=f"step_{i}",
                    step_type=StepType.ACTION,
                    action=str(step_data),
                )

            # 自动串联 next：仅当该步骤未显式指定 next 且不是末步时执行；
            # 条件分支步骤的 next 为列表（非 None），自动跳过，不覆盖。
            if step.next is None and i < len(steps_data) - 1:
                # 取下一项步骤配置。
                next_step_data = steps_data[i + 1]
                if isinstance(next_step_data, dict):
                    # 字典项：下一步骤有显式 id 用之，否则按序号编号。
                    step.next = next_step_data.get("id", f"step_{i + 1}")
                else:
                    # 非字典项：下一步骤必为自动编号的 step_{i + 1}。
                    step.next = f"step_{i + 1}"

            # 登记解析完成的步骤。
            steps.append(step)

        # 组装流程实例：各字段与配置字典逐项对应，缺失键按默认值兜底。
        return cls(
            # id：入参流程 ID。
            id=flow_id,
            # description：流程描述（缺省空串）。
            description=data.get("description", ""),
            # steps：上方解析完成的步骤列表。
            steps=steps,
            # name：流程显示名（缺省 None）。
            name=data.get("name"),
            # slot_initial_values：槽位初始值（缺省空字典）。
            slot_initial_values=data.get("slot_initial_values", {}),
            # persisted_slots：流程结束后保留的槽位（缺省空列表）。
            persisted_slots=data.get("persisted_slots", []),
            # metadata：额外元数据（缺省空字典）。
            metadata=data.get("metadata", {}),
        )

    def as_dict(self) -> dict[str, Any]:
        """转换为字典。

        业务流程：1. 组装必需字段（id/description/steps）；
        2. 其余字段仅在非空时写入，精简序列化结果。

        副作用：无（只读序列化，供持久化或调试输出）。

        返回：
            流程配置字典，steps 为各步骤 as_dict 的列表；
            可再次经 from_dict 重建等价流程。
        """
        # 必需字段：id、描述、步骤列表（递归序列化每个步骤）；
        # 同样显式标注 dict[str, Any]——字面量推断出的值类型过窄
        # （str | list[dict[str, Any]]），后续写入 list[str]/dict 会报类型不匹配。
        data: dict[str, Any] = {
            "id": self.id,
            "description": self.description,
            "steps": [step.as_dict() for step in self.steps],
        }

        # 以下可选字段仅在非空时写入键。
        # 有显示名才写 name 键。
        if self.name:
            data["name"] = self.name
        # 有槽位初始值才写 slot_initial_values 键。
        if self.slot_initial_values:
            data["slot_initial_values"] = self.slot_initial_values
        # 有持久化槽位才写 persisted_slots 键。
        if self.persisted_slots:
            data["persisted_slots"] = self.persisted_slots
        # 有元数据才写 metadata 键。
        if self.metadata:
            data["metadata"] = self.metadata

        return data

    def get_step(self, step_id: str) -> FlowStep | None:
        """获取指定ID的步骤。

        参数：
            step_id: 步骤ID，字符串。

        返回：
            对应步骤对象；不存在返回 None（O(1) 索引查找）。
        """
        # 直接查索引表：__post_init__ 已建立 ID→步骤映射。
        return self._step_index.get(step_id)

    def get_first_step(self) -> FlowStep | None:
        """获取第一个步骤。

        返回：
            首个步骤对象；步骤列表为空返回 None。
        """
        # 先判空再取首元素，避免空列表取下标越界。
        return self.steps[0] if self.steps else None

    def get_next_step(self, current_step_id: str) -> FlowStep | None:
        """获取下一个步骤。

        业务流程：先按 ID 取当前步骤，再按其 next 指向取目标步骤。

        参数：
            current_step_id: 当前步骤ID，字符串。

        返回：
            下一个步骤对象；当前步骤不存在、或无 next 指向
            （流程末尾/条件分支步骤）时返回 None。
        """
        # 取当前步骤。
        current_step = self.get_step(current_step_id)
        # 当前步骤存在且有 next 指向时，取目标步骤；否则返回 None。
        if current_step and current_step.next:
            return self.get_step(current_step.next)
        return None

    def get_collect_steps(self) -> list[FlowStep]:
        """获取所有收集信息的步骤。

        返回：
            类型为 COLLECT 的步骤列表（保持步骤顺序）；无则返回空列表。
        """
        # 列表推导：过滤出步骤类型为 COLLECT 的步骤。
        return [step for step in self.steps if step.step_type == StepType.COLLECT]

    def get_slots_to_collect(self) -> list[str]:
        """获取需要收集的所有槽位名。

        返回：
            所有收集步骤的槽位名列表（保持步骤顺序）；无则返回空列表。
        """
        # 列表推导：取每个收集步骤的槽位名（collect 非空才计入）。
        return [step.collect for step in self.steps if step.collect]

    def __iter__(self) -> Iterator[FlowStep]:
        """迭代所有步骤。

        返回：
            步骤列表的迭代器（支持 for step in flow 直接遍历）。
        """
        # 委托给内部步骤列表的迭代器。
        return iter(self.steps)

    def __len__(self) -> int:
        """返回步骤数量。

        返回：
            步骤列表长度（len(flow) 即步骤数）。
        """
        # 委托给内部步骤列表的长度。
        return len(self.steps)


@dataclass
class FlowsList:
    """Flow列表容器。

    职责：管理多个 Flow 的集合——按 ID 索引增删查，支持迭代与
          成员判断，是 Domain 持有全部已加载流程的载体。
    设计模式：数据传输对象（DTO）+ 索引表（_flow_index 加速按 ID 查找）。
    生命周期：由 Domain 加载全部 flows 配置时创建（可经 add_flow
              逐个登记），随 Domain 存续。

    Attributes:
        flows: Flow列表
    """

    # Flow 列表：保持注册顺序；可变默认值经 field 工厂独立创建。
    flows: list[Flow] = field(default_factory=list)

    def __post_init__(self):
        """初始化后处理。

        业务流程：dataclass 生成的 __init__ 在末尾自动调用本方法；
        遍历 flows 建立"流程ID → Flow 对象"索引，供 get_flow O(1) 查找。

        副作用：写入实例属性 _flow_index（私有索引表）。
        """
        # 初始化流程索引：键为流程 ID，值为 Flow 对象。
        self._flow_index: dict[str, Flow] = {}
        # 遍历流程列表，逐个登记到索引。
        for flow in self.flows:
            self._flow_index[flow.id] = flow

    @property
    def flow_ids(self) -> list[str]:
        """获取所有Flow ID。

        返回：
            全部流程 ID 列表（保持注册顺序）。
        """
        # 由索引表取全部键并转为列表。
        return list(self._flow_index.keys())

    def get_flow(self, flow_id: str) -> Flow | None:
        """获取指定ID的Flow。

        参数：
            flow_id: Flow ID，字符串。

        返回：
            对应 Flow 对象；不存在返回 None（O(1) 索引查找）。
        """
        # 直接查索引表。
        return self._flow_index.get(flow_id)

    def add_flow(self, flow: Flow) -> None:
        """添加Flow。

        业务流程：同时登记到列表（保序）与索引表（加速查找）。

        副作用：修改实例的 flows 列表与 _flow_index 索引。

        参数：
            flow: 要添加的 Flow 对象；同 ID 重复添加时索引表条目
                  被后者覆盖（列表仍保留两条，调用方需保证 ID 唯一）。
        """
        # 追加到列表末尾，保持注册顺序。
        self.flows.append(flow)
        # 登记到索引表：同 ID 后者覆盖前者。
        self._flow_index[flow.id] = flow

    def remove_flow(self, flow_id: str) -> Flow | None:
        """移除Flow。

        业务流程：先从索引表弹出，命中再从列表移除，
        保证列表与索引两处状态一致。

        副作用：修改实例的 flows 列表与 _flow_index 索引。

        参数：
            flow_id: 要移除的 Flow ID，字符串。

        返回：
            被移除的 Flow 对象；ID 不存在返回 None（不抛异常）。
        """
        # 从索引表弹出：不存在时得到 None（pop 的默认值兜底）。
        flow = self._flow_index.pop(flow_id, None)
        # 仅在索引命中时才同步移除列表条目，避免误删。
        if flow:
            self.flows.remove(flow)
        return flow

    def has_flow(self, flow_id: str) -> bool:
        """检查是否存在指定Flow。

        参数：
            flow_id: Flow ID，字符串。

        返回：
            存在返回 True，否则返回 False。
        """
        # 直接查索引表成员。
        return flow_id in self._flow_index

    def __iter__(self) -> Iterator[Flow]:
        """迭代所有Flow。

        返回：
            Flow 列表的迭代器（支持 for flow in flows_list 直接遍历）。
        """
        # 委托给内部流程列表的迭代器。
        return iter(self.flows)

    def __len__(self) -> int:
        """返回Flow数量。

        返回：
            Flow 列表长度（len(flows_list) 即流程数）。
        """
        # 委托给内部流程列表的长度。
        return len(self.flows)

    def __contains__(self, flow_id: str) -> bool:
        """检查Flow是否存在。

        参数：
            flow_id: Flow ID，字符串（in 运算符右侧为本类实例时自动调用）。

        返回：
            存在返回 True，否则返回 False（与 has_flow 等价，
            支持 "流程ID" in flows_list 写法）。
        """
        # 委托给索引表成员判断。
        return flow_id in self._flow_index


# 导出清单：from flow import * 时仅暴露以下四个名字（按字母序排列）。
__all__ = [
    # 列表项1：对话流程。
    "Flow",
    # 列表项2：流程步骤。
    "FlowStep",
    # 列表项3：流程列表容器。
    "FlowsList",
    # 列表项4：步骤类型枚举。
    "StepType",
]
