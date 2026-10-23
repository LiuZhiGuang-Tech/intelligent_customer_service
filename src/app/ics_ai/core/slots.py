"""
slots - 槽位系统

核心功能：定义对话状态中的"槽位"（Slot）类型体系——抽象基类 Slot、六个具体
          类型（文本/布尔/浮点/列表/分类/任意）、映射类型枚举 SlotMappingType、
          类型注册表 SLOT_TYPE_MAP 与工厂函数 create_slot。槽位用于存储对话
          过程中收集的信息（如订单号、退款金额），是对话状态（Tracker）的
          核心组成部分。

执行流程：本模块被导入时定义全部类与注册表；运行时由 Domain/Tracker 通过
          create_slot 工厂或 from_dict 反序列化创建槽位实例，槽位值统一经
          value 属性读写——写入时按子类规则做类型/范围验证，失败抛出
          InvalidSlotValueError。

副作用：仅定义类、枚举与字典等内存对象，不读写文件/数据库、不发起网络请求、
        不修改全局状态。

槽位映射类型（SlotMappingType）：
- from_llm: 由LLM从用户输入中提取并填充
- controlled: 由Action填充，不由LLM自动提取

主要依赖：app.ics_ai.shared.constants（槽位类型常量）、
          app.ics_ai.shared.exceptions（InvalidSlotValueError）。
"""

# 导入 ABC：标记 Slot 为抽象基类（仅作类型层级标识；本模块未定义抽象方法，
# 因基类 _validate_value 已提供默认实现，由子类按需重写）。
from abc import ABC

# 导入 Enum：用于定义槽位映射类型枚举。
from enum import Enum

# 导入 Any：任意类型注解（typing 中未弃用的部分，无内置替代）；
# Optional/Union/Type 已按 PEP 604/585 改为 X | None、A | B、type[X] 写法。
from typing import Any

# 导入六个槽位类型常量：作为各子类的 type_name 标识与注册表键名，
# 保证字符串取值全系统一致（常量含义见 shared/constants.py）。
from app.ics_ai.shared.constants import (
    SLOT_TYPE_ANY,
    SLOT_TYPE_BOOL,
    SLOT_TYPE_CATEGORICAL,
    SLOT_TYPE_FLOAT,
    SLOT_TYPE_LIST,
    SLOT_TYPE_TEXT,
)

# 导入无效槽位值异常：value 属性写入校验失败时抛出。
from app.ics_ai.shared.exceptions import InvalidSlotValueError


class SlotMappingType(str, Enum):
    """槽位映射类型枚举。

    职责：定义槽位如何被填充，区分 LLM 自动提取与 Action 代码控制两条填充路径。
    设计模式：继承 str 的混入枚举——成员可直接当字符串参与比较与序列化
              （如 mapping_type.value 写入字典、与 YAML 配置值比对）。
    生命周期：随模块导入创建，进程内全局唯一，全程序只读复用。

    成员：
    - FROM_LLM: LLM从用户输入中提取值填充
    - CONTROLLED: Action代码控制填充，LLM不自动提取
    """

    # 列表项1：由 LLM 从用户消息中提取值并自动填充（订单号、退款原因等）。
    FROM_LLM = "from_llm"
    # 列表项2：由 Action 代码在业务执行时写入（如查询接口的返回值），
    # LLM 不允许直接设置，防止业务数据被对话内容污染。
    CONTROLLED = "controlled"


class Slot(ABC):
    """槽位抽象基类。

    职责：定义槽位的公共属性（名称/初始值/映射类型/描述）、经 property
          强制验证的值读写、重置/判空/序列化/反序列化等通用行为。
    设计模式：模板方法——_validate_value 由子类按类型规则重写，
              value.setter 统一调用；另含 from_dict 工厂方法。
    生命周期：由 Domain 加载配置或 Tracker 反序列化时创建，
              随所属 Tracker/Domain 存续。

    属性：
        name: 槽位名称（唯一标识，字符串，不允许为空）
        value: 当前值（经 property 读写，写入时验证；None 表示未设置）
        initial_value: 初始值（reset() 恢复到此值，允许 None）
        influence_conversation: 是否影响对话流程（布尔，默认 True）
        mappings: 槽位映射规则（字典列表，允许 None/空列表）
        mapping_type: 映射类型 (from_llm/controlled)
        description: 槽位描述（供LLM理解用途，允许 None）
    """

    # 类级默认类型名：子类各自覆盖为对应常量（text/bool/...），
    # 用于错误提示与序列化的 type 字段。
    type_name: str = "any"

    def __init__(
        self,
        name: str,
        initial_value: Any = None,
        influence_conversation: bool = True,
        mappings: list[dict[str, Any]] | None = None,
        mapping_type: SlotMappingType | str = SlotMappingType.FROM_LLM,
        description: str | None = None,
    ) -> None:
        """初始化槽位

        业务流程：1. 保存名称/初始值等基础属性；2. 对 None 的 mappings
        兜底为空列表；3. 将字符串形式的映射类型统一转为枚举。

        副作用：仅写入实例属性，无 IO 操作。

        参数：
            name: 槽位名称，字符串，不允许为空
            initial_value: 初始值，任意类型，允许 None（表示未设置）
            influence_conversation: 是否影响对话流程，默认 True
            mappings: 槽位映射规则列表，允许 None（按空列表处理）
            mapping_type: 映射类型，枚举或字符串（from_llm/controlled），
                          传入其他字符串将抛 ValueError
            description: 槽位描述，供 LLM 理解用途，允许 None
        """
        # 槽位名称：唯一标识，Tracker/Domain 按此名读写槽位。
        self.name = name
        # 初始值：reset() 时恢复的默认值。
        self.initial_value = initial_value
        # 当前值：私有属性，统一经 value property 读写以强制类型验证。
        self._value = initial_value
        # 是否影响对话：False 时该槽位变化不参与流程判断。
        self.influence_conversation = influence_conversation
        # 映射规则：None 兜底为空列表——既避免 None 判空散落各处，
        # 也避免可变默认参数在实例间共享。
        self.mappings = mappings or []

        # 槽位映射类型支持：字符串入参（YAML/domain 配置场景）转枚举，
        # 统一内部表示；值不在枚举范围内时由枚举构造抛 ValueError。
        if isinstance(mapping_type, str):
            mapping_type = SlotMappingType(mapping_type)
        # 保存统一为枚举后的映射类型。
        self.mapping_type = mapping_type
        # 槽位描述：拼入 LLM 提示词，帮助其理解该槽位收集什么信息。
        self.description = description

    # property 装饰器：把 getter 方法包装为只读属性，
    # 调用方以 slot.value 方式访问。
    @property
    def value(self) -> Any:
        """获取槽位当前值

        返回：
            当前值，任意类型；None 表示槽位尚未设置。
        """
        # getter 直接返回私有属性，不做任何加工。
        return self._value

    # setter 装饰器：与上方 @property 同名配对，赋值语句
    # slot.value = x 会进入本方法，从而强制执行类型验证。
    @value.setter
    def value(self, new_value: Any) -> None:
        """设置槽位值

        业务流程：1. None 直接放行（语义为清空槽位）；2. 非 None 值调用
        _validate_value 按子类规则验证；3. 验证通过才写入。

        副作用：修改实例的 _value 属性。

        参数：
            new_value: 新值，任意类型；None 允许（清空槽位）。

        异常：验证失败抛 InvalidSlotValueError，不写入、不返回。
        """
        # 空值兜底：None 代表"未设置/清空"，不参与类型验证，
        # 因此任何槽位都可以被重置为未设置状态。
        if new_value is not None and not self._validate_value(new_value):
            # 验证失败：携带槽位名、非法值与期望类型抛出，便于定位配置错误。
            raise InvalidSlotValueError(
                f"槽位 '{self.name}' 的值 '{new_value}' 类型无效，"
                f"期望类型: {self.type_name}"
            )
        # 验证通过（或值为 None）：写入私有属性。
        self._value = new_value

    def _validate_value(self, value: Any) -> bool:
        """验证槽位值是否有效

        业务流程：基类默认放行所有值；子类重写此方法实现具体的
        类型/范围验证逻辑（value.setter 会统一调用）。

        参数：
            value: 待验证的值，任意类型

        返回：
            有效返回 True，否则返回 False（不抛异常，由调用方决定报错方式）。
        """
        # 基类默认实现：不做任何限制。
        return True

    def reset(self) -> None:
        """重置槽位为初始值

        副作用：将 _value 恢复为 initial_value（多轮对话重启流程时调用）。
        """
        # 直接写私有属性：初始值在构造时已确定，无需再验证。
        self._value = self.initial_value

    def is_set(self) -> bool:
        """检查槽位是否已设置(非空)

        返回：
            _value 非 None 返回 True；未设置/被清空返回 False。
        """
        # 以 None 作为"未设置"的唯一判据。
        return self._value is not None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式

        业务流程：1. 组装基础字段（名称/类型/当前值/初始值/对话影响/
        映射类型字符串）；2. 描述非空时才附加，精简序列化结果。

        副作用：无（只读序列化，供持久化或调试输出）。

        返回：
            描述槽位状态的字典；mapping_type 已转为字符串值。
        """
        # 组装序列化字典：type 取类级 type_name，mapping_type 取枚举的字符串值。
        data = {
            # 键 name：槽位名称。
            "name": self.name,
            # 键 type：槽位类型名（子类覆盖后的 type_name）。
            "type": self.type_name,
            # 键 value：当前值。
            "value": self._value,
            # 键 initial_value：初始值（供反序列化后 reset 使用）。
            "initial_value": self.initial_value,
            # 键 influence_conversation：是否影响对话。
            "influence_conversation": self.influence_conversation,
            # 键 mapping_type：映射类型的字符串值（枚举转 str 便于持久化）。
            "mapping_type": self.mapping_type.value,
        }
        # 描述可选：未配置时不写入，减少存储与传输的冗余字段。
        if self.description:
            data["description"] = self.description
        # 返回组装完成的字典。
        return data

    # classmethod 装饰器：工厂方法的惯用形式，cls 接收被调用的类
    # （本模块中调用 Slot.from_dict(...) 时 cls 即 Slot 本身）。
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Slot":
        """从字典创建槽位实例

        业务流程：1. 读取 type 字段确定槽位类型（缺省按 any）；
        2. 从注册表查找对应槽位类（未知类型兜底 AnySlot）；
        3. 用字典字段构造实例；4. 若含 value 则直接写入。

        副作用：创建新对象，不修改入参字典。

        参数：
            data: to_dict 产出的字典；name 为必需键，
                  其余键允许缺失（按各字段默认值处理）。

        返回：
            对应类型的 Slot 实例；name 缺失时抛 KeyError。

        说明：工厂方法 + 注册表模式，使反序列化无需 if-else 穷举类型。
        """
        # 类型缺省处理：旧数据/精简数据无 type 字段时按任意类型兜底。
        slot_type = data.get("type", SLOT_TYPE_ANY)
        # 类型查找兜底：注册表未收录的类型名不抛异常，退化为 AnySlot，
        # 保证旧会话数据始终能恢复。
        slot_class = SLOT_TYPE_MAP.get(slot_type, AnySlot)

        # 用对应槽位类构造实例：逐字段取值并给默认值，
        # 与 __init__ 签名一一对应。
        slot = slot_class(
            # name：必需键，缺失时抛 KeyError（调用方需保证数据完整）。
            name=data["name"],
            # initial_value：缺省 None（未设置）。
            initial_value=data.get("initial_value"),
            # influence_conversation：缺省 True，与构造函数默认一致。
            influence_conversation=data.get("influence_conversation", True),
            # mappings：缺省 None（构造时兜底为空列表）。
            mappings=data.get("mappings"),
            # mapping_type：缺省 LLM 填充，与构造函数默认一致。
            mapping_type=data.get("mapping_type", SlotMappingType.FROM_LLM),
            # description：缺省 None（无描述）。
            description=data.get("description"),
        )

        # 设置当前值：直接写私有属性绕过 setter 验证——反序列化数据
        # 假定此前已通过验证；且历史数据可能不符合当前类型约束，
        # 硬校验会导致旧会话无法恢复。
        if "value" in data:
            slot._value = data["value"]

        # 返回重建的槽位实例。
        return slot

    def is_from_llm(self) -> bool:
        """检查槽位是否由LLM填充

        返回：
            映射类型为 FROM_LLM 时返回 True。
        """
        # 与枚举成员比较（str 混入枚举，可直接相等比较）。
        return self.mapping_type == SlotMappingType.FROM_LLM

    def is_controlled(self) -> bool:
        """检查槽位是否由Action控制填充

        返回：
            映射类型为 CONTROLLED 时返回 True。
        """
        # CONTROLLED 槽位只允许 Action 代码写入。
        return self.mapping_type == SlotMappingType.CONTROLLED

    def __repr__(self) -> str:
        """调试用字符串表示：类名 + 槽位名 + 当前值。"""
        # 拼接类名与关键状态，便于日志定位。
        return f"{self.__class__.__name__}(name={self.name}, value={self._value})"


class TextSlot(Slot):
    """文本槽位

    职责：存储字符串类型的值（如订单号、收货地址）。
    验证规则：仅接受 str 实例（生命周期与序列化行为随基类）。
    """

    # 类型标识覆盖为 text，用于错误提示与序列化。
    type_name = SLOT_TYPE_TEXT

    def _validate_value(self, value: Any) -> bool:
        """验证是否为字符串

        参数：
            value: 待验证的值

        返回：
            str 实例返回 True，否则 False。
        """
        # 严格类型检查：int/bool 等均不放行。
        return isinstance(value, str)


class BoolSlot(Slot):
    """布尔槽位

    职责：存储布尔类型的值（如"是否包邮"）。
    验证规则：仅接受 bool 实例。
    """

    # 类型标识覆盖为 bool。
    type_name = SLOT_TYPE_BOOL

    def _validate_value(self, value: Any) -> bool:
        """验证是否为布尔值

        参数：
            value: 待验证的值

        返回：
            bool 实例返回 True，否则 False。
        """
        # 注意 isinstance(1, bool) 为 False：整型不会伪装成布尔值通过。
        return isinstance(value, bool)


class FloatSlot(Slot):
    """浮点槽位

    职责：存储数值类型的值（如退款金额），支持设置取值范围。
    验证规则：先验类型（int/float 皆可），再验上下界范围。

    属性：
        min_value: 允许的最小值；None 表示不设下限
        max_value: 允许的最大值；None 表示不设上限
    """

    # 类型标识覆盖为 float。
    type_name = SLOT_TYPE_FLOAT

    def __init__(
        self,
        name: str,
        initial_value: Any = None,
        influence_conversation: bool = True,
        mappings: list[dict[str, Any]] | None = None,
        mapping_type: SlotMappingType | str = SlotMappingType.FROM_LLM,
        description: str | None = None,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        """初始化浮点槽位

        业务流程：1. 调用基类构造保存公共属性；2. 保存范围边界。

        参数：
            name/initial_value/influence_conversation/mappings/
            mapping_type/description: 含义同基类 Slot.__init__。
            min_value: 最小值，允许 None（不设下限）
            max_value: 最大值，允许 None（不设上限）
        """
        # 复用基类初始化：公共属性逻辑不重复实现，按位置逐一传参——
        # 依次为 槽位名/初始值/是否影响对话/映射规则/映射类型/描述。
        super().__init__(
            name,
            initial_value,
            influence_conversation,
            mappings,
            mapping_type,
            description,
        )
        # 范围下限：None 表示不限。
        self.min_value = min_value
        # 范围上限：None 表示不限。
        self.max_value = max_value

    def _validate_value(self, value: Any) -> bool:
        """验证是否为有效数值

        业务流程：1. 类型校验（int 或 float）；2. 下限校验；
        3. 上限校验；任一环节失败即返回 False。

        参数：
            value: 待验证的值

        返回：
            通过返回 True，否则 False。
        """
        # 第一步 类型校验：int/float 均接受（用户说"退50"解析为 int 也合法）。
        if not isinstance(value, (int, float)):
            return False

        # 第二步 下限校验：仅当配置了 min_value 才比较（None 不限制）。
        if self.min_value is not None and value < self.min_value:
            return False

        # 第三步 上限校验：仅当配置了 max_value 才比较。
        if self.max_value is not None and value > self.max_value:
            return False

        # 三项校验全部通过。
        return True


class ListSlot(Slot):
    """列表槽位

    职责：存储列表类型的值（如多件商品 ID）。
    验证规则：仅接受 list 实例；提供 append 便捷追加。
    """

    # 类型标识覆盖为 list。
    type_name = SLOT_TYPE_LIST

    def __init__(
        self,
        name: str,
        initial_value: Any = None,
        influence_conversation: bool = True,
        mappings: list[dict[str, Any]] | None = None,
        mapping_type: SlotMappingType | str = SlotMappingType.FROM_LLM,
        description: str | None = None,
    ) -> None:
        """初始化列表槽位

        业务流程：1. 将 None 初始值兜底为空列表；2. 调用基类构造。

        参数：各参数含义同基类 Slot.__init__。
        """
        # 空值兜底：列表槽位的初始值统一为 [] 而非 None，
        # 保证 append/reset 后的取值行为一致。
        if initial_value is None:
            initial_value = []
        # 复用基类初始化，按位置传参——
        # 依次为 槽位名/兜底后的初始值/是否影响对话/映射规则/映射类型/描述。
        super().__init__(
            name,
            initial_value,
            influence_conversation,
            mappings,
            mapping_type,
            description,
        )

    def _validate_value(self, value: Any) -> bool:
        """验证是否为列表

        参数：
            value: 待验证的值

        返回：
            list 实例返回 True，否则 False。
        """
        # 仅接受列表类型。
        return isinstance(value, list)

    def append(self, item: Any) -> None:
        """向列表添加元素

        业务流程：1. 若当前值为 None（被显式清空过）先重建空列表；
        2. 追加元素。注意：直接操作 _value 不经 setter，元素本身不校验。

        副作用：修改 _value 指向的列表（原地追加）。

        参数：
            item: 待追加元素，任意类型。
        """
        # 空值兜底：值被置 None 后仍可继续追加。
        if self._value is None:
            self._value = []
        # 原地追加到列表末尾。
        self._value.append(item)


class CategoricalSlot(Slot):
    """分类槽位

    职责：只允许设置预定义值列表中的值（如"仅退款/退货退款"）。
    验证规则：允许列表为空时不限制；否则严格成员校验。

    属性：
        values: 允许的值列表
    """

    # 类型标识覆盖为 categorical。
    type_name = SLOT_TYPE_CATEGORICAL

    def __init__(
        self,
        name: str,
        initial_value: Any = None,
        influence_conversation: bool = True,
        mappings: list[dict[str, Any]] | None = None,
        mapping_type: SlotMappingType | str = SlotMappingType.FROM_LLM,
        description: str | None = None,
        values: list[Any] | None = None,
    ) -> None:
        """初始化分类槽位

        业务流程：1. 调用基类构造保存公共属性；2. 对 None 的 values
        兜底为空列表（语义为"未配置约束"）。

        参数：
            name/initial_value/influence_conversation/mappings/
            mapping_type/description: 含义同基类 Slot.__init__。
            values: 允许的值列表，允许 None（按空列表处理）。
        """
        # 复用基类初始化，按位置传参——
        # 依次为 槽位名/初始值/是否影响对话/映射规则/映射类型/描述。
        super().__init__(
            name,
            initial_value,
            influence_conversation,
            mappings,
            mapping_type,
            description,
        )
        # 允许值列表：None 兜底为空列表，避免后续判空。
        self.values = values or []

    def _validate_value(self, value: Any) -> bool:
        """验证值是否在允许列表中

        业务流程：1. 允许列表为空视为未配置约束，放行任意值；
        2. 否则要求 value 必须是列表成员。

        参数：
            value: 待验证的值

        返回：
            通过返回 True，否则 False。
        """
        # 未配置约束时放行：允许列表为空是合法的"宽松模式"。
        if not self.values:
            return True
        # 严格成员校验：值必须在预定义集合内。
        return value in self.values


class AnySlot(Slot):
    """任意类型槽位

    职责：接受任何类型的值，不进行类型验证（如存放接口返回的原始字典）。
    验证规则：恒通过。
    """

    # 类型标识覆盖为 any。
    type_name = SLOT_TYPE_ANY

    def _validate_value(self, value: Any) -> bool:
        """接受任何值

        参数：
            value: 待验证的值

        返回：
            恒返回 True。
        """
        # 不做任何限制：任意类型的值均合法。
        return True


# 槽位类型注册表：类型名 -> 槽位类；create_slot 工厂与 from_dict 反序列化
# 共用此映射，新增槽位类型只需在此登记一行。
SLOT_TYPE_MAP: dict[str, type[Slot]] = {
    # text -> 文本槽位。
    SLOT_TYPE_TEXT: TextSlot,
    # bool -> 布尔槽位。
    SLOT_TYPE_BOOL: BoolSlot,
    # float -> 浮点槽位。
    SLOT_TYPE_FLOAT: FloatSlot,
    # list -> 列表槽位。
    SLOT_TYPE_LIST: ListSlot,
    # categorical -> 分类槽位。
    SLOT_TYPE_CATEGORICAL: CategoricalSlot,
    # any -> 任意类型槽位（兜底）。
    SLOT_TYPE_ANY: AnySlot,
}


def create_slot(
    name: str,
    slot_type: str = SLOT_TYPE_ANY,
    mapping_type: SlotMappingType | str = SlotMappingType.FROM_LLM,
    description: str | None = None,
    **kwargs: Any,
) -> Slot:
    """创建槽位实例

    业务流程：1. 按类型名从注册表查找槽位类（未知类型兜底 AnySlot）；
    2. 用公共参数与 kwargs 透传参数构造实例。

    副作用：创建新对象，无 IO 操作。

    参数：
        name: 槽位名称
        slot_type: 槽位数据类型 (text/bool/float/list/categorical/any)；
                   未知类型名不抛异常，按 any 处理
        mapping_type: 映射类型 (from_llm/controlled)，非法字符串抛 ValueError
        description: 槽位描述（供LLM理解用途）
        **kwargs: 额外的槽位配置，原样透传给具体槽位类的构造函数
                  （如 FloatSlot 的 min_value/max_value、
                  CategoricalSlot 的 values）

    返回：
        对应类型的 Slot 实例。

    示例：
        >>> # 创建由LLM填充的文本槽位
        >>> slot = create_slot("food_type", "text", mapping_type="from_llm")
        >>> slot.value = "pizza"

        >>> # 创建由Action控制的槽位
        >>> slot = create_slot("order_id", "text", mapping_type="controlled")
    """
    # 注册表查找 + 未知类型兜底：保证工厂函数对任何输入都能产出可用槽位。
    slot_class = SLOT_TYPE_MAP.get(slot_type, AnySlot)
    # 构造实例：公共参数显式传递，类型特有参数经 kwargs 透传。
    return slot_class(
        name=name, mapping_type=mapping_type, description=description, **kwargs
    )


# 导出白名单：按"基类 -> 各类型槽位 -> 映射枚举 -> 注册表 -> 工厂函数"
# 顺序声明，约束 `from app.ics_ai.core.slots import *` 的导出范围。
__all__ = [
    # 列表项1：槽位抽象基类。
    "Slot",
    # 列表项2：文本槽位。
    "TextSlot",
    # 列表项3：布尔槽位。
    "BoolSlot",
    # 列表项4：浮点槽位。
    "FloatSlot",
    # 列表项5：列表槽位。
    "ListSlot",
    # 列表项6：分类槽位。
    "CategoricalSlot",
    # 列表项7：任意类型槽位。
    "AnySlot",
    # 列表项8：槽位映射类型枚举。
    "SlotMappingType",
    # 列表项9：类型注册表。
    "SLOT_TYPE_MAP",
    # 列表项10：槽位工厂函数。
    "create_slot",
]
