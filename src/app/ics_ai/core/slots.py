"""
槽位系统模块
功能：
定义对话状态中的”槽位“（Slot）类型体系——抽象基类 Slot、六个具体类型（文本/布尔/浮点/列表/分类/任意）、
映射类型枚举 SlotMappingType、类型注册表 SLOT_TYPE_MAP 与 工厂函数 create_slot。
槽位用于存储对话过程中收集的信息（如订单号、退款金额），是对话状态（Tracker）的核心组成部分。
"""
from __future__ import annotations
from enum import Enum
from typing import Any
from app.ics_ai.shared.constants import (
    SLOT_TYPE_ANY,
    SLOT_TYPE_BOOL,
    SLOT_TYPE_CATEGORICAL,
    SLOT_TYPE_FLOAT,
    SLOT_TYPE_LIST,
    SLOT_TYPE_TEXT,
)
from app.ics_ai.shared.exceptions import InvalidSlotValueError
class SlotMappingType(str, Enum):
    """
    槽位映射类型枚举
    """
    # 列表项1：由 LLM 从用户消息中提取值并自动填充
    FROM_LLM = "from_llm"
    # 列表项2：由 Action 代码在业务执行时写入
    CONTROLLED = "controlled"
class Slot:
    """
    槽位抽象基类
    """
    # 类级默认类型名：子类各自覆盖为对应常量（text/bool...）
    # 用于错误提示与序列化的 type 字段
    type_name: str = SLOT_TYPE_ANY
    def __init__(
        self,
        name: str,  # 槽位名称，字符串，不允许为空
        initial_value: Any = None,  # 初始值，任意类型，允许为None（表示未设置）
        influence_conversation: bool = True,  # 是否影响对话流程，布尔值，默认True
        mappings: list[dict[str, Any]]
        | None = None,  # 映射表，列表，允许为None（按空列表处理）
        mapping_type: SlotMappingType
        | str = SlotMappingType.FROM_LLM,  # 映射类型，枚举或字符串（from_llm/controlled）
        description: str
        | None = None,  # 槽位描述，供 LLM 理解用途，允许为None（表示未设置）
    ):
        self.name = name
        self.initial_value = initial_value
        # 当前值：私有属性，统一经 value property 读写以强制类型验证
        self._value = initial_value
        self.influence_conversation = influence_conversation
        self.mappings = mappings or []
        # 槽位映射类型支持：字符串入参（YAML/domain 配置场景）转枚举
        if isinstance(mapping_type, str):
            self.mapping_type = SlotMappingType(mapping_type)
        # 保存统一为枚举后的映射类型
        self.mapping_type = mapping_type
        self.description = description
    # property 装饰器：把 getter 方法包装为只读属性
    @property
    def value(self) -> Any:
        """
        获取槽位当前值
        :return: 槽位当前值，任意类型；None 表示槽位尚未设置
        """
        return self._value
    # setter 装饰器：与上方 @property 同名配对，赋值语句
    # slot.value = x 会进入本方法，从而强制执行类型验证
    @value.setter
    def value(self, new_value: Any) -> None:
        """
        设置槽位当前值
        :param new_value: 要设置的值，任意类型
        """
        # 控制兜底：None 代表”未设置/清空“，不参与类型校验
        # 因此任何槽位都可以被充值微微设置状态
        if new_value is not None and not self._validate_value(new_value):
            # 验证失败：携带槽位名、非法值与期望类型抛出，便于定位配置错误
            raise InvalidSlotValueError(
                f"槽位'{self.name}'的值'{new_value}'类型无效，"
                f"期望类型为'{self.type_name}'"
            )
        # 验证通过：写入私有属性
        self._value = new_value
    def _validate_value(self, value: Any) -> bool:
        """
        验证槽位值是否有效
        :param value: 要验证的值，任意类型
        :return: 如果值符合类型要求则返回True，否则返回False
        """
        # 基类默认实现：不做任何限制
        return True
    def reset(self) -> None:
        """
        重置槽位值为初始值
        """
        # 直接写私有属性：初始值在构造时已确定，无需再验证
        self._value = self.initial_value
    def is_set(self) -> bool:
        """
        检查槽位是否已设置（非None）
        """
        # 以 None 作为 未设置 的唯一依据
        return self._value is not None
    def to_dict(self) -> dict[str, Any]:
        """
        转换为字典格式
        :return: 槽位字典，包含槽位名称、当前值、是否影响对话流程、映射表、映射类型、描述
        """
        # 组装序列化字典
        data = {
            "name": self.name,
            "type": self.type_name,
            "value": self._value,
            "initial_value": self.initial_value,
            "influence_conversation": self.influence_conversation,
            "mapping_type": self.mapping_type.value,  # pyright: ignore[reportAttributeAccessIssue]
        }
        # 描述可选：未配置时不写入，减少存储与传输的冗余字段
        if self.description:
            data["description"] = self.description
        return data
    # classmethod 装饰器：工厂方法的管用形式，cls 接收被调用的类
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Slot:
        """
        从字典创建槽位实例
        :param data: 槽位字典，包含槽位名称、当前值、是否影响对话流程、映射表、映射类型、描述
        :return: 槽位实例
        """
        # 类型缺省处理：旧数据/精简数据无 type 字段时按任意类型兜底
        slot_type = data.get("type", SLOT_TYPE_ANY)
        # 类型查找兜底：注册表未收录的类型名不抛出异常，退化为 AnySlot
        # 保证旧会话数据始终能恢复
        slot_class = SLOT_TYPE_MAP.get(slot_type, AnySlot)
        # 用对应槽位类构造实例：逐字段取值并给默认值
        slot = slot_class(  # pyright: ignore[reportOptionalCall]
            name=data["name"],
            initial_value=data.get("initial_value"),
            influence_conversation=data.get("influence_conversation", True),
            mappings=data.get("mappings"),
            mapping_type=data.get("mapping_type", SlotMappingType.FROM_LLM),
            description=data.get("description"),
        )
        # 设置当前值：直接写私有属性绕过 setter 验证
        if "value" in data:
            slot._value = data["value"]
        return slot
    def is_from_llm(self) -> bool:
        """
        检查槽位是否从 LLM 推理得到
        :return: 如果槽位映射类型为 FROM_LLM 则返回True，否则返回False
        """
        # 与枚举成员比较
        return self.mapping_type == SlotMappingType.FROM_LLM
    def is_controlled(self) -> bool:
        """
        检查槽位是否由 Action 控制填充
        :return: 如果槽位映射类型为 FROM_LLM 则返回False，否则返回True
        """
        return self.mapping_type == SlotMappingType.CONTROLLED
    def __repr__(self) -> str:
        """调试用字符串表示：类名 + 槽位名 + 当前值"""
        return f"{self.__class__.__name__}(name={self.name}, value={self._value})"
class TextSlot(Slot):
    """文本槽位，用于存储文本类型的信息"""
    # 类型标识覆盖为 text，用于错误提示与序列化
    type_name = SLOT_TYPE_TEXT
    def _validate_value(self, value: Any) -> bool:
        """
        验证槽位值是否为文本类型
        :param value: 要验证的值，任意类型
        :return: 如果值为文本类型则返回True，否则返回False
        """
        return isinstance(value, str)
class BoolSlot(Slot):
    """布尔槽位，用于存储布尔类型的信息"""
    # 类型标识覆盖为 bool，用于错误提示与序列化
    type_name = SLOT_TYPE_BOOL
    def _validate_value(self, value: Any) -> bool:
        """
        验证槽位值是否为布尔类型
        :param value: 要验证的值，任意类型
        :return: 如果值为布尔类型则返回True，否则返回False
        """
        return isinstance(value, bool)
class FloatSlot(Slot):
    """
    浮点数槽位，用于存储浮点数类型的信息
    """
    type_name = SLOT_TYPE_FLOAT
    def __init__(
        self,
        name: str,
        initial_value: Any = None,
        influence_conversation: bool = True,
        mappings: list[dict[str, Any]] | None = None,
        mapping_type: SlotMappingType | str = SlotMappingType.FROM_LLM,
        description: str | None = None,
        max_value: float | None = None,
        min_value: float | None = None,
    ):
        """
        初始化浮点槽位
        :param name: 槽位名称，唯一标识槽位
        :param initial_value: 初始槽位值，任意类型
        :param influence_conversation: 是否影响对话流程，默认True
        :param mappings: 映射表，用于将槽位值转换为其他类型
        :param mapping_type: 映射类型，默认FROM_LLM
        :param description: 描述槽位的功能或作用，用于文档化
        :param max_value: 最大允许值，默认None
        :param min_value: 最小允许值，默认None
        :return: None
        """
        super().__init__(
            name,
            initial_value,
            influence_conversation,
            mappings,
            mapping_type,
            description,
        )
        self.max_value = max_value
        self.min_value = min_value
    def _validate_value(self, value: Any) -> bool:
        """
        验证槽位值是否为浮点数类型
        :param value: 要验证的值，任意类型
        :return: 如果值为浮点数类型则返回True，否则返回False
        """
        # 第一步 类型校验：int/float 均接受
        if not isinstance(value, (int, float)):
            return False
        # 第二步 下限校验：仅当配置了 min_value 才比较
        if self.min_value is not None and value < self.min_value:
            return False
        # 第三步 上限校验：仅当配置了 max_value 才比较
        if self.max_value is not None and value > self.max_value:
            return False
        # 三项校验全部通过
        return True
class ListSlot(Slot):
    """列表槽位，用于存储列表类型的信息"""
    # 类型标识覆盖为 list，用于错误提示与序列化
    type_name = SLOT_TYPE_LIST
    def __init__(
        self,
        name: str,
        initial_value: Any = None,
        influence_conversation: bool = True,
        mappings: list[dict[str, Any]] | None = None,
        mapping_type: SlotMappingType | str = SlotMappingType.FROM_LLM,
        description: str | None = None,
    ):
        """
        初始化列表槽位
        :param name: 槽位名称，唯一标识槽位
        :param initial_value: 初始槽位值，任意类型
        :param influence_conversation: 是否影响对话流程，默认True
        :param mappings: 映射表，用于将槽位值转换为其他类型
        :param mapping_type: 映射类型，默认FROM_LLM
        :param description: 描述槽位的功能或作用，用于文档化
        :return: None
        """
        # 空值兜底：列表槽位的初始值统一为 [] 而非 None
        if initial_value is None:
            initial_value = []
        # 调用父类初始化方法
        super().__init__(
            name,
            initial_value,
            influence_conversation,
            mappings,
            mapping_type,
            description,
        )
    def _validate_value(self, value: Any) -> bool:
        """
        验证槽位值是否为列表类型
        :param value: 要验证的值，任意类型
        :return: 如果值为列表类型则返回True，否则返回False
        """
        return isinstance(value, list)
    def append(self, item: Any) -> None:
        """
        向列表槽位中添加元素
        :param item: 要添加的元素，任意类型
        :return: None
        """
        if self._value is None:
            self._value = []
        self._value.append(item)
class CategoricalSlot(Slot):
    """分类槽位，用于存储分类类型的信息"""
    # 类型标识覆盖为 categorical，用于错误提示与序列化
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
    ):
        """
        初始化分类槽位
        :param name: 槽位名称，唯一标识槽位
        :param initial_value: 初始槽位值，任意类型
        :param influence_conversation: 是否影响对话流程，默认True
        :param mappings: 映射表，用于将槽位值转换为其他类型
        :param mapping_type: 映射类型，默认FROM_LLM
        :param description: 描述槽位的功能或作用，用于文档化
        :param values: 允许的分类值，默认None
        :return: None
        """
        # 调用父类初始化方法
        super().__init__(
            name,
            initial_value,
            influence_conversation,
            mappings,
            mapping_type,
            description,
        )
        self.values = values or []
    def _validate_value(self, value: Any) -> bool:
        """
        验证槽位值是否为分类类型
        :param value: 要验证的值，任意类型
        :return: 如果值为分类类型则返回True，否则返回False
        """
        # 未配置约束时放行：允许列表为空是合法的”宽松模式“
        if not self.values:
            # values 为空，允许为空值
            return True
        # 严格模式校验：值必须再预定义集合内
        return value in self.values
class AnySlot(Slot):
    """任意类型槽位，用于存储任意类型的信息"""
    # 类型标识覆盖为 any，用于错误提示与序列化
    type_name = SLOT_TYPE_ANY
    def _validate_value(self, value: Any) -> bool:
        """
        验证槽位值是否为任意类型
        :param value: 要验证的值，任意类型
        :return: 恒返回True
        """
        # 不做任何限制：任意类型的值均合法
        return True
# 槽位类型注册表：类型名 -> 槽位类；
# create_slot 工厂与 from_dict 反序列化
# 共用此映射，新增槽位类型只需要在此等级一行
SLOT_TYPE_MAP: dict[str, type[Slot]] = {
    SLOT_TYPE_TEXT: TextSlot,  # 文本槽位
    SLOT_TYPE_BOOL: BoolSlot,  # 布尔槽位
    SLOT_TYPE_FLOAT: FloatSlot,  # 浮点数槽位
    SLOT_TYPE_LIST: ListSlot,  # 列表槽位
    SLOT_TYPE_CATEGORICAL: CategoricalSlot,  # 分类槽位
    SLOT_TYPE_ANY: AnySlot,  # 任意类型槽位（兜底）
}
def create_slot(
    name: str,
    slot_type: str = SLOT_TYPE_ANY,
    mapping_type: SlotMappingType | str = SlotMappingType.FROM_LLM,
    description: str | None = None,
    **kwargs: Any,
) -> Slot:
    """
    创建槽位实例
    :param name: 槽位名称，唯一标识槽位
    :param slot_type: 槽位类型，默认ANY
    :param mapping_type: 映射类型，默认FROM_LLM
    :param description: 描述槽位的功能或作用，用于文档化
    :param kwargs: 其他槽位初始化参数
    :return: 槽位实例
    """
    # 注册表查询 + 未知类型兜底：保证工厂函数对任何输入都能产出可用的槽位
    slot_class = SLOT_TYPE_MAP.get(slot_type, AnySlot)
    # 构造实例：公共参数显示传递，类型特有参数经 kwargs 透传
    return slot_class(
        name=name,
        mapping_type=mapping_type,
        description=description,
        **kwargs,
    )
# 导出白名单：按 ”基类 -> 各类型槽位 -> 映射枚举 -> 注册表 -> 工厂函数“ 顺序声明，
# 约束 “from app.ics_ai.core.slots import *” 导出范围
__all__ = [
    "Slot",
    "TextSlot",
    "BoolSlot",
    "FloatSlot",
    "ListSlot",
    "CategoricalSlot",
    "AnySlot",
    "SlotMappingType",
    "SLOT_TYPE_MAP",
    "create_slot",
]