"""
slot_commands - 槽位命令族模块

核心功能：定义对话槽位读写的两个命令——SetSlotCommand（设置槽位
          值：text 类型槽位强制转字符串，写入失败吞异常返回空事件）、
          ResetSlotCommand（重置槽位为初始值，槽位不存在时静默返回
          空事件）；另提供模块级工具函数 clean_extracted_value，把
          DSL 捕获的字符串值归一化为 Python 类型（None/bool/数字/
          字符串）。

执行流程：
    1. 导入期：两个命令类经 @register_command 以 set_slot/reset_slot
       为键登记进 base 模块的 _COMMAND_REGISTRY；
    2. 构造期：LLM 输出的命令字典经 from_dict 构造（必需字段缺失
       抛 ValueError），或命令 DSL 文本经基类 from_dsl（re.match +
       IGNORECASE）→ regex_pattern → _from_regex_match 构造（支持
       "set slot <name> <value>" 与 SetSlot(<name>, <value>) 两种
       写法）；
    3. 执行期：CommandProcessor 调用 run()——SetSlotCommand 向
       tracker 写入槽位值并产出 slot_set 事件，ResetSlotCommand
       调用槽位对象的 reset() 并产出 slot_reset 事件。

副作用：
    - 导入期：向 base 模块全局注册表 _COMMAND_REGISTRY 写入两个
      命令类（全局可变状态，跨会话持续生效）；
    - run 的具体实现：SetSlotCommand 修改 tracker 槽位值，
      ResetSlotCommand 调用槽位 reset()（均为 tracker 状态变更）；
      两者失败时均静默返回空事件列表（不抛异常、不中断命令流）；
    - 无文件/数据库读写、无网络请求。

主要依赖：
    - re：from_dsl 的 DSL 文本正则解析；
    - dataclasses：@dataclass 自动生成命令类的数据类方法；
    - typing：Any + TYPE_CHECKING（注解已按 PEP 585/604 新式写法）；
    - app.ics_ai.dialogue_understanding.commands.base：命令基类
      Command 与注册装饰器 register_command；
    - app.ics_ai.core.tracker：DialogueStateTracker（仅注解用，
      TYPE_CHECKING 隔离）。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，TYPE_CHECKING
# 分支中的 tracker 前向引用无需引号即可书写，PEP 604 联合类型注解
# （X | None）也不受运行时版本限制。
from __future__ import annotations

# 导入正则引擎：基类 from_dsl 的模式匹配（regex_pattern +
# _from_regex_match）依赖它；本文件仅 _from_regex_match 的
# re.Match[str] 类型注解直接引用 re。
import re

# 导入 dataclass 装饰器：两个命令类靠它自动生成 __init__ / __repr__。
from dataclasses import dataclass

# 导入类型注解工具：Any 类型通配符；TYPE_CHECKING 隔离仅注解导入
# （PEP 585/604 新式注解下 Dict/List/Optional 已不需要）。
from typing import Any, TYPE_CHECKING

# 导入命令基类与注册装饰器：两个命令类继承 Command 并经
# register_command 注册。
from app.ics_ai.dialogue_understanding.commands.base import Command, register_command

# 类型检查期专属导入：tracker 类型只出现在方法注解中，运行时不
# 导入（避免与 core 包形成循环依赖）。
if TYPE_CHECKING:
    # 对话状态追踪器：命令 run 的操作对象（仅注解用）。
    from app.ics_ai.core.tracker import DialogueStateTracker


def clean_extracted_value(value: str) -> Any:
    """清理并归一化从 DSL 中提取的值。

    业务流程：1. None 直接透传；2. strip 去首尾空白；3. 剥去完整
              包裹的单/双引号；4. 特殊词归一化（null/none→None、
              true→True、false→False，大小写不敏感）；5. 尝试数字
              转换（含"."转 float，否则转 int，失败保持字符串）。

    副作用：无（纯函数，不修改任何状态）。

    参数：
        value: DSL 捕获组提取的原始字符串值，允许 None。

    返回：
        归一化后的值——None/bool/int/float/str 之一；数字格式非法
        时保持原字符串返回（不抛异常）。
    """
    # 空值兜底：捕获组未命中（None）时直接透传。
    if value is None:
        return None

    # 去除首尾空白：DSL 书写可能带多余空格。
    value = value.strip()

    # 剥引号：仅当首尾为配对的单引号或双引号时去除（完整包裹才剥，
    # 避免误伤引号不配对的文本）；剥引号先于特殊词判断，因此带引号
    # 的 "null" 同样会归一化为 None。
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        value = value[1:-1]

    # 特殊词归一化：LLM 可能以文本形式输出空值/布尔值，统一转
    # Python 类型（lower 兼容 NULL/True 等大小写变体）。
    value_lower = value.lower()
    # 空值词：null 与 none 均归一化为 None。
    if value_lower == "null" or value_lower == "none":
        return None
    # 布尔词：true/false 归一化为对应布尔值。
    if value_lower == "true":
        return True
    if value_lower == "false":
        return False

    # 数字转换：含小数点按 float，否则按 int；非法数字格式（如
    # "abc"）捕获 ValueError 后保持字符串原样返回。
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    # 非特殊词且非数字：作为普通字符串返回。
    return value


@register_command
@dataclass
class SetSlotCommand(Command):
    """设置槽位命令。

    职责：把用户消息中识别出的信息写入对话状态槽位——LLM 识别到
          用户提供的信息（如订单号）后生成此命令；run() 经
          tracker.set_slot 写入槽位值（text 类型槽位强制转字符串），
          失败时记日志并返回空事件（不中断命令流）。
    设计模式：命令模式 + 注册表工厂（@register_command 按 DSL 名
              set_slot 登记）；即时数据命令——run() 直接修改
              tracker 状态。
    生命周期：LLM 输出解析（from_dict/from_dsl）或按钮 payload
              直构（understand 节点的 parse_set_slots_payload）时
              创建，CommandProcessor 执行 run() 后随单轮消息处理
              流程销毁。

    使用方（构造/校验位置）：
        - CommandProcessor：isinstance 校验槽位在 Domain 中的存在性，
          force_slot_filling 模式下仅保留设置当前槽位的实例；
        - command_parser：解析 LLM 字典输出时校验 name 必填；
        - understand 节点：/SetSlots(...) payload 绕过 LLM 直接构造。

    Attributes:
        name: 槽位名称
        value: 槽位值
        extractor: 提取器类型（llm, nlu, form等）
    """

    # 槽位名称：必须与 Domain 中定义的槽位名一致（写入不存在的槽位
    # 会在 run 中失败）；正则约束为字母/下划线开头。
    name: str
    # 槽位值：类型任意（字符串/数字/布尔/None 均可），text 类型槽位
    # 会在 run 中被强制转为字符串。
    value: Any
    # 提取器类型：记录值的来源通道（llm/nlu/form 等），默认 LLM 提取；
    # 该字段不参与 DSL 序列化与解析（DSL 往返后恒为默认值）。
    extractor: str = "llm"

    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。

        返回：
            命令名 set_slot——注册表键与 DSL 语法动词。
        """
        # 注册表键：LLM 字典输出中 "command" 字段的取值即此名。
        return "set_slot"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SetSlotCommand":
        """从字典创建命令。

        业务流程：读取 name/value（必需）与 extractor（可选，缺省
                  "llm"）构造实例；必需键缺失时把 KeyError 转译为
                  带字段说明的 ValueError。

        副作用：无（纯构造，仅异常转译）。

        参数：
            data: LLM 输出的命令字典，须含 "name"/"value" 键。

        返回：
            SetSlotCommand 实例。

        异常：
            ValueError：data 缺少 "name" 或 "value" 必需键时抛出。
        """
        # 必需键用下标访问（缺失即 KeyError），捕获后转译为更友好的
        # ValueError 并保留原始异常链（from e）。
        try:
            return SetSlotCommand(
                name=data["name"],
                value=data["value"],
                extractor=data.get("extractor", "llm"),
            )
        except KeyError as e:
            # 转译：告知调用方具体缺失的必需字段。
            raise ValueError(f"Missing required field for SetSlotCommand: {e}") from e

    def run(
        self,
        tracker: DialogueStateTracker,
        flows: Any | None = None,
    ) -> list[dict[str, Any]]:
        """执行设置槽位命令。

        业务流程：1. 查询槽位定义对象，text 类型槽位把值强制转为
                  字符串；2. 经 tracker.set_slot 写入槽位值；3. 返回
                  slot_set 事件（携带名称/值/提取器）。写入失败时
                  记错误日志并返回空事件列表。

        副作用：修改 tracker 状态（写入槽位值）；失败时写日志。

        参数：
            tracker: 对话状态追踪器（槽位定义与写入目标）。
            flows: 可用的Flow列表，允许 None（本命令未使用）。

        返回：
            单元素事件列表（event=slot_set，含 name/value/extractor
            与时间戳占位）；写入失败时返回空列表（不抛异常）。
        """
        # 获取槽位对象以检查类型：tracker.slots 为槽位定义字典，
        # 未定义的槽位名得到 None。
        slot_obj = tracker.slots.get(self.name)
        # 待写入值：先取命令原值，后续按槽位类型可能转换。
        value_to_set = self.value

        # 如果槽位是 text 类型，将值转换为字符串：
        # hasattr 探测槽位对象是否声明 type_name 属性（鸭子类型兼容，
        # 非标准槽位对象跳过类型转换）。
        if slot_obj and hasattr(slot_obj, "type_name"):
            # text 槽位且值非 None：统一转字符串（保证槽位值类型一致，
            # 如按钮 payload 传入的 int 订单号）。
            if slot_obj.type_name == "text" and value_to_set is not None:
                value_to_set = str(value_to_set)

        # 设置槽位值：槽位不存在等失败场景在此捕获，避免单条命令
        # 异常中断整个命令流的执行。
        try:
            # 写入槽位：真正的状态变更发生在这里。
            tracker.set_slot(self.name, value_to_set)
            # 返回槽位设置事件列表：
            return [
                {
                    # 事件类型标记：由下游 Policy/Action 识别。
                    "event": "slot_set",
                    # 事件负载：槽位名与实际写入值（可能经类型转换）。
                    "name": self.name,
                    "value": value_to_set,
                    # 事件负载：值来源通道（llm/nlu/form 等）。
                    "extractor": self.extractor,
                    # 事件时间戳占位：由 tracker 后续补齐。
                    "timestamp": None,
                }
            ]
        except Exception as e:
            # 槽位不存在或设置失败：记错误日志后吞掉异常。
            # 局部导入 logging：标准库模块有缓存，重复导入开销极低，
            # 把依赖范围最小化到失败分支。
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"SetSlotCommand failed to set slot '{self.name}': {e}")
            # 失败返回空事件：调用方按"无事件"继续处理后续命令。
            return []

    def to_dsl(self) -> str:
        """转换为DSL字符串。

        业务流程：按值类型选择输出格式——字符串值双引号包裹、
                  None 输出 null 字面量、其余类型直接输出字面量。

        副作用：无。

        返回：
            DSL 文本："set slot <name> <value>" 形式（extractor
            不参与 DSL 序列化）。
        """
        # 根据值类型决定是否添加引号：
        # 字符串值：双引号包裹（与 regex_pattern 分支 A 的引号规则
        # 对应，解析端 clean_extracted_value 会剥引号还原）。
        if isinstance(self.value, str):
            return f'set slot {self.name} "{self.value}"'
        # None 值：显式输出 null 字面量（解析端归一化回 None）。
        elif self.value is None:
            return f"set slot {self.name} null"
        # 其他类型（int/float/bool）：直接输出字面量（bool 输出
        # True/False，解析端归一化回布尔值，往返对称）。
        else:
            return f"set slot {self.name} {self.value}"

    @classmethod
    def regex_pattern(cls) -> str:
        """返回正则表达式模式。

        返回：
            双分支正则——"set slot <name> <value>" 与旧别名
            "SetSlot(<name>, <value>)"；整体无 ^ 行首锚定（解析端
            from_dsl 用 re.match 隐式行首锚定，分支 A 以 $ 收尾）。
        """
        # 正则逐段拆解（两分支覆盖同一命令的新旧两种 LLM 输出写法；
        # 基类 from_dsl 以 re.match + IGNORECASE 使用本模式——大小写
        # 不敏感，隐式行首锚定）：
        # (?: A | B )        —— 非捕获分组：仅分组不捕获，包裹两个等价写法分支；
        # —— 分支 A：短语形式 set slot <name> <value> ——
        # set\s+slot        —— DSL 动词短语（词内空白任意，IGNORECASE
        #                      下大小写不敏感）；
        # \s+               —— 动词与槽位名之间的空白；
        # ['"`]?            —— 槽位名的起始引号，可选（不捕获，单/双/
        #                      反引号均可）；
        # ([a-zA-Z_][a-zA-Z0-9_-]*)  —— 捕获组1：槽位名（字母/下划线
        #                      开头，后接字母/数字/下划线/连字符）；
        # ['"`]?            —— 槽位名的收尾引号，可选（不捕获）；
        # \s+               —— 槽位名与值之间的空白（至少一个）；
        # ['"`]?            —— 值的起始引号，可选（不捕获）；
        # (.+?)             —— 捕获组2：值文本（至少一个字符；惰性匹配
        #                      配合末尾 $——尽早停止扩张，把收尾引号
        #                      留给组后的 ['"`]?，与 to_dsl 输出的
        #                      引号形式往返对称）；
        # ['"`]?            —— 值的收尾引号，可选（不捕获）；
        # $                 —— 值结尾锚定（分支 A 要求值到行尾）；
        # —— 分支 B：旧别名函数形式 SetSlot(<name>, <value>) ——
        # SetSlot\(         —— 旧别名动词与左括号（IGNORECASE 下实际
        #                      大小写不敏感）；
        # ['"]?             —— 槽位名的起始引号，可选（不捕获，仅单/
        #                      双引号）；
        # ([a-zA-Z_][a-zA-Z0-9_-]*)  —— 捕获组3：槽位名（规则同组1）；
        # ['"]?             —— 槽位名的收尾引号，可选（不捕获）；
        # ,\s*              —— 逗号分隔（逗号后空白任意）；
        # ['"]?             —— 值的起始引号，可选（不捕获）；
        # (.*?)             —— 捕获组4：值文本（惰性匹配——尽早停止
        #                      扩张，把收尾引号留给组后的 ['"]?，保证
        #                      与 to_dsl 输出的引号形式往返对称；
        #                      贪婪 .* 会把收尾引号吞进组内）；
        # ['"]?             —— 值的收尾引号，可选（不捕获）；
        # \)                —— 右括号（分支 B 无 $ 锚定，行尾多余文本
        #                      不阻碍匹配）。
        return r"""(?:set\s+slot\s+['"`]?([a-zA-Z_][a-zA-Z0-9_-]*)['"`]?\s+['"`]?(.+?)['"`]?$|SetSlot\(['"]?([a-zA-Z_][a-zA-Z0-9_-]*)['"]?,\s*['"]?(.*?)['"]?\))"""

    @classmethod
    def _from_regex_match(cls, match: re.Match[str]) -> "SetSlotCommand | None":
        """从正则匹配创建命令。

        业务流程：1. 槽位名取组1（set slot 格式），为空则取组3
                  （SetSlot 旧别名格式）；2. 值取组2/组4（分支对应
                  关系同名称），经 clean_extracted_value 归一化；
                  3. 名称命中则构造实例。

        副作用：无（纯构造）。

        参数：
            match: from_dsl 中 re.match 的匹配结果（分组含参数）。

        返回：
            SetSlotCommand 实例；名称未命中时返回 None（防御性
            分支，正常匹配下不可达）。
        """
        # 组分配：组1/2 属于 set slot 格式，组3/4 属于 SetSlot 旧别名
        # 格式；未命中的分支其组为 None，用 or 取先命中者。
        name = match.group(1) or match.group(3)
        value = match.group(2) or match.group(4)

        # 命中槽位名：构造命令（值经 clean_extracted_value 归一化为
        # Python 类型；值为空串/None 时兜底 None）。
        if name:
            return SetSlotCommand(
                name=name.strip(),
                value=clean_extracted_value(value) if value else None,
            )
        # 未命中任何分支：防御性返回 None（re.match 命中本模式时
        # 必有名称组，正常不可达）。
        return None

    def __hash__(self) -> int:
        """返回命令的哈希值。

        业务流程：以 (name, str(value)) 二元组计算哈希。

        副作用：无。

        返回：
            哈希值——与 __eq__ 保持一致（等价对象哈希相同）。
        """
        # 以 (name, str(value)) 计算哈希：@dataclass 生成 __eq__ 后
        # __hash__ 被置为 None（实例不可哈希），此处手动恢复可哈希性，
        # 使命令对象可放入集合/字典键（供命令去重等场景使用）。
        return hash((self.name, str(self.value)))

    def __eq__(self, other: object) -> bool:
        """判断两命令是否等价。

        业务流程：类型守卫后，比较槽位名（精确相等）与槽位值
                  （字符串化后大小写不敏感相等）。

        副作用：无。

        参数：
            other: 待比较对象，允许任意类型。

        返回：
            等价返回 True，否则 False——"123" 与 123、True 与
            "true" 均视为等价（值语义比较而非类型严格比较）。
        """
        # 类型守卫：非 SetSlotCommand 实例一律不等。
        if not isinstance(other, SetSlotCommand):
            return False
        # 名称精确相等 + 值字符串化后忽略大小写比较（str() 归一
        # 抹平 123/"123" 的类型差异，lower() 忽略 True/true 差异）。
        return (
            other.name == self.name
            and str(other.value).lower() == str(self.value).lower()
        )


@register_command
@dataclass
class ResetSlotCommand(Command):
    """重置槽位命令。

    职责：把指定槽位恢复为初始值——run() 调用槽位对象的 reset()
          方法；槽位不存在时静默返回空事件（无守卫异常、不报错）。
    设计模式：命令模式 + 注册表工厂（@register_command 按 DSL 名
              reset_slot 登记）；即时数据命令——run() 直接修改
              tracker 状态。
    生命周期：LLM 输出解析（from_dict/from_dsl）时创建，
              CommandProcessor 执行 run() 后随单轮消息处理流程
              销毁。

    使用现状：当前项目内无外部使用方（commands/__init__.py 与
              CommandProcessor 均未引用），属保留的 DSL 完整性
              命令——为 LLM 提供声明"清空槽位"意图的语法通道。

    Attributes:
        name: 要重置的槽位名称
    """

    # 要重置的槽位名称：须与 Domain 中定义的槽位名一致；正则约束
    # 为字母/下划线开头。
    name: str

    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。

        返回：
            命令名 reset_slot——注册表键与 DSL 语法动词。
        """
        # 注册表键：LLM 字典输出中 "command" 字段的取值即此名。
        return "reset_slot"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResetSlotCommand":
        """从字典创建命令。

        业务流程：读取 name（必需）构造实例；缺失时把 KeyError
                  转译为带字段说明的 ValueError。

        副作用：无（纯构造，仅异常转译）。

        参数：
            data: LLM 输出的命令字典，须含 "name" 键。

        返回：
            ResetSlotCommand 实例。

        异常：
            ValueError：data 缺少 "name" 必需键时抛出。
        """
        # 必需键用下标访问（缺失即 KeyError），捕获后转译为
        # ValueError 并保留原始异常链。
        try:
            return ResetSlotCommand(name=data["name"])
        except KeyError as e:
            # 转译：告知调用方具体缺失的必需字段。
            raise ValueError(
                f"Missing required field 'name' for ResetSlotCommand"
            ) from e

    def run(
        self,
        tracker: DialogueStateTracker,
        flows: Any | None = None,
    ) -> list[dict[str, Any]]:
        """执行重置槽位命令。

        业务流程：1. 查询槽位定义对象；2. 存在则调用其 reset()
                  恢复初始值并返回 slot_reset 事件；3. 不存在则
                  静默返回空事件列表。

        副作用：修改 tracker 状态（调用槽位 reset()）。

        参数：
            tracker: 对话状态追踪器（槽位定义来源）。
            flows: 可用的Flow列表，允许 None（本命令未使用）。

        返回：
            单元素事件列表（event=slot_reset，含 name 与时间戳
            占位）；槽位不存在时返回空列表。
        """
        # 查询槽位定义对象：未定义的槽位名得到 None。
        slot = tracker.slots.get(self.name)
        # 槽位存在：恢复初始值并产出事件（与 SetSlotCommand 的
        # try/except 风格不同，此处用 if 守卫静默跳过不存在槽位）。
        if slot:
            # 恢复槽位初始值：真正的状态变更发生在这里。
            slot.reset()
            # 返回槽位重置事件列表：
            return [
                {
                    # 事件类型标记：由下游 Policy/Action 识别。
                    "event": "slot_reset",
                    # 事件负载：被重置的槽位名。
                    "name": self.name,
                    # 事件时间戳占位：由 tracker 后续补齐。
                    "timestamp": None,
                }
            ]
        # 槽位不存在：静默返回空事件（不报错、不中断命令流）。
        return []

    def to_dsl(self) -> str:
        """转换为DSL字符串。

        副作用：无。

        返回：
            DSL 文本：短语形式 reset slot <name>。
        """
        # 短语形式：与 regex_pattern 的动词短语对应。
        return f"reset slot {self.name}"

    @classmethod
    def regex_pattern(cls) -> str:
        """返回正则表达式模式。

        返回：
            短语形式正则——无 $ 行尾锚定（槽位名后允许跟随其他
            文本）；解析端 from_dsl 用 re.match + IGNORECASE 使用。
        """
        # 正则逐段拆解（基类 from_dsl 以 re.match + IGNORECASE 使用
        # 本模式——大小写不敏感，隐式行首锚定）：
        # reset\s+slot       —— DSL 动词短语（词内空白任意，IGNORECASE
        #                       下大小写不敏感）；
        # \s+                —— 动词与槽位名之间的空白；
        # ['"`]?             —— 槽位名的起始引号，可选（不捕获，单/双/
        #                       反引号均可）；
        # ([a-zA-Z_][a-zA-Z0-9_-]*)  —— 捕获组1：槽位名（字母/下划线
        #                       开头，后接字母/数字/下划线/连字符）；
        # ['"`]?             —— 槽位名的收尾引号，可选（不捕获）；
        # （无 $ 锚定：槽位名之后的内容不参与匹配——re.match 语义下
        # 本模式为前缀匹配，与 SetSlotCommand 分支 A 的 $ 收尾不同。）
        return r"""reset\s+slot\s+['"`]?([a-zA-Z_][a-zA-Z0-9_-]*)['"`]?"""

    @classmethod
    def _from_regex_match(cls, match: re.Match[str]) -> "ResetSlotCommand | None":
        """从正则匹配创建命令。

        业务流程：提取捕获组一（槽位名），去首尾空白后构造；
                  组缺失时返回 None。

        副作用：无（纯构造）。

        参数：
            match: from_dsl 中 re.match 的匹配结果。

        返回：
            ResetSlotCommand 实例；名称组缺失时返回 None（防御性
            分支，正常匹配下不可达）。
        """
        # 组1：槽位名（模式仅一个捕获组）。
        name = match.group(1)
        # 命中槽位名：去空白后构造。
        if name:
            return ResetSlotCommand(name=name.strip())
        # 名称组缺失：防御性返回 None（re.match 命中本模式时组1
        # 必非空，正常不可达）。
        return None


# 导出：模块对外暴露的公共 API（按字母序，便于 IDE 提示与 diff 整洁）。
__all__ = [
    "ResetSlotCommand",
    "SetSlotCommand",
    "clean_extracted_value",
]
