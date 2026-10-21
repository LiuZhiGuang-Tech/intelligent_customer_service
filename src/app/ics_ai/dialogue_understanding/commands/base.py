"""
base - 命令（Command）基类模块

核心功能：定义对话系统所有命令的抽象契约与注册机制——Command
          抽象基类约定 command_name / from_dict / run 三个核心
          契约方法及 DSL 解析、字典序列化、相等性比较等通用能力；
          模块级注册表 + @register_command 装饰器实现命令类的
          动态注册与按名解析；command_from_dict /
          parse_command_from_text 提供字典与 DSL 文本两条反序列
          化入口。

执行流程：
    1. 装配期：各具体命令类经 @register_command 装饰，以
       command_name() 为键登记进 _COMMAND_REGISTRY；
    2. 反序列化：LLM 输出的命令字典经 command_from_dict（按
       "command" 字段查注册表构造），或命令文本经
       parse_command_from_text（按注册顺序逐类正则匹配），
       得到 Command 实例；
    3. 执行期：CommandProcessor 调用 command.run(tracker, flows)
       直接修改 Tracker 状态（压栈帧/设槽位）并返回事件列表。

Command 与 Action 的职责边界（本模块核心设计意图）：
    - Command（命令）：由 LLM 根据用户输入生成，经
      CommandProcessor 执行，直接修改 Tracker 状态（如压入
      栈帧、设置槽位）；
    - Action（动作）：由 Policy 根据栈帧状态选择，经 Agent
      执行，返回 ActionResult（事件、响应）；
    - 完整链路：用户输入 → LLM 生成 Command → CommandProcessor
      执行并更新 Tracker（可能压入栈帧）→ Policy 检测栈帧选择
      Action → Agent 执行 Action 生成响应；
    - 栈帧化 Action：部分 Action 只压栈帧（如 SearchStackFrame），
      由 Policy 检测栈帧并执行实际操作（如检索），实现动作与
      响应生成的解耦。示例："帮我查一下订单" → LLM 生成
      KnowledgeAnswerCommand → CommandProcessor 确定 next_action
      为 action_trigger_search → ActionTriggerSearch 压入
      SearchStackFrame → EnterpriseSearchPolicy 检测到该栈帧，
      执行检索并生成响应。

副作用：
    - 模块级注册表 _COMMAND_REGISTRY 为全局可变状态：
      @register_command 在类定义期写入，跨会话持续生效；
    - run 的具体实现（子类）会修改 tracker 状态；本模块自身
      无 IO、不调用网络。

主要依赖：
    - dataclasses：模块本体（as_dict 的 asdict 序列化）+
      dataclass 装饰器。
    - re：DSL 文本的正则解析。
    - abc：ABC + abstractmethod 定义命令契约。
    - typing：Any + TYPE_CHECKING（注解已按 PEP 585/604 新式写法）。
    - app.ics_ai.core.tracker：命令操作的状态对象（仅注解用）。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，支持前向引用
# （如 TYPE_CHECKING 分支中的 tracker 类型与 Command 自引用）。
from __future__ import annotations

# 导入 dataclasses 模块本体：as_dict 中调用 dataclasses.asdict
# 将命令实例递归序列化为字典。
import dataclasses

# 导入正则引擎：from_dsl 的模式匹配与 parse_command_from_text
# 的逐类尝试解析均依赖它。
import re

# 导入抽象基类工具：ABC 标记命令基类，abstractmethod 强制子类
# 实现 command_name / from_dict / run 三个契约方法。
from abc import ABC, abstractmethod

# 导入 dataclass 装饰器：Command 及其子类靠它自动生成
# __init__ / __repr__ 等数据类方法。
from dataclasses import dataclass

# 导入类型注解工具：Any 类型通配符；TYPE_CHECKING 隔离仅注解
# 导入（PEP 585/604 新式注解下 Dict/List/Optional/Type 已不需要）。
from typing import TYPE_CHECKING, Any

# 类型检查期专属导入：tracker 类型只出现在方法注解中，运行时不
# 导入（避免与 core 包形成循环依赖）。
if TYPE_CHECKING:
    # 对话状态追踪器：命令 run 的操作对象（仅注解用）。
    from app.ics_ai.core.tracker import DialogueStateTracker

# 命令注册表：命令名 → 命令类。模块级全局可变状态，装配期由
# @register_command 写入，运行期只读（查询走 get/copy）。
_COMMAND_REGISTRY: dict[str, type[Command]] = {}


# 这是一个泛型函数（generic function）：判定依据是函数名后紧跟方括号
# 形式的"类型参数列表"（PEP 695 语法，Python 3.12+ 起支持）——运行时
# 该函数对象会携带 __type_params__ 属性记录声明的类型参数，类型检查器
# 据此把"入参类型与返回类型经 C 绑定"的关系纳入检查（传入什么子类，
# 返回类型就是什么子类）。等价于旧式 T = TypeVar("T", bound=...) 声明
# + 签名中引用 T 的写法，但类型参数直接内联于函数签名，无需额外的
# 模块级 TypeVar 变量。签名逐段拆解如下：
#   [C: type["Command"]] —— 类型参数声明：
#     · C：本函数私有的类型变量，被入参与返回值共用；
#     · type["Command"]：C 的上界（bound）。注意约束的是"类对象自身
#       的类型"而非实例类型——即实参 cls 必须是一个类，且该类的实例
#       都是 Command 及其子类；引号形式为前向引用（Command 定义于
#       本行之后），PEP 695 的上界求值被延迟，运行时不会 NameError。
#   (cls: C) -> C —— 入参与返回共用同一个 C：装饰器把 @dataclass 加工
#       后的具体子类类型（如 type[StartFlowCommand]）原样透传回去，
#       类型检查器因此保留子类 dataclass 合成的 __init__ 签名不被降级
#       （为什么不写成 -> type[Command]，详见下方 docstring 设计说明）。
def register_command[C: type["Command"]](cls: C) -> C:
    """命令注册装饰器。

    业务流程：以 cls.command_name() 为键写入 _COMMAND_REGISTRY
              （同名重复注册时后者覆盖前者）；原样返回命令类，
              不改变被装饰类的定义。

    设计说明（PEP 695 泛型）：类型参数 C 的上界为 type["Command"]
    （类对象自身的类型），入参与返回共用 C，保证装饰后子类类型不被
    "降级"——若声明为 type[Command]，类型检查器会把 StartFlowCommand
    等子类视为基类类型，导致子类 dataclass 合成的 __init__ 参数
    （如 flow）被判定为不存在（basedpyright reportCallIssue）；
    若上界写成 "Command"（缺 type），则把类对象写入注册表时会因
    "类对象与实例类型不兼容" 报类型错误。

    副作用：修改模块级注册表 _COMMAND_REGISTRY（全局可变状态，
            跨会话持续生效）。

    参数：
        cls: 要注册的命令类（Command 子类，须实现 command_name）。

    返回：
        原样返回入参类（具体子类类型经泛型 C 保留，可继续作为
        普通类使用）。
    """
    # 以类方法 command_name() 的返回值为键注册。
    _COMMAND_REGISTRY[cls.command_name()] = cls
    # 原样返回类：装饰器语法糖，不影响类本身的使用。
    return cls


def get_command_class(name: str) -> type[Command] | None:
    """根据命令名获取命令类。

    业务流程：查注册表并返回（dict.get 语义，未命中得 None）。

    副作用：无。

    参数：
        name: 命令名称（如 "start_flow"、"set_slot"、"cancel_flow"）。

    返回：
        命令类；未注册的名称返回 None（调用方需处理空值）。
    """
    # 注册表查询：get 避免未命中时抛 KeyError。
    return _COMMAND_REGISTRY.get(name)


def get_all_command_classes() -> dict[str, type[Command]]:
    """获取所有已注册的命令类。

    业务流程：返回注册表的浅拷贝——外部对副本的增删不影响
              内部注册表。

    副作用：无。

    返回：
        命令名到命令类的映射副本。
    """
    # 浅拷贝：保护模块级注册表不被外部直接修改。
    return _COMMAND_REGISTRY.copy()


@dataclass
class Command(ABC):
    """命令基类。

    职责：定义命令的统一契约——命令是本架构中可执行的原子操作，
          所有具体命令类型（start_flow/set_slot 等）都应继承本类；
          同时提供 as_dict 序列化、from_dsl 正则解析、比较与
          哈希等通用能力。
    设计模式：命令模式 + 注册表工厂——命令对象由 LLM 输出经
              解析器构造，经 @register_command 注册后可按名
              动态创建（command_from_dict）。
    生命周期：LLM 生成器产出命令文本 → 解析器构造命令对象 →
              CommandProcessor 执行（run）后丢弃，随单轮消息
              处理流程创建与销毁。

    命令的生命周期细节：
        1. LLM 生成器根据用户输入生成命令文本；
        2. 命令解析器将文本解析为命令对象；
        3. 命令处理器执行命令并更新对话状态。
    """

    # classmethod + abstractmethod 组合：子类必须以类方法形式
    # 提供唯一的命令名（同时是注册表键）。
    @classmethod
    @abstractmethod
    def command_name(cls) -> str:
        """返回命令名称。

        命令名用于识别命令类型，必须全局唯一，也是注册表键
        （如 "start_flow"、"set_slot"、"cancel_flow"）。

        返回：
            命令名称字符串；子类未实现时抛 NotImplementedError。
        """
        # 兜底保护：子类未覆盖时抛出明确错误。
        raise NotImplementedError()

    @classmethod
    def command_type(cls) -> str:
        """返回命令类型（command_name 的别名）。

        业务流程：直接转发 command_name()，仅为兼容调用方的
                  两种命名习惯而存在。

        副作用：无。

        返回：
            命令类型字符串（与 command_name 一致）。
        """
        # 别名转发：单一事实来源仍是 command_name。
        return cls.command_name()

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> Command:
        """从字典创建命令实例。

        业务流程：由子类实现——从 data 中提取命令参数并构造
                  实例（data 通常来自 LLM 输出的 JSON）。

        副作用：由子类决定（纯构造，一般无）。

        参数：
            data: 包含命令参数的字典；参数缺失或无效时子类应
                  抛 ValueError。

        返回：
            命令实例；子类未实现时抛 NotImplementedError，
            参数缺失或无效时抛 ValueError。
        """
        # 兜底保护：子类未覆盖时抛出明确错误。
        raise NotImplementedError()

    def as_dict(self) -> dict[str, Any]:
        """将命令转换为字典。

        业务流程：1. dataclasses.asdict 递归序列化实例字段；
                  2. 注入 "command" 键标记命令类型（与
                  command_from_dict 的读取约定对应）。

        副作用：无。

        返回：
            包含全部字段与 "command" 类型标记的字典。
        """
        # asdict 递归转换：嵌套 dataclass/列表/字典一并展开。
        data = dataclasses.asdict(self)
        # 注入命令类型键：反序列化时据此查找注册表。
        data["command"] = self.command_name()
        return data

    @abstractmethod
    def run(
        self,
        tracker: DialogueStateTracker,
        flows: Any | None = None,
    ) -> list[dict[str, Any]]:
        """执行命令。

        业务流程：由子类实现——在对话状态追踪器上执行该命令
                  （如压栈帧、设槽位），并返回产生的事件。

        副作用：由子类决定——通常修改 tracker 状态（这是
                Command 与 Action 的本质区别之一）。

        参数：
            tracker: 对话状态追踪器，必需。
            flows: 可用的 Flow 列表，允许 None（仅部分命令需要，
                   如 start_flow 需按名查找 Flow）。

        返回：
            执行命令产生的事件列表（每项为事件字典）；
            子类未实现时抛 NotImplementedError。
        """
        # 兜底保护：子类未覆盖时抛出明确错误。
        raise NotImplementedError()

    @classmethod
    def from_dsl(cls, text: str) -> Command | None:
        """从 DSL 文本解析命令。

        业务流程：1. 取本类的正则模式（regex_pattern）；
                  2. 模式为 None 表示不支持 DSL 解析，返回 None；
                  3. 忽略大小写做 match；4. 匹配成功则交给
                  _from_regex_match 构造实例。

        副作用：无。

        参数：
            text: DSL 格式的命令文本（如 "start_flow{...}"）。

        返回：
            命令实例；不支持 DSL 或不匹配时返回 None。
        """
        # 取子类声明的正则模式；None 表示该命令不支持 DSL。
        pattern = cls.regex_pattern()
        # 模式缺失：直接返回 None（不是解析失败，是未启用 DSL）。
        if not pattern:
            return None

        # re.IGNORECASE：命令文本大小写不敏感，提升解析容错。
        match = re.match(pattern, text, re.IGNORECASE)
        # 匹配成功 → 交给子类的构造钩子；否则返回 None。
        if match:
            return cls._from_regex_match(match)
        return None

    @classmethod
    def _from_regex_match(cls, match: re.Match[str]) -> Command | None:
        """从正则匹配结果创建命令。

        业务流程：由子类覆盖实现——从 match 分组中提取参数并
                  构造实例；基类默认不支持，直接抛出。

        副作用：由子类决定（纯构造，一般无）。

        参数：
            match: from_dsl 中 re.match 的匹配结果（分组含参数）。

        返回：
            命令实例；子类未覆盖时抛 NotImplementedError。
        """
        # 兜底保护：未覆盖解析钩子的命令类不支持从匹配构造。
        raise NotImplementedError()

    def to_dsl(self) -> str:
        """将命令转换为 DSL 文本。

        业务流程：默认仅输出命令名；带参数的命令应由子类覆盖
                  以输出完整 DSL。

        副作用：无。

        返回：
            DSL 格式的命令字符串（默认为命令名本身）。
        """
        # 默认实现：无参数命令的 DSL 就是命令名。
        return f"{self.command_name()}"

    @classmethod
    def regex_pattern(cls) -> str | None:
        """返回用于解析此命令的正则表达式模式。

        业务流程：基类默认返回 None（不支持 DSL 解析）；
                  支持 DSL 的子类覆盖并返回模式串。

        副作用：无。

        返回：
            正则表达式字符串；不支持 DSL 解析时返回 None。
        """
        # 默认不支持 DSL：子类按需覆盖。
        return None

    def __hash__(self) -> int:
        """计算命令的哈希值。

        说明：仅按命令名哈希——同类型命令共享哈希口径，
              便于在集合/字典中按命令类型参与运算。

        返回：
            命令名的哈希值。
        """
        # 以命令名为哈希依据：与 __eq__ 的字段级等价口径配套。
        return hash(self.command_name())

    def __eq__(self, other: object) -> bool:
        """判断两个命令是否相等。

        业务流程：1. 非命令对象直接判不等；2. 同为命令时按
                  as_dict 的全字段比较（含 "command" 类型标记）。

        副作用：无。

        参数：
            other: 待比较对象（任意类型）。

        返回：
            True 表示字段级相等；否则 False。
        """
        # 类型守卫：非 Command 实例一律不等（避免属性访问错误）。
        if not isinstance(other, Command):
            return False
        # 字段级比较：asdict 展开 + 命令名标记一起参与比较。
        return self.as_dict() == other.as_dict()

    def __repr__(self) -> str:
        """返回命令的字符串表示。

        返回：
            "类名({字段字典})" 形式的调试字符串。
        """
        # 调试友好的表示：类名 + 完整字段字典。
        return f"{self.__class__.__name__}({self.as_dict()})"


def command_from_dict(data: dict[str, Any]) -> Command:
    """从字典创建命令对象（字典反序列化主入口）。

    业务流程：1. 读取 data["command"] 命令类型标记，缺失时抛
              ValueError；2. 按标记查注册表，未知类型抛
              ValueError；3. 调用该命令类的 from_dict 完成构造。

    副作用：无（构造与注册表查询均无 IO）。

    参数：
        data: 包含命令信息的字典（须含 "command" 键标记类型，
              其余键为命令参数）。

    返回：
        命令对象；"command" 缺失或类型未知时抛 ValueError。
    """
    # 读取命令类型标记（as_dict 注入的 "command" 键）。
    command_name = data.get("command")
    # 缺失标记：无法确定命令类型，快速失败。
    if not command_name:
        raise ValueError("Missing 'command' field in data")

    # 按标记查注册表拿命令类。
    command_cls = get_command_class(command_name)
    # 未注册的类型：同样快速失败，避免静默吞错。
    if command_cls is None:
        raise ValueError(f"Unknown command type: {command_name}")

    # 委托命令类自身的 from_dict 完成参数级构造。
    return command_cls.from_dict(data)


def parse_command_from_text(text: str) -> Command | None:
    """从文本解析命令（DSL 反序列化入口）。

    业务流程：1. 去除首尾空白；2. 按注册顺序遍历所有已注册
              命令类，逐类尝试 from_dsl 正则解析；3. 首个解析
              成功者立即返回；解析异常（未实现钩子/参数非法）
              的类静默跳过继续；4. 全部失败返回 None。

    副作用：无。

    参数：
        text: 待解析的命令文本（允许带首尾空白）。

    返回：
        命令对象；所有命令类都无法匹配时返回 None。

    说明：注册表为 dict，遍历顺序即注册顺序——多个命令类的
          正则模式存在重叠时，先注册者优先命中。
    """
    # 归一化输入：去除首尾空白，避免正则因空白错位。
    text = text.strip()
    # 按注册顺序逐类尝试：dict 保持插入序（即注册序）。
    for command_cls in _COMMAND_REGISTRY.values():
        try:
            # 尝试用该类的正则模式解析文本。
            command = command_cls.from_dsl(text)
            # 命中即返回：先注册者优先。
            if command is not None:
                return command
        # 该类未实现解析钩子（NotImplementedError）或参数非法
        # （ValueError）时跳过，继续尝试下一个命令类。
        except (NotImplementedError, ValueError):
            continue
    # 所有命令类均无法解析：返回 None 由调用方降级处理。
    return None


# 导出清单：本模块对外暴露的公共 API（基类 + 注册/解析工具）。
__all__ = [
    # 命令基类。
    "Command",
    # 注册装饰器与注册表查询。
    "register_command",
    "get_command_class",
    "get_all_command_classes",
    # 反序列化入口（字典 / DSL 文本）。
    "command_from_dict",
    "parse_command_from_text",
]
