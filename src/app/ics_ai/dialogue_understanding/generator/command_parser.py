"""
command_parser - 命令解析器模块

核心功能：将 LLM 输出的自由文本解析为 Command 对象列表——
          ParseResult 承载解析产物（命令/错误/原始行）；CommandParser
          负责文本清理（去 markdown 围栏/反引号/列表标记/编号）、
          分行（换行符 + 分号）、逐行解析（标准注册表路径 + 逐类
          正则兜底）与基础校验；default_parser 单例 + parse_commands/
          parse_single_command 提供模块级便捷入口。

执行流程：
    1. 装配期：模块导入时创建 default_parser 单例，并在构造函数中
       对命令注册表做一次快照（get_all_command_classes 浅拷贝）；
    2. 解析期：LLMCommandGenerator.generate 调用 parser.parse →
       _clean_text 清理 → _split_lines 分行 → 逐行跳过注释/空行 →
       _parse_line 还原命令（未识别或异常的行记入 errors）；
    3. 消费期：解析结果封装进 GenerationResult，命令交由
       CommandProcessor 执行，errors 记入元数据供调试。

副作用：
    - 模块级 default_parser 为导入期创建的全局单例，其注册表快照
      依赖导入顺序（命令模块需先于本模块或同批导入）；
    - logger 仅在解析异常时输出 debug 级日志，无其他 IO。

主要依赖：
    - re：文本清理的三组正则与兜底解析的模式匹配；
    - logging：解析失败的调试日志；
    - dataclasses：@dataclass + field(default_factory=...) 定义解析结果；
    - typing：新式注解下已无需 Dict/List/Optional/Tuple（已移除）；
    - app.ics_ai.dialogue_understanding.commands.base：命令基类、
      注册表快照与标准解析入口。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，Command 前向
# 引用无需引号即可书写。
from __future__ import annotations

# 导入正则引擎：_clean_text 的三组清理正则与 _parse_line 的兜底
# 模式匹配均依赖它。
import re

# 导入日志工具：解析异常时输出 debug 级日志。
import logging

# 导入 dataclass 装饰器与 field：解析结果数据类定义；
# field(default_factory=...) 为可变默认值（列表）指定每次实例化
# 时的工厂函数，避免可变默认值在实例间共享。
from dataclasses import dataclass, field

# 导入命令基类与注册表工具：Command 为结果元素类型；
# get_all_command_classes 提供注册表快照；parse_command_from_text
# 是逐行解析的标准路径（按注册顺序逐类 from_dsl）。
from app.ics_ai.dialogue_understanding.commands.base import (
    Command,
    get_all_command_classes,
    parse_command_from_text,
)

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """解析结果。

    职责：作为命令解析的统一结果信封（DTO）——同时携带成功解析的
          命令、失败行及其错误信息、清理后的原始行，供上层按需
          消费与排障。
    设计模式：数据传输对象（DTO）+ 派生属性——success 与
              has_errors 由列表长度实时派生，不重复存储。
    生命周期：随单次 parse 调用创建，由调用方消费后丢弃。

    Attributes:
        commands: 成功解析的命令列表
        errors: 解析错误列表（每项为 (原始行文本, 错误信息) 二元组）
        raw_lines: 清理并分行后的原始输入行
    """

    # 成功解析的命令列表：default_factory 每次实例化新建，避免共享。
    commands: list[Command] = field(default_factory=list)
    # 解析错误列表：元素为 (原始行, 错误信息) 二元组；
    # 无法识别的行与解析时抛异常的行都记入此处。
    errors: list[tuple[str, str]] = field(default_factory=list)
    # 清理并分行后的原始输入行：与 commands/errors 对照排障用。
    raw_lines: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """是否成功解析了至少一个命令。

        返回：
            commands 非空时返回 True；全行失败返回 False。
        """
        # 派生判定：命令列表非空即成功。
        return len(self.commands) > 0

    @property
    def has_errors(self) -> bool:
        """是否有解析错误。

        返回：
            errors 非空时返回 True（注意：存在错误不代表无成功命令，
            两者可同时成立）。
        """
        # 派生判定：错误列表非空即有错误。
        return len(self.errors) > 0


class CommandParser:
    """命令解析器。

    职责：将 LLM 输出的文本解析为命令对象列表——容忍 LLM 输出的
          各种"噪声"（markdown 围栏、反引号、列表标记、编号、分号
          分隔），并将无法解析的行记录为错误而非直接失败。
    设计模式：门面（Facade）+ 注册表快照——对外仅暴露 parse/
              parse_single/validate_commands；构造时快照命令注册表，
              解析按快照逐类尝试。
    生命周期：LLMCommandGenerator 创建并长期持有（或使用模块级
              default_parser 单例），跨多轮消息复用。

    支持的命令格式：
    1. DSL格式: start flow booking, set slot name "John"
    2. 函数格式: StartFlow(booking), SetSlot(name, "John")
    """

    def __init__(self):
        """初始化解析器。

        业务流程：对模块级命令注册表做一次浅拷贝快照——后续解析
                  基于该快照，运行期新增注册的命令不会自动进入
                  本解析器（快照时机依赖导入顺序）。

        副作用：创建 self._command_classes 实例属性（注册表副本）。
        """
        # 注册表快照：copy 隔离外部增删；解析遍历顺序即快照时的注册顺序。
        self._command_classes = get_all_command_classes()

    def parse(self, text: str) -> ParseResult:
        """解析文本中的命令。

        业务流程：1. 清理文本（去 markdown 围栏/反引号/列表标记）；
                  2. 分行（换行符与分号均作分隔）；3. 逐行跳过空行
                  与注释行，尝试解析为命令；4. 未识别的行与解析
                  异常均记入 errors（不中断整体解析）。

        副作用：无（纯解析，仅记录 debug 日志）。

        参数：
            text: LLM 输出的文本（可含 markdown 噪声，允许多行）。

        返回：
            解析结果信封（命令 + 错误 + 原始行）。
        """
        # 初始化结果信封：逐行解析产物最终登记于此。
        result = ParseResult()

        # 清理文本：移除 markdown 围栏与反引号等噪声。
        text = self._clean_text(text)

        # 按行分割：换行符与分号均视为命令分隔。
        lines = self._split_lines(text)
        # 登记原始行：与解析产物对照排障。
        result.raw_lines = lines

        # 逐行解析：单行失败不影响其余行。
        for line in lines:
            # 去除首尾空白后判断是否为有效行。
            line = line.strip()
            # 空行：直接跳过。
            if not line:
                continue

            # 跳过注释行：# 与 // 开头的行视为注释（LLM 偶发输出的说明性文字）。
            if line.startswith("#") or line.startswith("//"):
                continue

            try:
                # 尝试解析当前行。
                command = self._parse_line(line)
                # 解析成功：登记命令。
                if command:
                    result.commands.append(command)
                else:
                    # 无法识别的行：记入错误列表（行文本 + 统一错误信息）。
                    result.errors.append((line, "Unrecognized command format"))
            # 任意解析异常：捕获后登记为该行的错误，继续解析后续行。
            except Exception as e:
                result.errors.append((line, str(e)))
                logger.debug(f"Failed to parse line: {line}, error: {e}")

        return result

    def parse_single(self, text: str) -> Command | None:
        """解析单个命令。

        业务流程：复用 parse 的完整流程，仅取首个成功命令。

        副作用：无。

        参数：
            text: 命令文本（单行或包含多行，仅取首个成功者）。

        返回：
            命令对象；解析失败或无命令时返回 None（调用方需处理空值）。
        """
        # 复用完整解析：保证与 parse 行为一致。
        result = self.parse(text)
        # 有命令取首个；否则 None。
        return result.commands[0] if result.commands else None

    def _clean_text(self, text: str) -> str:
        """清理文本。

        业务流程：1. 移除 markdown 代码块围栏（```lang 及 ```）；
                  2. 移除内联代码的反引号；3. 逐行移除列表标记
                  （-/*/*）与编号（"1. "）——LLM 常以这些格式包裹
                  命令输出。

        副作用：无。

        参数：
            text: 原始文本。

        返回：
            清理后的文本（行结构保留）。
        """
        # 移除markdown代码块：匹配 ``` 开头（可带语言名）至行尾的围栏标记。
        text = re.sub(r"```[\w]*\n?", "", text)

        # 移除内联代码的反引号（如 `cannot_handle` -> cannot_handle）：
        # 捕获组保留反引号内的实际内容。
        text = re.sub(r"`([^`]+)`", r"\1", text)

        # 移除开头的编号（如 "1. ", "- "）：逐行清理列表噪声。
        lines = []
        for line in text.split("\n"):
            # 移除列表标记：行首（允许缩进）的 -/*/* 及其后空白。
            line = re.sub(r"^\s*[-*•]\s*", "", line)
            # 移除编号：行首（允许缩进）的数字 + 点 + 空白。
            line = re.sub(r"^\s*\d+\.\s*", "", line)
            lines.append(line)

        # 重新拼接：保持行结构供后续分行解析。
        return "\n".join(lines)

    def _split_lines(self, text: str) -> list[str]:
        """分割文本行。

        业务流程：1. 按换行符分割；2. 行内含分号时再按分号拆分
                  （兼容单行多命令写法）；3. 去除各行首尾空白并
                  丢弃空行。

        副作用：无。

        参数：
            text: 待分割的文本。

        返回：
            非空且已 strip 的行列表。
        """
        # 按换行符分割
        lines = text.split("\n")

        # 也可能用分号分隔：含分号的行再拆为多行。
        result = []
        for line in lines:
            # 行内出现分号：按分号拆分（分号视为命令分隔符）。
            if ";" in line:
                result.extend(line.split(";"))
            else:
                result.append(line)

        # 统一 strip 并过滤空行：仅保留有效内容行。
        return [l.strip() for l in result if l.strip()]

    def _parse_line(self, line: str) -> Command | None:
        """解析单行命令。

        业务流程：1. 首选标准路径 parse_command_from_text——按注册
                  顺序逐类 from_dsl 正则解析；2. 失败则走兜底路径——
                  遍历注册表快照，逐类取 regex_pattern 直接匹配并
                  调用 _from_regex_match 构造；3. 两层均失败返回 None。

        副作用：无。

        参数：
            line: 单行命令文本（已 strip）。

        返回：
            命令对象；无法解析时返回 None（由上层记入 errors）。

        说明：兜底路径在行为上与首选路径基本等价（from_dsl 即
              regex_pattern + _from_regex_match 的封装），属防御性
              冗余——差异仅在于兜底层不吞 ValueError（参数缺失等
              构造错误会上抛，由 parse 记为该行错误）。
        """
        # 去除首尾空白：保证模式匹配从命令文本起始。
        line = line.strip()

        # 尝试使用注册的命令类解析（标准路径：先注册者优先）。
        command = parse_command_from_text(line)
        # 命中即返回。
        if command:
            return command

        # 尝试逐个命令类的模式匹配（兜底路径）。
        for command_name, command_cls in self._command_classes.items():
            try:
                # 取该类的正则模式；None 表示不支持 DSL 解析。
                pattern = command_cls.regex_pattern()
                # 有模式才尝试匹配：大小写不敏感提升容错。
                if pattern:
                    match = re.match(pattern, line, re.IGNORECASE)
                    # 匹配成功：直接调用构造钩子创建命令。
                    if match:
                        return command_cls._from_regex_match(match)
            # 未实现解析钩子（NotImplementedError）或无正则支持
            # （AttributeError）的类静默跳过，继续尝试下一个。
            except (NotImplementedError, AttributeError):
                continue

        # 无法解析：返回 None 由上层记录错误。
        return None

    def validate_commands(self, commands: list[Command]) -> list[Command]:
        """验证命令列表。

        业务流程：逐个调用 _is_valid_command，仅保留有效命令。

        副作用：无。

        参数：
            commands: 待验证的命令列表。

        返回：
            有效的命令列表（可能为空）。
        """
        # 初始化有效列表：逐个命令做基础校验。
        valid_commands = []
        for cmd in commands:
            # 校验通过才保留。
            if self._is_valid_command(cmd):
                valid_commands.append(cmd)
        return valid_commands

    def _is_valid_command(self, command: Command) -> bool:
        """检查命令是否有效。

        业务流程：1. 空命令直接无效；2. 按字段探测校验——带 flow
                  字段的命令（如 StartFlowCommand）flow 不得为空，
                  带 name 字段的命令（如 SetSlotCommand）name 不得
                  为空；3. 字段访问异常视为无效。

        副作用：无。

        参数：
            command: 待检查的命令对象（允许 None）。

        返回：
            True 表示通过基础校验；False 表示无效。
        """
        # 基本验证：命令不是None
        if command is None:
            return False

        # 检查必要字段：鸭子类型探测（有该字段才校验）。
        # 注：flow/name 仅声明在 Command 子类上，pyright 系检查器不支持
        # 经 hasattr 收窄到未声明的子类类型，故下方两处属性访问属已知
        # 误报，用压制注释保留鸭子类型守卫的运行时语义不变。
        try:
            # StartFlowCommand 需要 flow 字段：flow 为空则命令不可执行。
            if hasattr(command, "flow") and not command.flow:  # pyright: ignore[reportAttributeAccessIssue]
                return False
            # SetSlotCommand 需要 name 字段：name 为空则命令不可执行。
            if hasattr(command, "name") and not command.name:  # pyright: ignore[reportAttributeAccessIssue]
                return False
        # 字段访问抛异常（如属性为反逻辑的自定义对象）：视为无效。
        except Exception:
            return False

        return True


# 创建默认解析器实例：模块级单例（导入期创建，注册表快照以导入
# 时的命令集合为准）。
default_parser = CommandParser()


def parse_commands(text: str) -> list[Command]:
    """便捷函数：解析命令。

    业务流程：委托 default_parser.parse，仅返回命令列表（丢弃
              错误与原始行信息）。

    副作用：无。

    参数：
        text: LLM 输出的文本。

    返回：
        成功解析的命令列表（可能为空；错误信息需用
        default_parser.parse 才能获取）。
    """
    # 委托单例解析：取命令列表。
    result = default_parser.parse(text)
    return result.commands


def parse_single_command(text: str) -> Command | None:
    """便捷函数：解析单个命令。

    业务流程：委托 default_parser.parse_single。

    副作用：无。

    参数：
        text: 命令文本。

    返回：
        命令对象；解析失败时返回 None。
    """
    # 委托单例的单命令解析。
    return default_parser.parse_single(text)


# 导出
__all__ = [
    # 命令解析器。
    "CommandParser",
    # 解析结果信封。
    "ParseResult",
    # 模块级便捷函数。
    "parse_commands",
    "parse_single_command",
    # 默认解析器单例。
    "default_parser",
]
