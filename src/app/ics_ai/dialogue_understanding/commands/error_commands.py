"""
error_commands - 错误命令族模块

核心功能：定义系统错误场景的三个命令——ErrorCommand（通用错误事件）、
          InternalErrorCommand（内部错误，如 LLM 调用失败/超时）、
          ParseErrorCommand（命令解析失败）。命令本身不抛异常也不写
          日志，仅作为"错误事件载体"在 run 时返回事件列表，由下游
          命令执行流（CommandProcessor）收集进结果统一处理。

执行流程：
    1. 导入期：三个命令类经 @register_command 登记进命令注册表
       （仅 ErrorCommand 提供 DSL 正则，可按 error 名从文本解析）；
    2. 生成期：LLMCommandGenerator.generate 的异常分级映射创建
       InternalErrorCommand（捕获 LLMException）与 ErrorCommand
       （捕获其余意外异常）；
    3. 执行期：CommandProcessor 调用 run 产出错误事件字典（带降级
       原因 DegradationReason.INTERNAL_ERROR），事件流入结果。

副作用：无——run 只读自身字段返回事件，不修改 Tracker/对话栈、
        不读写文件/数据库、不发起网络请求。

主要依赖：
    - commands.base：Command 基类与 @register_command 注册装饰器；
    - shared.constants.DegradationReason：降级原因枚举常量；
    - core.tracker：DialogueStateTracker（仅注解用，TYPE_CHECKING 隔离）。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，TYPE_CHECKING 分支
# 中的 tracker 前向引用无需引号即可书写，PEP 604 联合类型注解
# （X | None）也不受运行时版本限制。
from __future__ import annotations

# 导入 re：仅用于 _from_regex_match 的 re.Match[str] 类型注解
# （正则模式以字符串形式由 regex_pattern 提供）。
import re

# 导入 dataclass 装饰器：三个错误命令靠它自动生成 __init__ 等方法。
from dataclasses import dataclass

# 导入类型注解工具：Any 类型通配符；TYPE_CHECKING 隔离仅注解导入
# （注解已按 PEP 585/604 新式写法，Dict/List/Optional 不再需要）。
from typing import Any, TYPE_CHECKING

# Command 基类：统一命令接口（command_name/from_dict/run/to_dsl）；
# register_command 装饰器：把命令类按 DSL 名登记进命令注册表。
from app.ics_ai.dialogue_understanding.commands.base import Command, register_command

# 降级原因枚举：错误事件携带的 degradation_reason 字段取值来源。
from app.ics_ai.shared.constants import DegradationReason

# 类型检查期专属导入：只出现在方法注解中，运行时不导入（避免
# 与 core 包形成循环依赖）。
if TYPE_CHECKING:
    # 对话状态追踪器：run 的入参类型（本命令族未使用该参数）。
    from app.ics_ai.core.tracker import DialogueStateTracker


@register_command
@dataclass
class ErrorCommand(Command):
    """错误命令。

    职责：承载通用系统错误信息并以事件形式上报——LLM 命令生成
          阶段发生意外异常时由生成器创建，执行时产出 error_occurred
          事件（带降级原因），供下游统一处理。
    设计模式：数据类 + 注册表装饰器（@register_command 按 DSL 名
              error 登记，支持 to_dsl/正则解析双向转换）。
    生命周期：异常映射或 LLM 输出解析时创建，随命令流执行后销毁。

    Attributes:
        error_type: 错误类型
        message: 错误消息
    """

    # 错误类型：错误的分类标识（如 generation_error），默认未知错误。
    error_type: str = "unknown_error"
    # 错误消息：人类可读的错误描述；None 表示无消息（DSL 输出省略）。
    message: str | None = None

    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。

        返回：
            命令名 error——命令注册表键与 DSL 语法动词。
        """
        # 注册表键与 DSL 动词：error(...)。
        return "error"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ErrorCommand":
        """从字典创建命令。

        业务流程：按键读取 error_type/message 构造实例，缺失键按
                  字段默认值兜底。

        副作用：无（纯构造）。

        参数：
            data: as_dict 产出的字典，各键允许缺失。

        返回：
            ErrorCommand 实例。
        """
        # 逐键读取：error_type 缺省 unknown_error；message 允许缺失（None）。
        return ErrorCommand(
            error_type=data.get("error_type", "unknown_error"),
            message=data.get("message"),
        )

    def run(
        self,
        tracker: DialogueStateTracker,
        flows: Any | None = None,
    ) -> list[dict[str, Any]]:
        """执行错误命令。

        业务流程：组装一条 error_occurred 事件字典（错误类型/消息/
                  降级原因/时间戳占位）并返回。

        副作用：无（纯事件产出——错误命令不压栈帧、不改 Tracker，
                事件由 CommandProcessor 收集进结果）。

        参数：
            tracker: 对话状态追踪器（本命令未使用，保持接口一致）。
            flows: 可用的Flow列表，允许 None（本命令未使用）。

        返回：
            单元素事件列表：{"event": "error_occurred", ...}。
        """
        # 事件字典：degradation_reason 标记内部错误降级；timestamp
        # 固定 None（预留字段，当前不填充）。
        return [
            {
                "event": "error_occurred",
                "error_type": self.error_type,
                "message": self.message,
                "degradation_reason": DegradationReason.INTERNAL_ERROR,
                "timestamp": None,
            }
        ]

    def to_dsl(self) -> str:
        """转换为DSL字符串。

        业务流程：有消息时输出双参数形式 error("类型", "消息")，
                  无消息时输出单参数形式。

        副作用：无。

        返回：
            DSL 文本（可被 regex_pattern 正则解析还原）。
        """
        # 有消息：双参数形式（类型与消息均为引号包裹）。
        if self.message:
            return f'error("{self.error_type}", "{self.message}")'
        # 无消息：单参数形式。
        return f'error("{self.error_type}")'

    @classmethod
    def regex_pattern(cls) -> str:
        """返回正则表达式模式。

        返回：
            整行锚定的 error DSL 正则——参数一必选、参数二可选，
            引号可选；捕获组用 [^'"]+ 排除引号字符，收尾引号不会
            泄漏进捕获组（与回答类命令的贪婪 .* 写法不同，往返对称）。
        """
        # 正则逐段拆解：
        # ^error\(                    —— DSL 动词与左括号，整行锚定；
        # (['"]?)                     —— 参数一起始引号，可选（不捕获）；
        # ([^'"]+)                    —— 参数一内容：至少一个非引号字符（组1）；
        # ['"]?                       —— 参数一收尾引号，可选；
        # (?:,\s*['"]?([^'"]+)['"]?)? —— 参数二整体可选：逗号+空白+
        #                                可选引号+非引号内容（组2）+可选收尾引号；
        # \)$                         —— 右括号与行尾。
        return r"""^error\(['"]?([^'"]+)['"]?(?:,\s*['"]?([^'"]+)['"]?)?\)$"""

    @classmethod
    def _from_regex_match(cls, match: re.Match[str]) -> ErrorCommand | None:
        """从正则匹配创建命令。

        业务流程：提取捕获组一/二（类型与消息），去首尾空白后构造；
                  组一缺失时兜底 unknown_error，组二缺失时消息为 None。

        副作用：无（纯构造）。

        参数：
            match: regex_pattern 的匹配结果（命令解析器回填接口约定）。

        返回：
            ErrorCommand 实例；类型注解按解析器接口保留 None 可能
            （本实现的组一至少一个字符，实际不会返回 None）。
        """
        # 组一：错误类型（必选参数）。
        error_type = match.group(1)
        # 组二：错误消息（可选参数，未提供时为 None）。
        message = match.group(2)
        # 去空白后构造：组一为空兜底 unknown_error；组二为空消息置 None。
        return ErrorCommand(
            error_type=error_type.strip() if error_type else "unknown_error",
            message=message.strip() if message else None,
        )


@register_command
@dataclass
class InternalErrorCommand(Command):
    """内部错误命令。

    职责：承载系统内部错误信息（如 LLM 调用失败、超时等）——LLM
          命令生成器捕获 LLMException 时创建，执行时产出
          internal_error 事件（带降级原因）。
    设计模式：数据类 + 注册表装饰器。
    生命周期：异常映射时创建，随命令流执行后销毁。

    已知不对称（不改代码仅记录）：仅提供 to_dsl 序列化，未实现
    regex_pattern/_from_regex_match，无法从 DSL 文本解析还原。

    Attributes:
        exception_type: 异常类型名
        exception_message: 异常消息
    """

    # 异常类型名：抛出异常的类名（如 TimeoutError），默认通用内部错误。
    exception_type: str = "InternalError"
    # 异常消息：str(e) 的文本内容；None 表示无消息（DSL 输出省略）。
    exception_message: str | None = None

    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。

        返回：
            命令名 internal_error——注册表键与 DSL 语法动词。
        """
        # 注册表键与 DSL 动词：internal_error(...)。
        return "internal_error"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InternalErrorCommand":
        """从字典创建命令。

        业务流程：按键读取 exception_type/exception_message 构造
                  实例，缺失键按字段默认值兜底。

        副作用：无（纯构造）。

        参数：
            data: as_dict 产出的字典，各键允许缺失。

        返回：
            InternalErrorCommand 实例。
        """
        # 逐键读取：类型缺省 InternalError；消息允许缺失（None）。
        return InternalErrorCommand(
            exception_type=data.get("exception_type", "InternalError"),
            exception_message=data.get("exception_message"),
        )

    def run(
        self,
        tracker: DialogueStateTracker,
        flows: Any | None = None,
    ) -> list[dict[str, Any]]:
        """执行内部错误命令。

        业务流程：组装一条 internal_error 事件字典（异常类型/消息/
                  降级原因/时间戳占位）并返回。

        副作用：无（纯事件产出，不修改任何状态）。

        参数：
            tracker: 对话状态追踪器（本命令未使用，保持接口一致）。
            flows: 可用的Flow列表，允许 None（本命令未使用）。

        返回：
            单元素事件列表：{"event": "internal_error", ...}。
        """
        # 事件字典：degradation_reason 标记内部错误降级；timestamp
        # 固定 None（预留字段，当前不填充）。
        return [
            {
                "event": "internal_error",
                "exception_type": self.exception_type,
                "exception_message": self.exception_message,
                "degradation_reason": DegradationReason.INTERNAL_ERROR,
                "timestamp": None,
            }
        ]

    def to_dsl(self) -> str:
        """转换为DSL字符串。

        业务流程：有异常消息时输出双参数形式，无消息时输出单参数
                  形式（仅供调试/日志展示——本命令未注册解析正则）。

        副作用：无。

        返回：
            DSL 文本（注意：无对应正则，不能从文本解析还原）。
        """
        # 有异常消息：双参数形式。
        if self.exception_message:
            return (
                f'internal_error("{self.exception_type}", "{self.exception_message}")'
            )
        # 无异常消息：单参数形式。
        return f'internal_error("{self.exception_type}")'


@register_command
@dataclass
class ParseErrorCommand(Command):
    """解析错误命令。

    职责：承载命令解析失败场景的原始文本与错误消息——预留的解析
          失败事件载体（当前模块外无直接使用方，命令生成器的空解析
          兜底走 CannotHandleCommand）。
    设计模式：数据类 + 注册表装饰器。
    生命周期：解析失败时创建（预留），随命令流执行后销毁。

    已知不对称（不改代码仅记录）：仅提供 to_dsl 序列化，未实现
    regex_pattern/_from_regex_match，无法从 DSL 文本解析还原。

    Attributes:
        raw_text: 原始文本
        error_message: 解析错误消息
    """

    # 原始文本：解析失败的 LLM 原始输出，默认空串。
    raw_text: str = ""
    # 解析错误消息：失败原因描述；None 表示无消息（DSL 输出省略）。
    error_message: str | None = None

    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。

        返回：
            命令名 parse_error——注册表键与 DSL 语法动词。
        """
        # 注册表键与 DSL 动词：parse_error(...)。
        return "parse_error"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParseErrorCommand":
        """从字典创建命令。

        业务流程：按键读取 raw_text/error_message 构造实例，缺失键
                  按字段默认值兜底。

        副作用：无（纯构造）。

        参数：
            data: as_dict 产出的字典，各键允许缺失。

        返回：
            ParseErrorCommand 实例。
        """
        # 逐键读取：raw_text 缺省空串；error_message 允许缺失（None）。
        return ParseErrorCommand(
            raw_text=data.get("raw_text", ""),
            error_message=data.get("error_message"),
        )

    def run(
        self,
        tracker: DialogueStateTracker,
        flows: Any | None = None,
    ) -> list[dict[str, Any]]:
        """执行解析错误命令。

        业务流程：组装一条 parse_error 事件字典（原始文本/错误消息/
                  时间戳占位）并返回。

        副作用：无（纯事件产出，不修改任何状态）。

        参数：
            tracker: 对话状态追踪器（本命令未使用，保持接口一致）。
            flows: 可用的Flow列表，允许 None（本命令未使用）。

        返回：
            单元素事件列表：{"event": "parse_error", ...}。
        """
        # 事件字典：本命令不带降级原因字段（与其余两个错误命令不同）；
        # timestamp 固定 None（预留字段，当前不填充）。
        return [
            {
                "event": "parse_error",
                "raw_text": self.raw_text,
                "error_message": self.error_message,
                "timestamp": None,
            }
        ]

    def to_dsl(self) -> str:
        """转换为DSL字符串。

        业务流程：输出单参数形式 parse_error("原始文本")
                  （仅供调试/日志展示——本命令未注册解析正则）。

        副作用：无。

        返回：
            DSL 文本（注意：无对应正则，不能从文本解析还原）。
        """
        # 单参数形式：仅携带原始文本。
        return f'parse_error("{self.raw_text}")'


# 导出：模块对外暴露的公共 API（按字母序，便于 IDE 提示与 diff 整洁）。
__all__ = [
    "ErrorCommand",
    "InternalErrorCommand",
    "ParseErrorCommand",
]
