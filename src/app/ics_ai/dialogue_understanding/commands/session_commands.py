"""
session_commands - 会话命令族模块

核心功能：定义会话生命周期与交互辅助的五个命令——
          SessionStartCommand/RestartCommand（会话开始/重启，run 仅返回
          事件标记，实际操作由 ActionSessionStart/ActionRestart 执行）、
          ClarifyCommand（澄清追问，纯事件不压栈帧）、HumanHandoffCommand
          （人工转接，压 HumanHandoffStackFrame 交由 Policy 检测）、
          NoopCommand（显式空操作）。

执行流程：
    1. 导入期：五个命令类经 @register_command 登记进命令注册表，
       均具备 to_dsl/regex_pattern/_from_regex_match 三件套，可按
       各自 DSL 名从文本解析还原；
    2. 生成期：LLM 输出解析或系统内部创建命令实例；
    3. 执行期：除 NoopCommand 返回空列表外均产出事件字典；其中
       HumanHandoffCommand 额外压入转接栈帧；会话开始/重启的真正
       执行发生在对应 Action 层。

副作用：
    - 仅 HumanHandoffCommand.run 修改对话栈（压入
      HumanHandoffStackFrame）；
    - 其余命令 run 无状态修改（纯事件产出/空操作）；
    - 无文件/数据库读写、无网络请求。

主要依赖：
    - commands.base：Command 基类与 @register_command 注册装饰器；
    - stack.stack_frame.HumanHandoffStackFrame：转接帧（仅
      HumanHandoffCommand 执行期局部导入，依赖范围最小化）；
    - core.tracker：DialogueStateTracker（仅注解用，TYPE_CHECKING 隔离）。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，TYPE_CHECKING 分支
# 中的 tracker 前向引用无需引号即可书写，PEP 604 联合类型注解
# （X | None）也不受运行时版本限制。
from __future__ import annotations

# 导入 re：仅用于 _from_regex_match 的 re.Match[str] 类型注解
# （正则模式以字符串形式由 regex_pattern 提供）。
import re

# 导入 dataclass 装饰器与 field：命令数据类定义；field(default_factory=...)
# 为可变默认值（选项列表）指定每次实例化时的工厂函数，避免可变
# 默认值在实例间共享。
from dataclasses import dataclass, field

# 导入类型注解工具：Any 类型通配符；TYPE_CHECKING 隔离仅注解导入
# （注解已按 PEP 585/604 新式写法，Dict/List/Optional 不再需要）。
from typing import Any, TYPE_CHECKING

# Command 基类：统一命令接口（command_name/from_dict/run/to_dsl）；
# register_command 装饰器：把命令类按 DSL 名登记进命令注册表。
from app.ics_ai.dialogue_understanding.commands.base import Command, register_command

# 类型检查期专属导入：只出现在方法注解中，运行时不导入（避免
# 与 core 包形成循环依赖）。
if TYPE_CHECKING:
    # 对话状态追踪器：run 的入参类型（仅 HumanHandoffCommand 实际使用）。
    from app.ics_ai.core.tracker import DialogueStateTracker


@register_command
@dataclass
class SessionStartCommand(Command):
    """会话开始命令。

    职责：标记新会话开始、请求初始化对话状态——run 仅产出
          session_start_requested 事件，真正的初始化由
          ActionSessionStart 执行（命令只表达意图不执行操作）。
    设计模式：数据类 + 注册表装饰器（@register_command 按 DSL 名
              session_start 登记）。
    生命周期：会话启动或 LLM 输出解析时创建，随命令流执行后销毁。
    """

    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。

        返回：
            命令名 session_start——注册表键与 DSL 语法动词。
        """
        # 注册表键与 DSL 动词：session_start（无参形式）。
        return "session_start"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionStartCommand":
        """从字典创建命令。

        业务流程：本命令无字段，忽略字典内容直接构造。

        副作用：无（纯构造）。

        参数：
            data: as_dict 产出的字典（内容被忽略，允许任意）。

        返回：
            SessionStartCommand 实例（无状态，所有实例等价）。
        """
        # 无字段命令：直接构造（data 仅满足接口签名）。
        return SessionStartCommand()

    def run(
        self,
        tracker: DialogueStateTracker,
        flows: Any | None = None,
    ) -> list[dict[str, Any]]:
        """执行会话开始命令。

        业务流程：组装一条 session_start_requested 事件字典返回；
                  不做任何状态修改。

        副作用：无（纯事件产出——初始化由 ActionSessionStart 执行）。

        参数：
            tracker: 对话状态追踪器（本命令未使用，保持接口一致）。
            flows: 可用的Flow列表，允许 None（本命令未使用）。

        返回：
            单元素事件列表：{"event": "session_start_requested", ...}。
        """
        # 不直接重启，让 ActionSessionStart 来执行（命令只表达意图）。
        # 事件字典：timestamp 固定 None（预留字段，当前不填充）。
        return [
            {
                "event": "session_start_requested",
                "timestamp": None,
            }
        ]

    def to_dsl(self) -> str:
        """转换为DSL字符串。

        副作用：无。

        返回：
            DSL 文本：无参动词形式 session_start。
        """
        # 无参动词：与正则 ^session_start$ 对应。
        return "session_start"

    @classmethod
    def regex_pattern(cls) -> str:
        """返回正则表达式模式。

        返回：
            整行锚定的无参动词正则（不允许携带参数）。
        """
        # 正则逐段拆解：
        # ^              —— 行首锚定（配合末尾 $ 整行全匹配，防部分匹配）；
        # session_start  —— DSL 动词（与注册表 command_name 同名）；
        # $              —— 行尾锚定（无参动词，不允许携带任何参数）。
        return r"""^session_start$"""

    @classmethod
    def _from_regex_match(cls, match: re.Match[str]) -> SessionStartCommand | None:
        """从正则匹配创建命令。

        业务流程：本命令无字段，忽略匹配内容直接构造。

        副作用：无（纯构造）。

        参数：
            match: regex_pattern 的匹配结果（命令解析器回填接口约定）。

        返回：
            SessionStartCommand 实例（无状态，实际不会返回 None）。
        """
        # 无字段命令：直接构造（match 仅满足接口签名）。
        return SessionStartCommand()


@register_command
@dataclass
class ClarifyCommand(Command):
    """澄清命令。

    职责：当用户输入不清晰或需要更多信息时，标记当前轮次需要向
          用户请求澄清——纯事件产出，不压栈帧（CommandProcessor
          对该类型显式设置 next_action=action_clarify）。
    设计模式：数据类 + 注册表装饰器（@register_command 按 DSL 名
              clarify 登记）。
    生命周期：LLM 输出解析时创建，随命令流执行后销毁。

    Attributes:
        question: 澄清问题（可选）
        options: 供用户选择的选项列表（可选）
    """

    # 澄清问题：向用户展示的追问文本；None 表示不携带问题（裸 clarify）。
    question: str | None = None
    # 供用户选择的选项列表：空列表表示自由文本回答；default_factory
    # 避免可变默认值在实例间共享。
    options: list[str] = field(default_factory=list)

    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。

        返回：
            命令名 clarify——注册表键与 DSL 语法动词。
        """
        # 注册表键与 DSL 动词：clarify(...)。
        return "clarify"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClarifyCommand":
        """从字典创建命令。

        业务流程：按键读取 question/options 构造实例，缺失键按
                  字段默认值兜底。

        副作用：无（纯构造）。

        参数：
            data: as_dict 产出的字典，各键允许缺失。

        返回：
            ClarifyCommand 实例。
        """
        # 逐键读取：question 允许缺失（None）；options 缺省空列表。
        return ClarifyCommand(
            question=data.get("question"),
            options=data.get("options", []),
        )

    def run(
        self,
        tracker: DialogueStateTracker,
        flows: Any | None = None,
    ) -> list[dict[str, Any]]:
        """执行澄清命令。

        业务流程：组装一条 clarification_requested 事件字典（问题/
                  选项/时间戳占位）返回；不压栈帧、不改状态。

        副作用：无（纯事件产出——下一步动作由 CommandProcessor
                显式设置为 action_clarify）。

        参数：
            tracker: 对话状态追踪器（本命令未使用，保持接口一致）。
            flows: 可用的Flow列表，允许 None（本命令未使用）。

        返回：
            单元素事件列表：{"event": "clarification_requested", ...}。
        """
        # 纯事件产出：与 HumanHandoffCommand 不同，不压任何栈帧。
        # 事件字典：question/options 透传自身字段；timestamp 固定
        # None（预留字段，当前不填充）。
        return [
            {
                "event": "clarification_requested",
                "question": self.question,
                "options": self.options,
                "timestamp": None,
            }
        ]

    def to_dsl(self) -> str:
        """转换为DSL字符串。

        业务流程：有问题时输出引号形式 clarify("问题")，无问题时
                  输出裸动词。

        副作用：无。

        返回：
            DSL 文本（注意：options 不参与 DSL 序列化）。
        """
        # 有问题：携带问题的引号形式。
        if self.question:
            return f'clarify("{self.question}")'
        # 无问题：裸动词（与正则可选参数分支对应）。
        return "clarify"

    @classmethod
    def regex_pattern(cls) -> str:
        """返回正则表达式模式。

        返回：
            整行锚定的 clarify DSL 正则——参数整体可选（裸动词或
            带一个引号包裹的问题）。
        """
        # 正则逐段拆解：
        # ^                 —— 行首锚定（配合末尾 $ 整行全匹配，防部分匹配）；
        # clarify           —— DSL 动词（与注册表 command_name 同名）；
        # (?: ... )?        —— 非捕获分组 + 可选量词：参数部分整体可选
        #                      （裸动词 clarify 兼容，可选组连左右括号
        #                      一起省略）；
        # \(                —— 左括号；
        # ['"]?             —— 起始引号，可选（不捕获，单双引号均可）；
        # (.*?)             —— 捕获组1：问题文本（惰性匹配——尽早停止
        #                      扩张，把收尾引号留给组后的 ['"]?，保证
        #                      与 to_dsl 输出的引号形式往返对称；
        #                      贪婪 .* 会把收尾引号吞进组内）；
        # ['"]?             —— 收尾引号，可选（不捕获）；
        # \)                —— 右括号（位于可选组内，随参数整体可省略）；
        # $                 —— 行尾锚定。
        return r"""^clarify(?:\(['"]?(.*?)['"]?\))?$"""

    @classmethod
    def _from_regex_match(cls, match: re.Match[str]) -> ClarifyCommand | None:
        """从正则匹配创建命令。

        业务流程：提取捕获组一（问题文本），去首尾空白后构造；
                  组缺失（裸动词）时问题为 None。

        副作用：无（纯构造）。

        参数：
            match: regex_pattern 的匹配结果（命令解析器回填接口约定）。

        返回：
            ClarifyCommand 实例（options 不参与 DSL 解析，恒为空
            列表；类型注解按解析器接口保留 None 可能性）。
        """
        # 组1：澄清问题（裸动词匹配时为 None）。
        question = match.group(1)
        # 去空白后构造：组为空时问题置 None（裸 clarify 语义）。
        return ClarifyCommand(question=question.strip() if question else None)


@register_command
@dataclass
class HumanHandoffCommand(Command):
    """人工转接命令。

    职责：当系统无法处理用户请求或用户明确要求时，请求转接人工
          客服——run 直接压入 HumanHandoffStackFrame，Policy 在
          predict 时检测该栈帧并执行对应操作（命令只负责压帧）。
    设计模式：数据类 + 注册表装饰器（@register_command 按 DSL 名
              human_handoff 登记）。
    生命周期：LLM 输出解析时创建，随命令流执行后销毁。

    Attributes:
        reason: 转接原因
    """

    # 转接原因：人类可读的转接说明；None 表示无原因（裸 human_handoff）。
    reason: str | None = None

    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。

        返回：
            命令名 human_handoff——注册表键与 DSL 语法动词。
        """
        # 注册表键与 DSL 动词：human_handoff(...)。
        return "human_handoff"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HumanHandoffCommand":
        """从字典创建命令。

        业务流程：按键读取 reason 构造实例，缺失键按默认值兜底。

        副作用：无（纯构造）。

        参数：
            data: as_dict 产出的字典，各键允许缺失。

        返回：
            HumanHandoffCommand 实例。
        """
        # 逐键读取：reason 允许缺失（None）。
        return HumanHandoffCommand(reason=data.get("reason"))

    def run(
        self,
        tracker: DialogueStateTracker,
        flows: Any | None = None,
    ) -> list[dict[str, Any]]:
        """执行人工转接命令。

        业务流程：1. 压入 HumanHandoffStackFrame（携带转接原因）；
                  2. 返回 human_handoff_requested 事件标记。

        副作用：修改对话栈（tracker.dialogue_stack 压入转接帧，
                Policy 检测到该帧后决定转接动作）。

        参数：
            tracker: 对话状态追踪器（经其 dialogue_stack 压帧）。
            flows: 可用的Flow列表，允许 None（本命令未使用）。

        返回：
            单元素事件列表：{"event": "human_handoff_requested", ...}。
        """
        # 直接压入 HumanHandoffStackFrame，Policy 会在 predict 时检测并处理。
        # 栈帧类延迟到执行期局部导入：本文件其余命令不依赖对话栈，
        # 局部导入把该依赖范围最小化到实际需要它的命令。
        from app.ics_ai.dialogue_understanding.stack.stack_frame import (
            HumanHandoffStackFrame,
        )

        # 压入转接帧：携带转接原因（None 表示无原因）。
        tracker.dialogue_stack.push(HumanHandoffStackFrame(reason=self.reason))  # pyright: ignore[reportArgumentType]

        # 事件产出：转接请求已受理（真正转接由 Policy 检测帧后的动作执行）。
        # timestamp 固定 None（预留字段，当前不填充）。
        return [
            {
                "event": "human_handoff_requested",
                "reason": self.reason,
                "timestamp": None,
            }
        ]

    def to_dsl(self) -> str:
        """转换为DSL字符串。

        业务流程：有原因时输出引号形式 human_handoff("原因")，
                  无原因时输出裸动词。

        副作用：无。

        返回：
            DSL 文本。
        """
        # 有原因：携带原因的引号形式。
        if self.reason:
            return f'human_handoff("{self.reason}")'
        # 无原因：裸动词（与正则可选参数分支对应）。
        return "human_handoff"

    @classmethod
    def regex_pattern(cls) -> str:
        """返回正则表达式模式。

        返回：
            整行锚定的 human_handoff DSL 正则——参数整体可选。
        """
        # 正则逐段拆解（与 ClarifyCommand 同构）：
        # ^                 —— 行首锚定（配合末尾 $ 整行全匹配，防部分匹配）；
        # human_handoff     —— DSL 动词（与注册表 command_name 同名）；
        # (?: ... )?        —— 非捕获分组 + 可选量词：参数部分整体可选
        #                      （裸动词 human_handoff 兼容，可选组连左右
        #                      括号一起省略）；
        # \(                —— 左括号；
        # ['"]?             —— 起始引号，可选（不捕获，单双引号均可）；
        # (.*?)             —— 捕获组1：原因文本（惰性匹配——尽早停止
        #                      扩张，把收尾引号留给组后的 ['"]?，保证
        #                      与 to_dsl 输出的引号形式往返对称；
        #                      贪婪 .* 会把收尾引号吞进组内）；
        # ['"]?             —— 收尾引号，可选（不捕获）；
        # \)                —— 右括号（位于可选组内，随参数整体可省略）；
        # $                 —— 行尾锚定。
        return r"""^human_handoff(?:\(['"]?(.*?)['"]?\))?$"""

    @classmethod
    def _from_regex_match(cls, match: re.Match[str]) -> "HumanHandoffCommand | None":
        """从正则匹配创建命令。

        业务流程：提取捕获组一（原因文本），去首尾空白后构造；
                  组缺失（裸动词）时原因为 None。

        副作用：无（纯构造）。

        参数：
            match: regex_pattern 的匹配结果（命令解析器回填接口约定）。

        返回：
            HumanHandoffCommand 实例（类型注解按解析器接口保留
            None 可能性）。
        """
        # 组1：转接原因（裸动词匹配时为 None）。
        reason = match.group(1)
        # 去空白后构造：组为空时原因置 None（裸 human_handoff 语义）。
        return HumanHandoffCommand(reason=reason.strip() if reason else None)


@register_command
@dataclass
class RestartCommand(Command):
    """重启命令。

    职责：请求重置当前对话状态（清空所有槽位和Flow）——run 仅产出
          restart_requested 事件，真正的重启操作由 ActionRestart 执行
          （命令只表达意图不执行操作）。
    设计模式：数据类 + 注册表装饰器（@register_command 按 DSL 名
              restart 登记）。
    生命周期：LLM 输出解析时创建，随命令流执行后销毁。
    """

    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。

        返回：
            命令名 restart——注册表键与 DSL 语法动词。
        """
        # 注册表键与 DSL 动词：restart（无参形式）。
        return "restart"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RestartCommand":
        """从字典创建命令。

        业务流程：本命令无字段，忽略字典内容直接构造。

        副作用：无（纯构造）。

        参数：
            data: as_dict 产出的字典（内容被忽略，允许任意）。

        返回：
            RestartCommand 实例（无状态，所有实例等价）。
        """
        # 无字段命令：直接构造（data 仅满足接口签名）。
        return RestartCommand()

    def run(
        self,
        tracker: DialogueStateTracker,
        flows: Any | None = None,
    ) -> list[dict[str, Any]]:
        """执行重启命令。

        业务流程：组装一条 restart_requested 事件字典返回；不做
                  任何状态修改。

        副作用：无（纯事件产出——重启由 ActionRestart 执行）。

        参数：
            tracker: 对话状态追踪器（本命令未使用，保持接口一致）。
            flows: 可用的Flow列表，允许 None（本命令未使用）。

        返回：
            单元素事件列表：{"event": "restart_requested", ...}。
        """
        # 不直接重启，让 ActionRestart 来执行（命令只表达意图）。
        # 事件字典：timestamp 固定 None（预留字段，当前不填充）。
        return [
            {
                "event": "restart_requested",
                "timestamp": None,
            }
        ]

    def to_dsl(self) -> str:
        """转换为DSL字符串。

        副作用：无。

        返回：
            DSL 文本：无参动词形式 restart。
        """
        # 无参动词：与正则 ^restart$ 对应。
        return "restart"

    @classmethod
    def regex_pattern(cls) -> str:
        """返回正则表达式模式。

        返回：
            整行锚定的无参动词正则（不允许携带参数）。
        """
        # 正则逐段拆解：
        # ^              —— 行首锚定（配合末尾 $ 整行全匹配，防部分匹配）；
        # restart        —— DSL 动词（与注册表 command_name 同名）；
        # $              —— 行尾锚定（无参动词，不允许携带任何参数）。
        return r"""^restart$"""

    @classmethod
    def _from_regex_match(cls, match: re.Match[str]) -> "RestartCommand | None":
        """从正则匹配创建命令。

        业务流程：本命令无字段，忽略匹配内容直接构造。

        副作用：无（纯构造）。

        参数：
            match: regex_pattern 的匹配结果（命令解析器回填接口约定）。

        返回：
            RestartCommand 实例（无状态，实际不会返回 None）。
        """
        # 无字段命令：直接构造（match 仅满足接口签名）。
        return RestartCommand()


@register_command
@dataclass
class NoopCommand(Command):
    """空操作命令。

    职责：显式表示"不做任何事"——run 返回空事件列表，供需要
          无动作占位的场景使用（CommandProcessor 的响应类型归类
          无对应分支，不改变 response_type）。
    设计模式：数据类 + 注册表装饰器（@register_command 按 DSL 名
              noop 登记）。
    生命周期：LLM 输出解析时创建，随命令流执行后销毁。
    """

    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。

        返回：
            命令名 noop——注册表键与 DSL 语法动词。
        """
        # 注册表键与 DSL 动词：noop（无参形式）。
        return "noop"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NoopCommand":
        """从字典创建命令。

        业务流程：本命令无字段，忽略字典内容直接构造。

        副作用：无（纯构造）。

        参数：
            data: as_dict 产出的字典（内容被忽略，允许任意）。

        返回：
            NoopCommand 实例（无状态，所有实例等价）。
        """
        # 无字段命令：直接构造（data 仅满足接口签名）。
        return NoopCommand()

    def run(
        self,
        tracker: DialogueStateTracker,
        flows: Any | None = None,
    ) -> list[dict[str, Any]]:
        """执行空操作命令。

        业务流程：直接返回空列表，不产出事件、不修改状态。

        副作用：无。

        参数：
            tracker: 对话状态追踪器（本命令未使用，保持接口一致）。
            flows: 可用的Flow列表，允许 None（本命令未使用）。

        返回：
            空事件列表（本命令的"空"即语义本身）。
        """
        # 空操作：无事件、无状态变更。
        return []

    def to_dsl(self) -> str:
        """转换为DSL字符串。

        副作用：无。

        返回：
            DSL 文本：无参动词形式 noop。
        """
        # 无参动词：与正则 ^noop$ 对应。
        return "noop"

    @classmethod
    def regex_pattern(cls) -> str:
        """返回正则表达式模式。

        返回：
            整行锚定的无参动词正则（不允许携带参数）。
        """
        # 正则逐段拆解：
        # ^              —— 行首锚定（配合末尾 $ 整行全匹配，防部分匹配）；
        # noop           —— DSL 动词（与注册表 command_name 同名）；
        # $              —— 行尾锚定（无参动词，不允许携带任何参数）。
        return r"""^noop$"""

    @classmethod
    def _from_regex_match(cls, match: re.Match[str]) -> "NoopCommand | None":
        """从正则匹配创建命令。

        业务流程：本命令无字段，忽略匹配内容直接构造。

        副作用：无（纯构造）。

        参数：
            match: regex_pattern 的匹配结果（命令解析器回填接口约定）。

        返回：
            NoopCommand 实例（无状态，实际不会返回 None）。
        """
        # 无字段命令：直接构造（match 仅满足接口签名）。
        return NoopCommand()


# 导出：模块对外暴露的公共 API（按字母序，便于 IDE 提示与 diff 整洁）。
__all__ = [
    "ClarifyCommand",
    "HumanHandoffCommand",
    "NoopCommand",
    "RestartCommand",
    "SessionStartCommand",
]
