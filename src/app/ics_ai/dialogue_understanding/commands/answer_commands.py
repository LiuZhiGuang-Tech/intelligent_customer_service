"""
answer_commands - 回答类命令模块

核心功能：定义四种"以生成回答/响应为目标"的命令——
          ChitChatAnswerCommand（闲聊回复）、CannotHandleCommand
          （无法处理降级）、KnowledgeAnswerCommand（知识库检索回答
          RAG）、FreeFormAnswerCommand（自由回答）；均继承 Command
          基类并经 @register_command 注册，支持字典反序列化
          （from_dict）与 DSL 文本正则解析（regex_pattern +
          _from_regex_match）两条构造路径。

执行流程：
    1. 装配期：模块被导入时，四个命令类经 @register_command 装饰器
       以 command_name() 为键登记进 base 模块的 _COMMAND_REGISTRY；
    2. 反序列化：LLM 输出的命令字典经各自的 from_dict 构造，或命令
       DSL 文本经 from_dsl 正则匹配构造（覆盖 "chitchat"、
       'cannot_handle("原因")'、"SearchAndReply(...)" 旧别名等写法）；
    3. 执行期：CommandProcessor 调用 run()——前三类命令直接向
       tracker.dialogue_stack 压入对应栈帧（ChitChatStackFrame /
       CannotHandleStackFrame / SearchStackFrame），由
       EnterpriseSearchPolicy 检测栈帧并生成响应，事件中的
       degradation_reason 字段记录降级原因供下游使用；
       FreeFormAnswerCommand 仅返回事件标记，不压栈帧。

设计说明（本模块核心设计意图）——"压栈帧 + Policy 检测"模式：
    回答类命令不直接生成回复文本，而是压入语义化栈帧声明意图，由
    EnterpriseSearchPolicy 统一检测栈帧并生成响应，实现"命令声明
    意图、策略生成响应"的解耦（与 base.py 文件头 Command/Action
    链路中的"栈帧化"设计呼应）。

副作用：
    - 导入期：向 base 模块的全局注册表 _COMMAND_REGISTRY 写入四个
      命令类（全局可变状态，跨会话持续生效）；
    - run 的具体实现：ChitChat / CannotHandle / KnowledgeAnswer
      三类命令会向 tracker.dialogue_stack 压栈（修改 tracker 状态）；
      本模块自身无 IO、不调用网络。

主要依赖：
    - re：from_dsl 的 DSL 文本正则解析。
    - dataclasses：@dataclass 自动生成命令类的数据类方法。
    - typing：Any + TYPE_CHECKING（注解已按 PEP 585/604 新式写法）。
    - app.ics_ai.dialogue_understanding.commands.base：命令基类
      Command 与注册装饰器 register_command。
    - app.ics_ai.shared.constants：DegradationReason 降级原因常量。
    - app.ics_ai.dialogue_understanding.stack.stack_frame：各栈帧类
      （run 内延迟导入）；app.ics_ai.core.tracker：run 的操作对象
      （仅注解用）。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，支持前向引用
# （如 TYPE_CHECKING 分支中的 tracker 类型与各命令类自引用）。
from __future__ import annotations

# 导入正则引擎：from_dsl 的模式匹配（regex_pattern + _from_regex_match）依赖它。
import re

# 导入 dataclass 装饰器：四个命令类靠它自动生成 __init__ / __repr__ 等数据类方法。
from dataclasses import dataclass

# 导入类型注解工具：Any 类型通配符；TYPE_CHECKING 隔离仅注解导入
# （PEP 585/604 新式注解下 Dict/List/Optional 已不需要）。
from typing import Any, TYPE_CHECKING

# 导入命令基类与注册装饰器：四个命令类继承 Command 并经 register_command 注册。
from app.ics_ai.dialogue_understanding.commands.base import Command, register_command

# 导入降级原因常量：run 返回的事件中用 degradation_reason 字段记录分类。
from app.ics_ai.shared.constants import DegradationReason

# 类型检查期专属导入：tracker 类型只出现在方法注解中，运行时不
# 导入（避免与 core 包形成循环依赖）。
if TYPE_CHECKING:
    # 对话状态追踪器：命令 run 的操作对象（仅注解用）。
    from app.ics_ai.core.tracker import DialogueStateTracker


@register_command
@dataclass
class ChitChatAnswerCommand(Command):
    """闲聊回答命令。

    职责：封装"处理用户闲聊输入"的意图——输入不属于任何业务 Flow
          时由 LLM 生成该命令；run() 向对话栈压入 ChitChatStackFrame，
          由 EnterpriseSearchPolicy 检测该帧并使用 LLM 生成自然闲聊
          回复。
    设计模式：命令模式 + 注册表工厂；压栈帧 + Policy 检测模式——
              run() 直接修改 tracker 状态（压栈），属"即时数据命令"。
    生命周期：LLM 生成器产出命令文本/字典 → from_dict 或 from_dsl
              构造实例 → CommandProcessor 执行 run() 后随单轮消息
              处理流程销毁。
    """

    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。

        返回：
            "chitchat"：本命令在注册表中的唯一键。
        """
        # 注册表键：LLM 字典输出中 "command" 字段的取值即此名。
        return "chitchat"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChitChatAnswerCommand:
        """从字典创建命令。

        业务流程：本命令无参数字段，忽略 data 内容直接构造实例。

        副作用：无（纯构造）。

        参数：
            data: LLM 输出的命令字典（无有效参数键，允许任意内容）。

        返回：
            ChitChatAnswerCommand 实例。
        """
        # 无参命令：不读取 data 任何键，直接构造。
        return ChitChatAnswerCommand()

    def run(
        self,
        tracker: DialogueStateTracker,
        flows: Any | None = None,
    ) -> list[dict[str, Any]]:
        """执行闲聊回答命令。

        业务流程：1. 向 tracker.dialogue_stack 压入 ChitChatStackFrame
                  声明闲聊处理状态；2. 返回 chitchat_triggered 事件
                  （携带降级原因常量）。

        副作用：修改 tracker 状态（向对话栈压入闲聊栈帧）。

        参数：
            tracker: 对话状态追踪器，必需。
            flows: 可用的 Flow 列表，本命令不使用，允许 None。

        返回：
            单元素事件列表（event=chitchat_triggered，含
            degradation_reason 与时间戳占位）。
        """
        # 函数内延迟导入：将 stack 包的加载推迟到命令执行期，避免
        # commands 包在装配期强依赖具体栈帧类型（模块缓存保证重复
        # 导入开销极低）。
        from app.ics_ai.dialogue_understanding.stack.stack_frame import (
            ChitChatStackFrame,
        )

        # 压入闲聊栈帧：声明"当前处于闲聊处理"状态，供 Policy 检测。
        tracker.dialogue_stack.push(ChitChatStackFrame())

        # 返回闲聊触发事件列表：
        return [
            {
                # 事件类型标记：由下游 Policy/Action 识别。
                "event": "chitchat_triggered",
                # 事件负载：闲聊属于非业务请求的降级分类常量。
                "degradation_reason": DegradationReason.CHITCHAT,
                # 事件时间戳占位：由 tracker 后续补齐。
                "timestamp": None,
            }
        ]

    def to_dsl(self) -> str:
        """转换为DSL字符串。

        业务流程：输出裸命令名（与基类默认实现一致，显式覆盖以对齐
                  本类 DSL 风格，与 regex_pattern 对应可往返解析）。

        副作用：无。

        返回：
            DSL 格式的命令字符串。
        """
        # 无参数命令：DSL 即命令名本身。
        return "chitchat"

    @classmethod
    def regex_pattern(cls) -> str:
        """返回用于解析此命令的正则表达式模式。

        业务流程：返回全串匹配（^...$）的模式串，支持两种等价写法——
                  1. 裸命令名 "chitchat"；
                  2. 函数式 "ChitChat()"（无参）。

        副作用：无。

        返回：
            正则表达式字符串（无捕获组，本命令无参数）。
        """
        # ^...$ 全串锚定防止部分匹配；分支覆盖裸名与函数式两种写法。
        return r"""^(?:chitchat|ChitChat\(\))$"""

    @classmethod
    def _from_regex_match(cls, match: re.Match[str]) -> ChitChatAnswerCommand | None:
        """从正则匹配结果创建命令。

        业务流程：本命令无参数，忽略匹配结果（模式无捕获组）直接
                  构造实例。

        副作用：无（纯构造）。

        参数：
            match: from_dsl 中 re.match 的匹配结果（本命令不使用）。

        返回：
            ChitChatAnswerCommand 实例（不返回 None）。
        """
        # 无参命令：不提取分组，直接构造。
        return ChitChatAnswerCommand()


@register_command
@dataclass
class CannotHandleCommand(Command):
    """无法处理命令。

    职责：封装"系统无法理解或处理用户输入"的降级意图；run() 向
          对话栈压入 CannotHandleStackFrame（携带原因），由
          EnterpriseSearchPolicy 检测该帧并生成降级响应。
    设计模式：命令模式 + 注册表工厂；压栈帧 + Policy 检测模式——
              run() 直接修改 tracker 状态（压栈），属"即时数据命令"。
    生命周期：同其余命令——LLM 产出 → from_dict 或 from_dsl 构造
              实例 → CommandProcessor 执行 run() 后随单轮消息处理
              流程销毁。

    字段说明：
        reason: 无法处理的原因。
    """

    # 无法处理的原因（对应 DegradationReason 常量取值）；缺省取默认降级原因。
    reason: str = DegradationReason.DEFAULT

    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。

        返回：
            "cannot_handle"：本命令在注册表中的唯一键。
        """
        # 注册表键：LLM 字典输出中 "command" 字段的取值即此名。
        return "cannot_handle"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CannotHandleCommand:
        """从字典创建命令。

        业务流程：读取 data["reason"]（可缺省，缺省取默认降级原因）
                  构造实例。

        副作用：无（纯构造）。

        参数：
            data: LLM 输出的命令字典；"reason" 键可缺失。

        返回：
            CannotHandleCommand 实例。
        """
        # reason 为可选字段：dict.get 未命中时取默认降级原因常量。
        return CannotHandleCommand(reason=data.get("reason", DegradationReason.DEFAULT))

    def run(
        self,
        tracker: DialogueStateTracker,
        flows: Any | None = None,
    ) -> list[dict[str, Any]]:
        """执行无法处理命令。

        业务流程：1. 向 tracker.dialogue_stack 压入
                  CannotHandleStackFrame（携带本命令的 reason）；
                  2. 返回 cannot_handle 事件（携带原因与降级常量）。

        副作用：修改 tracker 状态（向对话栈压入无法处理栈帧）。

        参数：
            tracker: 对话状态追踪器，必需。
            flows: 可用的 Flow 列表，本命令不使用，允许 None。

        返回：
            单元素事件列表（event=cannot_handle，含 reason、
            degradation_reason 与时间戳占位）。
        """
        # 函数内延迟导入：将 stack 包的加载推迟到命令执行期，避免
        # commands 包在装配期强依赖具体栈帧类型。
        from app.ics_ai.dialogue_understanding.stack.stack_frame import (
            CannotHandleStackFrame,
        )

        # 压入无法处理栈帧：携带原因，供 Policy 生成针对性的降级响应。
        tracker.dialogue_stack.push(CannotHandleStackFrame(reason=self.reason))

        # 返回无法处理事件列表：
        return [
            {
                # 事件类型标记：由下游 Policy/Action 识别。
                "event": "cannot_handle",
                # 事件负载：无法处理的具体原因。
                "reason": self.reason,
                # 事件负载：降级分类常量（无法处理）。
                "degradation_reason": DegradationReason.CANNOT_HANDLE,
                # 事件时间戳占位：由 tracker 后续补齐。
                "timestamp": None,
            }
        ]

    def to_dsl(self) -> str:
        """转换为DSL字符串。

        业务流程：reason 非空且非默认值时输出 'cannot_handle("原因")'
                  带参写法；否则输出裸命令名（两种写法均被
                  regex_pattern 覆盖，可往返解析）。

        副作用：无。

        返回：
            DSL 格式的命令字符串。
        """
        # 带 reason 的完整写法：仅在显式提供了非默认原因时输出。
        if self.reason and self.reason != DegradationReason.DEFAULT:
            # 引号包裹原因，与 regex_pattern 的捕获格式对应。
            return f'cannot_handle("{self.reason}")'
        # 默认原因或空原因：省略参数的裸命令名写法。
        return "cannot_handle"

    @classmethod
    def regex_pattern(cls) -> str:
        """返回用于解析此命令的正则表达式模式。

        业务流程：返回全串匹配（^...$）的模式串，支持两种写法——
                  1. "cannot_handle(...)"（原因可选，入捕获组 1）；
                  2. "CannotHandle(...)"（原因可选，入捕获组 2）。

        副作用：无。

        返回：
            正则表达式字符串（两个捕获组均可选，组内允许空匹配）。
        """
        # 正则逐段拆解（两分支覆盖同一命令的新旧两种 LLM 输出写法）：
        # ^                  —— 行首锚定（配合末尾 $ 整行全匹配，防部分匹配）；
        # (?: A | B )        —— 非捕获分组：仅分组不捕获，包裹两个等价写法分支；
        # —— 分支 A：小写主名形式 cannot_handle(...) ——
        # cannot_handle      —— DSL 动词（与注册表 command_name 同名）；
        # (?: ... )?         —— 参数部分整体可选（裸动词 cannot_handle 兼容，
        #                       可选组连左右括号一起省略）；
        # \(                 —— 左括号；
        # ['"]?              —— 起始引号，可选（不捕获，单双引号均可）；
        # (.*?)              —— 捕获组1：原因文本（惰性匹配——尽早停止扩张，
        #                       把收尾引号留给组后的 ['"]?，保证与 to_dsl
        #                       输出的引号形式往返对称；贪婪 .* 会把收尾
        #                       引号吞进组内）；
        # ['"]?              —— 收尾引号，可选（不捕获）；
        # \)                 —— 右括号（位于可选组内，随参数整体可省略）；
        # —— 分支 B：旧别名 PascalCase 形式 CannotHandle(...) ——
        # CannotHandle       —— 旧别名动词（兼容历史 LLM 输出写法）；
        # \(                 —— 左括号（此分支括号必需——与分支 A 不同，
        #                       可选性仅覆盖括号内参数）；
        # ['"]?              —— 起始引号，可选（不捕获）；
        # (.*?)              —— 捕获组2：原因文本（惰性，语义同组1）；
        # ['"]?              —— 收尾引号，可选（不捕获）；
        # \)                 —— 右括号（必需；空参 CannotHandle() 时中间
        #                       可选组整体匹配空，组 2 得空串，由
        #                       _from_regex_match 兜底为默认原因）；
        # $                  —— 行尾锚定。
        return r"""^(?:cannot_handle(?:\(['"]?(.*?)['"]?\))?|CannotHandle\((?:['"]?(.*?)['"]?)?\))$"""

    @classmethod
    def _from_regex_match(cls, match: re.Match[str]) -> CannotHandleCommand | None:
        """从正则匹配结果创建命令。

        业务流程：1. 取第一个捕获组（cannot_handle 格式的原因），为
                  空（含空串）则取第二个捕获组（CannotHandle 格式）；
                  2. 两组均空/为空串时兜底为默认降级原因；3. 命中的
                  原因 strip 去空白后构造实例。

        副作用：无（纯构造）。

        参数：
            match: from_dsl 中 re.match 的匹配结果。

        返回：
            CannotHandleCommand 实例（本实现不返回 None）。
        """
        # 兜底链：组 1 优先，其次组 2，均无（None 或空串均为假值）取默认原因。
        reason = match.group(1) or match.group(2) or DegradationReason.DEFAULT
        # 命中原因时 strip 去空白构造；reason 为 DEFAULT（非空串）时同样走 strip。
        return CannotHandleCommand(
            reason=reason.strip() if reason else DegradationReason.DEFAULT
        )


@register_command
@dataclass
class KnowledgeAnswerCommand(Command):
    """知识库回答命令。

    职责：封装"基于知识库检索生成回答（RAG）"的意图；run() 向对话
          栈压入 SearchStackFrame，由 EnterpriseSearchPolicy 检测该帧
          并执行检索与回答生成。
    设计模式：命令模式 + 注册表工厂；压栈帧 + Policy 检测模式——
              run() 直接修改 tracker 状态（压栈），属"即时数据命令"。
    生命周期：同其余命令——LLM 产出 → from_dict 或 from_dsl 构造
              实例 → CommandProcessor 执行 run() 后随单轮消息处理
              流程销毁。

    字段说明：
        query: 用于知识库检索的查询（可选，默认使用用户最新消息）。
    """

    # 检索查询词；None 表示不显式指定，检索时回退到用户最新消息文本。
    query: str | None = None

    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。

        返回：
            "knowledge_answer"：本命令在注册表中的唯一键。
        """
        # 注册表键：LLM 字典输出中 "command" 字段的取值即此名。
        return "knowledge_answer"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeAnswerCommand:
        """从字典创建命令。

        业务流程：读取 data["query"]（可缺省，缺省得 None）构造实例。

        副作用：无（纯构造）。

        参数：
            data: LLM 输出的命令字典；"query" 键可缺失。

        返回：
            KnowledgeAnswerCommand 实例（query 可能为 None）。
        """
        # query 为可选字段：dict.get 未命中时得 None（回退到最新消息）。
        return KnowledgeAnswerCommand(query=data.get("query"))

    def run(
        self,
        tracker: DialogueStateTracker,
        flows: Any | None = None,
    ) -> list[dict[str, Any]]:
        """执行知识库回答命令。

        业务流程：1. 向 tracker.dialogue_stack 压入 SearchStackFrame
                  （帧不存储 query——检索时从 latest_message 获取）；
                  2. 确定实际查询词：优先本命令的 query，为空时取
                  tracker.latest_message 的文本；3. 返回
                  knowledge_search_triggered 事件（携带查询词与降级
                  常量）。

        副作用：修改 tracker 状态（向对话栈压入搜索栈帧）。

        参数：
            tracker: 对话状态追踪器，必需（用于压栈与读取最新消息）。
            flows: 可用的 Flow 列表，本命令不使用，允许 None。

        返回：
            单元素事件列表（event=knowledge_search_triggered，含
            query、degradation_reason 与时间戳占位）。
        """
        # 函数内延迟导入：将 stack 包的加载推迟到命令执行期，避免
        # commands 包在装配期强依赖具体栈帧类型。
        from app.ics_ai.dialogue_understanding.stack.stack_frame import SearchStackFrame

        # 压入搜索栈帧：声明"当前处于知识检索"状态，供 Policy 检测；
        # 帧构造不传 query，与 SearchStackFrame 的设计约定一致。
        tracker.dialogue_stack.push(SearchStackFrame())

        # 确定实际查询词：先取命令显式携带的 query。
        query = self.query
        # 未显式指定且存在最新用户消息时：回退取其文本作为查询词。
        if not query and tracker.latest_message:
            # 从最新消息对象读取文本内容。
            query = tracker.latest_message.text

        # 返回知识检索触发事件列表：
        return [
            {
                # 事件类型标记：由下游 Policy/Action 识别。
                "event": "knowledge_search_triggered",
                # 事件负载：实际使用的检索查询词（可能仍为 None）。
                "query": query,
                # 事件负载：降级分类常量（无相关答案场景）。
                "degradation_reason": DegradationReason.NO_RELEVANT_ANSWER,
                # 事件时间戳占位：由 tracker 后续补齐。
                "timestamp": None,
            }
        ]

    def to_dsl(self) -> str:
        """转换为DSL字符串。

        业务流程：query 非空时输出 'knowledge_answer("查询")' 带参
                  写法；为 None 输出裸命令名（两种写法均被
                  regex_pattern 覆盖，可往返解析）。

        副作用：无。

        返回：
            DSL 格式的命令字符串。
        """
        # 带 query 的完整写法：引号包裹查询词，与 regex_pattern 对应。
        if self.query:
            return f'knowledge_answer("{self.query}")'
        # 未指定查询词：省略参数的裸命令名写法。
        return "knowledge_answer"

    @classmethod
    def regex_pattern(cls) -> str:
        """返回用于解析此命令的正则表达式模式。

        业务流程：返回全串匹配（^...$）的模式串，支持两种写法——
                  1. "knowledge_answer(...)"（查询词可选，入捕获组 1）；
                  2. "SearchAndReply(...)"（旧版别名，查询词可选，
                  入捕获组 2）。

        副作用：无。

        返回：
            正则表达式字符串（两个捕获组均可选，组内允许空匹配）。
        """
        # 正则逐段拆解（与 CannotHandleCommand 同构，两分支覆盖
        # 新名与旧别名两种 LLM 输出写法）：
        # ^                  —— 行首锚定（配合末尾 $ 整行全匹配，防部分匹配）；
        # (?: A | B )        —— 非捕获分组：仅分组不捕获，包裹两个等价写法分支；
        # —— 分支 A：小写主名形式 knowledge_answer(...) ——
        # knowledge_answer   —— DSL 动词（与注册表 command_name 同名）；
        # (?: ... )?         —— 参数部分整体可选（裸动词 knowledge_answer
        #                       兼容，可选组连左右括号一起省略）；
        # \(                 —— 左括号；
        # ['"]?              —— 起始引号，可选（不捕获，单双引号均可）；
        # (.*?)              —— 捕获组1：查询词（惰性匹配——尽早停止扩张，
        #                       把收尾引号留给组后的 ['"]?，保证与 to_dsl
        #                       输出的引号形式往返对称；贪婪 .* 会把收尾
        #                       引号吞进组内）；
        # ['"]?              —— 收尾引号，可选（不捕获）；
        # \)                 —— 右括号（位于可选组内，随参数整体可省略）；
        # —— 分支 B：旧别名形式 SearchAndReply(...) ——
        # SearchAndReply     —— 旧别名动词（兼容历史 LLM 输出写法）；
        # \(                 —— 左括号（此分支括号必需——与分支 A 不同，
        #                       可选性仅覆盖括号内参数）；
        # ['"]?              —— 起始引号，可选（不捕获）；
        # (.*?)              —— 捕获组2：查询词（惰性，语义同组1）；
        # ['"]?              —— 收尾引号，可选（不捕获）；
        # \)                 —— 右括号（必需；空参 SearchAndReply() 时
        #                       中间可选组整体匹配空，组 2 得空串，由
        #                       _from_regex_match 归一为 None）；
        # $                  —— 行尾锚定。
        return r"""^(?:knowledge_answer(?:\(['"]?(.*?)['"]?\))?|SearchAndReply\((?:['"]?(.*?)['"]?)?\))$"""

    @classmethod
    def _from_regex_match(cls, match: re.Match[str]) -> KnowledgeAnswerCommand | None:
        """从正则匹配结果创建命令。

        业务流程：1. 取第一个捕获组（knowledge_answer 格式的查询词），
                  为空（含空串）则取第二个捕获组（SearchAndReply 旧
                  别名格式）；2. 命中查询词则 strip 去空白后构造；
                  3. 两组均空构造 query=None 实例（回退到最新消息）。

        副作用：无（纯构造）。

        参数：
            match: from_dsl 中 re.match 的匹配结果。

        返回：
            KnowledgeAnswerCommand 实例（query 可能为 None，本实现
            不返回 None）。
        """
        # 第一个分组是 knowledge_answer 格式的查询词，第二个是 SearchAndReply 旧别名：优先取前者。
        query = match.group(1) or match.group(2)
        # 命中查询词：strip 去空白后构造；否则构造 query=None 实例。
        return KnowledgeAnswerCommand(query=query.strip() if query else None)


@register_command
@dataclass
class FreeFormAnswerCommand(Command):
    """自由回答命令。

    职责：封装"生成不受 Flow 约束的自由回答"的意图；run() 仅返回
          事件标记，不压栈帧——后续响应生成由下游流程自行处理。
    设计模式：命令模式 + 注册表工厂；事件标记模式（既非压栈的
              "即时数据命令"，也非由特定 Action 接管的"动作触发
              命令"，仅向下游广播意图事件）。
    生命周期：同其余命令——LLM 产出 → from_dict 或 from_dsl 构造
              实例 → CommandProcessor 执行 run() 后随单轮消息处理
              流程销毁。
    """

    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。

        返回：
            "free_form_answer"：本命令在注册表中的唯一键。
        """
        # 注册表键：LLM 字典输出中 "command" 字段的取值即此名。
        return "free_form_answer"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FreeFormAnswerCommand:
        """从字典创建命令。

        业务流程：本命令无参数字段，忽略 data 内容直接构造实例。

        副作用：无（纯构造）。

        参数：
            data: LLM 输出的命令字典（无有效参数键，允许任意内容）。

        返回：
            FreeFormAnswerCommand 实例。
        """
        # 无参命令：不读取 data 任何键，直接构造。
        return FreeFormAnswerCommand()

    def run(
        self,
        tracker: DialogueStateTracker,
        flows: Any | None = None,
    ) -> list[dict[str, Any]]:
        """执行自由回答命令。

        业务流程：仅返回 free_form_answer_triggered 事件标记，不压
                  栈帧、不修改 tracker。

        副作用：无（不修改 tracker 状态）。

        参数：
            tracker: 对话状态追踪器，本命令不使用其状态。
            flows: 可用的 Flow 列表，本命令不使用，允许 None。

        返回：
            单元素事件列表（event=free_form_answer_triggered，含
            时间戳占位）。
        """
        # 返回自由回答触发事件列表（无负载字段，仅类型标记与时间戳占位）：
        return [
            {
                # 事件类型标记：由下游 Policy/Action 识别。
                "event": "free_form_answer_triggered",
                # 事件时间戳占位：由 tracker 后续补齐。
                "timestamp": None,
            }
        ]

    def to_dsl(self) -> str:
        """转换为DSL字符串。

        业务流程：输出裸命令名（与基类默认实现一致，显式覆盖以对齐
                  本类 DSL 风格，与 regex_pattern 对应可往返解析）。

        副作用：无。

        返回：
            DSL 格式的命令字符串。
        """
        # 无参数命令：DSL 即命令名本身。
        return "free_form_answer"

    @classmethod
    def regex_pattern(cls) -> str:
        """返回用于解析此命令的正则表达式模式。

        业务流程：返回全串匹配（^...$）的模式串，仅支持裸命令名
                  "free_form_answer" 一种写法。

        副作用：无。

        返回：
            正则表达式字符串（无捕获组，本命令无参数）。
        """
        # ^...$ 全串锚定防止部分匹配；仅裸命令名一种写法。
        return r"""^free_form_answer$"""

    @classmethod
    def _from_regex_match(cls, match: re.Match[str]) -> FreeFormAnswerCommand | None:
        """从正则匹配结果创建命令。

        业务流程：本命令无参数，忽略匹配结果（模式无捕获组）直接
                  构造实例。

        副作用：无（纯构造）。

        参数：
            match: from_dsl 中 re.match 的匹配结果（本命令不使用）。

        返回：
            FreeFormAnswerCommand 实例（不返回 None）。
        """
        # 无参命令：不提取分组，直接构造。
        return FreeFormAnswerCommand()


# 导出
__all__ = [
    "ChitChatAnswerCommand",
    "CannotHandleCommand",
    "KnowledgeAnswerCommand",
    "FreeFormAnswerCommand",
]
