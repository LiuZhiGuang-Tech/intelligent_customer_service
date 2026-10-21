"""
llm_generator - LLM 命令生成器模块

核心功能：命令生成阶段的默认实现——继承 CommandGenerator 抽象基类，
          每轮消息将对话上下文交给 LLM，再把 LLM 输出的 DSL 文本解析
          为 Command 对象列表；生成失败/解析为空时有分级兜底，保证
          下游命令流始终拿到可执行命令（"永远返回结果"契约）。

执行流程：
    1. 装配期：from_config（工厂方法）或 __init__ 直接构造——配置存入
       LLMGeneratorConfig；PromptBuilder 与 CommandParser 缺省时按
       默认参数创建；LLM 客户端支持注入，否则延迟到首次访问
       llm_client 属性时才经工厂创建（懒加载）。
    2. 每轮消息：generate 四步走——
       a. PromptBuilder.build_messages 构建提示词（上下文/槽位/Flow）；
       b. await llm_client.complete 异步调用 LLM 得到回复文本；
       c. CommandParser.parse 将回复解析为 Command 列表；
       d. 解析结果为空时兜底 CannotHandleCommand（reason=parse_failed）。
    3. 异常分级：LLMException → InternalErrorCommand；其余意外异常 →
       ErrorCommand，错误信息写入 result.metadata，不向调用方抛出。

副作用：
    - generate 内通过 llm_client 调用外部 LLM 服务（网络 IO）；
    - 使用 logging 记录调试/信息/警告/错误日志；
    - 无文件/数据库读写；模块导入时仅定义类与日志器，无全局状态修改。

主要依赖：
    - app.ics_ai.dialogue_understanding.generator.base_generator：生成器
      契约与结果 DTO（CommandGenerator/GeneratorConfig/GenerationResult）；
    - generator.prompt_builder.PromptBuilder：提示词构建；
    - generator.command_parser.CommandParser：LLM 输出解析；
    - commands.answer_commands / commands.error_commands：兜底命令与
      错误命令（CannotHandle/InternalError/Error）；
    - app.ics_ai.shared.llm：create_llm_client 工厂与 LLMClient 协议；
    - app.ics_ai.shared.exceptions：LLMException 业务异常。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，TYPE_CHECKING 分支
# 中的 tracker/domain 前向引用无需引号即可书写，PEP 604 联合类型注解
# （X | None）也不受运行时版本限制。
from __future__ import annotations

# 导入 logging：模块级日志（提示词长度/LLM 回复/解析错误/异常告警）。
import logging

# 导入 dataclass 装饰器：LLMGeneratorConfig 靠它自动生成 __init__ 等
# 数据类方法。
from dataclasses import dataclass

# 导入类型注解工具：Any 类型通配符；TYPE_CHECKING 隔离仅注解导入
# （注解已按 PEP 585/604 新式写法，Dict/List/Optional 不再需要）。
from typing import Any, TYPE_CHECKING

# 生成器契约与结果 DTO：CommandGenerator 定义 generate 抽象入口与
# generate_sync 同步桥接，GeneratorConfig 为公共配置基类（历史轮数/
# 槽位与 Flow 开关），GenerationResult 为生成结果信封。
from app.ics_ai.dialogue_understanding.generator.base_generator import (
    CommandGenerator,
    GeneratorConfig,
    GenerationResult,
)

# 提示词构建器：把 Tracker/Domain/Flows 组装为 LLM 提示词消息列表。
from app.ics_ai.dialogue_understanding.generator.prompt_builder import PromptBuilder

# 命令解析器：把 LLM 输出的 DSL 文本解析为 Command 对象列表。
from app.ics_ai.dialogue_understanding.generator.command_parser import CommandParser

# 无法处理命令：无用户消息/解析为空时的兜底命令（静默降级，不报错）。
from app.ics_ai.dialogue_understanding.commands.answer_commands import (
    CannotHandleCommand,
)

# 错误命令族：InternalErrorCommand 承载 LLM 异常（内部错误），
# ErrorCommand 承载其余意外异常（通用错误），交由下游统一执行。
from app.ics_ai.dialogue_understanding.commands.error_commands import (
    ErrorCommand,
    InternalErrorCommand,
)

# LLM 客户端工厂：按 type（openai/qwen/azure/anthropic）创建对应实现，
# 其余参数（model/api_key/api_base 等）经 **kwargs 透传给具体客户端。
from app.ics_ai.shared.llm import create_llm_client

# LLM 客户端协议：统一各后端的异步补全接口（complete/complete_sync）。
from app.ics_ai.shared.llm.base_client import LLMClient

# LLM 业务异常：LLM 调用失败（连接/超时/响应解析等）的异常基类，
# generate 中单独分级捕获并映射为 InternalErrorCommand。
from app.ics_ai.shared.exceptions import LLMException

# 类型检查期专属导入：这些类型只出现在方法注解中，运行时不导入
# （避免与 core 包形成循环依赖或引入不必要的运行时开销）。
if TYPE_CHECKING:
    # 对话状态追踪器：generate 的状态来源（用户消息/历史/活动 Flow）。
    from app.ics_ai.core.tracker import DialogueStateTracker

    # 业务域定义：generate 的可选上下文来源（槽位元信息等）。
    from app.ics_ai.core.domain import Domain

# 模块级日志记录器：日志名即本模块完整导入路径。
logger = logging.getLogger(__name__)


@dataclass
class LLMGeneratorConfig(GeneratorConfig):
    """LLM生成器配置。

    职责：承载 LLM 命令生成器的全部可调参数——LLM 后端类型与模型名、
          鉴权与接口地址、采样与生成长度、超时与深度思考开关；
          并继承 GeneratorConfig 的提示词裁剪开关（max_history_turns/
          include_slots/include_flows）。
    设计模式：数据类 + 继承（配置参数对象，避免构造函数散装传参）。
    生命周期：随生成器构造创建并长期只读复用。

    Attributes:
        type: LLM类型 (openai, qwen, azure, anthropic)
        model: 模型名称
        api_key: API密钥
        api_base: 自定义API基础URL（用于vLLM等）
        temperature: 采样温度
        max_tokens: 最大生成token数
        timeout: 超时时间（秒）
        enable_thinking: 启用深度思考模式
    """

    # LLM 后端类型：决定 create_llm_client 创建哪种客户端实现
    # （openai/qwen/azure/anthropic，openai 兼容 vLLM 等自建服务）。
    type: str = "openai"
    # 模型名称：传给 LLM 服务的具体模型标识。
    model: str = "gpt-4o-mini"
    # API 密钥：鉴权凭证；None 时由客户端实现回退到环境变量等默认方式。
    api_key: str | None = None
    # 自定义 API 基础 URL：指向 vLLM 等自建/代理服务；None 用官方端点。
    api_base: str | None = None
    # 采样温度：0 表示贪心解码（确定性输出），利于命令 DSL 稳定可解析。
    temperature: float = 0.0
    # 单次生成最大 token 数：命令输出较短，256 足够。
    max_tokens: int = 256
    # 请求超时时间：单位秒，超时抛 LLM 超时异常（generate 中分级捕获）。
    timeout: float = 30.0
    # 是否启用深度思考模式：仅对支持该特性的模型生效。
    enable_thinking: bool = False


class LLMCommandGenerator(CommandGenerator):
    """LLM命令生成器。

    职责：使用大语言模型将用户输入转换为对话系统命令——编排
          "提示词构建 → LLM 调用 → 结果解析 → 兜底与错误映射"
          四个环节，各环节分别委托给 PromptBuilder/LLMClient/
          CommandParser 专职协作对象，最终产出 GenerationResult。
    设计模式：模板方法（generate 固定四步流程，细节委托协作对象）
              + 懒加载（llm_client 延迟创建）+ 依赖注入（客户端/
              构建器/解析器均可注入替换，便于测试）。
    生命周期：随 Agent/生成器工厂创建并跨多轮消息长期复用；
              LLM 客户端首次访问时创建并缓存于实例。

    工作流程：
    1. 构建Prompt（包含上下文、槽位、Flow信息）
    2. 调用LLM生成命令文本
    3. 解析命令文本为命令对象
    """

    def __init__(
        self,
        config: LLMGeneratorConfig | None = None,
        llm_client: LLMClient | None = None,
        prompt_builder: PromptBuilder | None = None,
        command_parser: CommandParser | None = None,
    ):
        """初始化LLM命令生成器。

        业务流程：1. 配置缺省时创建默认 LLMGeneratorConfig（覆盖基类
                  __init__，配置类型收窄为 LLMGeneratorConfig）；2. LLM
                  客户端注入则直接使用，否则置 None 延迟创建；3. 提示词
                  构建器缺省时按配置裁剪开关创建；4. 命令解析器缺省时
                  创建默认实例。

        副作用：创建 config/_llm_client/prompt_builder/command_parser
                四个实例属性；不建立网络连接（客户端懒加载）。

        参数：
            config: 生成器配置，允许 None（用默认配置）。
            llm_client: LLM 客户端，允许 None（延迟创建）；注入常用于
                        测试时替换桩客户端。
            prompt_builder: 提示词构建器，允许 None（按 config 创建）。
            command_parser: 命令解析器，允许 None（默认创建）。
        """
        # 配置兜底：未显式传入时使用默认配置（openai + gpt-4o-mini 等）。
        self.config = config or LLMGeneratorConfig()

        # 初始化LLM客户端：外部注入优先；否则置 None 走懒加载，
        # 避免构造期就读取密钥/初始化网络客户端。
        if llm_client:
            self._llm_client = llm_client
        else:
            self._llm_client = None  # 延迟初始化

        # 初始化Prompt构建器：未注入时按配置中的历史轮数与信息开关创建，
        # 保证提示词裁剪行为与生成器配置一致。
        self.prompt_builder = prompt_builder or PromptBuilder(
            max_history_turns=self.config.max_history_turns,
            include_slots=self.config.include_slots,
            include_flows=self.config.include_flows,
        )

        # 初始化命令解析器：未注入时创建默认实例（DSL 正则解析）。
        self.command_parser = command_parser or CommandParser()

    @property
    def llm_client(self) -> LLMClient:
        """获取LLM客户端（延迟初始化）。

        业务流程：内部客户端为 None 时按配置经 create_llm_client 工厂
                  创建并缓存；之后每次访问直接返回缓存实例。

        副作用：首次访问创建 LLM 客户端实例并写入 self._llm_client；
                客户端实现可能读取环境变量密钥，但此处不发网络请求
                （真正的网络 IO 发生在 complete 调用时）。

        返回：
            LLMClient 实例（进程内复用）。
        """
        # 懒加载：仅在首次访问时创建客户端，之后复用缓存实例。
        if self._llm_client is None:
            self._llm_client = create_llm_client(
                # 后端类型：openai/qwen/azure/anthropic，决定具体实现类。
                type=self.config.type,
                # 以下参数经 **kwargs 透传给具体客户端构造函数。
                model=self.config.model,
                api_key=self.config.api_key,
                api_base=self.config.api_base,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
                enable_thinking=self.config.enable_thinking,
            )
        return self._llm_client

    async def generate(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        flows: list[Any] | None = None,
    ) -> GenerationResult:
        """使用LLM生成命令。

        业务流程：
            0. 无用户消息（空轮次）时直接兜底 CannotHandleCommand
               （reason=no_user_message），避免无效 LLM 调用；
            1. 构建Prompt：prompt_builder.build_messages 产出消息列表，
               格式化后存入 result.prompt 供日志/调试；
            2. 调用LLM：await llm_client.complete 异步获取回复，原始
               输出与模型/用量/时延元数据写入 result；
            3. 解析命令：command_parser.parse 将回复转为 Command 列表，
               解析错误记录进 result.metadata 并告警；
            4. 空结果兜底：未解析出任何命令时返回 CannotHandleCommand
               （reason=parse_failed），保证下游总有命令可执行。
            异常分级：LLMException → InternalErrorCommand；其余异常 →
            ErrorCommand(error_type=generation_error)；两者均写入
            result.metadata["error"]，不向调用方抛出。

        副作用：调用外部 LLM 服务（网络 IO）；写日志；只读 tracker/
                domain/flows，不修改任何外部状态。

        参数：
            tracker: 对话状态追踪器（用户消息/历史/活动 Flow 来源）。
            domain: Domain定义，允许 None。
            flows: 可用的Flow列表，允许 None。

        返回：
            GenerationResult：正常时 commands 为解析出的命令列表（或
            兜底命令），LLM/意外异常时 commands 为错误命令——任何
            情况都返回结果对象，不抛异常。
        """
        # 初始化结果载体：prompt/raw_output/metadata 随流程逐步填充。
        result = GenerationResult()

        # 检查是否有用户消息：空轮次（如仅事件触发）无内容可生成，
        # 直接兜底返回无法处理命令。
        if not tracker.latest_message:
            logger.warning("No user message to process")
            result.commands = [CannotHandleCommand(reason="no_user_message")]
            return result

        try:
            # 1. 构建Prompt：组装上下文/槽位/Flow 信息为消息列表。
            messages = self.prompt_builder.build_messages(tracker, domain, flows)  # pyright: ignore[reportArgumentType]
            # 将提示词格式化为可读文本存入结果，便于日志与调试回放。
            result.prompt = self._format_messages_for_log(messages)

            logger.debug(f"Generated prompt with {len(messages)} messages")

            # 2. 调用LLM：异步请求，温度/生成长度取自配置。
            llm_response = await self.llm_client.complete(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            # 记录原始输出：后续解析的输入，也是排查问题的第一现场。
            result.raw_output = llm_response.content
            # 记录 LLM 元数据：模型名、token 用量、时延，便于监控与计费分析。
            result.metadata["llm_response"] = {
                "model": llm_response.model,
                "usage": llm_response.usage,
                "latency": llm_response.latency,
            }

            logger.info(f"LLM response: {llm_response.content}")

            # 3. 解析命令：把 LLM 输出的 DSL 文本转换为 Command 对象列表。
            parse_result = self.command_parser.parse(llm_response.content)
            result.commands = parse_result.commands

            if parse_result.errors:
                # 解析存在错误：记录明细供上游排查（成功部分仍会执行）。
                result.metadata["parse_errors"] = parse_result.errors
                logger.warning(f"Parse errors: {parse_result.errors}")

            # 4. 如果没有解析出命令，返回cannot_handle：LLM 输出为空或
            # 全部解析失败时兜底，保证命令流不中断。
            if not result.commands:
                logger.warning("No commands parsed from LLM output")
                result.commands = [CannotHandleCommand(reason="parse_failed")]

        except LLMException as e:
            # LLM 专用异常（连接/超时/鉴权/响应解析等）：映射为内部错误
            # 命令，保持"永远返回结果"的生成契约。
            logger.error(f"LLM error: {e}")
            result.commands = [
                InternalErrorCommand(
                    exception_type=e.__class__.__name__,
                    exception_message=str(e),
                )
            ]
            result.metadata["error"] = str(e)

        except Exception as e:
            # 意外异常（构建提示词/解析等环节的非 LLM 错误）：映射为
            # 通用错误命令，同样不向调用方抛出。
            logger.error(f"Unexpected error in command generation: {e}")
            result.commands = [
                ErrorCommand(
                    error_type="generation_error",
                    message=str(e),
                )
            ]
            result.metadata["error"] = str(e)

        return result

    def _format_messages_for_log(self, messages: list[dict[str, str]]) -> str:
        """格式化消息用于日志。

        业务流程：逐条取角色与内容，超过 500 字符的内容截断并追加
                  省略号，最后按行拼接为多行文本。

        副作用：无（纯内存格式化）。

        参数：
            messages: 消息列表，每项含 role/content 键（键缺失按
                      "unknown"/空串兜底）。

        返回：
            形如 "[role]: content" 的多行拼接字符串。
        """
        # 逐条格式化结果的收集列表。
        parts = []
        for msg in messages:
            # 取消息角色：键缺失时以 unknown 标记（防御畸形消息）。
            role = msg.get("role", "unknown")
            # 取消息内容：键缺失时按空串处理。
            content = msg.get("content", "")
            # 截断长内容：提示词可能很长，日志中截断到 500 字符
            # 兼顾可读性与日志体积。
            if len(content) > 500:
                content = content[:500] + "..."
            # 组装单条格式并加入收集列表。
            parts.append(f"[{role}]: {content}")
        # 多行拼接：每条消息占一行。
        return "\n".join(parts)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LLMCommandGenerator":
        """从配置字典创建生成器。

        业务流程：按默认值兜底读取配置字典各键构造 LLMGeneratorConfig，
                  再以其构造生成器实例（LLM 客户端延迟创建）。

        副作用：无（纯构造，不读文件/环境变量、不发网络请求）。

        参数：
            config: 配置字典，键与 LLMGeneratorConfig 字段一一对应——
                    type/model/api_key/api_base/temperature/max_tokens/
                    timeout/enable_thinking/max_history_turns/
                    include_slots/include_flows，全部键允许缺失（按默认值兜底）。

        返回：
            LLMCommandGenerator 实例。
        """
        # 逐键读取配置：缺失时按与 LLMGeneratorConfig 字段一致的默认值兜底；
        # 键名与 LLMGeneratorConfig 字段一一对应（type/model/api_key/...）。
        generator_config = LLMGeneratorConfig(
            type=config.get("type", "openai"),
            model=config.get("model", "gpt-4o-mini"),
            api_key=config.get("api_key"),
            api_base=config.get("api_base"),
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 256),
            timeout=config.get("timeout", 30.0),
            enable_thinking=config.get("enable_thinking", False),
            max_history_turns=config.get("max_history_turns", 5),
            include_slots=config.get("include_slots", True),
            include_flows=config.get("include_flows", True),
        )
        # 以配置构造生成器：llm_client 未提供，延迟到首次访问时创建。
        return cls(config=generator_config)


# 导出：模块对外暴露的公共 API（生成器类与配置类）。
__all__ = [
    "LLMCommandGenerator",
    "LLMGeneratorConfig",
]
