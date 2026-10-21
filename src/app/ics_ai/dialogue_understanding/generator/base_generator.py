"""
base_generator - 命令生成器抽象基类模块

核心功能：定义对话理解阶段"命令生成"的抽象契约——
          CommandGenerator 抽象基类约定 generate 异步生成入口与
          generate_sync 同步桥接；GeneratorConfig 汇总生成器行为
          开关（历史轮数/槽位与 Flow 注入/采样温度）；GenerationResult
          作为结果信封统一承载命令列表、LLM 原始输出、提示词与元数据。

执行流程：
    1. 装配期：子类（如 LLMCommandGenerator）继承 CommandGenerator
       并实现 generate，构造时可选传入 GeneratorConfig（缺省取
       全默认值）；
    2. 生成期：调用方每轮消息调用 generate（异步上下文）或
       generate_sync（同步上下文）→ 子类渲染提示词并请求 LLM →
       解析输出得到 Command 列表 → 封装为 GenerationResult 返回；
    3. 消费期：调用方经 success/first_command 派生属性判定结果并
       将命令交给 CommandProcessor 执行。

副作用：
    - generate_sync 会获取或新建事件循环并阻塞至异步生成完成
      （仅适用于无运行中循环的同步上下文）；
    - 本模块自身无 IO、不调用网络（网络请求由子类实现）。

主要依赖：
    - abc：ABC + abstractmethod 定义生成器契约；
    - dataclasses：@dataclass + field(default_factory=...) 定义配置
      与结果两个数据类；
    - typing：Any + TYPE_CHECKING（注解已按 PEP 585/604 新式写法）；
    - asyncio：generate_sync 内局部导入，用于同步桥接异步生成。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，TYPE_CHECKING 分支
# 中的 tracker/domain/Command 前向引用无需引号即可书写。
from __future__ import annotations

# 导入抽象基类工具：ABC 标记生成器基类，abstractmethod 强制子类
# 实现 generate 异步生成方法。
from abc import ABC, abstractmethod

# 导入 dataclass 装饰器与 field：数据类定义；field(default_factory=...)
# 为可变默认值（命令列表/元数据字典）指定每次实例化时的工厂函数，
# 避免可变默认值在实例间共享。
from dataclasses import dataclass, field

# 导入类型注解工具：Any 类型通配符；TYPE_CHECKING 隔离仅注解导入
# （PEP 585/604 新式注解下 Dict/List/Optional 已不需要）。
from typing import Any, TYPE_CHECKING

# 类型检查期专属导入：这些类型只出现在方法注解中，运行时不导入
# （避免与 core/dialogue_understanding 包形成循环依赖）。
if TYPE_CHECKING:
    # 对话状态追踪器：生成命令所需的对话状态来源。
    from app.ics_ai.core.tracker import DialogueStateTracker

    # 业务域定义：提供槽位元信息等静态业务数据。
    from app.ics_ai.core.domain import Domain

    # 命令基类：生成结果 commands 列表的元素类型。
    from app.ics_ai.dialogue_understanding.commands.base import Command


@dataclass
class GeneratorConfig:
    """生成器配置。

    职责：集中管理命令生成器的行为开关——上下文裁剪（历史轮数/
          是否注入槽位与 Flow）与 LLM 采样参数（温度）。
    设计模式：数据类配置对象（参数对象模式）——避免生成器构造
              函数携带一长串散装参数。
    生命周期：随生成器创建并长期持有；缺省构造全默认值。

    Attributes:
        max_history_turns: 发送给LLM的最大历史轮数
        include_slots: 是否包含槽位信息
        include_flows: 是否包含Flow信息
        temperature: LLM采样温度
    """

    # 最大历史轮数：控制对话历史上下文长度（轮数越多提示词越长）。
    max_history_turns: int = 5
    # 是否向提示词注入槽位元信息：关闭可缩短提示词。
    include_slots: bool = True
    # 是否向提示词注入可用 Flow 清单：关闭可缩短提示词。
    include_flows: bool = True
    # LLM 采样温度：0.0 表示确定性输出（命令生成需要稳定可解析）。
    temperature: float = 0.0


@dataclass
class GenerationResult:
    """生成结果。

    职责：作为命令生成的统一结果信封（DTO）——同时携带结构化
          命令列表与生成过程的原始痕迹（LLM 原始输出/发送的提示词），
          便于上层执行与下游调试排障。
    设计模式：数据传输对象（DTO）+ 派生属性——success 与
              first_command 由 commands 实时派生，不重复存储。
    生命周期：随单轮 generate 调用创建，由调用方消费后丢弃。

    Attributes:
        commands: 生成的命令列表
        raw_output: LLM的原始输出
        prompt: 发送给LLM的提示词
        metadata: 额外的元数据
    """

    # 生成的命令列表：default_factory 每次实例化新建列表，避免共享。
    commands: list["Command"] = field(default_factory=list)
    # LLM 原始输出：未解析的原始文本（调试与回放用）。
    raw_output: str = ""
    # 发送给 LLM 的提示词：渲染后的完整提示词（调试与回放用）。
    prompt: str = ""
    # 额外元数据：如 token 用量、解析失败原因等自由扩展字段。
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """是否成功生成了命令。

        返回：
            commands 非空时返回 True；空列表（含解析失败/静默拒绝）
            返回 False。
        """
        # 派生判定：命令列表非空即成功。
        return len(self.commands) > 0

    @property
    def first_command(self) -> Command | None:
        """返回第一个命令。

        返回：
            commands 的首元素；命令列表为空时返回 None（调用方需
            处理空值）。
        """
        # 空列表保护：避免 IndexError，缺省返回 None。
        return self.commands[0] if self.commands else None


class CommandGenerator(ABC):
    """命令生成器抽象基类。

    职责：定义"用户输入 → 可执行命令"的生成契约——generate 为
          异步生成抽象入口（子类按策略实现：LLM/规则/NLU 等），
          generate_sync 提供同步桥接，name 提供生成器标识。
    设计模式：策略模式 + 模板方法——generate 由子类实现不同生成
              策略；generate_sync 以固定流程包裹抽象方法完成
              同/异步桥接（不随子类变化）。
    生命周期：随 Agent/生成器工厂创建并长期复用，跨多轮消息。

    命令生成器的职责边界：
        - 生成器只负责"产出命令"（解析 LLM 输出为 Command 对象）；
        - 命令的执行由 CommandProcessor 完成，二者解耦。
    """

    def __init__(self, config: GeneratorConfig | None = None):
        """初始化生成器。

        业务流程：持有配置对象；未传入时以全默认配置兜底。

        副作用：创建 self.config 实例属性。

        参数：
            config: 生成器配置，允许 None（None 时自动创建
                    GeneratorConfig() 默认配置）。
        """
        # 配置兜底：未显式配置时使用默认值（温度 0.0、历史 5 轮等）。
        self.config = config or GeneratorConfig()

    @abstractmethod
    async def generate(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        flows: list[Any] | None = None,
    ) -> GenerationResult:
        """生成命令。

        业务流程：由子类实现——组装提示词、请求 LLM（或其他策略）、
                  解析输出为 Command 对象并封装为 GenerationResult。

        副作用：由子类决定（LLM 实现会调用外部 API）。

        参数：
            tracker: 对话状态追踪器（用户消息/活动 Flow/槽位/历史来源）。
            domain: 业务域定义，允许 None。
            flows: 可用的 Flow 列表，允许 None。

        返回：
            生成结果信封（含命令列表与原始输出痕迹）；
            子类未实现时抛 NotImplementedError。
        """
        # 兜底保护：子类未覆盖时抛出明确错误。
        raise NotImplementedError()

    def generate_sync(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        flows: list[Any] | None = None,
    ) -> GenerationResult:
        """同步版本的generate方法。

        业务流程：1. 局部导入 asyncio；2. 获取当前事件循环——获取
                  失败（无运行中/已注册循环，如独立线程）时新建并
                  设为当前；3. run_until_complete 阻塞执行异步
                  generate 直至完成。

        副作用：获取/新建事件循环并阻塞当前线程至生成完成。

        参数：
            tracker: 对话状态追踪器。
            domain: 业务域定义，允许 None。
            flows: 可用的 Flow 列表，允许 None。

        返回：
            与 generate 相同的生成结果。

        异常：
            RuntimeError：当前线程已有运行中的事件循环时，
            run_until_complete 不允许嵌套调用而抛出——本方法仅适用
            于纯同步调用上下文（异步上下文请直接 await generate）。
        """
        # 局部导入 asyncio：仅同步桥接路径需要，保持模块顶层轻量。
        import asyncio

        # 优先复用当前事件循环（主线程已注册或运行中的循环）。
        try:
            loop = asyncio.get_event_loop()
        # 无可用循环（如工作线程首次调用）：新建并注册为当前循环。
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # 阻塞运行异步生成：直至 generate 协程完成并返回结果。
        return loop.run_until_complete(self.generate(tracker, domain, flows))

    @property
    def name(self) -> str:
        """返回生成器名称。

        返回：
            生成器类名（如 "LLMCommandGenerator"），用于日志与
            元数据标识。
        """
        # 以类名为标识：无需子类额外维护名称字段。
        return self.__class__.__name__


# 导出
__all__ = [
    # 命令生成器抽象基类。
    "CommandGenerator",
    # 生成器配置。
    "GeneratorConfig",
    # 生成结果信封。
    "GenerationResult",
]
