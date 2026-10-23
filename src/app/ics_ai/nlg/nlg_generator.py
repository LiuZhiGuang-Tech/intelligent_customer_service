"""
nlg_generator - NLG 生成器基类模块

核心功能：定义自然语言生成（NLG）的抽象契约——NLGGenerator(ABC)
          以 generate 为抽象方法（按 utter 响应动作名 + tracker 上下文
          产出 NLGResponse，并附 generate_sync 同步适配）、NLGConfig
          配置 DTO（LLM 开关与选型）、NLGResponse 响应 DTO（文本/
          按钮/图片/自定义数据/元数据 + as_dict 条件序列化）。

执行流程：
    1. 契约期：TemplateNLG（模板渲染）与 ResponseRephraser（LLM
       重述，agent.py load 装配的 nlg_generator 即此组合）实现
       generate 抽象方法；
    2. 调用期：动作执行层（agent.py 的响应回退链）以 utter_action
       （如 utter_greet）+ tracker（槽位填充上下文）调用 generate，
       得到 NLGResponse 后经 as_dict 转为消息字典进入响应列表；
    3. 同步场景：generate_sync 经事件循环阻塞驱动异步 generate
       （与 agent.handle_message_sync 同构的适配模式）。

副作用：本模块定义期无副作用（纯类型与抽象接口）；generate_sync
        在当前线程无事件循环时会新建并安装循环（运行期副作用，
        发生在子类实例调用时）。

主要依赖：
    - 标准库：abc（抽象基类）、dataclasses（DTO）、typing；
    - 类型检查期：app.ics_ai.core.tracker.DialogueStateTracker 与
      app.ics_ai.core.domain.Domain（仅注解用，TYPE_CHECKING 隔离）。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，TYPE_CHECKING
# 分支中的类型前向引用无需引号即可书写，PEP 604 联合类型注解
# （X | None）不受运行时版本限制。
from __future__ import annotations

# 导入抽象基类工具：ABC 基类与 abstractmethod 装饰器（generate 的
# 契约强制）。
from abc import ABC, abstractmethod

# 导入 dataclass 工具：两个 DTO 类与可变默认值工厂。
from dataclasses import dataclass, field

# 导入类型注解工具：Any 类型通配符 + TYPE_CHECKING 隔离仅注解导入
# （PEP 585/604 新式注解下 Dict/List/Optional 已不需要）。
from typing import Any, TYPE_CHECKING

# 类型检查期专属导入：这些类型只出现在注解中，运行时不导入
# （避免 NLG 包与 core 包在导入期相互依赖）。
if TYPE_CHECKING:
    # 对话状态追踪器：槽位填充上下文的来源（仅注解用）。
    from app.ics_ai.core.tracker import DialogueStateTracker

    # 业务域定义：响应模板的静态来源（仅注解用）。
    from app.ics_ai.core.domain import Domain


@dataclass
class NLGConfig:
    """NLG 配置。

    职责：承载 NLG 生成器的可调参数——是否用 LLM 生成、LLM 类型
          与模型名；TemplateNLG 等模板实现忽略 use_llm，LLM 重述
          实现据此选型。
    设计模式：数据传输对象（dataclass 参数对象）。
    生命周期：未显式传入时由 NLGGenerator.__init__ 兜底创建默认
              实例，随生成器长期只读持有。

    Attributes:
        use_llm: 是否使用 LLM 生成（默认 False——纯模板模式，
                 不依赖外部 API）。
        llm_type: LLM 类型（openai/qwen/azure/anthropic 等提供商
                  标识，与 shared.llm 的工厂约定对齐）。
        llm_model: LLM 模型名（提供商侧模型标识，默认 gpt-4o-mini）。
    """

    # LLM 生成开关：False 时子类应走纯模板/规则路径。
    use_llm: bool = False
    # LLM 类型：提供商标识（经 create_llm_client 工厂解析）。
    llm_type: str = "openai"
    # LLM 模型名：具体模型标识。
    llm_model: str = "qwen-plus"


@dataclass
class NLGResponse:
    """NLG 响应 DTO。

    职责：承载一条机器人回复的结构化产出——文本、按钮、图片、
          自定义数据与元数据；as_dict 把它转为动作响应层的标准
          消息字典（text 必含，其余键仅在非空时携带）。
    设计模式：数据传输对象（dataclass，可变默认值经 default_factory
              保证实例隔离）。
    生命周期：generate 实现内构造并随返回值交付调用方；as_dict
              产物直接进入 MessageResponse.messages / 动作响应。

    Attributes:
        text: 回复文本（生成结果的核心字段）。
        buttons: 按钮列表（每项为 {"title"/"payload"...} 结构的
                 字典，交互组件描述）。
        image: 图片 URL（允许 None——纯文本回复时缺省）。
        custom: 自定义数据（渠道特定的结构化附加内容，允许 None）。
        metadata: 元数据（生成来源/耗时等过程信息，默认空字典）。
    """

    # 回复文本：空串为缺省（子类生成后填充）。
    text: str = ""
    # 按钮列表：可变默认值经工厂创建（避免跨实例共享）。
    buttons: list[dict[str, Any]] = field(default_factory=list)
    # 图片 URL：None 表示无图片。
    image: str | None = None
    # 自定义数据：None 表示无附加结构化内容。
    custom: dict[str, Any] | None = None
    # 元数据：默认空字典（as_dict 对空字典不输出该键）。
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """转换为消息字典（条件序列化）。

        业务流程：text 恒输出 → buttons/image/custom/metadata 四键
                  仅在非空/非 None 时输出——保证下游响应字典不携带
                  空键（与动作响应层的最小载荷约定一致）。

        副作用：无（纯构造，不修改任何状态）。

        返回：
            消息字典（{"text": ...} 基底 + 非空附加键；buttons/
            image/custom/metadata 按内容取舍）。
        """
        # text 是必备键：恒输出（哪怕空串——下游按 text 键取文本）。
        # 显式注解 dict[str, Any]：字面量会被推断为 dict[str, str]，
        # 导致后续 list/dict 值赋值报 reportArgumentType（4 处一注解消除）。
        result: dict[str, Any] = {"text": self.text}
        # 按钮列表非空才输出（空列表对渠道无意义）。
        if self.buttons:
            result["buttons"] = self.buttons
        # 图片非 None 才输出。
        if self.image:
            result["image"] = self.image
        # 自定义数据非 None 才输出。
        if self.custom:
            result["custom"] = self.custom
        # 元数据非空字典才输出。
        if self.metadata:
            result["metadata"] = self.metadata
        # 返回条件序列化后的消息字典。
        return result


class NLGGenerator(ABC):
    """NLG 生成器抽象基类。

    职责：定义回复生成的统一契约——按 utter 响应动作名（如
          utter_greet）与 tracker 上下文产出 NLGResponse；子类
          实现具体生成策略（TemplateNLG 模板渲染 / ResponseRephraser
          LLM 重述）。
    设计模式：模板方法 + 策略——基类持有配置并提供 generate_sync
              同步适配，生成策略由子类注入。
    生命周期：由装配方（agent.py load / TemplateNLG 构造）创建，
              随 Agent 长期持有复用；无状态（上下文经参数传入）。
    """

    def __init__(self, config: NLGConfig | None = None):
        """初始化生成器。

        业务流程：持有配置——未传入时以默认 NLGConfig 兜底
                  （纯模板模式：use_llm=False）。

        副作用：无（纯赋值）。

        参数：
            config: NLG 配置，允许 None（默认 NLGConfig）。
        """
        # 配置兜底：默认走纯模板路径（不依赖 LLM）。
        self.config = config or NLGConfig()

    @abstractmethod
    async def generate(
        self,
        utter_action: str,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        **kwargs: Any,
    ) -> NLGResponse:
        """生成回复（抽象方法，子类必须实现）。

        业务流程（契约约定）：子类按 utter_action 从自身生成源
            （domain 响应模板 / LLM 重述链）取文本 → 经 tracker
            填充槽位变量 → 组装 NLGResponse 返回。

        副作用：由子类定义（LLM 实现会调用外部 API）。

        参数：
            utter_action: 响应动作名称（如 utter_greet——与 Domain
                          responses 模板键对齐）。
            tracker: 对话状态追踪器（槽位填充上下文，必传）。
            domain: 业务域定义（模板来源），允许 None（子类自行
                    兜底或仅按 utter_action 查找）。
            **kwargs: 额外参数（重述风格/语言等，由子类约定）。

        返回：
            NLGResponse——结构化回复产出。

        异常：子类未实现时经 ABC 实例化检查在构造期即报
              TypeError；运行期失败行为由子类定义。
        """
        # 抽象占位：ABC + abstractmethod 保证子类必须覆写。
        raise NotImplementedError()

    def generate_sync(
        self,
        utter_action: str,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        **kwargs: Any,
    ) -> NLGResponse:
        """同步版本的生成方法。

        业务流程：获取/复用当前线程事件循环 → run_until_complete
                  阻塞驱动异步 generate。

        副作用：事件循环缺失时新建并安装到当前线程；驱动的
                generate 具有其全部副作用。

        参数：同 generate（utter_action/tracker/domain/**kwargs）。

        返回：
            NLGResponse——同 generate。

        说明：仅适合同步上下文调用；已有运行中循环时会抛
              RuntimeError（异步框架内请直接用 generate）。
        """
        # 局部导入：仅同步适配路径需要 asyncio（延迟加载）。
        import asyncio

        # 优先复用当前线程已有的事件循环。
        try:
            loop = asyncio.get_event_loop()
        # 当前线程无循环 → 新建并安装（供本次及后续复用）。
        except RuntimeError:
            # 新建事件循环。
            loop = asyncio.new_event_loop()
            # 安装为当前线程循环。
            asyncio.set_event_loop(loop)

        # 阻塞驱动异步 generate 至完成并返回结果。
        return loop.run_until_complete(
            self.generate(utter_action, tracker, domain, **kwargs)
        )


# 导出：模块对外暴露的公共 API（抽象基类与两个 DTO）。
__all__ = [
    "NLGGenerator",
    "NLGConfig",
    "NLGResponse",
]
