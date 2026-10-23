"""
response_rephraser - LLM 响应重述器模块

核心功能：NLGGenerator 的 LLM 改写路径实现——对底层 NLG 生成器
          （如 TemplateNLG）产出的响应做条件触发的 LLM 重述，使
          回复更自然（风格可选友好/专业/随意/共情），带重述结果
          验证（长度倍率 + 槽位占位符保持）与多层降级。

执行流程：
    1. 装配期：agent.py 在 NLG 配置启用 LLM 时创建 ResponseRephraser
       （包装 TemplateNLG 作 base_generator，LLM 客户端可注入或
       延迟初始化）；
    2. 生成期：动作层调用 generate → 底层生成器产出原始响应 →
       _should_rephrase 三条件判定（开关/metadata 标记/长度阈值）
       → 触发则 rephrase（构建上下文 + 提示词 + LLM 调用 + 验证）
       → 成功打 rephrased 标记，失败逐层回退原文。

副作用：
    - LLM 网络请求：rephrase 经 llm_client 发起（首次访问经
      create_llm_client 工厂延迟创建，构造期不发请求）；
    - 无其他副作用（不改 tracker/Domain，验证失败与异常均降级
      返回原文）。

已知问题（2026-10-17 冒烟实证）：
    - rephrase 的 context 形参从未被函数体消费（死参数，保留以
      兼容调用方签名）。

已修复缺陷（2026-10-17，冒烟实证后修复）：
    - rephrase 曾调用不存在的 llm_client.chat → 已改为契约方法
      complete（重述功能由静默失效恢复为真正生效）；
    - _build_context 曾对 latest_message（UserMessage dataclass）
      误用 dict 的 .get → 已改为 .text 属性读取。

主要依赖：
    - app.ics_ai.nlg.nlg_generator：NLGGenerator 基类与 DTO；
    - app.ics_ai.shared.llm：create_llm_client 工厂与 LLMClient
      基类（重述的 LLM 能力来源）；
    - app.ics_ai.core.tracker / core.domain：仅注解用
      （TYPE_CHECKING 隔离）。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，TYPE_CHECKING
# 分支中的类型前向引用无需引号，PEP 604 联合类型注解不受运行时
# 版本限制。
from __future__ import annotations

# 导入日志器工厂：降级路径与重述过程的运行时留痕。
import logging

# 导入正则引擎：_validate_rephrase 中槽位占位符集合的提取比对。
import re

# 导入 dataclass 工具：RephraserConfig 配置类的自动方法生成。
from dataclasses import dataclass

# 导入类型注解工具：Any 类型通配符（PEP 585/604 新式注解下
# Dict/List/Optional 已不需要）。
from typing import Any, TYPE_CHECKING

# 从 NLG 基类模块导入：抽象基类（本类是其 LLM 改写路径实现）与
# 响应/配置 DTO。
from app.ics_ai.nlg.nlg_generator import NLGGenerator, NLGConfig, NLGResponse

# 从统一 LLM 包导入：客户端工厂（延迟初始化用）与基类类型
# （llm_client 注入参数的类型注解）。
from app.ics_ai.shared.llm import create_llm_client
from app.ics_ai.shared.llm.base_client import LLMClient

# 类型检查期专属导入：这些类型只出现在方法注解中，运行时不导入
# （避免 NLG 包与 core 包在导入期相互依赖）。
if TYPE_CHECKING:
    # 对话状态追踪器：重述上下文的数据源（仅注解用）。
    from app.ics_ai.core.tracker import DialogueStateTracker

    # 业务域定义：透传底层生成器用（仅注解用）。
    from app.ics_ai.core.domain import Domain

# 模块级日志器：遵循命名约定供框架统一配置。
logger = logging.getLogger(__name__)


@dataclass
class RephraserConfig:
    """重述器配置。

    职责：承载 LLM 重述的行为参数——开关、模型选型、触发阈值、
          风格与语言、槽位保持约定（数据传输对象，纯存储）。
    设计模式：DTO——@dataclass 自动生成 __init__/__repr__。
    生命周期：ResponseRephraser 构造时创建（未传则用默认值），
              随生成器实例跨轮复用。

    Attributes:
        enabled: 重述总开关（False 时 generate 直接透传底层响应）。
        llm_type: LLM 类型（openai/qwen/azure/anthropic，经工厂解析）。
        llm_model: LLM 模型名。
        temperature: 生成温度（重述允许更高随机性，默认 0.7）。
        rephrase_threshold: 触发重述的最小文本长度（字符数，
                            默认 10——短回复不浪费 LLM 调用）。
        preserve_slots: 重述时是否保留 {xxx} 槽位占位符（True 时
                        验证环节比对占位符集合一致性）。
        style: 重述风格（friendly/professional/casual/empathetic，
               未知值在提示词中原样透传）。
        language: 目标语言（提示词中声明，默认中文）。
    """

    # 重述总开关：False 时跳过全部 LLM 调用。
    enabled: bool = True
    # LLM 类型：后端提供商标识。
    llm_type: str = "openai"
    # LLM 模型名：具体模型标识。
    llm_model: str = "qwen-plus"
    # 生成温度：重述属生成类任务，默认 0.7 高于命令生成。
    temperature: float = 0.7
    # 触发阈值：文本长度（字符数）达到该值才触发重述。
    rephrase_threshold: int = 10
    # 槽位保持开关：True 时占位符不参与改写并参与验证。
    preserve_slots: bool = True
    # 重述风格：STYLE_DESCRIPTIONS 的键或自由文本。
    style: str = "friendly"
    # 目标语言：提示词中的语言声明。
    language: str = "zh"


# 重述提示词模板：五项固定要求 + 槽位保持指令插值位（slot_instruction
# 按 preserve_slots 动态拼接第 6 条）。
REPHRASE_PROMPT_TEMPLATE = """请将以下机器人回复重述得更加自然和{style}。

原始回复：
{original_text}

重述要求：
1. 保持原意不变
2. 使语言更加自然流畅
3. 采用{style}的语气
4. 使用{language}
5. 保持简洁，不要过度扩展
{slot_instruction}

请直接输出重述后的文本，不要包含任何解释：
"""

# 风格描述映射：配置风格名 → 提示词中的中文风格短语
# （未收录的风格名经 STYLE_DESCRIPTIONS.get(style, style) 原样透传）。
STYLE_DESCRIPTIONS = {
    "friendly": "友好、亲切",
    "professional": "专业、正式",
    "casual": "轻松、随意",
    "empathetic": "富有同理心、温暖",
}


class ResponseRephraser(NLGGenerator):
    """响应重述器。

    职责：对底层 NLG 生成器的响应做条件触发的 LLM 重述——先取
          底层响应，判定需要重述时经提示词调用 LLM 改写文本，
          验证通过后以新文本重组响应（保留按钮/图片等附加字段，
          metadata 追加 rephrased 标记与原始文本）。
    设计模式：装饰器模式——包装另一个 NLGGenerator 增强其输出；
              也支持无包装的独立模式（直接调 rephrase 改写任意
              文本）。LLM 客户端经属性延迟初始化（惰性单例）。
    生命周期：随 agent 装配创建，跨多轮对话复用；LLM 客户端在
              首次重述时创建后缓存于 _llm_client。

    使用示例：
    ```python
    # 装饰器模式（TemplateNLG 无状态，domain 经 generate 调用透传）
    template_nlg = TemplateNLG()
    rephraser = ResponseRephraser(base_generator=template_nlg)
    response = await rephraser.generate("utter_greet", tracker, domain)

    # 独立模式
    rephraser = ResponseRephraser()
    rephrased = await rephraser.rephrase("您好，请问有什么可以帮您？")
    ```
    """

    def __init__(
        self,
        config: RephraserConfig | None = None,
        base_generator: NLGGenerator | None = None,
        llm_client: LLMClient | None = None,
    ):
        """初始化重述器。

        业务流程：调基类无参构造（self.config 用默认 NLGConfig——
                  本类的行为参数由 rephrase_config 承载，基类配置
                  不参与本实现决策）→ 存储重述配置（None 兜底默认）
                  → 存储底层生成器与 LLM 客户端（均可为 None，
                  前者决定 generate 可用性，后者延迟初始化）。

        副作用：无（构造期不发网络请求，LLM 客户端延迟创建）。

        参数：
            config: 重述器配置，允许 None（默认 RephraserConfig()）。
            base_generator: 底层 NLG 生成器，允许 None（此时
                            generate 返回空响应——仅独立模式可用）。
            llm_client: 外部注入的 LLM 客户端，允许 None（首次
                        访问 llm_client 属性时按配置经工厂创建）。
        """
        # 基类构造：不传 config，self.config 为默认 NLGConfig
        # （本实现的生效配置是 self.rephrase_config）。
        super().__init__()
        # 重述配置：未传时兜底默认值。
        self.rephrase_config = config or RephraserConfig()
        # 底层生成器：装饰器模式的目标对象（None 即独立模式）。
        self.base_generator = base_generator
        # LLM 客户端缓存：None 表示尚未创建（见 llm_client 属性）。
        self._llm_client = llm_client

    @property
    def llm_client(self) -> LLMClient:
        """获取 LLM 客户端（延迟初始化）。

        业务流程：已有缓存直接返回；否则按重述配置（类型/模型/
                  温度）经 create_llm_client 工厂创建并缓存——
                  避免构造期创建客户端（未用重述则零开销）。

        副作用：首次访问时创建 LLMClient 实例并写入 _llm_client
                （构造期不发网络请求，建连发生在 complete 调用期）。

        返回：
            LLMClient 实例（缓存后复用）。
        """
        # 惰性单例：首次访问按配置创建，后续直接复用缓存。
        if self._llm_client is None:
            self._llm_client = create_llm_client(
                type=self.rephrase_config.llm_type,
                model=self.rephrase_config.llm_model,
                temperature=self.rephrase_config.temperature,
            )
        return self._llm_client

    async def generate(
        self,
        utter_action: str,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        **kwargs: Any,
    ) -> NLGResponse:
        """生成回复（底层生成 + 条件重述）。

        业务流程：1. 无底层生成器 → 记警告返回空响应（装饰器模式
                  未配置）；2. 委托底层生成器产出原始响应；3.
                  _should_rephrase 判定无需重述 → 原样返回；4. 需要
                  重述则构建上下文并调用 rephrase → 成功时以新文本
                  重组响应（附加字段透传，metadata 合并原值并追加
                  rephrased/original_text）→ rephrase 内部异常已在
                  其自身 except 中降级返回原文（此处仍按成功路径
                  重组）；5. 本层 try 兜底：_build_context 等环节
                  异常时返回原始响应。

        副作用：委托底层生成器（其副作用由其自身定义）；触发重述
                时发起 LLM 网络请求。

        参数：
            utter_action: 响应动作名称（透传底层生成器）。
            tracker: 对话状态追踪器（透传底层生成器 + 构建重述
                     上下文），允许 None（仅当底层生成器不使用时）。
            domain: 业务域定义，允许 None（透传底层生成器）。
            **kwargs: 额外参数（原样透传底层生成器）。

        返回：
            NLGResponse——重述成功带 rephrased=True 标记；任一环节
            失败降级为原始响应或空响应（不抛异常）。
        """
        # 降级一：装饰器模式未配置底层生成器，返回空响应。
        if self.base_generator is None:
            logger.warning("No base generator configured, returning empty response")
            return NLGResponse(text="")

        # 委托底层生成器产出原始响应（模板路径或其它 NLG 实现）。
        original_response = await self.base_generator.generate(
            utter_action, tracker, domain, **kwargs
        )

        # 条件判定：不触发重述则原样返回（开关/阈值未命中）。
        if self._should_rephrase(original_response):
            try:
                # 重述文本：context 用于 LLM 个性化（注意 rephrase
                # 当前实现未消费该参数，见文件头缺陷记录）。
                rephrased_text = await self.rephrase(
                    original_response.text,
                    context=self._build_context(tracker),
                )

                # 以重述文本重组响应：保留原响应的按钮/图片/自定义
                # 三附加字段，metadata 合并原值并追加重述留痕
                # （rephrased 标记 + 原始文本便于审计回溯）。
                return NLGResponse(
                    text=rephrased_text,
                    buttons=original_response.buttons,
                    image=original_response.image,
                    custom=original_response.custom,
                    metadata={
                        **original_response.metadata,
                        "rephrased": True,
                        "original_text": original_response.text,
                    },
                )
            except Exception as e:
                # 降级二：上下文构建/重组环节异常 → 放弃重述返回
                # 原始响应（rephrase 内部异常已在其自身 except 中
                # 降级返回原文，不会进入本分支）。
                logger.warning(f"Rephrase failed, using original: {e}")
                return original_response

        # 未触发重述：原样返回底层响应。
        return original_response

    def _should_rephrase(self, response: NLGResponse) -> bool:
        """判定响应是否需要重述。

        业务流程：1. 总开关关闭 → False；2. 响应 metadata 带
                  rephrase=True 标记 → True（模板/动作层显式指定）；
                  3. 文本长度达到阈值 → True（默认 10 字符）；4.
                  其余 → False。

        副作用：无（纯判定）。

        参数：
            response: 待判定的原始响应。

        返回：
            是否应该重述（三条件依次判定，命中即短路）。
        """
        # 条件一：总开关关闭则一律不重述。
        if not self.rephrase_config.enabled:
            return False

        # 条件二：metadata 显式标记优先（阈值判定之前的强制通道）。
        if response.metadata.get("rephrase", False):
            return True

        # 条件三：文本长度达到阈值才值得消耗一次 LLM 调用
        # （短回复重述收益低）。
        if len(response.text) >= self.rephrase_config.rephrase_threshold:
            return True

        # 三条件均未命中：不重述。
        return False

    async def rephrase(
        self,
        text: str,
        style: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """重述单段文本。

        业务流程：1. 空文本/纯空白直接返回（无重述价值）；2.
                  解析风格描述（参数覆盖配置，未知风格名原样透传
                  进提示词）；3. 按 preserve_slots 拼接槽位保持
                  指令；4. 填充提示词模板；5. 经 llm_client.complete
                  调用 LLM（LLMResponse）；6. 验证重述结果（长度
                  倍率 + 槽位集合），通过则返回新文本，否则记警告
                  返回原文；7. 任何异常 → 记错误返回原文。

        副作用：发起 LLM 网络请求（经 llm_client 属性，首次访问
                触发客户端创建）。

        参数：
            text: 待重述的原始文本。
            style: 重述风格，允许 None（用配置默认）；未收录的
                   风格名原样透传。
            context: 上下文信息，允许 None——当前实现未消费该参数
                     （死参数，保留以兼容调用方签名）。

        返回：
            重述后的文本；验证失败或异常时返回原文（不抛异常）。
        """
        # 空文本短路：空白串无重述价值，原样返回。
        if not text or not text.strip():
            return text

        # 风格解析：调用参数优先于配置默认；映射表未收录的名称
        # 经 get 第二参原样透传（提示词中仍可读）。
        style = style or self.rephrase_config.style
        style_desc = STYLE_DESCRIPTIONS.get(style, style)

        # 槽位保持指令：开关开启时拼接第 6 条要求（提示词的
        # slot_instruction 插值位）。
        slot_instruction = ""
        if self.rephrase_config.preserve_slots:
            slot_instruction = "6. 保留所有{xxx}格式的占位符不变"

        # 填充提示词模板：原文/风格短语/语言/槽位指令四插值位。
        prompt = REPHRASE_PROMPT_TEMPLATE.format(
            original_text=text,
            style=style_desc,
            language=self.rephrase_config.language,
            slot_instruction=slot_instruction,
        )

        # LLM 调用与验证：异常统一降级返回原文。
        try:
            # 组装单轮用户消息（提示词整体作为 content）。
            messages = [{"role": "user", "content": prompt}]
            # LLM 调用：complete 为 LLMClient 契约方法（原代码误写
            # 为不存在的 chat 导致重述静默失效，2026-10-17 修复）；
            # 返回 LLMResponse，文本经 content 属性读取。
            response = await self.llm_client.complete(messages)

            # 取生成文本并去除首尾空白。
            rephrased = response.content.strip()

            # 验证重述结果：通过则采用新文本。
            if self._validate_rephrase(text, rephrased):
                logger.debug(f"Rephrased: '{text[:50]}...' -> '{rephrased[:50]}...'")
                return rephrased
            # 验证失败（过度扩展/缩减/槽位丢失）：回退原文。
            else:
                logger.warning("Rephrase validation failed, using original")
                return text

        except Exception as e:
            # LLM 调用/解析异常：记错误并降级返回原文（不向调用方
            # 传播——重述失败不应影响回复主链路）。
            logger.error(f"LLM rephrase error: {e}")
            return text

    def _validate_rephrase(self, original: str, rephrased: str) -> bool:
        """验证重述结果的有效性。

        业务流程：1. 空结果 → False；2. 长度检查——短于原文 0.3
                  倍（过度缩减，信息丢失）或长于 3 倍（过度扩展，
                  违背简洁要求）均 False；3. preserve_slots 开启时
                  比对原文与重述文本的槽位占位符集合，原文有占位
                  符且集合不一致（占位符被改写/丢失）→ False；
                  4. 其余 → True。

        副作用：无（纯判定）。

        参数：
            original: 原始文本。
            rephrased: LLM 重述后的文本。

        返回：
            是否有效（无效时调用方回退原文）。
        """
        # 基本检查：空结果视为无效。
        if not rephrased or not rephrased.strip():
            return False

        # 长度检查：以倍率区间约束重述幅度（过短丢信息、过长
        # 违背"保持简洁"的提示词要求）。
        original_len = len(original)
        rephrased_len = len(rephrased)

        # 过度缩减：低于原文 0.3 倍。
        if rephrased_len < original_len * 0.3:
            return False
        # 过度扩展：高于原文 3 倍。
        if rephrased_len > original_len * 3:
            return False

        # 槽位保持检查：原文含占位符时，重述文本的占位符集合必须
        # 完全一致（防 LLM 改写或丢弃 {xxx} 占位符）。
        if self.rephrase_config.preserve_slots:
            # 提取两段文本的占位符名集合（\w+ 命名，与模板层一致）。
            original_slots = set(re.findall(r"\{(\w+)\}", original))
            rephrased_slots = set(re.findall(r"\{(\w+)\}", rephrased))

            # 原文有占位符且集合不一致 → 无效（原文无占位符时不
            # 约束，允许 LLM 自由发挥）。
            if original_slots and original_slots != rephrased_slots:
                return False

        # 全部检查通过：结果有效。
        return True

    def _build_context(
        self,
        tracker: DialogueStateTracker,
    ) -> dict[str, Any]:
        """构建重述上下文（个性化改写的依据）。

        业务流程：1. 以 sender_id 为基础字段；2. tracker 存在最新
                  用户消息时补充 last_user_message（取 UserMessage
                  的 text 属性）；3. 存在活跃 Flow 时补充
                  active_flow（计算属性，从对话栈推导）。

        副作用：无（只读 tracker）。

        参数：
            tracker: 对话状态追踪器，允许 None（调用方 generate
                     在独立模式下可能传 None——访问 sender_id
                     即抛 AttributeError，由调用方 except 兜底）。

        返回：
            上下文字典（字段按 tracker 数据可用性增减）；
            tracker 无效时抛 AttributeError（不在此捕获）。
        """
        # 基础上下文：会话标识。
        context = {
            "sender_id": tracker.sender_id,
        }

        # 补充最新用户消息：latest_message 非 None 才读取；其类型为
        # UserMessage dataclass，text 经 .text 属性读取（原代码误用
        # dict 的 .get 方法导致 AttributeError，2026-10-17 修复）。
        if tracker.latest_message:
            context["last_user_message"] = tracker.latest_message.text

        # 补充当前流程：active_flow 为计算属性，无活跃 Flow 时
        # 为 None（falsy 不进入）。
        if tracker.active_flow:
            context["active_flow"] = tracker.active_flow

        return context


# 包级公开名单：本模块对外暴露重述器类与其配置 DTO。
__all__ = [
    "ResponseRephraser",
    "RephraserConfig",
]
