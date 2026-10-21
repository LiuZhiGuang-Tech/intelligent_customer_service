"""
enterprise_search_policy - 企业搜索策略

核心功能：基于知识库检索实现 RAG（检索增强生成）问答，并内置四级降级机制——
          知识库检索失败时依次降级到闲聊回复、默认兜底回复；同时承接栈帧
          分发职责，处理 Search / ChitChat / CannotHandle / Completed /
          HumanHandoff 五种栈帧并产出相应的策略预测。

执行流程：
    1. 上层（PolicyEnsemble）调用 predict 时，先检查"最新动作是否刚执行过"
       ——刚执行过动作且当前栈帧无需立即处理时放弃预测（避免重复响应）。
    2. 读取栈顶帧，按其类型分发到五个私有处理器之一：
       SearchStackFrame        → _handle_search_frame（RAG 检索主链路）
       ChitChatStackFrame      → _handle_chitchat_frame（闲聊回复）
       CannotHandleStackFrame  → _handle_cannot_handle_frame（兜底降级）
       CompletedStackFrame     → _handle_completed_frame（完成后追问）
       HumanHandoffStackFrame  → _handle_human_handoff_frame（人工转接）
    3. RAG 链路内部：_search 检索知识库并按相似度阈值过滤 →
       _generate_rag_answer 拼接上下文 + 提示词模板调用 LLM →
       回答含 [NO_RAG_ANSWER] 标记或 LLM 异常时降级到闲聊。
    4. LLM 客户端经 llm_client property 懒加载（首次访问才创建），
       避免构造策略时就初始化网络客户端。

副作用：
    - 模块导入：无 IO（LLMClient 仅注解引用，retrieval 仅类型检查期引用）。
    - predict 及各 _handle_* 处理器：读写对话栈（pop 栈帧）、写入 pattern
      执行历史（tracker.record_pattern）、调用 LLM 外部服务、调用检索器
      外部服务；必要时读取 Domain 响应模板。
    - 懒加载 property llm_client：首次访问时创建 LLM 客户端实例（写实例属性）。

主要依赖：
    - logging / dataclasses / typing / random：标准库基础能力。
    - app.ics_ai.policies.base_policy：策略基类契约（Policy/PolicyConfig/PolicyPrediction）。
    - app.ics_ai.shared.constants：降级原因枚举与默认兜底动作常量。
    - app.ics_ai.shared.llm：统一 LLM 客户端工厂（LangChain 封装，懒加载调用）。
    - app.ics_ai.dialogue_understanding.stack.stack_frame：五种栈帧类型
      （函数内运行时导入，供 isinstance 分发）。
    - app.ics_ai.retrieval.base_retriever.SearchResult：检索结果类型，
      仅用于类型注解——retrieval 子包尚未就位，置于 TYPE_CHECKING 分支。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，支持前向引用
# （如 "DialogueStateTracker"）并消除运行时注解求值开销。
from __future__ import annotations

# 导入日志器工厂：为本模块创建命名 logger（日志输出）。
import logging

# 导入 dataclass 工具：两个配置数据类靠它自动生成 __init__/__repr__；
# field(default_factory=...) 为可变默认值提供独立工厂。
from dataclasses import dataclass, field

# 导入类型注解工具：Any 任意类型；TYPE_CHECKING 为"仅类型检查期成立"的
# 常量，用于隔离只服务于注解的导入（避免运行时导入尚不存在的子包）。
from typing import TYPE_CHECKING, Any

# 导入策略基类契约：Policy 抽象基类、PolicyConfig 配置基类、
# PolicyPrediction 预测结果数据类。
from app.ics_ai.policies.base_policy import Policy, PolicyConfig, PolicyPrediction

# 导入共享常量：DegradationReason 降级原因枚举（写入 prediction.metadata
# 供下游统计）、ACTION_DEFAULT_FALLBACK 默认兜底动作名。
from app.ics_ai.shared.constants import ACTION_DEFAULT_FALLBACK, DegradationReason

# 导入 LLM 客户端工厂：懒加载 property 中调用，按配置类型创建客户端。
from app.ics_ai.shared.llm import create_llm_client

# 导入 LLM 客户端基类：作为 llm_client 属性与 __init__ 注入参数的类型注解。
from app.ics_ai.shared.llm.base_client import LLMClient

# 类型检查期专属导入：这些类型只出现在方法注解中；其中 SearchResult 所在的
# retrieval 子包尚未实现，运行时导入会 ImportError，故全部置于
# TYPE_CHECKING 分支（配合文件头 from __future__ import annotations，
# 注解延迟求值，运行时不解析这些名字）。
if TYPE_CHECKING:
    # 领域定义：读取 utter_default / utter_can_do_something_else /
    # utter_human_handoff 响应模板（仅注解用）。
    from app.ics_ai.core.domain import Domain

    # 对话状态追踪器：predict 系列方法的数据来源（仅注解用）。
    from app.ics_ai.core.tracker import DialogueStateTracker

    # 流程集合：predict 签名与基类保持一致（本策略不直接使用，仅注解用）。
    from app.ics_ai.dialogue_understanding.flow import FlowsList

    # 检索结果数据类：_search / _generate_rag_answer 的返回与参数注解。
    from app.ics_ai.retrieval.base_retriever import SearchResult

# 本模块命名 logger：日志前缀自动带模块路径，便于按策略过滤日志。
logger = logging.getLogger(__name__)


@dataclass
class _InternalRetrievalConfig:
    """内部检索配置（简化版）。

    职责：承载知识库检索的最小参数集，供 _search 方法读取。
    设计模式：数据传输对象（DTO），无行为逻辑；下划线前缀表示模块私有，
              不对外暴露（对外统一通过 EnterpriseSearchPolicyConfig.retrieval 访问）。
    生命周期：随 EnterpriseSearchPolicyConfig 的 default_factory 创建，
              与所属配置对象同生命周期。

    Attributes:
        enabled: 是否启用知识库检索；False 时 predict 直接跳过检索链路。
        top_k: 每次检索返回的最大文档条数。
        similarity_threshold: 相似度过滤阈值（0.0-1.0）；低于该值的结果被丢弃。
    """

    # 是否启用检索：默认 True；关闭后策略只走降级链路。
    enabled: bool = True
    # 检索条数上限：默认 3，兼顾上下文长度与相关性。
    top_k: int = 3
    # 相似度阈值：默认 0.5；None 分数的检索结果视为通过（不过滤）。
    similarity_threshold: float = 0.5


@dataclass
class EnterpriseSearchPolicyConfig(PolicyConfig):
    """企业搜索策略配置。

    职责：在基类 PolicyConfig（priority/max_history）之上扩展本策略专属
          配置——检索参数、LLM 选型参数、引用/相关性/闲聊开关。
    设计模式：数据传输对象（DTO）+ 继承扩展（复用基类字段并追加子类字段）。
    生命周期：随策略创建时构造（__init__ 未传入时按默认值新建），
              与所属策略实例同生命周期。

    Attributes:
        priority: 策略优先级，数值越大优先级越高。
        retrieval: 检索配置（enabled/top_k/similarity_threshold）。
        llm_type: LLM 类型标识 (openai/qwen/azure/anthropic)。
        llm_model: LLM 模型名。
        llm_temperature: 生成温度；0.0 表示确定性输出（客服场景避免发散）。
        enable_citation: 是否启用引用标注（当前实现未消费，预留字段）。
        enable_relevancy_check: 是否启用相关性检查（当前实现未消费，预留字段）。
        chitchat_enabled: 检索无果时是否允许降级到闲聊回复。
    """

    # 策略优先级：默认 50（中等），低于 FlowPolicy 的 100——
    # 流程优先于检索，保证命中 Flow 时检索策略让位。
    priority: int = 50

    # 检索配置子对象：可变默认值经 field 工厂独立创建，避免实例间共享。
    retrieval: _InternalRetrievalConfig = field(
        default_factory=_InternalRetrievalConfig
    )

    # LLM 类型：默认 openai（兼容 OpenAI API / vLLM 等服务）。
    llm_type: str = "openai"
    # LLM 模型名：默认 gpt-4o-mini（轻量、低成本）。
    llm_model: str = "gpt-4o-mini"
    # 生成温度：默认 0.0，客服回答要求稳定可复现。
    llm_temperature: float = 0.0
    # 引用标注开关：预留字段（当前链路未读取）。
    enable_citation: bool = False
    # 相关性检查开关：预留字段（当前链路未读取）。
    enable_relevancy_check: bool = True
    # 闲聊降级开关：默认 True，检索无果时先尝试闲聊再兜底。
    chitchat_enabled: bool = True


class EnterpriseSearchPolicy(Policy):
    """企业搜索策略。

    基于知识库检索实现 RAG 功能，并包含内置的降级机制。

    降级链（从高到低）：
    1. Flow匹配 → 执行Flow（由高优先级的 FlowPolicy 承接，本策略让位）
    2. 知识库检索 → 生成RAG回答（confidence=0.9）
    3. 闲聊识别 → 生成闲聊回复（confidence=0.7）
    4. 无法处理 → 返回默认回复（confidence=0.5）

    职责：实现基类的 predict 决策逻辑——按栈顶帧类型分发到对应的
          处理器，产出"发送文本 / 兜底动作 / 放弃"三类预测；同时提供
          检索、RAG 生成、闲聊生成三个可复用的私有能力方法。
    设计模式：抽象基类 + 策略模式（面向 Policy 契约编程）；predict 内部
              按栈帧类型分发是轻量的"责任分发"变体；LLM 客户端懒加载
              是延迟初始化（Lazy Initialization）模式。
    生命周期：框架启动时创建并注入配置/检索器（依赖可后置注入
              set_retriever），随会话/服务存续长期复用。

    Attributes:
        DEFAULT_PRIORITY: 类级默认优先级，与配置默认值 50 保持一致。
        RAG_PROMPT_TEMPLATE: RAG 回答的提示词模板（含 [NO_RAG_ANSWER] 哨兵）。
        CHITCHAT_PROMPT_TEMPLATE: 闲聊回复的提示词模板。
    """

    # 类级默认优先级：与 EnterpriseSearchPolicyConfig.priority 默认值一致。
    DEFAULT_PRIORITY = 50

    # RAG提示词模板：str.format 占位符为 {context}（检索文档拼接）与
    # {question}（用户问题）；[NO_RAG_ANSWER] 是哨兵值——LLM 认为文档
    # 信息不足时回复该标记，调用方据此判断是否降级。
    RAG_PROMPT_TEMPLATE = """你是一个专业的客服助手，正在根据知识库文档回答用户问题。

### 参考文档
{context}

### 用户问题
{question}

### 回答要求
严格基于上述文档内容回答：
1. 直接回答问题，不要添加问候语或寒暄
2. 禁止使用 emoji 表情符号
3. 使用专业、简洁的语气
4. 只陈述文档中明确提到的信息
5. 如果文档包含具体的产品名称、品牌、规格等，必须准确引用
6. 最多2-3句话，避免冗余
7. 如果文档信息不足以回答问题，仅回复"[NO_RAG_ANSWER]"

回答：
"""

    # 闲聊提示词模板：占位符 {message} 为用户原话；要求简短友好，
    # 与 RAG 回答的专业语气区分开。
    CHITCHAT_PROMPT_TEMPLATE = """你是一个友好的AI助手。用户发送了一条闲聊消息，请用自然、友好的方式回复。

用户: {message}

请回复（保持简短友好）：
"""

    def __init__(
        self,
        config: EnterpriseSearchPolicyConfig | None = None,
        llm_client: LLMClient | None = None,
        retriever: Any | None = None,
        **kwargs: Any,
    ):
        """初始化企业搜索策略。

        业务流程：
            1. 调用基类 __init__ 完成 config 兜底与策略名初始化；
            2. 将 self.config 重声明为子类类型（供类型检查器窄化，
               解锁子类专属字段如 retrieval / llm_type 的访问）；
            3. 保存注入的 LLM 客户端与检索器（均可为 None，
               LLM 走懒加载，检索器支持 set_retriever 后置注入）。

        副作用：仅写入实例属性，无 IO 操作（LLM 客户端不在此时创建）。

        Args:
            config: 策略配置；None 时使用 EnterpriseSearchPolicyConfig() 默认值。
            llm_client: 外部注入的 LLM 客户端；None 时由 llm_client
                        property 按配置懒加载创建。
            retriever: 检索器实例（鸭子类型，需提供 async search 方法）；
                       None 时跳过知识库检索，直接走降级链路。
            **kwargs: 额外参数，透传给基类（基类接收后忽略）。
        """
        # 调用基类 __init__：处理 config 兜底（None → 默认配置）+ 设置 _name = 类名。
        super().__init__(config or EnterpriseSearchPolicyConfig(), **kwargs)

        # 类型窄化重声明：基类 __init__ 将 self.config 标注为 PolicyConfig，
        # 此行以子类类型重新声明，让类型检查器识别子类专属字段
        # （retrieval / llm_type 等）；运行时语义等价于无操作自赋值。
        self.config: EnterpriseSearchPolicyConfig = self.config

        # 保存注入的 LLM 客户端：None 表示尚未创建，由 property 懒加载。
        self._llm_client = llm_client

        # 保存注入的检索器：None 表示未配置，_search 会跳过检索。
        self._retriever = retriever

    @property
    def llm_client(self) -> LLMClient:
        """获取LLM客户端（延迟初始化）。

        业务流程：实例属性 _llm_client 为 None 时按配置调用工厂函数
                  create_llm_client 创建并缓存；否则直接复用。
        副作用：首次访问时创建 LLM 客户端实例（写 self._llm_client）。

        返回：
            LLMClient 实例（缓存复用，多次访问返回同一对象）。
        """
        # 懒加载判断：尚未创建过客户端时按配置初始化。
        if self._llm_client is None:
            # 调用工厂函数：类型/模型/温度均取自策略配置。
            self._llm_client = create_llm_client(
                type=self.config.llm_type,
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
            )
        # 返回缓存实例（首次返回新建，之后返回同一对象）。
        return self._llm_client

    def does_support_stack_frame(self, frame: Any | None = None) -> bool:
        """检查策略是否支持处理指定栈帧。

        支持五种栈帧：SearchStackFrame（检索）、ChitChatStackFrame（闲聊）、
        CannotHandleStackFrame（兜底）、CompletedStackFrame（完成后追问）、
        HumanHandoffStackFrame（人工转接）；其余类型（如 FlowStackFrame）
        返回 False，交由 FlowPolicy 等其他策略承接。

        业务流程：函数内导入五种栈帧类型 → isinstance 元组判断 → 返回布尔值。
        副作用：无（纯类型判断）。

        参数：
            frame: 要检查的栈帧实例；None 时 isinstance 判定为 False
                   （None 不属于任何栈帧类型），即不支持。

        返回：
            是否支持处理该栈帧；frame 为五种支持类型之一时 True，否则 False。
        """
        # 函数内导入：栈帧子模块依赖较重，且避免模块加载期的循环导入；
        # 仅在本方法被调用时才执行导入。
        from app.ics_ai.dialogue_understanding.stack.stack_frame import (
            SearchStackFrame,
            ChitChatStackFrame,
            CannotHandleStackFrame,
            CompletedStackFrame,
            HumanHandoffStackFrame,
        )

        # isinstance 元组判断：frame 是五种支持类型之一即返回 True；
        # frame 为 None 或其他类型（如 FlowStackFrame）返回 False。
        return isinstance(
            frame,
            (
                SearchStackFrame,
                ChitChatStackFrame,
                CannotHandleStackFrame,
                CompletedStackFrame,
                HumanHandoffStackFrame,
            ),
        )

    async def predict(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        flows: FlowsList | None = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """预测下一步动作。

        检测栈帧类型并分发处理：
        - SearchStackFrame → 执行检索
        - ChitChatStackFrame → 生成闲聊回复
        - CannotHandleStackFrame → 返回降级响应
        - CompletedStackFrame → 询问是否还有其他需求
        - HumanHandoffStackFrame → 执行人工转接

        业务流程：
            a. 前置放弃检查：最新动作刚执行过（非 action_listen）且当前
               栈帧不属于"需立即处理"类型（Completed/Handoff）时放弃，
               防止动作执行后同轮重复产出预测；
            b. 提取用户消息文本（latest_message.text，可能为空串）；
            c. 按栈顶帧类型依次 isinstance 匹配，命中即分发到对应
               _handle_* 处理器并 await 返回；
            d. 五种类型均未命中时放弃预测。

        副作用：由分发目标决定（见各 _handle_* 方法的副作用说明）；
                本方法自身仅读取 tracker 状态。

        参数：
            tracker: 对话状态追踪器，必需（读栈顶帧/最新动作/最新消息）。
            domain: Domain定义，允许为 None；透传给
                    _handle_completed_frame / _handle_cannot_handle_frame /
                    _handle_human_handoff_frame 用于读取响应模板。
            flows: Flow列表，允许为 None；本策略不消费，仅为与基类
                   签名保持一致。
            **kwargs: 额外参数，接收后忽略（保持接口兼容）。

        返回：
            PolicyPrediction（来自具体处理器）或 abstain 放弃结果。
        """
        # 函数内导入栈帧类型：仅 predict 被调用时才加载，规避模块级循环导入。
        from app.ics_ai.dialogue_understanding.stack.stack_frame import (
            SearchStackFrame,
            ChitChatStackFrame,
            CannotHandleStackFrame,
            CompletedStackFrame,
            HumanHandoffStackFrame,
        )

        # 获取栈顶帧：分发决策的依据（可能是任意栈帧类型或空栈返回 None）。
        top_frame = tracker.dialogue_stack.top()

        # 检查是否已经有 bot 响应（如果刚执行了动作，则放弃）。
        # 但是对于需要立即处理的栈帧（如 CompletedStackFrame），不应放弃——
        # 因为 Completed/Handoff 帧正是动作执行完成后才压栈的"收尾帧"，
        # 若因"刚执行过动作"而放弃，追问/转接将永远无法触发。
        from app.ics_ai.dialogue_understanding.stack.stack_frame import (
            CompletedStackFrame as CompletedFrame,
            HumanHandoffStackFrame as HandoffFrame,
        )

        # "需立即处理"判定：栈顶是完成帧或转接帧 → 即使刚执行过动作也不放弃。
        needs_immediate_handling = isinstance(top_frame, (CompletedFrame, HandoffFrame))

        # 前置放弃检查：三个条件同时满足才放弃——
        # 1) 存在最新动作名；2) 最新动作不是监听（即刚执行过实际动作）；
        # 3) 当前栈帧不需要立即处理。
        if (
            tracker.latest_action_name
            and tracker.latest_action_name != "action_listen"
            and not needs_immediate_handling
        ):
            # 记录放弃原因（调试级别日志）。
            logger.debug(
                f"Action {tracker.latest_action_name} just executed, abstaining"
            )
            # 返回放弃预测：上层据 is_abstain 忽略本策略。
            return PolicyPrediction.abstain(self.name)

        # 获取用户消息（统一从latest_message获取）；无消息时保持空串，
        # 由各处理器自行处理空消息分支。
        user_message = ""
        if tracker.latest_message:
            # 取最新用户消息的纯文本内容。
            user_message = tracker.latest_message.text

        # 根据栈帧类型分发处理：isinstance 链按序匹配，命中即 await
        # 对应处理器并直接返回其预测结果。
        if isinstance(top_frame, CompletedStackFrame):
            # 完成帧 → 追问"还有什么可以帮您"。
            return await self._handle_completed_frame(tracker, top_frame, domain)

        if isinstance(top_frame, HumanHandoffStackFrame):
            # 转接帧 → 生成人工转接话术。
            return await self._handle_human_handoff_frame(tracker, top_frame, domain)

        if isinstance(top_frame, ChitChatStackFrame):
            # 闲聊帧 → LLM 闲聊回复。
            return await self._handle_chitchat_frame(tracker, user_message)

        if isinstance(top_frame, CannotHandleStackFrame):
            # 无法处理帧 → 兜底话术。
            return await self._handle_cannot_handle_frame(tracker, top_frame, domain)

        if isinstance(top_frame, SearchStackFrame):
            # 检索帧 → RAG 主链路（检索 + 生成 + 降级）。
            return await self._handle_search_frame(tracker, user_message)

        # 没有特定栈帧，放弃处理：栈顶为 None 或其他策略管辖的帧类型。
        return PolicyPrediction.abstain(self.name)

    async def _handle_search_frame(
        self,
        tracker: DialogueStateTracker,
        user_message: str,
    ) -> PolicyPrediction:
        """处理SearchStackFrame - 执行检索。

        核心功能：RAG 主链路——检索知识库 → LLM 生成回答；检索无果时
                  依次降级到闲聊、默认兜底。四种出口都会弹出本栈帧，
                  保证不会残留死帧。

        业务流程：
            1. 空消息直接放弃（无可检索内容）；
            2. 检索开启且配置了检索器 → _search 获取过滤后结果；
            3. 有结果 → _generate_rag_answer 生成回答；回答存在且不含
               [NO_RAG_ANSWER] 哨兵 → 弹帧 + 记录 pattern + 返回
               action_send_text（confidence=0.9）；
            4. RAG 无果且闲聊开启 → _generate_chitchat_answer；有回复则
               弹帧 + 返回 action_send_text（confidence=0.7，标记 CHITCHAT）；
            5. 均无果 → 弹帧 + 返回 ACTION_DEFAULT_FALLBACK（CANNOT_HANDLE）；
            6. 任意异常 → 弹帧（尽力而为）+ 返回兜底动作（INTERNAL_ERROR）。

        副作用：弹出对话栈帧；写 pattern 执行历史；调用检索器与 LLM
                外部服务；异常路径同样弹帧。

        参数：
            tracker: 对话状态追踪器（读栈/写 pattern）。
            user_message: 用户消息文本；空串时放弃。

        返回：
            PolicyPrediction——RAG 回答（0.9）/ 闲聊回答（0.7）/
            兜底动作（0.5）/ 异常兜底（0.3）之一，或空消息时的 abstain。
        """
        # 空消息守卫：无内容可检索，直接放弃（不弹帧，栈帧留待后续轮次）。
        if not user_message:
            return PolicyPrediction.abstain(self.name)

        # 入口日志：记录本次检索的用户问题，便于链路追踪。
        logger.info(
            f"[EnterpriseSearchPolicy] SearchStackFrame processing: {user_message}"
        )

        try:
            # 尝试知识库检索：两个前置条件——配置开关开启 + 检索器已注入。
            if self.config.retrieval.enabled and self._retriever:
                # 调用私有检索方法：内部完成 top_k 检索与阈值过滤。
                search_results = await self._search(user_message, tracker)

                # 有检索结果才进入 RAG 生成（空列表直接走降级）。
                if search_results:
                    # 记录检索命中条数（信息级日志）。
                    logger.info(
                        f"[EnterpriseSearchPolicy] 检索到 {len(search_results)} 条结果，开始生成RAG回答"
                    )
                    # 调用 RAG 生成：拼接上下文 + 提示词模板 + LLM 调用。
                    answer = await self._generate_rag_answer(
                        user_message, search_results
                    )
                    # 记录 RAG 回答前 200 字符（避免日志过长）。
                    logger.info(
                        f"[EnterpriseSearchPolicy] RAG回答: {answer[:200] if answer else 'None'}..."
                    )

                    # 双重校验：回答非空 且 不含 [NO_RAG_ANSWER] 哨兵
                    # （哨兵表示 LLM 判定文档信息不足，需降级）。
                    if answer and "[NO_RAG_ANSWER]" not in answer:
                        # 检索成功，弹出栈帧（本帧使命完成）。
                        tracker.dialogue_stack.pop()
                        # 记录 Pattern 执行历史（下游统计各链路触发占比）。
                        tracker.record_pattern("search")
                        # 调试日志：确认弹帧。
                        logger.debug(
                            "SearchStackFrame popped after successful retrieval"
                        )

                        # 返回 RAG 回答预测：confidence=0.9（高置信），
                        # metadata 携带回答文本、降级原因与检索明细。
                        return PolicyPrediction(
                            action="action_send_text",
                            confidence=0.9,
                            policy_name=self.name,
                            metadata={
                                "text": answer,
                                "degradation_reason": DegradationReason.DEFAULT,
                                "search_results": [r.content for r in search_results],
                            },
                        )

            # 降级到闲聊：RAG 无果（无检索结果 / 无回答 / 含哨兵）且
            # 配置允许闲聊时尝试。
            if self.config.chitchat_enabled:
                # 调用闲聊生成：LLM 按闲聊模板回复。
                chitchat_answer = await self._generate_chitchat_answer(user_message)
                # 有回复才算降级成功。
                if chitchat_answer:
                    # 弹出栈帧（本帧使命完成）。
                    tracker.dialogue_stack.pop()
                    # 记录 Pattern 执行历史（标记为 search 链路的闲聊降级）。
                    tracker.record_pattern("search")
                    # 调试日志：确认走闲聊降级弹帧。
                    logger.debug("SearchStackFrame popped after chitchat fallback")

                    # 返回闲聊回答预测：confidence=0.7（中置信），
                    # 标记降级原因为 CHITCHAT 供下游统计。
                    return PolicyPrediction(
                        action="action_send_text",
                        confidence=0.7,
                        policy_name=self.name,
                        metadata={
                            "text": chitchat_answer,
                            "degradation_reason": DegradationReason.CHITCHAT,
                        },
                    )

            # 无法处理：检索与闲聊均无果 → 弹帧 + 默认兜底动作。
            tracker.dialogue_stack.pop()
            # 记录 Pattern 执行历史。
            tracker.record_pattern("search")
            # 返回兜底动作预测：confidence=0.5，标记 CANNOT_HANDLE。
            return PolicyPrediction(
                action=ACTION_DEFAULT_FALLBACK,
                confidence=0.5,
                policy_name=self.name,
                metadata={"degradation_reason": DegradationReason.CANNOT_HANDLE},
            )

        except Exception as e:
            # 异常兜底：检索/生成链路任何异常都不应让策略崩溃，
            # 转为兜底动作并记录错误详情。
            logger.error(f"Search frame error: {e}")
            try:
                # 尽力弹帧：即使异常也要移除本帧，避免死帧残留。
                tracker.dialogue_stack.pop()
                # 记录 Pattern 执行历史（异常路径同样计入 search）。
                tracker.record_pattern("search")
            except Exception:
                # 弹帧本身失败（如空栈）：静默忽略，优先返回兜底预测。
                pass
            # 返回异常兜底预测：confidence=0.3（最低），metadata 携带错误信息。
            return PolicyPrediction(
                action=ACTION_DEFAULT_FALLBACK,
                confidence=0.3,
                policy_name=self.name,
                metadata={
                    "degradation_reason": DegradationReason.INTERNAL_ERROR,
                    "error": str(e),
                },
            )

    async def _handle_chitchat_frame(
        self,
        tracker: DialogueStateTracker,
        user_message: str,
    ) -> PolicyPrediction:
        """处理ChitChatStackFrame - 生成闲聊回复。

        业务流程：
            1. 先弹帧 + 记录 pattern（无论后续生成成败，本帧都终结）；
            2. 空消息 → 返回固定问候语（confidence=0.8）；
            3. 有消息 → LLM 闲聊生成；成功返回（confidence=0.9）；
            4. 生成异常或返回空 → 固定友好回复兜底（confidence=0.7）。

        副作用：弹出对话栈帧；写 pattern 执行历史；调用 LLM 外部服务。

        参数：
            tracker: 对话状态追踪器（读栈/写 pattern）。
            user_message: 用户消息文本；空串时走固定问候分支。

        返回：
            PolicyPrediction——action 均为 action_send_text，
            confidence 依次为 0.8 / 0.9 / 0.7。
        """
        # 调试日志：记录闲聊处理入口与用户消息。
        logger.debug(f"ChitChatStackFrame processing: {user_message}")

        # 弹出栈帧：先弹帧再生成，保证生成失败时帧也不会残留。
        tracker.dialogue_stack.pop()
        # 记录 Pattern 执行历史（标记 chitchat 链路）。
        tracker.record_pattern("chitchat")

        # 空消息分支：无法生成个性化回复，返回固定问候语。
        if not user_message:
            return PolicyPrediction(
                action="action_send_text",
                confidence=0.8,
                policy_name=self.name,
                metadata={"text": "你好！有什么可以帮您的吗？"},
            )

        try:
            # 调用闲聊生成：LLM 按闲聊模板回复用户消息。
            chitchat_answer = await self._generate_chitchat_answer(user_message)
            # 生成成功 → 返回 LLM 回答（高置信）。
            if chitchat_answer:
                return PolicyPrediction(
                    action="action_send_text",
                    confidence=0.9,
                    policy_name=self.name,
                    metadata={
                        "text": chitchat_answer,
                        "degradation_reason": DegradationReason.CHITCHAT,
                    },
                )
        except Exception as e:
            # 生成异常：记录错误，落入下方固定回复兜底。
            logger.error(f"Chitchat generation error: {e}")

        # 默认回复：LLM 异常或返回空时的固定友好话术兜底。
        return PolicyPrediction(
            action="action_send_text",
            confidence=0.7,
            policy_name=self.name,
            metadata={"text": "你好！很高兴和你聊天。"},
        )

    async def _handle_cannot_handle_frame(
        self,
        tracker: DialogueStateTracker,
        frame: Any,
        domain: Domain | None,
    ) -> PolicyPrediction:
        """处理CannotHandleStackFrame - 返回降级响应。

        业务流程：
            1. 弹帧 + 记录 pattern（标记 cannot_handle）；
            2. 从 domain 读取 utter_default 响应模板，随机选一条作为
               兜底话术；domain 缺失或无模板时用内置默认话术；
            3. 返回 action_send_text（confidence=0.5）。

        副作用：弹出对话栈帧；写 pattern 执行历史。

        参数：
            tracker: 对话状态追踪器（读栈/写 pattern）。
            frame: CannotHandleStackFrame 实例（鸭子类型读取 reason 字段）。
            domain: Domain定义，允许为 None（None 时用内置默认话术）。

        返回：
            PolicyPrediction——action_send_text，metadata 携带兜底文本、
            降级原因与帧内 reason。
        """
        # 调试日志：记录无法处理的原因（getattr 兜底防字段缺失）。
        logger.debug(
            f"CannotHandleStackFrame processing, reason: {getattr(frame, 'reason', '')}"
        )

        # 弹出栈帧：本帧使命完成。
        tracker.dialogue_stack.pop()
        # 记录 Pattern 执行历史。
        tracker.record_pattern("cannot_handle")

        # 尝试从domain获取默认回复：先设内置话术为兜底默认值。
        fallback_text = "抱歉，我没有理解您的意思。请换一种方式表达。"
        if domain:
            # 读取 utter_default 模板列表（Domain API，缺失返回空列表）。
            responses = domain.get_response("utter_default")
            if responses:
                # 函数内导入 random：仅本分支需要，用于模板随机轮换。
                import random

                # 随机选一条模板并取其文本（多模板轮换避免机械重复）。
                fallback_text = random.choice(responses).text

        # 返回兜底话术预测：confidence=0.5，metadata 携带文本/原因/帧 reason。
        return PolicyPrediction(
            action="action_send_text",
            confidence=0.5,
            policy_name=self.name,
            metadata={
                "text": fallback_text,
                "degradation_reason": DegradationReason.CANNOT_HANDLE,
                "reason": getattr(frame, "reason", ""),
            },
        )

    async def _handle_completed_frame(
        self,
        tracker: DialogueStateTracker,
        frame: Any,
        domain: Domain | None,
    ) -> PolicyPrediction:
        """处理CompletedStackFrame - 询问是否还有其他需求。

        当Flow完成后，系统会询问用户是否还有其他需要帮助的。

        业务流程：
            1. 读取帧内 previous_flow_name（getattr 兜底防缺失）；
            2. 弹帧 + 记录 pattern（标记 completed）；
            3. 从 domain 读取 utter_can_do_something_else 模板随机选用，
               缺失时用内置追问话术；
            4. 返回 action_send_text（confidence=0.9）。

        副作用：弹出对话栈帧；写 pattern 执行历史。

        参数：
            tracker: 对话状态追踪器（读栈/写 pattern）。
            frame: CompletedStackFrame 实例（鸭子类型读取 previous_flow_name）。
            domain: Domain定义，允许为 None（None 时用内置追问话术）。

        返回：
            PolicyPrediction——action_send_text，metadata 携带追问文本
            与来源流程名。
        """
        # 读取来源流程名：getattr 第三参兜底，防字段缺失时报错。
        previous_flow = getattr(frame, "previous_flow_name", "")
        # 调试日志：记录追问对应的来源流程。
        logger.debug(f"CompletedStackFrame processing, previous_flow: {previous_flow}")

        # 弹出栈帧：本帧使命完成。
        tracker.dialogue_stack.pop()
        # 记录 Pattern 执行历史。
        tracker.record_pattern("completed")

        # 尝试从domain获取完成响应：先设内置追问话术为默认值。
        completed_text = "还有什么我可以帮您的吗？"
        if domain:
            # 读取 utter_can_do_something_else 模板列表。
            responses = domain.get_response("utter_can_do_something_else")
            if responses:
                # 函数内导入 random：仅本分支需要。
                import random

                # 随机选一条模板并取其文本。
                completed_text = random.choice(responses).text

        # 返回追问预测：confidence=0.9，metadata 携带追问文本与来源流程名。
        return PolicyPrediction(
            action="action_send_text",
            confidence=0.9,
            policy_name=self.name,
            metadata={
                "text": completed_text,
                "previous_flow": previous_flow,
            },
        )

    async def _handle_human_handoff_frame(
        self,
        tracker: DialogueStateTracker,
        frame: Any,
        domain: Domain | None,
    ) -> PolicyPrediction:
        """处理HumanHandoffStackFrame - 执行人工转接。

        当需要转接人工客服时，生成转接响应。

        业务流程：
            1. 读取帧内 reason（getattr 兜底防缺失）；
            2. 弹帧 + 记录 pattern（标记 human_handoff）；
            3. 从 domain 读取 utter_human_handoff 模板随机选用，
               缺失时用内置转接话术；
            4. 返回 action_send_text（confidence=0.95，全策略最高，
               因为转接意图必须被忠实执行）。

        副作用：弹出对话栈帧；写 pattern 执行历史。

        参数：
            tracker: 对话状态追踪器（读栈/写 pattern）。
            frame: HumanHandoffStackFrame 实例（鸭子类型读取 reason）。
            domain: Domain定义，允许为 None（None 时用内置转接话术）。

        返回：
            PolicyPrediction——action_send_text，metadata 携带转接文本、
            human_handoff 标志与转接原因。
        """
        # 读取转接原因：getattr 第三参兜底，防字段缺失时报错。
        reason = getattr(frame, "reason", "")
        # 调试日志：记录转接原因。
        logger.debug(f"HumanHandoffStackFrame processing, reason: {reason}")

        # 弹出栈帧：本帧使命完成。
        tracker.dialogue_stack.pop()
        # 记录 Pattern 执行历史。
        tracker.record_pattern("human_handoff")

        # 尝试从domain获取转接响应：先设内置转接话术为默认值。
        handoff_text = "好的，正在为您转接人工客服，请稍候..."
        if domain:
            # 读取 utter_human_handoff 模板列表。
            responses = domain.get_response("utter_human_handoff")
            if responses:
                # 函数内导入 random：仅本分支需要。
                import random

                # 随机选一条模板并取其文本。
                handoff_text = random.choice(responses).text

        # 返回转接预测：confidence=0.95（最高），metadata 携带文本/标志/原因。
        return PolicyPrediction(
            action="action_send_text",
            confidence=0.95,
            policy_name=self.name,
            metadata={
                "text": handoff_text,
                "human_handoff": True,
                "reason": reason,
            },
        )

    async def _search(
        self,
        query: str,
        tracker: DialogueStateTracker | None = None,
    ) -> list[SearchResult]:
        """执行知识库搜索。

        核心功能：调用检索器获取 top_k 条候选，并按相似度阈值过滤
                  低分结果；检索器未配置或调用异常时返回空列表。

        业务流程：
            1. 检索器未注入 → 记录调试日志并返回空列表；
            2. 构建 tracker_state（tracker 序列化字典，供检索器获取
               用户信息与历史对话做个性化检索）；
            3. await 检索器 search（鸭子类型调用）；
            4. 列表推导过滤：score 为 None 视为通过（检索器未评分），
               score >= 阈值保留；
            5. 任意异常 → 记录错误并返回空列表（调用方据此走降级）。

        副作用：调用检索器外部服务（向量库/搜索服务）。

        参数：
            query: 搜索查询文本；空串由调用方（_handle_search_frame）先行拦截。
            tracker: 对话状态追踪器，允许为 None；用于序列化用户上下文。

        返回：
            过滤后的 SearchResult 列表；无检索器/无结果/异常时均为空列表。
        """
        # 检索器未配置守卫：直接跳过检索（调用方会走降级链路）。
        if not self._retriever:
            logger.debug("未配置检索器，跳过知识库搜索")
            return []

        try:
            # 入口日志：记录检索器具体类型，便于诊断检索链路。
            logger.info(
                f"[EnterpriseSearchPolicy] 调用检索器: {type(self._retriever).__name__}"
            )
            # 入口日志：记录查询文本与 top_k 配置。
            logger.info(
                f"[EnterpriseSearchPolicy] 查询: '{query}', top_k={self.config.retrieval.top_k}"
            )

            # 构建 tracker_state 用于检索器获取用户信息和历史对话：
            # tracker 存在时序列化为字典，否则传 None（检索器自行降级）。
            tracker_state = tracker.to_dict() if tracker else None

            # 调用检索器：鸭子类型 async 接口——query + top_k + 用户上下文。
            results = await self._retriever.search(
                query,
                top_k=self.config.retrieval.top_k,
                tracker_state=tracker_state,
            )

            # 记录检索器原始返回条数。
            logger.info(f"[EnterpriseSearchPolicy] 检索器返回 {len(results)} 条结果")

            # 过滤低相似度结果（score 为 None 时视为通过过滤）：
            # None 分数表示检索器未评分，保守保留；有分数则须 >= 阈值。
            threshold = self.config.retrieval.similarity_threshold
            filtered = [r for r in results if r.score is None or r.score >= threshold]

            # 记录过滤后剩余条数与所用阈值。
            logger.info(
                f"[EnterpriseSearchPolicy] 过滤后剩余 {len(filtered)} 条结果 "
                f"(阈值: {self.config.retrieval.similarity_threshold})"
            )

            # 返回过滤后的结果列表。
            return filtered

        except Exception as e:
            # 异常兜底：检索失败不抛出，返回空列表让调用方走降级链路。
            logger.error(f"[EnterpriseSearchPolicy] 搜索错误: {e}")
            return []

    async def _generate_rag_answer(
        self,
        question: str,
        search_results: list[SearchResult],
    ) -> str | None:
        """生成RAG回答。

        业务流程：
            1. 空结果守卫：直接返回 None（无可引用内容）；
            2. 拼接上下文：每条结果格式化为"编号. 来源\\n内容"，
               条目间以空行分隔；
            3. 用 RAG_PROMPT_TEMPLATE 渲染提示词（context + question）；
            4. 调用 LLM（单条 user 消息）；成功返回回答文本；
            5. 异常 → 记录错误并返回 None（调用方据此降级到闲聊）。

        副作用：调用 LLM 外部服务。

        参数：
            question: 用户问题文本。
            search_results: 过滤后的检索结果列表；空列表时返回 None。

        返回：
            LLM 生成的回答文本；可能含 [NO_RAG_ANSWER] 哨兵
            （由调用方检测）；异常时返回 None。
        """
        # 空结果守卫：没有检索内容就无法生成有据回答。
        if not search_results:
            return None

        # 构建上下文：逐条格式化收集。
        context_parts = []
        # enumerate 从 1 起编号：上下文中的条目编号便于 LLM 引用。
        for i, result in enumerate(search_results, 1):
            # 提取来源标识（文档名/链接等）。
            source = result.source
            # 提取文档正文内容。
            content = result.content
            # 格式：编号. 来源\n内容（换行分隔来源与正文）。
            context_parts.append(f"{i}. {source}\n{content}")

        # 条目间双换行拼接为完整上下文块。
        context = "\n\n".join(context_parts)
        # 记录完整 RAG 上下文（信息级日志，便于排查生成质量）。
        logger.info(f"[EnterpriseSearchPolicy] RAG上下文:\n{context}")

        # 构建提示词：模板填充上下文与用户问题。
        prompt = self.RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        # 调用LLM：异常不外抛，转为 None 由调用方降级。
        try:
            # 单条 user 角色消息调用 LLM 完成。
            response = await self.llm_client.complete(
                [{"role": "user", "content": prompt}]
            )
            # 记录回答前 200 字符（避免日志过长）。
            logger.info(
                f"[EnterpriseSearchPolicy] RAG回答: {response.content[:200] if response.content else 'None'}..."
            )
            # 返回回答文本（可能为 None——LLM 返回空内容时）。
            return response.content
        except Exception as e:
            # 生成异常：记录错误并返回 None（触发调用方降级）。
            logger.error(f"RAG generation error: {e}")
            return None

    async def _generate_chitchat_answer(self, message: str) -> str | None:
        """生成闲聊回答。

        业务流程：
            1. 用 CHITCHAT_PROMPT_TEMPLATE 渲染提示词（填充用户消息）；
            2. 调用 LLM（单条 user 消息）；
            3. 异常 → 记录错误并返回 None（调用方用固定话术兜底）。

        副作用：调用 LLM 外部服务。

        参数：
            message: 用户闲聊消息文本。

        返回：
            LLM 生成的闲聊回复文本；异常时返回 None。
        """
        # 构建提示词：模板填充用户消息。
        prompt = self.CHITCHAT_PROMPT_TEMPLATE.format(message=message)

        try:
            # 单条 user 角色消息调用 LLM 完成。
            response = await self.llm_client.complete(
                [{"role": "user", "content": prompt}]
            )
            # 返回闲聊回复文本。
            return response.content
        except Exception as e:
            # 生成异常：记录错误并返回 None（触发调用方固定话术兜底）。
            logger.error(f"Chitchat generation error: {e}")
            return None

    def set_retriever(self, retriever: Any) -> None:
        """设置检索器。

        业务流程：直接覆盖实例属性 _retriever——支持策略创建后再注入
                  检索器（如依赖装配晚于策略构造的场景）。
        副作用：写入实例属性 self._retriever。

        Args:
            retriever: 检索器实例（鸭子类型，需提供 async search 方法）。
        """
        # 覆盖保存检索器实例。
        self._retriever = retriever


# 导出清单：from enterprise_search_policy import * 时仅暴露以下名字。
# 注意：SearchResult 因 retrieval 子包尚未实现而置于 TYPE_CHECKING 导入，
# 无法在运行时由本模块提供，故不在 __all__ 中（待 retrieval 就位后
# 可恢复导出）。
__all__ = [
    # 列表项1：企业搜索策略主类。
    "EnterpriseSearchPolicy",
    # 列表项2：企业搜索策略配置数据类。
    "EnterpriseSearchPolicyConfig",
]
