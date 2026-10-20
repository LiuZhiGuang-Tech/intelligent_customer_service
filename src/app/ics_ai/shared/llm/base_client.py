"""
base_client - LLM 客户端基类模块

核心功能：定义 LLM 客户端的抽象契约——LLMClient(ABC)（统一构造
          参数、complete/complete_sync 双通道补全抽象、validate
          配置校验）与 LLMResponse 响应 DTO（内容/用量/延迟/思考
          过程 + 四个便捷 property）；是 shared/llm 包全部具体
          客户端的共同基类。

执行流程：
    1. 契约期：LangChainClient（langchain_client.py，唯一具体
       实现）继承本基类，按 type 构造对应 LangChain Chat 模型；
    2. 工厂期：create_llm_client（包 __init__.py）按 type 字符串
       分发构造；
    3. 调用期：动作层/生成器/NLG 重述经 complete（async）或
       complete_sync 发送消息列表，获得 LLMResponse——Token 用量
       经 property 快捷读取，思考内容经 thinking_content 读取；
    4. 异常契约：连接/超时/响应解析失败的专用异常类型定义于
       app.ics_ai.shared.exceptions（LLMConnectionError/
       LLMTimeoutError/LLMResponseError 等），由子类的错误处理
       分类抛出。

副作用：本模块定义期无副作用（纯类型与抽象接口）；子类的
        complete/complete_sync 会发起外部 LLM API 网络请求。

主要依赖：
    - 标准库：abc（抽象基类）、dataclasses（DTO）、typing；
    - 项目内：app.ics_ai.shared.exceptions（异常契约的定义处）。
"""

# 导入抽象基类工具：ABC 基类与 abstractmethod 装饰器（complete/
# complete_sync 的契约强制）。
from abc import ABC, abstractmethod

# 导入 dataclass 工具：LLMResponse 与可变默认值工厂。
from dataclasses import dataclass, field

# 导入类型注解工具：Any 类型通配符（PEP 585/604 新式注解下
# Dict/List/Optional 已不需要；未用的 Text 已随注释任务移除）。
from typing import Any


@dataclass
class LLMResponse:
    """LLM 响应 DTO。

    职责：封装一次 LLM API 调用的完整产出——生成文本、模型名、
          Token 用量、延迟、原始响应与元数据（含思考过程），是
          客户端层与业务层（生成器/策略/NLG）之间的响应契约。
    设计模式：数据传输对象（dataclass，必填 content 打头，其余
              字段带默认值）+ 便捷只读视图（四个 property 对
              usage/metadata 做取值兜底）。
    生命周期：子类 complete/_parse_response 内构造并随返回值
              交付调用方。

    Attributes:
        content: 生成的文本内容（唯一必填字段）。
        model: 使用的模型名称（空串为缺省，子类填充）。
        usage: Token 使用统计（键为 prompt_tokens/
               completion_tokens/total_tokens，值为整数计数）。
        latency: 响应延迟（单位秒，浮点数）。
        raw_response: 原始 API 响应对象（用于调试，允许 None）。
        metadata: 额外元数据（如思考过程 thinking_content 键）。
    """

    # 生成文本：LLM 输出的核心内容（业务层消费的主体）。
    content: str
    # 模型名称：记录实际使用的模型（审计/日志用途）。
    model: str = ""
    # Token 用量统计：三键字典（缺失键经 property 兜底为 0）。
    usage: dict[str, int] = field(default_factory=dict)
    # 响应延迟：单位秒（time.time 差值，浮点）。
    latency: float = 0.0
    # 原始 API 响应：保留完整响应对象供调试，允许 None。
    raw_response: Any | None = None
    # 额外元数据：思考过程等附加信息（键值任意）。
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_tokens(self) -> int:
        """提示词 Token 数。

        返回：
            usage 字典的 prompt_tokens 键值；键缺失时兜底为 0
            （int 类型）。
        """
        # 兜底取值：用量未上报时返回 0 而非 KeyError。
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        """生成 Token 数。

        返回：
            usage 字典的 completion_tokens 键值；键缺失时兜底为 0。
        """
        # 兜底取值：同上。
        return self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        """总 Token 数。

        返回：
            usage 字典的 total_tokens 键值；键缺失时兜底为 0。
        """
        # 兜底取值：同上。
        return self.usage.get("total_tokens", 0)

    @property
    def thinking_content(self) -> str | None:
        """思考过程内容（如果启用了 thinking 模式）。

        返回：
            metadata 字典的 thinking_content 键值（str，由子类
            从响应的 additional_kwargs 提取写入）；未启用或未
            上报时返回 None。
        """
        # 兜底取值：普通对话无思考内容时返回 None。
        return self.metadata.get("thinking_content")


class LLMClient(ABC):
    """LLM 客户端抽象基类。

    职责：定义 LLM 客户端的统一接口——构造期固定六个核心配置并
          收集扩展参数；声明 complete（异步）/complete_sync（同步）
          两个补全通道契约；提供 validate 配置自检与统一 repr。
    设计模式：模板方法 + 策略——基类固定配置管理，补全行为由
              子类注入（LangChainClient 经 LangChain 适配多后端）。
    生命周期：经 create_llm_client 工厂或直接构造创建，随装配方
              （Agent/策略/生成器）长期持有复用；无每请求状态
              （请求上下文经参数传入）。

    Attributes:
        model: 模型名称（提供商侧标识）。
        temperature: 温度参数（采样随机度，浮点）。
        max_tokens: 最大生成 Token 数（整数上限）。
        timeout: 请求超时（单位秒）。
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 30,
        **kwargs: Any,
    ) -> None:
        """初始化 LLM 客户端。

        业务流程：六个核心参数逐个持有 → **kwargs 整体收集进
                  extra_config（Azure 端点等提供商特有参数由子类
                  按键自行读取）。

        副作用：无（纯赋值；网络连接延迟到调用期建立）。

        参数：
            model: 模型名称（必传）。
            api_key: API 密钥（必传——validate 以非空为有效判据）。
            api_base: API 基础 URL，允许 None（缺省走提供商官方
                      地址；vLLM 等自部署服务经此指定）。
            temperature: 温度参数（默认 0.0——确定性输出）。
            max_tokens: 最大生成 Token 数（默认 1024）。
            timeout: 请求超时秒数（默认 30）。
            **kwargs: 额外配置参数（收集进 extra_config）。
        """
        # 模型名称：调用期透传给后端。
        self.model = model
        # API 密钥：鉴权凭据（validate 的有效性判据之一）。
        self.api_key = api_key
        # API 基础 URL：None 时用后端默认地址。
        self.api_base = api_base
        # 温度参数：采样随机度（0.0 最确定）。
        self.temperature = temperature
        # 最大生成 Token 数：输出长度上限。
        self.max_tokens = max_tokens
        # 请求超时：单位秒。
        self.timeout = timeout
        # 额外配置：整体收集（子类按键读取，如 azure_endpoint）。
        self.extra_config = kwargs

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """异步生成文本补全（抽象方法，子类必须实现）。

        核心方法：发送消息列表给 LLM 并获取响应。

        业务流程（契约约定）：子类把标准消息格式转换为后端格式 →
            调用后端 API → 解析为 LLMResponse 返回；失败时按错误
            类型分类抛出专用异常。

        副作用：发起外部 LLM API 网络请求（子类实现内）。

        参数：
            messages: 消息列表，格式为
                      [{"role": "user/assistant/system",
                        "content": "..."}]。
            **kwargs: 额外的 API 参数。

        返回：
            LLMResponse 对象（content/usage/latency 等已填充）。

        异常：契约约定的失败分类（定义于
              app.ics_ai.shared.exceptions）——LLMConnectionError
              连接失败、LLMTimeoutError 请求超时、LLMResponseError
              响应解析失败；子类实现还会细分认证/限流异常。
        """
        # 抽象占位：ABC + abstractmethod 保证子类必须覆写。
        pass

    @abstractmethod
    def complete_sync(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """同步生成文本补全（抽象方法，子类必须实现）。

        同步版本的 complete 方法，用于非异步上下文（脚本/同步
        调用链）。

        副作用：发起外部 LLM API 网络请求（子类实现内）。

        参数：
            messages: 消息列表（格式同 complete）。
            **kwargs: 额外的 API 参数。

        返回：
            LLMResponse 对象。

        异常：同 complete 的契约分类。
        """
        # 抽象占位：ABC + abstractmethod 保证子类必须覆写。
        pass

    def validate(self) -> bool:
        """验证客户端配置。

        业务流程：api_key 与 model 均非空即视为有效（宽松校验——
                  仅做存在性检查，不做格式/连通性验证）。

        副作用：无（纯判断）。

        返回：
            配置有效返回 True，否则返回 False（任一必填项为空）。
        """
        # 双条件与：密钥和模型名都非空才算配置就绪。
        return bool(self.api_key and self.model)

    def __repr__(self) -> str:
        # 统一 repr：类名 + 模型名（便于日志排查用的实例摘要）。
        return f"{self.__class__.__name__}(model={self.model})"
