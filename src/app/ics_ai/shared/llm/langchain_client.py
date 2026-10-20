"""
langchain_client - 统一 LLM 客户端模块

核心功能：定义 LangChainClient——LLMClient 基类的唯一具体实现，
          经 LangChain 框架适配四种后端（openai：OpenAI API/vLLM/
          OpenAI 兼容服务、qwen：阿里云 DashScope 通义千问、azure：
          Azure OpenAI、anthropic：Anthropic Claude），提供补全
          双通道（complete/complete_sync）、懒加载实例管理、
          thinking 深度思考模式与错误分类转换。

执行流程：
    1. 装配期：create_llm_client（包 __init__.py）按 type 构造
       本客户端（类型白名单构造期校验）；
    2. 调用期：complete/complete_sync → _get_llm 懒加载建后端
       实例（_create_llm 按 type 分发到四个 _create_*_llm）→
       _convert_messages 把标准消息转为 LangChain 消息对象 →
       ainvoke/invoke 调用后端 → _parse_response 提取内容/用量/
       思考内容组装 LLMResponse；
    3. 异常期：调用失败经 _handle_error 按错误消息关键词分类，
       转抛 shared.exceptions 中的专用异常（超时/认证/限流/
       连接）；响应解析失败由 _parse_response 转抛
       LLMResponseError。

副作用：
    - complete/complete_sync 发起外部 LLM API 网络请求（经
      LangChain 后端）；
    - _get_llm 首次调用时写实例缓存 _llm（模块进程内持久）；
    - 后端构造依赖延迟导入（langchain_openai 等，缺失时抛
      带安装指引的 ImportError）。

主要依赖：
    - 标准库：time（延迟计时）、typing；
    - app.ics_ai.shared.llm.base_client：LLMClient 基类与
      LLMResponse DTO；
    - app.ics_ai.shared.exceptions：五类 LLM 专用异常；
    - 延迟导入：langchain_openai / langchain_community /
      langchain_anthropic / langchain_core（按所选后端加载）。
"""

# 导入时间工具：complete/complete_sync 的延迟计时（time.time 差值）。
import time

# 导入类型注解工具：Any 类型通配符（PEP 585/604 新式注解下
# Dict/List/Optional 已不需要）。
from typing import Any

# 从基类子模块导入：LLMClient 抽象基类（本类的父类契约）与
# LLMResponse 响应 DTO（补全产出的载体）。
from app.ics_ai.shared.llm.base_client import LLMClient, LLMResponse

# 从异常子模块导入：五类 LLM 专用异常（_handle_error/_parse_response
# 的转换目标，定义于 shared.exceptions）。
from app.ics_ai.shared.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)


class LangChainClient(LLMClient):
    """统一 LLM 客户端。

    职责：经 LangChain 框架封装多后端 LLM 调用——把基类的抽象
          契约落地为四类型分发（openai/qwen/azure/anthropic）、
          统一消息格式转换、统一响应解析与统一错误分类。
    设计模式：适配器（LangChain 后端 → LLMClient 契约）+ 工厂
              方法（_create_*_llm 系列）+ 惰性初始化（_llm 缓存）。
    生命周期：经工厂或直接构造（类型非法即抛 ValueError）创建，
              随装配方长期持有；后端实例首调时懒创建并缓存。

    使用示例：
        # OpenAI API
        >>> client = LangChainClient(
        ...     type="openai",
        ...     model="gpt-4",
        ...     api_key="sk-xxx",
        ... )

        # 阿里云DashScope (支持thinking模式)
        >>> client = LangChainClient(
        ...     type="qwen",
        ...     model="qwen-plus",
        ...     api_key="sk-xxx",
        ...     enable_thinking=True,
        ... )

        # vLLM自部署 (OpenAI兼容接口，支持thinking模式)
        >>> client = LangChainClient(
        ...     type="openai",
        ...     model="Qwen/Qwen3-8B",
        ...     api_base="http://localhost:8000/v1",
        ...     api_key="EMPTY",
        ...     enable_thinking=True,  # vLLM需要启动时加 --enable-reasoning
        ... )
    """

    # 支持的类型：类型白名单（构造期校验依据，_create_llm 的分发
    # 键集合；列表类属性随类共享）。
    SUPPORTED_TYPES = ["openai", "qwen", "azure", "anthropic"]

    def __init__(
        self,
        type: str = "openai",
        model: str = "gpt-3.5-turbo",
        api_key: str = "",
        api_base: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 30,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> None:
        """初始化 LLM 客户端。

        业务流程：六核心参数上交基类持有 → 本类扩展三字段（type
                  归一化小写 / enable_thinking 开关 / _llm 懒加载
                  占位）→ 类型白名单校验（非法即抛 ValueError）。

        副作用：无网络请求（后端实例延迟到 _get_llm 建立）；
                非法 type 直接抛异常。

        参数：
            type: LLM 类型 (openai/qwen/azure/anthropic)
                - openai: OpenAI API 或兼容服务(如 vLLM)
                - qwen: 阿里云 DashScope 通义千问
                - azure: Azure OpenAI
                - anthropic: Anthropic Claude
            model: 模型名称。
            api_key: API 密钥。
            api_base: 自定义 API 基础 URL（用于 vLLM 等自部署服务），
                      允许 None。
            temperature: 温度参数（采样随机度）。
            max_tokens: 最大生成 Token 数。
            timeout: 请求超时（单位秒）。
            enable_thinking: 启用深度思考/推理模式
                - qwen 类型：DashScope 原生支持
                - openai 类型 + api_base：通过 chat_template_kwargs
                  传递（vLLM）
            **kwargs: 额外配置参数（如 azure_endpoint、api_version
                      等，收集进基类 extra_config）。

        异常：type 不在白名单时抛 ValueError（含支持类型清单）。
        """
        # 核心配置上交基类：model/api_key/api_base/temperature/
        # max_tokens/timeout 与 extra_config（**kwargs）由基类持有。
        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs,
        )

        # 类型归一化：lower 兼容 "OpenAI" 等大小写变体（分发与
        # 校验统一用小写键）。
        self.type = type.lower()
        # 深度思考开关：qwen 原生 / openai+vLLM 透传（见各工厂）。
        self.enable_thinking = enable_thinking
        # 懒加载占位：None 表示后端实例未创建（首调 _get_llm 时建）。
        self._llm = None

        # 验证类型：白名单外直接拒绝（fail-fast——构造期暴露配置
        # 错误，避免留到调用期）。
        if self.type not in self.SUPPORTED_TYPES:
            # 抛带支持清单的异常（便于排查拼写问题）。
            raise ValueError(
                f"不支持的LLM类型: {self.type}。"
                f"支持的类型: {', '.join(self.SUPPORTED_TYPES)}"
            )

    def _get_llm(self):
        """获取 LangChain LLM 实例（懒加载）。

        业务流程：缓存命中直接返回；未创建（None）时经 _create_llm
                  建实例并写缓存。

        副作用：首次调用写实例缓存 _llm（进程内持久，后续调用
                零构建开销）。

        返回：
            LangChain Chat 模型实例（具体类型随所选后端）。
        """
        # 缓存判定：未创建才走工厂（惰性初始化模式）。
        if self._llm is None:
            # 经类型分发工厂创建后端实例。
            self._llm = self._create_llm()
        # 返回缓存实例。
        return self._llm

    def _create_llm(self):
        """创建 LangChain LLM 实例。

        业务流程：按 self.type 四路分发到对应工厂方法；else 兜底
                  分支理论不可达（构造期已校验白名单），保留为
                  防御性兜底（默认走 openai 工厂）。

        副作用：触发对应后端的延迟导入（依赖缺失抛 ImportError）。

        返回：
            LangChain Chat 模型实例。
        """
        # 四路分发：类型键逐一匹配对应工厂。
        if self.type == "openai":
            return self._create_openai_llm()
        elif self.type == "qwen":
            return self._create_qwen_llm()
        elif self.type == "azure":
            return self._create_azure_llm()
        elif self.type == "anthropic":
            return self._create_anthropic_llm()
        else:
            # 防御性兜底：白名单校验保证不会走到这里，默认 openai。
            return self._create_openai_llm()

    def _create_openai_llm(self):
        """创建 ChatOpenAI 实例。

        支持：
        - OpenAI 官方 API
        - vLLM 等 OpenAI 兼容服务（通过 api_base 配置）
        - vLLM 的 thinking 模式（通过 extra_body 传递
          chat_template_kwargs）

        业务流程：延迟导入 ChatOpenAI（缺失抛带安装指引的
                  ImportError）→ 组装五要素参数 → api_base 映射为
                  base_url →（api_base + thinking 同开时）注入
                  vLLM 专用 extra_body → 构造实例。

        副作用：无网络请求（仅构造配置对象）。

        返回：
            ChatOpenAI 实例。
        """
        try:
            # 延迟导入：仅在真正使用 openai 后端时加载依赖。
            from langchain_openai import ChatOpenAI
        except ImportError:
            # 依赖缺失：包装为带安装指引的异常（清华镜像加速）。
            raise ImportError(
                "使用OpenAI类型需要安装langchain-openai包。"
                "请运行: pip install langchain-openai -i https://pypi.tuna.tsinghua.edu.cn/simple"
            )

        # 组装基础参数：五要素一一对应基类配置。
        # 显式注解 dict[str, Any]：basedpyright 对无注解字典会按
        # 字面量收窄为 dict[str, str | float | int]，** 解包后所有
        # 实参变成联合类型，导致构造调用全线 reportArgumentType 误报。
        llm_kwargs: dict[str, Any] = {
            "model": self.model,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }

        # 自定义API地址（用于vLLM等）：api_base 的 LangChain 参数
        # 名是 base_url（与基类字段名不同，需显式映射）。
        if self.api_base:
            # 键名转换后并入参数。
            llm_kwargs["base_url"] = self.api_base

        # vLLM thinking模式支持
        # 当使用自定义api_base且启用thinking时，通过extra_body传递配置
        # vLLM需要启动时加 --enable-reasoning --reasoning-parser qwen3
        # （双条件：官方 API 的 openai 类型无此通道，仅自部署场景）。
        if self.api_base and self.enable_thinking:
            # extra_body 经 model_kwargs 注入请求体——vLLM 据此
            # 开启推理模板。
            llm_kwargs["model_kwargs"] = {
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}
            }

        # 构造并返回 ChatOpenAI 实例。
        return ChatOpenAI(**llm_kwargs)

    def _create_qwen_llm(self):
        """创建 ChatTongyi 实例（阿里云 DashScope 通义千问）。

        原生支持 enable_thinking 深度思考模式。

        业务流程：延迟导入 ChatTongyi → 组装参数（密钥参数名是
                  dashscope_api_key）→ thinking 开启时经
                  model_kwargs 透传（DashScope 原生通道）→ 构造。

        副作用：无网络请求（仅构造配置对象）。

        返回：
            ChatTongyi 实例。
        """
        try:
            # 延迟导入：qwen 后端位于 langchain-community。
            from langchain_community.chat_models import ChatTongyi
        except ImportError:
            # 依赖缺失：附带 dashscope SDK 的安装指引。
            raise ImportError(
                "使用Qwen类型需要安装langchain-community包。"
                "请运行: pip install langchain-community dashscope -i https://pypi.tuna.tsinghua.edu.cn/simple"
            )

        # 组装参数：密钥键名为 dashscope_api_key（DashScope 约定）。
        # 显式注解 dict[str, Any]：basedpyright 对无注解字典会按
        # 字面量收窄为 dict[str, str | float | int]，** 解包后所有
        # 实参变成联合类型，导致构造调用全线 reportArgumentType 误报。
        llm_kwargs: dict[str, Any] = {
            "model": self.model,
            "dashscope_api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # DashScope原生支持thinking模式：经 model_kwargs 直传开关。
        if self.enable_thinking:
            # 原生通道注入。
            llm_kwargs["model_kwargs"] = {"enable_thinking": True}

        # 构造并返回 ChatTongyi 实例。
        return ChatTongyi(**llm_kwargs)

    def _create_azure_llm(self):
        """创建 AzureChatOpenAI 实例。

        业务流程：延迟导入 → 组装四要素参数 → 从 extra_config
                  读取 Azure 特有两键（azure_endpoint/api_version，
                  经构造 **kwargs 传入）条件透传 → 构造。

        副作用：无网络请求（仅构造配置对象）。

        返回：
            AzureChatOpenAI 实例。
        """
        try:
            # 延迟导入：Azure 变体同在 langchain-openai 包。
            from langchain_openai import AzureChatOpenAI
        except ImportError:
            # 依赖缺失：带安装指引。
            raise ImportError(
                "使用Azure类型需要安装langchain-openai包。"
                "请运行: pip install langchain-openai -i https://pypi.tuna.tsinghua.edu.cn/simple"
            )

        # 组装基础参数：四要素（timeout 由 SDK 默认处理，不透传）。
        # 显式注解 dict[str, Any]：basedpyright 对无注解字典会按
        # 字面量收窄为 dict[str, str | float | int]，** 解包后所有
        # 实参变成联合类型，导致构造调用全线 reportArgumentType 误报。
        llm_kwargs: dict[str, Any] = {
            "model": self.model,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Azure特定配置：两键存在才透传（部署端点与 API 版本）。
        if "azure_endpoint" in self.extra_config:
            # 端点地址（Azure 资源的专属 URL）。
            llm_kwargs["azure_endpoint"] = self.extra_config["azure_endpoint"]
        if "api_version" in self.extra_config:
            # API 版本（Azure 按版本发布功能）。
            llm_kwargs["api_version"] = self.extra_config["api_version"]

        # 构造并返回 AzureChatOpenAI 实例。
        return AzureChatOpenAI(**llm_kwargs)

    def _create_anthropic_llm(self):
        """创建 ChatAnthropic 实例。

        业务流程：延迟导入 → 直传四要素参数构造（无额外通道）。

        副作用：无网络请求（仅构造配置对象）。

        返回：
            ChatAnthropic 实例。
        """
        try:
            # 延迟导入：anthropic 后端独立包。
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            # 依赖缺失：带安装指引。
            raise ImportError(
                "使用Anthropic类型需要安装langchain-anthropic包。"
                "请运行: pip install langchain-anthropic -i https://pypi.tuna.tsinghua.edu.cn/simple"
            )

        # 构造传参用 pydantic alias（model_name/max_tokens_to_sample）：
        # 与字段名（model/max_tokens）运行时双通道等价（实测值均正确
        # 传入），而 basedpyright 按 alias 生成签名——用 alias 可消
        # 除"参数不存在/缺少必填"误报。
        # pyright 误报压制：basedpyright 把 default_request_timeout/
        # stop_sequences（均有默认值）误判为必填，报"缺少传值"——
        # 属库签名解析怪癖，压制不改可执行代码。
        return ChatAnthropic(  # pyright: ignore[reportCallIssue]
            model_name=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens_to_sample=self.max_tokens,
        )

    def _convert_messages(self, messages: list[dict[str, str]]) -> list[Any]:
        """转换消息格式为 LangChain 格式。

        业务流程：延迟导入三类消息对象 → 逐条按 role 映射——
                  system → SystemMessage、assistant → AIMessage、
                  其余（含缺省与 user）→ HumanMessage。

        副作用：无（纯转换，构造消息对象列表）。

        参数：
            messages: 标准消息列表 [{"role": "user", "content":
                      "..."}]。

        返回：
            LangChain 消息对象列表（list[Any]，元素为
            SystemMessage/AIMessage/HumanMessage 之一）。
        """
        try:
            # 延迟导入：langchain-core 是消息协议的最小依赖。
            from langchain_core.messages import (
                AIMessage,
                HumanMessage,
                SystemMessage,
            )
        except ImportError:
            # 依赖缺失：带安装指引。
            raise ImportError(
                "需要安装langchain-core包。"
                "请运行: pip install langchain-core -i https://pypi.tuna.tsinghua.edu.cn/simple"
            )

        # 结果列表：逐条转换后收集。
        result = []
        # 遍历标准消息字典。
        for msg in messages:
            # 读取角色（缺省按用户消息处理）与内容（缺省空串）。
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # 三路映射：role 决定 LangChain 消息类型。
            if role == "system":
                # 系统提示 → SystemMessage。
                result.append(SystemMessage(content=content))
            elif role == "assistant":
                # 助手历史 → AIMessage。
                result.append(AIMessage(content=content))
            else:  # user
                # 用户输入（及未知角色兜底）→ HumanMessage。
                result.append(HumanMessage(content=content))

        # 返回转换后的消息对象列表。
        return result

    async def complete(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """异步生成文本补全（基类契约实现）。

        业务流程：懒加载后端实例 → 消息格式转换 → 记起始时间 →
                  ainvoke 异步调用（失败经 _handle_error 分类转抛，
                  其必抛异常故调用后控制流不会继续）→ 计算延迟 →
                  解析为 LLMResponse。

        副作用：发起外部 LLM API 网络请求。

        参数：
            messages: 消息列表 [{"role": "user", "content": "..."}]。
            **kwargs: 额外 API 参数（当前透传未用，保留扩展位）。

        返回：
            LLMResponse 对象（content/usage/latency 等已填充）。

        异常：LLMTimeoutError/LLMAuthenticationError/
              LLMRateLimitError/LLMConnectionError（_handle_error
              分类）；LLMResponseError（解析失败）。
        """
        # 懒加载后端实例（首次调用时才构建）。
        llm = self._get_llm()
        # 标准消息转 LangChain 消息对象。
        langchain_messages = self._convert_messages(messages)

        # 记录起始时间（延迟统计基准）。
        start_time = time.time()

        try:
            # 异步调用后端补全。
            response = await llm.ainvoke(langchain_messages)
        except Exception as e:
            # 错误分类转换：_handle_error 内必抛专用异常（不会
            # 返回 None），故此处无需 return，控制流必然中断。
            self._handle_error(e)

        # 计算请求延迟（秒）。
        latency = time.time() - start_time

        # pyright 误报压制：_handle_error 运行时必抛专用异常（上方
        # except 分支控制流必然中断），response 必已绑定；但签名
        # 注解为 None 无法向检查器表达该契约（NoReturn 方案按用户
        # 决定不采用），故压制不改可执行代码。
        return self._parse_response(response, latency)  # pyright: ignore[reportPossiblyUnboundVariable]

    def complete_sync(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """同步生成文本补全（基类契约实现）。

        业务流程：与 complete 同构，唯一差异是调用后端的同步
                  invoke（阻塞当前线程直至响应）。

        副作用：发起外部 LLM API 网络请求（同步阻塞）。

        参数：
            messages: 消息列表 [{"role": "user", "content": "..."}]。
            **kwargs: 额外 API 参数（当前透传未用，保留扩展位）。

        返回：
            LLMResponse 对象。

        异常：同 complete 的四类分类异常与解析异常。
        """
        # 懒加载后端实例。
        llm = self._get_llm()
        # 标准消息转 LangChain 消息对象。
        langchain_messages = self._convert_messages(messages)

        # 记录起始时间。
        start_time = time.time()

        try:
            # 同步调用后端补全（阻塞）。
            response = llm.invoke(langchain_messages)
        except Exception as e:
            # 错误分类转换（_handle_error 必抛，控制流中断）。
            self._handle_error(e)

        # 计算请求延迟（秒）。
        latency = time.time() - start_time

        # pyright 误报压制：_handle_error 运行时必抛专用异常（上方
        # except 分支控制流必然中断），response 必已绑定；但签名
        # 注解为 None 无法向检查器表达该契约（NoReturn 方案按用户
        # 决定不采用），故压制不改可执行代码。
        return self._parse_response(response, latency)  # pyright: ignore[reportPossiblyUnboundVariable]

    def _parse_response(self, response: Any, latency: float) -> LLMResponse:
        """解析 LangChain 响应。

        业务流程：提取文本（content 属性存在则取，否则整体转串）
                  → 提取 Token 用量（response_metadata 的
                  token_usage 三键）→ 提取思考内容（additional_kwargs
                  的三个候选键按序探测，命中即止）→ 组装 LLMResponse；
                  全程异常包装为 LLMResponseError。

        副作用：无（纯解析构造）。

        参数：
            response: LangChain 响应对象（或任意可转串对象）。
            latency: 请求延迟（单位秒，调用方计时所得）。

        返回：
            LLMResponse 对象（raw_response 保留原始对象）。

        异常：解析失败时抛 LLMResponseError（含原始异常信息）。
        """
        try:
            # 提取文本：content 属性存在取之，否则整体转字符串兜底。
            content = (
                response.content if hasattr(response, "content") else str(response)
            )

            # 提取Token用量信息：默认空字典（后端未上报则全 0 兜底）。
            usage = {}
            # 用量挂载在 response_metadata 下（LangChain 约定）。
            if hasattr(response, "response_metadata"):
                # 取元数据字典。
                resp_metadata = response.response_metadata
                # token_usage 键存在才提取（不同后端可能缺失）。
                if "token_usage" in resp_metadata:
                    # 取用量字典。
                    token_usage = resp_metadata["token_usage"]
                    # 三键映射为 LLMResponse.usage 的标准键。
                    usage = {
                        "prompt_tokens": token_usage.get("prompt_tokens", 0),
                        "completion_tokens": token_usage.get("completion_tokens", 0),
                        "total_tokens": token_usage.get("total_tokens", 0),
                    }

            # 提取thinking内容（深度思考/推理内容）
            metadata = {}
            # 思考内容挂载在 additional_kwargs 下。
            if hasattr(response, "additional_kwargs"):
                # 取附加参数字典。
                additional = response.additional_kwargs
                # Qwen/vLLM的推理内容可能在不同字段：三候选键按序
                # 探测，首个命中即写入并停止（优先级从左到右）。
                for key in ["reasoning_content", "thinking_content", "thinking"]:
                    # 命中判定：键存在才写入。
                    if key in additional:
                        # 统一写入 thinking_content 键（对外契约统一）。
                        metadata["thinking_content"] = additional[key]
                        # 命中即止（不叠加多来源）。
                        break

            # 组装统一响应 DTO（raw_response 保留原始对象供调试）。
            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                latency=latency,
                raw_response=response,
                metadata=metadata,
            )
        except Exception as e:
            # 解析失败：包装为专用异常（保留原始错误信息）。
            raise LLMResponseError(f"解析响应失败: {e}")

    def _handle_error(self, error: Exception) -> None:
        """处理 API 错误（分类转换后必抛异常，不返回）。

        业务流程：按原始错误消息的关键词四路分类——含 "timeout"
                  或 "timed out" → 超时异常；含 "auth"/"key" →
                  认证异常；含 "rate" → 限流异常；其余 → 连接异常
                  （附错误类型类名）。

        副作用：无外部副作用（异常转换）。

        参数：
            error: 原始异常（LangChain/后端 SDK 抛出的任意异常）。

        异常：本方法必然抛出下列之一（调用处控制流就此中断）——
              LLMTimeoutError/LLMAuthenticationError/
              LLMRateLimitError/LLMConnectionError。
        """
        # 错误消息全文（关键词分类依据）与异常类名（兜底分支留痕）。
        error_message = str(error)
        error_type = type(error).__name__

        # 分类一：消息含 "timeout" 或 "timed out"（不区分大小写）
        # → 超时。双模式匹配：openai SDK 的真实超时消息是
        # "Request timed out."，不含字面 "timeout"，仅匹配单词
        # 会把它漏到兜底连接异常（2026-10-17 修复）。
        if "timeout" in error_message.lower() or "timed out" in error_message.lower():
            # 抛超时异常。
            raise LLMTimeoutError(f"请求超时: {error_message}")
        # 分类二：含 "auth" 或 "key" → 认证（密钥无效/缺失）。
        elif "auth" in error_message.lower() or "key" in error_message.lower():
            # 抛认证异常。
            raise LLMAuthenticationError(f"认证失败: {error_message}")
        # 分类三：含 "rate" → 限流（速率/配额超限）。
        elif "rate" in error_message.lower():
            # 抛限流异常。
            raise LLMRateLimitError(f"速率限制: {error_message}")
        else:
            # 兜底：无法识别的错误统一归为连接异常（附原始类名）。
            raise LLMConnectionError(f"请求失败({error_type}): {error_message}")
