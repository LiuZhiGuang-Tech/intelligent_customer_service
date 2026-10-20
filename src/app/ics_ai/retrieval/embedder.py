"""
embedder - 文本向量化模块

核心功能：提供文本 → 向量的嵌入转换能力——EmbedderConfig 承载配置、
          Embedder 抽象基类定义 embed / embed_batch 契约及同步适配方法、
          OpenAIEmbedder 走 OpenAI 兼容协议（默认硅基流动托管的
          BAAI/bge-m3）、LocalEmbedder 走本地 sentence-transformers 模型
          （依赖缺失时降级为随机向量）；create_embedder 工厂函数按
          provider 装配具体实现。

执行流程：
    1. 使用方通过 create_embedder(provider, **kwargs) 创建向量化器；
    2. 异步链路：await embed(text) 单条转换 / await embed_batch(texts)
       批量转换（OpenAI 实现内部按 batch_size 分批调用 API）；
    3. 同步链路：embed_sync / embed_batch_sync 在当前线程事件循环上
       驱动异步实现（循环缺失时新建并安装）；
    4. 归一化：normalize=True 时对每个向量做 L2 归一化（零范数跳过），
       保证余弦相似度计算的数值稳定性。

副作用：
    - 模块导入：执行 load_dotenv() 把项目根目录 .env 注入进程环境变量
      （幂等，不覆盖已存在的变量）；numpy 为导入期第三方依赖。
    - OpenAIEmbedder：惰性创建 AsyncOpenAI 客户端（默认指向硅基流动
      SILICONFLOW_BASE_URL 端点）；embed/embed_batch 调用硅基流动
      embeddings 外部服务（网络请求）。
    - LocalEmbedder：惰性加载 sentence-transformers 本地模型（读磁盘 +
      可能下载模型）；sentence-transformers 未安装时降级为随机向量
      （记录 warning 日志）。
    - embed_sync / embed_batch_sync：事件循环缺失时会新建循环并
      asyncio.set_event_loop 安装到当前线程（改变线程循环绑定状态）。

主要依赖：
    - os / abc / dataclasses / logging / asyncio：标准库基础能力。
    - typing：Any 通配符（类型注解已按 PEP 585/604 新式写法）。
    - dotenv：load_dotenv 读取项目根目录 .env（模块导入期执行）。
    - numpy：向量数值计算（数组构造、L2 范数、归一化）。
    - openai（惰性）：AsyncOpenAI 客户端，仅 OpenAIEmbedder 使用；
      通过 base_url 对接硅基流动等 OpenAI 兼容服务。
    - sentence_transformers（惰性）：本地嵌入模型，仅 LocalEmbedder 使用，
      缺失时随机降级。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，支持前向引用
# 并消除运行时注解求值开销。
from __future__ import annotations

# 导入 asyncio：同步适配方法（embed_sync 等）内部驱动协程使用。
import asyncio

# 导入日志器工厂：为本模块创建命名 logger（降级警告走此通道）。
import logging

# 导入 os：读取 .env 注入的环境变量（SILICONFLOW_* / EMBEDDING_MODEL_PATH）。
import os

# 导入抽象基类工具：ABC 标记向量化器基类，abstractmethod 强制子类
# 实现 embed / embed_batch 两个核心方法。
from abc import ABC, abstractmethod

# 导入 dataclass 工具：EmbedderConfig 靠它自动生成 __init__/__repr__；
# field(default_factory=...) 为可变默认值（dict）提供独立工厂。
from dataclasses import dataclass, field

# 导入类型注解工具：Any 类型通配符（extra 配置与工厂 kwargs 用）。
from typing import Any

# 导入 numpy：向量数值计算——数组构造 np.array、L2 范数 np.linalg.norm、
# 随机降级 np.random.randn、dtype 转换 astype。
import numpy as np

# 导入 dotenv 的加载函数：把项目根目录 .env 文件注入进程环境变量。
from dotenv import load_dotenv

# 模块导入时加载 .env：幂等操作（已存在的环境变量不会被覆盖）；
# 使 EmbedderConfig 的环境变量默认值工厂能读到 SILICONFLOW_* 等配置。
load_dotenv()

# 本模块命名 logger：LocalEmbedder 的降级警告由此输出。
logger = logging.getLogger(__name__)


@dataclass
class EmbedderConfig:
    """向量化器配置。

    职责：承载向量化器的通用配置项，供 Embedder 及其子类读取。
    设计模式：数据传输对象（DTO），无行为逻辑。
    生命周期：随向量化器创建时构造（子类 __init__ 未传入时按
              provider 补建默认值），与所属实例同生命周期。

    Attributes:
        provider: 提供商标识 (openai, local, custom)；
                  openai 泛指 OpenAI 兼容协议（含硅基流动等第三方服务）。
        model: 模型名称；默认 BAAI/bge-m3（硅基流动托管，向量维度 1024）。
        api_key: API 密钥；默认读环境变量 SILICONFLOW_API_KEY，
                 未设置时为 None（本地模型不需要密钥）。
        api_base: API 服务基地址（OpenAI 兼容协议端点）；
                  默认读环境变量 SILICONFLOW_BASE_URL（硅基流动 v1 端点），
                  None 时 AsyncOpenAI 走 OpenAI 官方默认地址。
        dimension: 向量维度；默认 1024（BAAI/bge-m3 的输出维度）。
        batch_size: 批处理大小（embed_batch 单次请求的最大文本条数）。
        normalize: 是否做 L2 归一化（True 保证余弦相似度可直接用点积）。
        extra: 扩展配置字典（自由键值对，供特定实现读取）。
    """

    # 提供商标识：默认 openai；create_embedder 按它分发实现类
    # （硅基流动走 OpenAI 兼容协议，provider 仍为 openai）。
    provider: str = "openai"
    # 嵌入模型名：默认硅基流动托管的 BAAI/bge-m3（1024 维）。
    model: str = "BAAI/bge-m3"
    # API 密钥：default_factory 在实例化时读环境变量（而非导入时求值一次），
    # 保证 .env 加载时序正确；未设置时为 None。
    api_key: str | None = field(
        default_factory=lambda: os.getenv("SILICONFLOW_API_KEY")
    )
    # API 基地址：同上读 SILICONFLOW_BASE_URL；None 时用 OpenAI 官方默认。
    api_base: str | None = field(
        default_factory=lambda: os.getenv("SILICONFLOW_BASE_URL")
    )
    # 向量维度：默认 1024（BAAI/bge-m3 的输出维度，需与向量库配置一致）。
    dimension: int = 1024
    # 批处理大小：默认 100（embed_batch 分批上限）。
    batch_size: int = 100
    # 归一化开关：默认 True。
    normalize: bool = True
    # 扩展配置：可变默认值经 field 工厂独立创建，避免实例间共享。
    extra: dict[str, Any] = field(default_factory=dict)


class Embedder(ABC):
    """向量化器抽象基类。

    职责：将文本转换为向量表示；定义嵌入能力的最小契约——
          embed（单条）/ embed_batch（批量）两个异步抽象方法、
          embed_sync / embed_batch_sync 同步适配层，以及 dimension
          property 对配置中向量维度的统一暴露。
    设计模式：抽象基类 + 模板方法变体（异步方法为强制契约，同步方法
              为通用适配层，子类无需重复实现）。
    生命周期：由 create_embedder 或调用方直接构造 → 惰性资源
              （客户端/模型）首次 embed 时初始化 → 服务存续期复用。

    Attributes:
        config: 向量化器配置实例（EmbedderConfig）。
    """

    def __init__(self, config: EmbedderConfig | None = None):
        """初始化向量化器。

        业务流程：config 为 None 时兜底新建默认配置，保证 self.config
                  始终可用（后续 dimension property 等依赖它）。

        副作用：仅写入实例属性，无 IO 操作。

        参数：
            config: 向量化器配置；None 时使用 EmbedderConfig() 默认值。
        """
        # 配置兜底：避免后续访问 self.config.dimension 时触发 None 错误。
        self.config = config or EmbedderConfig()

    @property
    def dimension(self) -> int:
        """向量维度。

        返回：
            配置中的 dimension 值（调用方据此校验返回向量的形状）。
        """
        # 从配置对象读取，便于运行期调整配置后即时生效。
        return self.config.dimension

    @abstractmethod
    async def embed(self, text: str) -> np.ndarray:
        """将单个文本转换为向量。

        业务流程：由子类实现——调用具体嵌入服务/模型，返回一维向量；
                  normalize=True 时做 L2 归一化。
        副作用：由子类决定（OpenAI 网络请求 / 本地模型推理）。

        参数：
            text: 输入文本；空串行为由子类决定（建议调用方先行拦截）。

        返回：
            一维 np.ndarray 向量（float 类型，形状 [dimension]）；
            子类实现失败时应抛异常（而非静默返回零向量）。
        """
        # 兜底保护：子类未覆盖时抛出明确错误。
        raise NotImplementedError()

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """批量将文本转换为向量。

        业务流程：由子类实现——按 batch_size 分批调用嵌入服务，
                  返回与输入等长且顺序对应的向量列表。
        副作用：由子类决定（批量网络请求 / 批量本地推理）。

        参数：
            texts: 输入文本列表；空列表时子类应返回空列表。

        返回：
            np.ndarray 向量列表（顺序与输入一一对应）。
        """
        # 兜底保护：子类未覆盖时抛出明确错误。
        raise NotImplementedError()

    def embed_sync(self, text: str) -> np.ndarray:
        """同步版本的embed。

        业务流程：1. 尝试获取当前线程事件循环；2. 获取失败（RuntimeError，
                  线程尚无循环）时新建循环并安装；3. run_until_complete
                  驱动异步 embed 直到返回。

        副作用：事件循环缺失时会创建新循环并安装到当前线程；
                驱动的 embed 实现可能产生网络/模型推理副作用。

        参数：
            text: 输入文本，透传给 embed。

        返回：
            np.ndarray 向量（即异步 embed 的返回值）；
            当前线程已有正在运行的事件循环时会抛 RuntimeError
            （本方法仅适合同步上下文调用）。
        """
        try:
            # 优先复用当前线程已有事件循环（避免反复创建泄漏资源）。
            loop = asyncio.get_event_loop()
        # 捕获"当前线程无事件循环"异常（Python 3.10+ 行为）。
        except RuntimeError:
            # 新建事件循环。
            loop = asyncio.new_event_loop()
            # 安装为当前线程循环，使后续调用复用同一循环。
            asyncio.set_event_loop(loop)
        # 阻塞驱动异步 embed 并取回向量结果。
        return loop.run_until_complete(self.embed(text))

    def embed_batch_sync(self, texts: list[str]) -> list[np.ndarray]:
        """同步版本的embed_batch。

        业务流程：与 embed_sync 相同——复用/新建线程事件循环后
                  run_until_complete 驱动异步 embed_batch。

        副作用：同 embed_sync（可能新建并安装事件循环；驱动的批量
                实现可能产生网络/推理副作用）。

        参数：
            texts: 输入文本列表，透传给 embed_batch。

        返回：
            np.ndarray 向量列表（即异步 embed_batch 的返回值）。
        """
        try:
            # 优先复用当前线程已有事件循环。
            loop = asyncio.get_event_loop()
        # 捕获"当前线程无事件循环"异常。
        except RuntimeError:
            # 新建事件循环。
            loop = asyncio.new_event_loop()
            # 安装为当前线程循环。
            asyncio.set_event_loop(loop)
        # 阻塞驱动异步批量嵌入并取回结果。
        return loop.run_until_complete(self.embed_batch(texts))


class OpenAIEmbedder(Embedder):
    """OpenAI 向量化器。

    职责：基于 AsyncOpenAI 客户端调用 OpenAI Embeddings 协议服务
          （默认对接硅基流动托管的 BAAI/bge-m3），实现单条与批量嵌入；
          批量按配置的 batch_size 分批请求。
    设计模式：继承 Embedder 模板；客户端惰性初始化
              （Lazy Initialization）——首次 embed 才创建连接。
    生命周期：构造时仅保存配置 → 首次嵌入时创建 _client →
              服务存续期复用同一客户端连接池。

    Attributes:
        _client: 惰性创建的 AsyncOpenAI 客户端；None 表示尚未初始化。
    """

    def __init__(self, config: EmbedderConfig | None = None):
        """初始化OpenAI向量化器。

        业务流程：config 为 None 时按 openai provider 补建默认配置，
                  再调用基类 __init__ 保存；客户端不在此创建（惰性）。

        副作用：仅写入实例属性，无 IO 操作。

        参数：
            config: 向量化器配置；None 时使用 openai 默认配置。
        """
        # 默认配置补建：确保 provider 标识为 openai。
        if config is None:
            config = EmbedderConfig(provider="openai")
        # 调用基类 __init__：保存配置（None 兜底已提前处理）。
        super().__init__(config)
        # 客户端占位：None 表示尚未创建，由 _get_client 惰性初始化。
        self._client = None

    def _get_client(self):
        """获取或创建OpenAI客户端。

        业务流程：_client 为 None 时局部导入 openai 包并创建
                  AsyncOpenAI 客户端（注入配置中的 api_key）；
                  openai 未安装时抛出带说明的 ImportError。

        副作用：首次调用时创建客户端实例（写 self._client）；
                openai 包的导入涉及磁盘/字节码缓存读取。

        返回：
            AsyncOpenAI 客户端实例（缓存复用）；
            openai 包缺失时抛 ImportError。
        """
        # 惰性初始化判断：尚未创建过客户端时才执行导入与构造。
        if self._client is None:
            try:
                # 局部导入 openai：仅使用 OpenAIEmbedder 时才要求该依赖。
                from openai import AsyncOpenAI

                # 创建异步客户端：密钥取自配置；base_url 指向 OpenAI 兼容
                # 端点（默认硅基流动 SILICONFLOW_BASE_URL），None 时
                # AsyncOpenAI 自动走 OpenAI 官方默认地址。
                self._client = AsyncOpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.api_base,
                )
            # openai 包未安装 → 给出明确的依赖缺失提示。
            except ImportError:
                raise ImportError("openai package required for OpenAIEmbedder")
        # 返回缓存实例（首次返回新建，之后返回同一对象）。
        return self._client

    async def embed(self, text: str) -> np.ndarray:
        """将文本转换为向量。

        业务流程：1. 获取（或创建）AsyncOpenAI 客户端；2. 调用
                  embeddings.create（单条文本输入）；3. np.array 包装
                  API 返回的嵌入列表；4. normalize=True 时做 L2 归一化
                  （零范数跳过除法，避免除零）；5. 返回向量。

        副作用：调用 OpenAI 兼容外部服务（网络请求）。

        参数：
            text: 输入文本。

        返回：
            一维 np.ndarray 向量（normalize=True 时模长为 1）；
            API 调用失败时异常向上传播（不静默）。
        """
        # 获取客户端（首次调用时惰性创建）。
        client = self._get_client()

        # 调用 OpenAI Embeddings API：模型与输入取自参数/配置。
        response = await client.embeddings.create(
            model=self.config.model,
            input=text,
        )

        # 将 API 返回的嵌入列表包装为 numpy 数组（float64 默认 dtype）。
        embedding = np.array(response.data[0].embedding)

        # 按配置决定是否归一化。
        if self.config.normalize:
            # 计算 L2 范数（向量模长）。
            norm = np.linalg.norm(embedding)
            # 零向量保护：范数为 0 时跳过除法（避免 NaN/Inf）。
            if norm > 0:
                # 原地除以模长，得到单位向量。
                embedding = embedding / norm

        # 返回（可能已归一化的）嵌入向量。
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """批量将文本转换为向量。

        业务流程：1. 空列表守卫直接返回空结果；2. 获取客户端；
                  3. 按 batch_size 切片分批调用 API（每批一次请求，
                  控制单请求负载）；4. 每批返回的嵌入逐条 np.array
                  包装并按配置归一化；5. 汇总返回（顺序与输入一致）。

        副作用：多次调用 OpenAI 兼容外部服务（批次数 = ceil(len/batch_size)）。

        参数：
            texts: 输入文本列表；空列表时返回空列表。

        返回：
            np.ndarray 向量列表，顺序与输入一一对应。
        """
        # 空输入守卫：避免无意义的 API 调用。
        if not texts:
            return []

        # 获取客户端（首次调用时惰性创建）。
        client = self._get_client()
        # 结果收集列表：按批次顺序追加，保持与输入的对应关系。
        results = []

        # 分批处理：range 按步长 batch_size 生成各批起始索引。
        for i in range(0, len(texts), self.config.batch_size):
            # 切片取当前批次的文本子集（末批可能不满）。
            batch = texts[i : i + self.config.batch_size]

            # 单批 API 请求：一次调用携带整批文本。
            response = await client.embeddings.create(
                model=self.config.model,
                input=batch,
            )

            # 逐条处理本批返回的嵌入（response.data 顺序与 batch 一致）。
            for item in response.data:
                # 包装为 numpy 数组。
                embedding = np.array(item.embedding)
                # 按配置归一化（与 embed 单条逻辑一致）。
                if self.config.normalize:
                    # 计算 L2 范数。
                    norm = np.linalg.norm(embedding)
                    # 零向量保护。
                    if norm > 0:
                        # 归一化为单位向量。
                        embedding = embedding / norm
                # 追加到结果列表。
                results.append(embedding)

        # 返回全部批次的向量集合（顺序与输入一致）。
        return results


def _default_local_model() -> str:
    """本地模式默认模型名（私有辅助函数）。

    业务流程：优先读环境变量 EMBEDDING_MODEL_PATH（.env 中配置的
              本地 bge-m3 快照目录，离线可用）；未设置时回退
              BAAI/bge-m3（HF 托管的同名模型，首次运行会下载权重）。

    副作用：读取环境变量，无 IO。

    返回：
        模型名或本地路径字符串。
    """
    # 本地路径优先；未配置时回退 HF 托管的 bge-m3。
    return os.getenv("EMBEDDING_MODEL_PATH") or "BAAI/bge-m3"


def _default_local_dimension(model_name: str) -> int:
    """本地模式默认向量维度（私有辅助函数）。

    业务流程：模型名含 "bge-m3" 时返回 1024（BAAI/bge-m3 的输出维度），
              其余小模型（如 MiniLM）返回 384；当前两条默认路径
              （本地快照 / HF 回退）均为 bge-m3 → 恒为 1024，
              函数保留参数以兼容未来更换模型族。

    副作用：无（纯字符串判断）。

    参数：
        model_name: 模型名或本地路径字符串。

    返回：
        向量维度整数（1024 或 384）。
    """
    # 按模型族判定维度：bge-m3 → 1024，其余 → 384。
    return 1024 if "bge-m3" in model_name else 384


class LocalEmbedder(Embedder):
    """本地向量化器。

    职责：基于 sentence-transformers 的本地模型实现单条与批量嵌入
          （适用于离线场景）；sentence-transformers 未安装时优雅降级
          为随机向量（记录 warning，保证链路不中断——适合开发/测试环境）。
    设计模式：继承 Embedder 模板；模型惰性加载；降级策略
              （Degradation）——依赖缺失时以 "random" 哨兵替代模型。
    生命周期：构造时仅保存配置 → 首次嵌入时加载本地模型（读磁盘/
              可能联网下载）→ 服务存续期复用同一模型。

    Attributes:
        _model: 惰性加载的 SentenceTransformer 实例，或降级哨兵
                字符串 "random"（鸭子类型混合状态，embed 时判断）。
    """

    def __init__(self, config: EmbedderConfig | None = None):
        """初始化本地向量化器。

        业务流程：config 为 None 时按 local provider 补建配置——模型
                  路径优先取环境变量 EMBEDDING_MODEL_PATH（.env 中配置的
                  本地 bge-m3 快照目录，离线可用），未设置时回退
                  BAAI/bge-m3（HF 托管，首次运行下载）；维度按模型族
                  自动判定（bge-m3 → 1024，MiniLM → 384）。

        副作用：仅写入实例属性，无 IO 操作。

        参数：
            config: 向量化器配置；None 时使用 local 默认配置。
        """
        # 默认配置补建：local provider + .env 本地模型路径（或轻量回退），
        # 维度按模型族自动判定（bge-m3 → 1024，MiniLM → 384）。
        if config is None:
            # 解析本地默认模型（优先 .env 的 EMBEDDING_MODEL_PATH）。
            local_model = _default_local_model()
            config = EmbedderConfig(
                provider="local",
                model=local_model,
                dimension=_default_local_dimension(local_model),
            )
        # 调用基类 __init__：保存配置。
        super().__init__(config)
        # 模型占位：None 表示尚未加载，由 _get_model 惰性初始化。
        self._model = None

    def _get_model(self):
        """获取或创建模型。

        业务流程：_model 为 None 时局部导入 sentence-transformers 并
                  按配置加载模型（model 为空时用 BAAI/bge-m3 默认）；
                  依赖未安装时记录 warning 并将 _model 置为 "random"
                  哨兵——后续 embed 走随机向量降级。

        副作用：首次调用时加载本地模型（读磁盘，首次运行可能
                联网下载模型权重）；或写入降级哨兵。

        返回：
            SentenceTransformer 实例或字符串 "random"（降级标记）。
        """
        # 惰性初始化判断：尚未加载过模型时才执行。
        if self._model is None:
            try:
                # 局部导入：仅使用 LocalEmbedder 时才要求该依赖。
                from sentence_transformers import SentenceTransformer

                # 模型名兜底：配置未指定时用 bge-m3 默认。
                model_name = self.config.model or "BAAI/bge-m3"
                # 加载本地模型（首次可能下载权重）。
                self._model = SentenceTransformer(model_name)
            # 依赖未安装 → 降级为随机向量模式（记录警告而非崩溃）。
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed, using random embeddings"
                )
                # "random" 哨兵：embed 中据此走随机向量分支。
                self._model = "random"
        # 返回缓存实例或降级哨兵。
        return self._model

    async def embed(self, text: str) -> np.ndarray:
        """将文本转换为向量。

        业务流程：1. 获取（或加载）模型；2. 模型为降级哨兵时返回
                  随机向量（float32，形状 [dimension]）；3. 正常路径
                  model.encode 编码文本；4. 按配置 L2 归一化；5. 返回。

        副作用：模型推理计算；降级路径使用随机数生成器。

        参数：
            text: 输入文本（降级模式下内容被忽略）。

        返回：
            一维 np.ndarray 向量；随机降级向量的模长不为 1
            （normalize 逻辑不作用于随机分支）。
        """
        # 获取模型（首次调用时惰性加载或降级）。
        model = self._get_model()

        # 降级分支：模型为 "random" 哨兵 → 生成随机向量。
        if model == "random":
            # randn 标准正态采样 + float32 类型（与模型输出 dtype 一致）。
            return np.random.randn(self.config.dimension).astype(np.float32)

        # 正常路径：编码文本为 numpy 向量。
        # 行尾压制说明：sentence-transformers 类型存根未声明
        # convert_to_numpy 参数，但运行时真实存在。
        embedding = model.encode(text, convert_to_numpy=True)  # pyright: ignore[reportCallIssue]

        # 按配置归一化（与 OpenAI 实现逻辑一致）。
        if self.config.normalize:
            # 计算 L2 范数。
            norm = np.linalg.norm(embedding)
            # 零向量保护。
            if norm > 0:
                # 归一化为单位向量。
                embedding = embedding / norm

        # 返回（可能已归一化的）嵌入向量。
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """批量将文本转换为向量。

        业务流程：1. 空列表守卫返回空结果；2. 获取模型；3. 降级哨兵时
                  逐条生成随机向量；4. 正常路径批量 encode 后逐条
                  归一化；5. 汇总返回（顺序与输入一致）。

        副作用：批量模型推理；降级路径使用随机数生成器。

        参数：
            texts: 输入文本列表；空列表时返回空列表。

        返回：
            np.ndarray 向量列表，顺序与输入一一对应。
        """
        # 空输入守卫。
        if not texts:
            return []

        # 获取模型（首次调用时惰性加载或降级）。
        model = self._get_model()

        # 降级分支：逐条生成随机向量（列表推导保持等长与顺序）。
        if model == "random":
            return [
                np.random.randn(self.config.dimension).astype(np.float32) for _ in texts
            ]

        # 正常路径：批量编码（一次推理调用，返回二维数组）。
        # 行尾压制说明：同上——存根缺 convert_to_numpy 声明。
        embeddings = model.encode(texts, convert_to_numpy=True)  # pyright: ignore[reportCallIssue]

        # 逐条归一化并收集。
        results = []
        # 遍历二维数组的每一行（每条文本对应的向量）。
        for embedding in embeddings:
            # 按配置归一化。
            if self.config.normalize:
                # 计算 L2 范数。
                norm = np.linalg.norm(embedding)
                # 零向量保护。
                if norm > 0:
                    # 归一化为单位向量。
                    embedding = embedding / norm
            # 追加到结果列表。
            results.append(embedding)

        # 返回全部向量（顺序与输入一致）。
        return results


def create_embedder(
    provider: str = "openai",
    **kwargs: Any,
) -> Embedder:
    """创建向量化器工厂函数。

    核心功能：按 provider 名称装配对应的 Embedder 实现，kwargs 透传
              给 EmbedderConfig 构造配置。
    业务流程：
        1. 以 provider + kwargs 构造 EmbedderConfig；
        2. provider == "openai" → OpenAIEmbedder；
        3. provider == "local" → LocalEmbedder；
        4. 其他值 → ValueError（白名单外提供商直接拒绝）。

    副作用：仅创建配置与实例，无 IO（实际模型/客户端由实例惰性创建）。

    参数：
        provider: 提供商名称 (openai/local)；默认 openai。
        **kwargs: 配置参数，透传给 EmbedderConfig
                  （model / api_key / dimension / batch_size / normalize 等）。

    返回：
        Embedder 实例（OpenAIEmbedder 或 LocalEmbedder）；
        provider 不在白名单时抛 ValueError。
    """
    # 构造统一配置对象：kwargs 中的合法键由 dataclass 自动接收。
    config = EmbedderConfig(provider=provider, **kwargs)

    # 按 provider 分发实现类。
    if provider == "openai":
        # OpenAI 兼容实现（默认对接硅基流动 BAAI/bge-m3）。
        return OpenAIEmbedder(config)
    elif provider == "local":
        # 本地模型实现（离线场景）：工厂路径下 config 已由类默认值构造，
        # 若调用方未显式传 model/dimension，则应用本地默认逻辑
        # （EMBEDDING_MODEL_PATH 优先 + 按模型族判定维度），与直接
        # 构造 LocalEmbedder() 的行为保持一致。
        if "model" not in kwargs:
            # 解析本地默认模型并回填配置。
            local_model = _default_local_model()
            config.model = local_model
            # 维度仅在调用方未指定时按模型族补默认。
            if "dimension" not in kwargs:
                config.dimension = _default_local_dimension(local_model)
        return LocalEmbedder(config)
    else:
        # 白名单外提供商：明确报错，避免静默走错实现。
        raise ValueError(f"Unknown embedder provider: {provider}")


# 导出清单：from embedder import * 时仅暴露以下名字。
__all__ = [
    # 列表项1：向量化器配置数据类。
    "EmbedderConfig",
    # 列表项2：向量化器抽象基类。
    "Embedder",
    # 列表项3：OpenAI 云端实现。
    "OpenAIEmbedder",
    # 列表项4：本地模型实现（含随机降级）。
    "LocalEmbedder",
    # 列表项5：装配工厂函数。
    "create_embedder",
]
