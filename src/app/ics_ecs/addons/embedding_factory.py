"""
embedding_factory.py - 嵌入器统一工厂（线上优先，本地兜底）

核心功能：
    为 GraphRAG 检索器（information_retrieval.py）与建索引脚本（create_indexing.py）
    提供唯一的嵌入器创建入口，统一返回 LangChain Embeddings 接口对象：
    - 线上优先：SILICONFLOW_API_KEY 已配置时，经硅基流动 OpenAI 兼容接口
      调用云端嵌入模型（缺省 BAAI/bge-m3，1024 维），无需本地模型文件；
    - 本地兜底：未配置线上 Key 时，回退 SentenceTransformer 加载本地
      bge-m3 快照（EMBEDDING_MODEL_PATH 指定，1024 维）。

执行流程（create_embedder）：
    1. load_dotenv 加载环境配置（幂等，已加载时无副作用）；
    2. 读取 SILICONFLOW_API_KEY，非空即走线上分支创建 OpenAIEmbeddings；
    3. 否则读取 EMBEDDING_MODEL_PATH 模型路径走本地分支，SentenceTransformer
      加载后经 LocalEmbedder 适配为 LangChain Embeddings 接口。

副作用：
    - 本模块无 import 期副作用（模型加载延迟到 create_embedder 调用时）；
    - 线上分支：每次 embed_documents/embed_query 调用产生一次 HTTP 请求；
    - 本地分支：创建时加载本地模型权重（数百 MB、有耗时，常驻内存）。

主要依赖：
    - langchain-core（Embeddings 接口）、langchain-openai（OpenAIEmbeddings）
    - sentence-transformers（本地兜底模型加载）
    - python-dotenv（.env 加载）
"""

# os：读取环境变量（SILICONFLOW_API_KEY / SILICONFLOW_BASE_URL / EMBEDDING_MODEL_PATH / EMBEDDING_MODEL_NAME）
import os

# dotenv：从 CWD 向上搜索并加载 .env，保证任何入口调用工厂时配置可用（幂等）
from dotenv import load_dotenv

# Embeddings：LangChain 嵌入器统一接口（embed_query/embed_documents 抽象方法），
# 工厂返回值与两个消费方的既有调用约定一致
from langchain_core.embeddings import Embeddings

# SentenceTransformer：本地兜底的句向量模型加载器（延迟到工厂调用时才 import 也行，
# 但该包已是项目必装依赖，直接顶层导入不增加 import 期负担——模型本身不在此加载）
from sentence_transformers import SentenceTransformer


class LocalEmbedder(Embeddings):
    """本地嵌入适配器：把 SentenceTransformer 包装为 LangChain Embeddings 接口。

    职责：为本地兜底分支提供与线上 OpenAIEmbeddings 相同的调用契约
          （embed_query 单文本 / embed_documents 批量文本）。
    设计模式：适配器模式——第三方 SentenceTransformer 接口适配项目内
          LangChain Embeddings 契约。
    生命周期：随 create_embedder 本地分支创建，随所属检索器/建索引脚本存活。
    """

    # 构造：加载本地模型并持有（实例化即有模型加载副作用）
    def __init__(self, path: str) -> None:
        # 加载 SentenceTransformer 模型（权重常驻内存，数百 MB、有耗时）
        self.model = SentenceTransformer(path)

    # 单文本嵌入：委托批量接口取第一条（满足基类抽象方法）
    def embed_query(self, text: str) -> list[float]:
        # 复用 embed_documents 处理单条，取首行向量
        return self.embed_documents([text])[0]

    # 批量文本嵌入：encode 后转为纯 float 嵌套列表（兼容 Neo4j 序列化）
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 批量推理：batch_size=64；normalize_embeddings=True 做 L2 归一化，
        # 使向量内积等价于余弦相似度，与向量索引的 cosine 相似度函数匹配
        embeddings = self.model.encode(texts, batch_size=64, normalize_embeddings=True)
        # ndarray 行转为 list[float]（Neo4j 驱动参数需要纯 Python 数值类型）
        return [list(map(float, emb)) for emb in embeddings]


def create_embedder() -> Embeddings:
    """创建嵌入器实例（线上优先，本地兜底）

    核心功能：按环境配置选择线上/本地嵌入器，返回统一的 LangChain Embeddings 对象。

    业务流程：1. load_dotenv 加载 .env（幂等，重复加载无副作用）；
              2. SILICONFLOW_API_KEY 非空 → 线上分支：创建 OpenAIEmbeddings
                 （模型取 EMBEDDING_MODEL_NAME，缺省 BAAI/bge-m3；基地址取
                 SILICONFLOW_BASE_URL，缺省硅基流动官方地址）；
              3. 否则 → 本地兜底：模型路径取 EMBEDDING_MODEL_PATH（.env 中配置的
                 本地 bge-m3 快照绝对路径，离线可用；未配置时回退 HF 托管名
                 BAAI/bge-m3，首次运行自动下载权重），
                 创建 LocalEmbedder 适配器。

    副作用：线上分支首次调用时才产生 HTTP 请求（本函数只建客户端不联网）；
            本地分支立即加载模型权重到内存；两者均无磁盘/全局状态写入。

    返回：LangChain Embeddings 实例（embed_query/embed_documents 可用）。
          两个分支抛出的异常（如模型路径不存在、Key 无效）不做捕获，直接向上传播。
    """
    # 幂等加载 .env：保证工厂在任意入口（服务装配/独立脚本）调用时都能读到配置
    load_dotenv()

    # 读取硅基流动 API Key：空串视为未配置，决定走线上还是本地分支
    api_key = os.getenv("SILICONFLOW_API_KEY", "")

    # 分支判定逻辑：有 Key 即走线上——教学/演示环境免装本地模型，路径无关；
    # 未配置 Key 的机器回退本地模型，保证检索功能仍可用
    if api_key:
        # 延迟导入：仅线上分支需要 langchain-openai（避免本地分支时引入 OpenAI 依赖链）
        from langchain_openai import OpenAIEmbeddings

        # 创建线上嵌入器：OpenAI 兼容接口（embeddings 端点 POST {base_url}/embeddings）
        # 注：api_key 字段注解为 SecretStr，os.getenv 返回 str；pydantic v2 字段 validator
        # 运行时会自动把 str 转 SecretStr，静态类型检查不识别此转换——属误报，压制
        return OpenAIEmbeddings(  # pyright: ignore[reportArgumentType]
            # 模型标识：硅基流动缺省提供 BAAI/bge-m3（1024 维），可经 .env 覆盖
            model=os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"),
            # API 密钥：硅基流动控制台申请，需与 base_url 配套；类型注解 SecretStr | None，
            # 运行时 pydantic v2 validator 自动把 str 转 SecretStr，静态检查报 str 不兼容为误报
            api_key=api_key,  # pyright: ignore[reportArgumentType]
            # 服务基地址：OpenAI 兼容版本路径，缺省硅基流动官方地址
            base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        )

    # 本地兜底分支：优先读 .env 中的 EMBEDDING_MODEL_PATH（bge-m3 本地快照，1024 维）；
    # 未配置时回退 HF 托管名 BAAI/bge-m3（首次运行自动下载权重）
    model_path = os.getenv("EMBEDDING_MODEL_PATH", "BAAI/bge-m3")

    # 创建本地适配器（构造时加载模型权重）；日志留痕实际路径，便于排查路径问题
    embedder = LocalEmbedder(model_path)
    return embedder
