"""
retrieval - 向量检索入口模块

核心功能：提供知识库检索的统一入口——InformationRetrieval 抽象基类定义
          connect + search 最小契约，SearchResult/Document 定义结果数据
          契约，Embedder 系列负责文本向量化，FlowRetriever 提供 Flow
          文档的专用检索实现；create_retriever 工厂函数支持按配置类路径
          动态装配检索器。

执行流程：
    1. 模块导入：聚合导出 base_retriever / embedder / flow_retriever
       三个子模块的全部公开符号（无 IO）；
    2. 使用方在 config.yml 的
       policies.EnterpriseSearchPolicy.vector_store 中指定检索器类路径，
       在 endpoints.yml 的 vector_store 中配置连接参数；
    3. create_retriever(class_path, connect_config) 按类路径动态加载
       检索器类 → 校验继承关系 → 实例化 → connect 初始化。

副作用：
    - 模块导入：无 IO（仅类定义与聚合导出）。
    - create_retriever：动态 import 目标模块（读文件系统/字节码缓存）、
      实例化检索器并调用 connect（具体 IO 由检索器子类决定）。

主要依赖：
    - importlib / logging：动态类加载与日志。
    - typing：Any/TYPE_CHECKING 等注解工具。
    - app.ics_ai.retrieval.base_retriever：基类与结果数据类。
    - app.ics_ai.retrieval.embedder：向量化器族。
    - app.ics_ai.retrieval.flow_retriever：Flow 文档专用检索器。

使用方式：
1. 创建自定义检索器类，继承 InformationRetrieval
2. 实现 connect() 和 search() 方法
3. 在 config.yml 的 policies.EnterpriseSearchPolicy.vector_store 中指定全类名
4. 在 endpoints.yml 的 vector_store 中配置连接参数

示例：
    class Neo4jRetriever(InformationRetrieval):
        def connect(self, config):
            self.driver = GraphDatabase.driver(
                config["uri"],
                auth=(config["user"], config["password"])
            )

        async def search(self, query, top_k=5):
            # 执行检索
            return [SearchResult(text=..., score=...)]
"""

# 导入动态加载工具：create_retriever 按类路径 importlib.import_module 加载。
import importlib

# 导入日志器工厂：为本包创建命名 logger。
import logging

# 导入类型注解工具：Any 通配符（connect_config 等开放结构的元素类型）。
from typing import Any

# 导入检索基类与结果数据类：本包的核心契约来源。
from app.ics_ai.retrieval.base_retriever import (
    Document,
    InformationRetrieval,
    Retriever,
    SearchResult,
)

# 导入向量化器族：嵌入配置/抽象基类/两个实现/工厂函数。
from app.ics_ai.retrieval.embedder import (
    Embedder,
    EmbedderConfig,
    LocalEmbedder,
    OpenAIEmbedder,
    create_embedder,
)

# 导入 Flow 专用检索器：面向对话流程文档的检索实现。
from app.ics_ai.retrieval.flow_retriever import FlowDocument, FlowRetriever

# 本包命名 logger：create_retriever 的加载/装配日志走此通道。
logger = logging.getLogger(__name__)


def create_retriever(
    class_path: str,
    connect_config: dict[str, Any] | None = None,
) -> InformationRetrieval | None:
    """根据类路径创建 Retriever 实例。

    从 config.yml 的 policies.EnterpriseSearchPolicy.vector_store 读取类路径，
    从 endpoints.yml 的 vector_store 读取连接配置。

    核心功能：检索器装配工厂——将配置中的类路径字符串解析为真实类，
              实例化并完成 connect 初始化。
    业务流程：
        1. class_path 为空 → 记录调试日志并返回 None（调用方自行降级）；
        2. rsplit(".", 1) 拆分模块路径与类名 → importlib 动态导入
           模块 → getattr 取类对象；解析/导入/取属性失败 → 抛 ImportError；
        3. issubclass 校验目标类必须继承 InformationRetrieval，
           否则抛 TypeError（保证返回值契约）；
        4. 实例化 → connect_config 非空时调用 connect 完成初始化；
        5. 装配过程中的任何异常都会记录后原样抛出（不静默吞错）。

    副作用：动态导入模块（读磁盘/字节码缓存）；实例化检索器并触发
            其 connect（可能建立网络连接等，由子类决定）。

    @param class_path: 检索器类的完整路径，如 "my_module.Neo4jRetriever"；
                       空串或 None 时返回 None。
    @param connect_config: 传递给 retriever.connect() 的配置字典；
                           None 时不调用 connect（由调用方稍后手动初始化）。

    @return: InformationRetrieval 实例（已 connect，若提供了配置）；
             class_path 为空时返回 None；类无法加载时抛 ImportError，
             继承关系不符时抛 TypeError。
    """
    # 空路径守卫：未配置检索器时静默降级（调用方按 None 处理）。
    if not class_path:
        logger.debug("未指定检索器类路径")
        return None

    # 动态加载类：解析失败/模块不存在/属性不存在均归为 ImportError。
    try:
        # rsplit 从右侧拆分一次：左半为模块路径，右半为类名
        # （避免模块名中含点号时误拆）。
        module_name, class_name = class_path.rsplit(".", 1)
        # 按模块路径动态导入。
        module = importlib.import_module(module_name)
        # 从模块中取出类对象。
        retriever_class = getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        # 记录加载失败详情后抛出统一格式的 ImportError。
        logger.error(f"加载检索器类 '{class_path}' 失败: {e}")
        raise ImportError(f"无法加载检索器类 '{class_path}': {e}")

    # 验证是否继承自 InformationRetrieval：保证返回值满足基类契约。
    if not issubclass(retriever_class, InformationRetrieval):
        raise TypeError(f"检索器类 '{class_path}' 必须继承自 InformationRetrieval")

    # 创建实例：实例化 + connect 初始化，异常原样抛出。
    try:
        # 无参构造：连接参数统一走 connect 注入。
        retriever = retriever_class()

        # 调用 connect 初始化连接：仅在提供了配置时执行。
        if connect_config:
            # 同步调用 connect（连接建立是一次性初始化，非每轮检索）。
            retriever.connect(connect_config)
            # 记录装配成功日志（含检索器类名）。
            logger.info(f"创建检索器: {class_name}")

        # 返回就绪的检索器实例。
        return retriever

    except Exception as e:
        # 记录创建失败详情后原样抛出（保留原始异常类型与堆栈）。
        logger.error(f"创建检索器实例失败: {e}")
        raise


# 导出清单：from retrieval import * 时仅暴露以下名字，按字母序排列
# （便于 IDE 补全与 diff 干净）。
__all__ = [
    # 基础文档数据类（检索结果统一契约）
    "Document",
    # 嵌入向量化器抽象基类
    "Embedder",
    # 嵌入器配置数据类
    "EmbedderConfig",
    # Flow 索引条目数据类
    "FlowDocument",
    # Flow 文档专用检索器
    "FlowRetriever",
    # 检索器抽象基类
    "InformationRetrieval",
    # 本地嵌入器实现
    "LocalEmbedder",
    # OpenAI 兼容云嵌入器实现
    "OpenAIEmbedder",
    # 向后兼容别名（指向 InformationRetrieval 同一对象）
    "Retriever",
    # 搜索结果数据类
    "SearchResult",
    # 嵌入器工厂函数
    "create_embedder",
    # 检索器装配工厂函数
    "create_retriever",
]
