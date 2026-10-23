"""
addons 包 — 电商客服 Demo 的 GraphRAG 增强能力层

核心功能：
    为 ics_ecs（电商客服）提供 GraphRAG 检索、向量嵌入器工厂、
    Neo4j 向量/全文索引构建等"附加能力"模块。

包内模块：
    - information_retrieval.py  → GraphRAG 类（本包核心能力）
    - embedding_factory.py      → create_embedder() 嵌入器统一工厂
    - create_indexing.py        → Neo4j 索引构建脚本（独立入口，不随服务导入）

执行流程（__init__.py 自身）：
    导入时仅执行一行子模块 re-export，无重量级副作用（GraphRAG 构造时
    才会触发 Neo4j 连接、LLM 客户端创建、嵌入器初始化）。

副作用：
    - 仅一行 import：子模块 information_retrieval 的 import 期会触发
      模块级常量、类型注解、嵌入器工厂的函数定义加载，不联网、不连库。
    - 本文件不触发 create_indexing.py（该模块仅在 __main__ 入口独立调用）。

公开 API（本包顶层 re-export）：
    GraphRAG → from .information_retrieval 提升的类别名
    注：当前项目内所有消费方均走全路径导入，未使用此 re-export；
        保留它是为了让外部调用方可写 `from app.ics_ecs.addons import GraphRAG`，
        与 LangChain 等库的"包顶层暴露核心类"风格对齐。
"""

# 从 information_retrieval 子模块 re-export GraphRAG 到包顶层，
# 使外部可简写 `from app.ics_ecs.addons import GraphRAG` 而非全路径；
# 当前项目内部未使用此入口（统一走子模块全路径），保留为公共 API 约定
from .information_retrieval import GraphRAG
