"""
information_retrieval.py - GraphRAG 知识图谱检索器（Text2Cypher 六步流程）

核心功能：
    继承 ics_ai 的 InformationRetrieval 基类契约（connect/search/close），
    实现基于 Neo4j 知识图谱的检索：混合检索定位入口节点 → LLM 生成/验证/
    校正 Cypher → 执行查询并格式化为 SearchResult 列表。
    消费方为 enterprise_search_policy（GraphRAG 创建失败时自动降级，不阻塞服务）。

实现思路：
    先确定查询知识图谱的入口节点，根据入口节点和用户意图生成Cypher语句进行查询
    1.LLM根据用户输入，确定需要的入口节点类型以及实体
    2.根据提供的入口节点类型和实体，使用混合检索获取候选入口节点信息
    3.LLM根据用户输入和入口节点信息生成Cypher查询语句
    4.LLM验证生成的Cypher语法、逻辑是否正确，罗列出错误信息
    5.LLM根据用户输入、入口节点、错误信息、先前Cypher语句来生成更正后的Cypher语句
    6.执行Cypher查询，返回查询结果
工作流程:
    用户提问：用户输入查询问题
    标签路由：LLM识别问题涉及的节点类型和实体
    节点检索：使用混合检索找到候选入口节点
    Cypher生成：LLM根据入口节点和问题生成Cypher查询语句
    语句验证：验证Cypher语句的正确性
    语句校正：如有错误则进行校正
    执行查询：在Neo4j中执行Cypher查询
    返回结果：将查询结果返回给用户

副作用：
    - connect()：建立 Neo4j 连接（driver/Neo4jGraph 长期持有）；调用 DashScope
      API（读取 DASHSCOPE_API_KEY 环境变量）创建 LLM；经嵌入器统一工厂
      （embedding_factory.create_embedder）创建嵌入模型——线上优先（硅基流动，
      SILICONFLOW_API_KEY/BASE_URL/EMBEDDING_MODEL_NAME 环境变量），
      未配置 Key 时回退本地模型（EMBEDDING_MODEL_PATH 环境变量）；
    - search()：每次调用消耗 LLM tokens；对 Neo4j 只读查询（explain/execute）；
    - 本模块不写库；__main__ 为独立联调入口（硬编码本地 Neo4j 凭据）。

主要依赖：
    - app.ics_ai.retrieval.base_retriever（InformationRetrieval 基类与 SearchResult 契约）
    - neo4j / neo4j_graphrag（驱动与 HybridRetriever；检索
      {标签}_vector/{标签}_fulltext 索引，索引由 addons/create_indexing.py 构建）
    - langchain-community（ChatTongyi、Neo4jGraph、CypherQueryCorrector）
    - pydantic（路由输出结构化）、jieba（全文查询分词）
"""

# 标准库导入（PEP 8 分组一：标准库，段内 straight 导入按字母序在前、from 导入在后）
# asyncio：并发执行多标签混合检索
import asyncio

# json：解析 LLM 输出的 JSON（路由结果、验证错误列表）
import json

# logging：标准库日志，输出检索流程进度
import logging

# os：读取环境变量（DASHSCOPE_API_KEY 等）
import os

# re：正则（JSON 提取、全文分词过滤）
import re

# typing：类型注解（PEP 604/585 迁移后仅 Any 需从 typing 导入，
# 其余以内建泛型 dict/list 替代旧式 Dict/List，| 联合替代 Optional/Text）
from typing import Any

# 第三方库导入（PEP 8 分组二：第三方库，同样 straight 在前、from 按模块路径字母序排列）
# dotenv：加载 .env（__main__ 测试与 connect 内 API Key 读取）
import dotenv

# jieba：中文分词，生成全文检索查询文本
import jieba

# CypherQueryCorrector/Schema：LangChain 的 Cypher 关系方向校正器及其 schema 描述
# （两者定义源均为 cypher_utils；cypher.py 仅转发导入且无 __all__，静态检查报"未导出"，
# 故直接从定义源模块导入，运行时为同一对象，行为不变）
from langchain_community.chains.graph_qa.cypher_utils import (
    CypherQueryCorrector,
    Schema,
)

# ChatTongyi：DashScope 通义千问的 LangChain 聊天模型封装
from langchain_community.chat_models.tongyi import ChatTongyi

# Neo4jGraph：LangChain 图数据库包装器，提取库 schema 供提示词使用
from langchain_community.graphs.neo4j_graph import Neo4jGraph

# 提示模板三件套：聊天提示模板 + 系统/人类消息模板（组装各阶段 prompt）
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# GraphDatabase：Neo4j 官方驱动工厂，connect() 创建连接
from neo4j import GraphDatabase

# CypherSyntaxError：捕获 explain 语法检查时的语法异常
from neo4j.exceptions import CypherSyntaxError

# HybridRetriever：neo4j_graphrag 的向量+全文混合检索器
from neo4j_graphrag.retrievers import HybridRetriever

# BaseModel/Field：pydantic 数据模型，定义路由输出的结构化 schema
from pydantic import BaseModel, Field

# 项目内部导入（PEP 8 分组三：项目内部，置于最后）
# 导入 ics_ai 基类
# InformationRetrieval：检索器基类契约（connect/search/close，embeddings 延迟初始化）
# SearchResult：检索结果数据类（text/score/metadata），本模块的统一返回载体
from app.ics_ai.retrieval.base_retriever import InformationRetrieval, SearchResult

# create_embedder：嵌入器统一工厂（线上硅基流动优先，本地 SentenceTransformer 兜底），
# 与建索引脚本共用同一嵌入方案，保证查询向量与索引维度一致
from app.ics_ecs.addons.embedding_factory import create_embedder

# 配置控制台日志
# 获取名为 "retrieval" 的命名 logger（与根 logger 隔离，便于按名过滤）
logger = logging.getLogger("retrieval")
# 设置日志级别为 INFO：流程进度可见，DEBUG 噪音不输出
logger.setLevel(logging.INFO)
# 守卫：仅在未挂载过 handler 时添加——重复导入不会重复输出日志
if not logger.handlers:
    # 定义日志格式："[级别]时间: 消息"
    formatter = logging.Formatter("[%(levelname)s]%(asctime)s: %(message)s")
    # 控制台流 handler：日志输出到标准错误/标准输出
    handler = logging.StreamHandler()
    # 为 handler 应用上述格式
    handler.setFormatter(formatter)
    # 挂载 handler 到 logger
    logger.addHandler(handler)


# 路由输出定义,使用Pydantic定义数据模型
# 职责：描述单个"标签-实体"路由项，作为 LLM 结构化输出（tool calling）的 schema。
# 生命周期：随 route_label 的 LLM 调用结果创建，仅在该流程内使用。
class RouteItem(BaseModel):
    # 节点类型标签（必填），如 "SKU"；description 供 LLM 理解字段含义
    label: str = Field(..., description='节点类型，比如"SKU"')
    # 实体文本（必填），如 "拯救者Y9000P"；用于后续混合检索定位入口节点
    entity: str = Field(..., description="实体文本")


# 职责：路由输出容器——LLM 一次可判定多个入口节点（标签+实体 对的列表）。
# 设计：pydantic 模型经 with_structured_output 转为工具定义约束 LLM 输出格式。
class RouteOutput(BaseModel):
    # 路由结果列表：每个元素是一个 (标签, 实体) 对
    outputs: list[RouteItem]


def get_chat_history(
    tracker_state: dict[str, Any], user_id: str | None
) -> str | list[str]:
    """从 tracker state 中提取聊天历史

    核心功能：把对话状态里的 user/bot 事件拼成 "角色:文本" 多行字符串，
              供标签路由 prompt 作为上下文（多轮指代消解）。

    业务流程：1. 无状态或无事件时返回空串；
              2. 遍历事件：user 事件按有无 user_id 标注 "user_id=xxx" 或 "user"，
                 bot 事件标注 "bot"；
              3. 取最近 5 条以换行连接返回。

    副作用：无——纯内存处理，只读入参。

    参数：
        tracker_state: 对话状态字典（含 events 列表），允许 None/空。
        user_id: 用户 ID 字符串，允许 None（此时角色标注为 "user"）。

    返回：最近 5 条聊天记录的换行连接字符串（str）；无历史时返回空列表 []
          （list[str]，与空串同为假值，调用方仅作上下文文本使用）。
    """
    # 累积 "角色:文本" 行的列表
    chat_history = []
    # 提取事件列表到局部变量：避免同一键重复 .get 三次；同时让静态检查器
    # 可跨守卫与循环追踪非 None 不变式（dict.get 返回 Any | None，
    # 直接迭代会报"None 不支持迭代"，经下方真值守卫收窄后迭代安全）
    events = tracker_state.get("events")
    # 空值兜底：无状态或无事件记录时直接返回空列表，避免后续遍历报错
    if not tracker_state or not events:
        # 空历史：返回空列表（list，与空串同为假值）
        return chat_history
    # 逐条遍历对话事件（每个事件为 {"event": 类型, "text": 文本} 字典）
    for event in events:
        # 判定逻辑：区分用户消息与机器人消息，分别标注角色前缀
        if event.get("event") == "user":
            # 用户消息：有 user_id 时标注 "user_id=xxx"（帮助 LLM 识别查询主体），否则统一标 "user"
            role = (
                "user_id=" + user_id if user_id else "user"
            )  # 如果有user_id则为"user_id=xxx"，否则为"user"
            # 拼接 "角色:去除首尾空白的用户文本" 行
            chat_history.append(f"{role}:{event.get('text').strip()}")
        elif event.get("event") == "bot":
            # 机器人消息：拼接 "bot:去除首尾空白的回复文本" 行
            chat_history.append(f"bot:{event.get('text').strip()}")

    # 返回最近5条聊天记录，用换行符连接成一个字符串。（奇数条，最后一条为用户最后的提问，前面是成对的问答）
    # 截取最近 5 条并换行连接：保留最新上下文，避免 prompt 过长
    return "\n".join(chat_history[-5:])


class GraphRAG(InformationRetrieval):
    """继承了 InformationRetrieval 基类

    职责：GraphRAG 检索器——标签路由、入口节点混合检索、Cypher 生成/验证/
          校正、执行查询并格式化结果，完整实现基类 connect/search/close 契约。

    设计模式：模板方法——search() 固定六步编排，各步骤以独立方法实现；
              工厂解耦——LLM/嵌入模型/驱动均由 connect(config) 装配。

    生命周期：构造仅准备 prompt 与标签清单（轻量）；connect() 建立连接后
              方可 search()；close() 释放 Neo4j 连接。随 enterprise_search_policy
              装配创建并持有。
    """

    def __init__(self, embeddings=None):
        # 调用基类构造：保存 embeddings（可为 None，由 _init_embeddings 延迟创建）
        super().__init__(embeddings)
        # 入口节点可选标签
        # 标签清单文本：嵌入路由 prompt，告诉 LLM 有哪些入口节点类型及示例，
        # 约束其只从此清单中选标签
        self.optional_label = (
            '- Category1:   一级分类，如"食品饮料"、"家用电器"、"手机"'
            '- Category2:   二级分类，如"大家电"、"香水彩妆"'
            '- Category3:   三级分类，如"手机"、"香水"、"笔记本"'
            '- Trademark:   品牌，如"华为"、"索芙特"、"金沙河"'
            '- SPU:         商品名称，如"华为Mate 40 pro"'
            '- SKU:         单品名称，如"联想(Lenovo) 拯救者Y9000P 2022 16英寸游戏笔记本电脑 i9-12900H RTX3070Ti 钛晶灰"'
            '- Attr:        商品属性值，如"70英寸"、"蓝色"、"非有机食品"'
            '- User:        用户ID，如"176"'
        )
        # 节点标签路由 prompt
        # PromptTemplate是基础的提示模板类，用于传统的文本生成场景：
        # ChatPromptTemplate是专为聊天模型设计的提示模板类（系统消息、用户消息、AI消息等，适合多轮对话）
        # 路由 prompt：系统消息说明任务与输出格式，人类消息承载用户查询（{query}）
        self.route_label_prompt = ChatPromptTemplate.from_messages(
            [
                # 系统消息模板：定义路由 Agent 的角色、任务与严格 JSON 输出约定
                SystemMessagePromptTemplate.from_template(
                    "你是一个智能检索路由Agent。"
                    "现在根据用户输入判断最可能需要的一个或多个标签以及每个标签对应的实体，作为后续Neo4j查询的入口节点\n"
                    "**注意：如果查询与用户相关，需要将用户信息加入入口节点**\n"
                    '以严格JSON格式输出结果，比如"[{{"label": "SPU", "entity": "iPhone 16 Pro"}}]"。'
                    "可选节点类型:\n{optional_label}"
                ),
                # 人类消息模板：承载用户查询（{query} 变量）
                HumanMessagePromptTemplate.from_template("{query}"),
            ]
        )
        # Cypher 生成 prompt
        # 生成 prompt：系统消息给 schema，要求仅返回 Cypher；人类消息给入口节点与查询
        self.generate_cypher_prompt = ChatPromptTemplate.from_messages(
            [
                # 系统消息模板：Cypher 专家角色 + schema 上下文 + 禁止返回嵌入向量等冗余属性
                SystemMessagePromptTemplate.from_template(
                    "你是一个Cypher专家，正在根据入口节点信息和用户输入，参照schema生成准确无误的Cypher查询语句。"
                    "**注意：查询结果中不可以包含嵌入向量等多余属性**\n"
                    "仅返回Cypher语句。\n"
                    "schema:\n{schema}"
                ),
                # 人类消息模板：入口节点信息 + 用户输入
                HumanMessagePromptTemplate.from_template(
                    "入口节点:\n{entry_nodes}\n\n用户输入:\n{query}\n\nCypher语句:"
                ),
            ]
        )
        # Cypher 验证 prompt
        # 验证 prompt：让 LLM 以审查者身份对照 schema 列出错误清单（JSON 列表）
        self.validate_cypher_prompt = ChatPromptTemplate.from_messages(
            [
                # 系统消息模板：五项审查清单（用户过滤/语法/关系方向/变量/结果可用性）
                SystemMessagePromptTemplate.from_template(
                    "你是一位Cypher专家，正在审查一位初级开发人员编写的Cypher语句。你需要根据schema和用户输入，检查如下内容：\n"
                    "* Cypher语句中是否需要包含用户信息作为过滤条件？\n"
                    "* Cypher语句中是否有任何语法错误？\n"
                    "* Cypher语句中的关系方向是否符合schema中的定义？\n"
                    "* Cypher语句中是否漏定义了变量或使用了未定义的变量？\n"
                    "* Cypher语句检索出的内容能否用于回答用户的问题？\n"
                    '以严格列表格式输出错误信息，比如"["错误1", "错误2"]"，始终解释schema与Cypher语句之间的差异。'
                    "如果确认没有问题，返回空内容即可。\n"
                    "schema:\n{schema}"
                ),
                # 人类消息模板：入口节点 + 用户输入 + 待验证语句
                HumanMessagePromptTemplate.from_template(
                    "入口节点:\n{entry_nodes}\n\n"
                    "用户输入:\n{query}\n\n"
                    "待验证的Cypher语句:\n{cypher}"
                ),
            ]
        )
        # Cypher 校正 prompt
        # 校正 prompt：携带验证错误清单，让 LLM 产出修正后的语句（仅返回 Cypher）
        self.correct_cypher_prompt = ChatPromptTemplate.from_messages(
            [
                # 系统消息模板：依据 schema 与错误信息更正语句
                SystemMessagePromptTemplate.from_template(
                    "你是一位Cypher专家，正在审查一位初级开发人员编写的Cypher语句。你需要根据schema以及提供的错误信息更正Cypher语句。"
                    "仅返回Cypher语句。\n"
                    "schema:\n{schema}"
                ),
                # 人类消息模板：入口节点 + 用户输入 + 错误清单 + 待更正语句
                HumanMessagePromptTemplate.from_template(
                    "入口节点:\n{entry_nodes}\n\n"
                    "用户输入:\n{query}\n\n"
                    "错误信息:\n{errors}\n\n"
                    "待更正的Cypher语句:\n{cypher}\n\n"
                    "更正后的Cypher语句:"
                ),
            ]
        )

    def _init_embeddings(self):
        """初始化嵌入模型（延迟加载）

        核心功能：在 embeddings 未注入时创建默认嵌入器（线上优先，本地兜底）。

        业务流程：1. 已有 embeddings 则跳过（幂等）；
              2. 调用嵌入器统一工厂 create_embedder：SILICONFLOW_API_KEY
                 已配置时走线上硅基流动（缺省 BAAI/bge-m3，1024 维），
                 否则回退 SentenceTransformer 本地模型（EMBEDDING_MODEL_PATH 路径）。

        副作用：线上分支仅创建客户端（首次调用时产生 HTTP 请求）；
                本地分支立即加载模型权重（数百 MB、常驻内存）。

        参数：无（读写实例属性 self.embeddings）。

        返回：无返回值（None）。
        """
        # 幂等守卫：已注入嵌入模型（构造传入或已初始化）则直接返回
        if self.embeddings is not None:
            # 跳过重复加载
            return

        # 调用嵌入器统一工厂：线上/本地的分支选择与适配逻辑集中在工厂内维护，
        # 保证本检索器与建索引脚本（create_indexing.py）使用同一嵌入方案
        self.embeddings = create_embedder()
        # 日志留痕：实际使用的嵌入器类型，便于排查线上/本地分支走向
        logger.info("嵌入模型已加载: %s", type(self.embeddings).__name__)

    def connect(self, config: dict[str, Any] | None = None) -> None:
        """连接检索系统：连接到Neo4j数据库并初始化相关组件

        核心功能：装配检索器全部运行时依赖——Neo4j 驱动、图 schema、
              Cypher 校正器、LLM、嵌入模型。

        业务流程：1. 解析连接配置（uri/user/password，带缺省值）并创建驱动；
              2. 经 Neo4jGraph 获取增强 schema 与关系列表；
              3. 用关系列表构造 CypherQueryCorrector；
              4. 创建 ChatTongyi LLM（coder 模型，读 DASHSCOPE_API_KEY）；
              5. 初始化嵌入模型。

        副作用：建立 Neo4j 连接（长持有）；读取 .env 与环境变量；实例化 LLM 客户端。

        参数：
            config: 连接配置字典，允许 None/空；支持键 uri 或 neo4j_url、
                    user（缺省 "neo4j"）、password（缺省空串）。

        返回：无返回值（None）。
        """
        # 空值兜底：config 为 None 时置空字典，统一按字典取键
        config = config or {}

        # 1、创建Neo4j驱动程序
        # 连接地址：优先取 "uri" 键，回退 "neo4j_url" 键，两者皆无时用本地缺省地址
        neo4j_url = config.get("uri") or config.get(
            "neo4j_url", "bolt://localhost:7687"
        )
        # 用户名：缺省 "neo4j"（本地默认账户）
        neo4j_user = config.get("user", "neo4j")
        # 密码：缺省空串（由调用方经 config 注入）
        neo4j_password = config.get("password", "")
        # 组装认证对 (用户名, 密码)
        neo4j_auth = (neo4j_user, neo4j_password)
        # Neo4j 驱动
        # 创建驱动实例并长持有（with 不可用——连接生命周期由 close() 管理）
        self.driver = GraphDatabase.driver(neo4j_url, auth=neo4j_auth)

        # 2、获取图数据库schema
        # Neo4j Graph 包装器
        # 创建 LangChain 图包装器：负责连接库并抽取 schema（供后续 prompt 引用）
        neo4j_graph = Neo4jGraph(
            neo4j_url,
            neo4j_auth[0],  # 连接Neo4j数据库的用户名
            neo4j_auth[1],  # 连接Neo4j数据库的密码
            enhanced_schema=True,  # 是否使用增强的schema信息，设为True 提供更详细的schema信息，包括节点标签、关系类型和属性等
        )
        # Neo4j schema
        # 保存文本化 schema：嵌入"生成/验证/校正"三段 prompt 的 {schema} 占位符
        self.neo4j_schema = neo4j_graph.schema
        # Neo4j 关系列表
        # 提取关系列表到局部变量：.get 返回 Any | None，直接迭代会报"None 不支持迭代"，
        # 用 or [] 兜底空值（relationships 键由 Neo4jGraph 构建 schema 时保证存在，
        # 兜底仅为静态收窄与防御极端情形）
        relationships = neo4j_graph.structured_schema.get("relationships") or []
        # 从结构化 schema 提取 (起点, 关系类型, 终点) 三元组列表——CypherQueryCorrector
        # 用它判断关系方向合法性
        corrector_schema = [
            # 每条关系取 (起点, 关系类型, 终点) 三元组构造 Schema
            Schema(el["start"], el["type"], el["end"])
            # 遍历关系列表
            for el in relationships
        ]

        # 3、初始化 Cypher查询校正器（langchain提供的api）
        # 基于关系三元组创建校正器：search() 末步用它校正 LLM 语句中的关系方向
        self.cypher_corrector = CypherQueryCorrector(corrector_schema)

        # 4、配置 LLM（使用coder模型，对语法处理效果更好）
        # 保持思考模式开启，有助于 Cypher 生成/校验等复杂任务的准确性
        # 固定选用 qwen3-coder-flash：账号实测可用（qwen3-coder-plus 在部分
        # 账号返回 404 Model not exist）；避免用日期快照名——快照未开通/下线
        # 时 DashScope 报误导性 404，稳定别名由平台自动指向最新可用快照
        model_name = "qwen3-coder-flash"
        # 加载 .env：确保 DASHSCOPE_API_KEY 在读取前已注入环境（幂等调用）
        dotenv.load_dotenv()
        # 读取 DashScope API Key（未配置时为 None，LLM 调用阶段才报错）
        model_api_key = os.getenv("DASHSCOPE_API_KEY")
        # 创建通义千问聊天模型实例（长持有，供路由/生成/验证/校正复用）
        # 传参形态说明：ChatTongyi 为 pydantic 模型，字段为 model_name(alias="model")/
        # dashscope_api_key(alias="api_key"，字段注解 SecretStr | None)；不同版本的
        # basedpyright 分别按别名或字段名生成构造签名，两种传参形态互斥报错
        # （"参数不存在"/"缺少传入值"），属签名生成误报——运行时 pydantic v2 别名
        # 校验恒开且 populate_by_name=True，两种形态完全等价；api_key 传 str 会经
        # 字段 validator 转 SecretStr，os.getenv 的 str | None 实参与 SecretStr 注解
        # 不匹配同为静态误报，故压制 reportCallIssue 与 reportArgumentType
        self.llm = ChatTongyi(model=model_name, api_key=model_api_key)  # pyright: ignore[reportCallIssue, reportArgumentType]

        # 5、初始化嵌入模型
        # 延迟加载本地嵌入模型（若构造时未注入 embeddings）
        self._init_embeddings()

        # 日志留痕：连接成功与实际地址
        logger.info("GraphRAG 已连接到 Neo4j: %s", neo4j_url)

    def _extract_json_from_text(self, text: str) -> str:
        """从文本中提取 JSON 部分。

        处理 LLM 输出可能包含的额外内容（如思考过程 <think>...</think>、markdown 代码块等）。

        核心功能：把 LLM 的自由文本回复清洗为可 json.loads 的 JSON 字符串。

        业务流程：1. 空文本兜底返回 "[]"；
              2. 移除 <think> 思考块；
              3. 已是 JSON 开头则直接返回；
              4. 提取 markdown 代码块内容；
              5. 正则捕获首个 JSON 数组/对象；
              6. 均失败则返回 "[]"。

        副作用：无——纯字符串处理。

        参数：
            text: LLM 原始回复文本，允许 None/空串。

        返回：JSON 字符串（数组或对象）；各兜底路径失败时返回 "[]"。
        """
        # 空值兜底：空输入直接返回空数组字符串（调用方可安全 json.loads）
        if not text:
            # 返回空数组占位
            return "[]"

        # 局部导入 re（模块顶部已导入，此处为原代码保留，不改动）
        import re

        # 去除首尾空白，便于后续前缀判定
        text = text.strip()

        # 1. 移除思考模式的 <think>...</think> 标签内容
        # 正则跨行匹配并删除 <think> 块（[\s\S] 兼容换行），防止思考文本干扰 JSON 解析
        text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

        # 2. 尝试直接解析
        # 判定逻辑：文本已以 JSON 数组/对象开头，视为纯 JSON 直接返回
        if text.startswith("[") or text.startswith("{"):
            # 直接返回原文
            return text

        # 3. 尝试从 markdown 代码块中提取
        # 正则匹配 ```json ... ``` 代码块（(?:json)? 兼容无语言标注），捕获内部内容
        code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        # 命中代码块则取捕获组（模型最常把 JSON 包在代码块里）
        if code_block_match:
            # 返回代码块内容（去首尾空白）
            return code_block_match.group(1).strip()

        # 4. 尝试找到 JSON 数组或对象
        # 贪婪匹配首个 "[" 到末个 "]"（或花括号对），从混杂文本中截出 JSON 主体
        json_match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
        # 命中则返回捕获的 JSON 片段
        if json_match:
            # 返回匹配片段
            return json_match.group(1)

        # 全部策略失败：返回空数组字符串兜底
        return "[]"

    @staticmethod
    def _clean_cypher(text: str) -> str:
        """从 LLM 回复中清洗出纯 Cypher 语句（思考块/围栏兜底净化）。

        核心功能：处理 qwen3-coder-flash 思考模式输出对语句提取的两种污染——
              ① <think> 思考块（内含围栏代码示例，会干扰"首个围栏"式提取）；
              ② markdown 围栏残渣（成对围栏未匹配时泄漏进语句体，
                 如语句末尾出现 ```，导致 explain/执行报语法错误）。

        业务流程：1. 空输入直接返回空串；
              2. 删除 <think>...</think> 思考块（未闭合的 <think> 一并删除）；
              3. 优先提取首个成对围栏（语言标签可选，兼容 CRLF 与大小写）；
              4. 无成对围栏时，剥掉句首孤立围栏标记与句尾孤立 ``` 残渣；
              5. 返回去首尾空白后的纯语句。

        副作用：无——纯字符串处理。

        参数：
            text: LLM 原始回复或既经上游抽取的语句文本，允许 None/空串。

        返回：清洗后的 Cypher 语句字符串；无可提取内容时返回空串。
        """
        # 空值兜底：None/空串无清洗意义，原样返回空串
        if not text:
            # 直接短路返回
            return ""

        # 1. 删除 <think> 思考块：[\s\S] 跨行匹配；(?:</think>|\Z) 兼容模型
        # 未输出闭合标签的情形（思考未闭合时删到文本末尾）
        text = re.sub(r"<think>[\s\S]*?(?:</think>|\Z)", "", text).strip()

        # 2. 提取首个成对围栏内容：
        #    - 语言标签 (?:[a-zA-Z]+[ \t]*\r?\n)?：兼容 ```cypher 与 ```CRLF、
        #      标签后多余空白；仅识别完整换行结尾的标签行，避免误吞语句体
        #    - 惰性捕获 ([\s\S]*?)：取到最近一个闭合 ```，即首个围栏块
        fenced_match = re.search(r"```(?:[a-zA-Z]+[ \t]*\r?\n)?([\s\S]*?)```", text)
        # 命中成对围栏则取捕获组作为语句体（模型最常把语句包在代码块里）
        if fenced_match:
            # 捕获组即围栏内的语句文本
            text = fenced_match.group(1)
        else:
            # 3. 无成对围栏的残渣净化：剥句首孤立的 ```cypher 开栏标记
            #    （^ 锚定开头，\s* 容忍前导空白与换行）
            text = re.sub(r"^\s*```[a-zA-Z]*[ \t]*\r?\n?", "", text)
            # 剥句尾孤立的 ``` 残渣（\s*$ 容忍尾部空白；该残渣混入语句体
            # 会使 explain/执行报 Invalid input '```' 语法错误）
            text = re.sub(r"```\s*$", "", text)

        # 返回去首尾空白后的纯语句（可能为空串，由上游空语句兜底承接）
        return text.strip()

    async def route_label(self, query):
        """
        路由标签识别：识别标签，抽取实体
        使用LLM识别用户查询中涉及的节点类型和实体

        核心功能：把用户查询解析为一组 (标签, 实体) 路由对，作为图谱入口节点候选。

        业务流程：1. 填充路由 prompt（可选标签清单 + 查询文本）；
              2. 优先 with_structured_output 结构化输出；
              3. 返回 None（模型不支持 tool call）或抛异常时，降级为普通调用 +
                 _extract_json_from_text 清洗 + 手工解析为 RouteItem。

        副作用：调用 DashScope LLM API（消耗 tokens）。

        参数：
            query: 用户查询文本（通常为含历史的上下文串），字符串。

        返回：RouteItem 列表（可能为空）；结构化降级解析失败会抛出异常不返回。
        """

        # 1、填充prompt中的变量
        # 注入可选标签清单与用户查询，生成待发送的聊天提示
        prompt = self.route_label_prompt.format_prompt(
            optional_label=self.optional_label, query=query
        )

        # 2、调用LLM，获得输出结果
        # with_structured_output方法：输出遵循指定的数据结构（指定的RouteOutput类）
        # 依赖于 function calling 或 tool calling 机制，LangChain 会将数据模型（如 RouteOutput）转换为工具定义（tool definition）
        # try 覆盖结构化调用与降级解析的全程：任何失败都走普通调用兜底
        try:
            # 结构化输出调用：LLM 结果经 tool calling 强制符合 RouteOutput schema
            llm_output = await self.llm.with_structured_output(RouteOutput).ainvoke(
                prompt
            )
            # 空值兜底：部分模型 tool call 不可用时返回 None，转普通调用解析
            if llm_output is None:
                # 日志留痕：降级原因
                logger.warning("LLM structured output 返回 None，尝试普通调用方式")
                # 普通 ainvoke：返回原始文本消息
                llm_output = await self.llm.ainvoke(prompt)
                # isinstance 收窄：LangChain content 类型为 str | list[...]（list 仅
                # 出现在多模态/工具块场景，本分支普通文本调用恒为 str），非 str 兜底
                # 空串——下游 json.loads 解析失败会被 except 捕获走降级重试
                content = llm_output.content
                # 从自由文本中清洗出 JSON 字符串
                json_str = self._extract_json_from_text(
                    content if isinstance(content, str) else ""
                )
                # json.loads 解析为字典列表，逐项解包构造 RouteItem（字段校验由 pydantic 完成）
                outputs = [RouteItem(**item) for item in json.loads(json_str)]
            else:
                # isinstance 收窄：with_structured_output(schema_cls) 的返回类型注解
                # 为宽泛联合（dict[Unknown, Unknown] | BaseModel），静态检查无法得知
                # 运行时恒为传入的 RouteOutput 实例，收窄后访问 .outputs 属性；
                # 异常形态（dict）兜底空路由列表，交由后续检索空结果降级
                outputs = (
                    llm_output.outputs if isinstance(llm_output, RouteOutput) else []
                )
        except Exception as e:
            # 如果模型不支持 tool call 或调用失败，使用普通调用方式
            # 日志留痕：记录降级原因（异常内容）后兜底重试
            logger.warning(f"LLM structured output 调用失败: {e}，尝试普通调用方式")
            # 普通 ainvoke 调用
            llm_output = await self.llm.ainvoke(prompt)
            # isinstance 收窄：LangChain content 类型为 str | list[...]（list 仅出现
            # 在多模态/工具块场景，本分支普通文本调用恒为 str），非 str 兜底空串
            content = llm_output.content
            # 清洗出 JSON 字符串
            json_str = self._extract_json_from_text(
                content if isinstance(content, str) else ""
            )
            # 解析并构造 RouteItem 列表
            outputs = [RouteItem(**item) for item in json.loads(json_str)]

        # 日志留痕：路由得到的标签-实体列表，便于排查检索偏差
        logger.info("入口节点标签与实体:%s", outputs)
        # 返回路由结果供 node_retrieval 使用
        return outputs

    async def node_retrieval(self, route_res, top_k):
        """
        节点检索：根据标签和实体，检索入口节点
        对于用户节点直接通过Cypher查询获取，对于其他类型的节点则使用混合检索（向量+全文）进行检索

        核心功能：按路由结果定位图谱入口节点——User 走精确 Cypher，其余标签走
              HybridRetriever 混合检索（消费 create_indexing.py 建的索引）。

        业务流程：1. 遍历路由对：实体为空跳过；User 标签按 user_id 精确查库；
              2. 其余 (标签, 实体) 成对收集；3. jieba 分词生成全文查询文本、
              向量化生成查询向量；4. 每标签建 HybridRetriever，asyncio.to_thread
              并发检索；5. 按标签归并结果（{标签}_name 或 _value + score）。

        副作用：对 Neo4j 执行只读查询；调用本地嵌入模型推理。

        参数：
            route_res: route_label 的输出，RouteItem 可迭代对象。
            top_k: 每个标签检索返回的节点数量上限，正整数。

        返回：dict——标签到节点信息列表（含得分）的映射；无待检索对时返回
              仅含 User 结果的字典（可能为空）。
        """
        # 用于存储需要检索的标签-实体对
        pairs = []  # 用于存储需要检索的标签-实体对
        # 用于存储检索到的节点结果，以标签为键
        retrieved_nodes = {}  # 用于存储检索到的节点结果，以标签为键

        # 逐条处理路由结果
        for i in route_res:
            # 遍历路由结果中的每一项，如果实体为空则跳过当前项
            if not i.entity:  # 遍历路由结果中的每一项，如果实体为空则跳过当前项
                # 空实体无检索意义，跳过
                continue
            # 如果标签是"User"，则直接使用Cypher查询在数据库中查找用户节点。
            if (
                i.label == "User"
            ):  # 如果标签是"User"，则直接使用Cypher查询在数据库中查找用户节点。
                # 精确查询：User 节点唯一标识明确（user_id），无需模糊检索；实体转 int 匹配数值主键
                user_node = self.driver.execute_query(
                    # 参数化 Cypher：按 user_id 精确匹配用户节点（$user_id 防注入）
                    "match (u:User) where u.user_id = $user_id return u;",
                    # 参数字典：user_id 键对应 Cypher 的 $user_id 占位符
                    {"user_id": int(i.entity)},
                )
                # 将结果添加到retrieved_nodes字典中
                # setdefault 归并：同标签多次出现时追加进同一列表
                retrieved_nodes.setdefault(i.label, []).append(
                    user_node
                )  # 将结果添加到retrieved_nodes字典中
            else:  # 如果不是用户节点，则将标签和实体作为一个元组添加到pairs列表中，供后续检索使用
                # 非 User 标签进入混合检索队列
                pairs.append((i.label, i.entity))

        # 如果没有需要检索的标签-实体对，直接返回已找到的节点（通常是用户节点）
        if (
            not pairs
        ):  # 如果没有需要检索的标签-实体对，直接返回已找到的节点（通常是用户节点）
            # 无混合检索任务，提前返回（避免空 zip/检索开销）
            return retrieved_nodes

        # 将标签-实体对分离成两个独立的列表：labels和entities
        # zip(*) 转置拆列，再转 list 便于 zip 配对后续列表
        labels, entities = zip(*pairs)
        # 转置结果转为两个独立列表
        labels, entities = list(labels), list(entities)

        # 对每个实体进行中文分词处理，只保留中英文和数字字符，用" OR "连接，生成全文检索查询文本
        # 列表推导逐实体处理：jieba 分词 → 正则过滤纯符号词 → 空格分隔词以 " OR " 连接
        # （Neo4j 全文索引语法按 OR 组合词命中）
        query_texts = [
            " OR ".join(
                [
                    # 去除词前后空白后输出
                    word.strip()
                    # jieba.lcut 精确模式分词
                    for word in jieba.lcut(entity)
                    # 判定逻辑：仅保留完整由中英文/数字构成的词（正则过滤标点等噪声）
                    if re.fullmatch(r"[a-zA-Z0-9\u4e00-\u9fa5]+", word.strip())
                ]
            )
            # 与实体列表一一对应产出查询文本
            for entity in entities
        ]
        # 对实体进行向量化处理，生成向量表示，用于向量检索
        # 批量嵌入实体（与全文文本按下标一一对应），供向量检索半边使用
        query_vectors = self.embeddings.embed_documents(entities)

        # 为每个标签创建混合检索任务
        tasks = []
        for label, query_text, query_vector in zip(labels, query_texts, query_vectors):
            # 创建HybridRetriever实例（neo4j_graphrag库）
            # 每标签一个检索器：索引名与 create_indexing.py 的命名约定闭环
            # （{标签小写}_vector 向量索引 + {标签小写}_fulltext 全文索引）
            retriever = HybridRetriever(
                # 已连接的 Neo4j 驱动
                self.driver,
                vector_index_name=label.lower() + "_vector",  # 向量索引名称
                fulltext_index_name=label.lower() + "_fulltext",  # 全文索引名称
            )
            tasks.append(
                # 将同步的检索操作包装为异步任务，以支持并发执行
                # asyncio.to_thread 包装同步检索：不阻塞事件循环，多标签并发
                asyncio.to_thread(
                    # 同步方法：同时传入全文查询文本与查询向量做混合检索
                    retriever.get_search_results,
                    # 全文检索半边的查询文本
                    query_text,
                    # 向量检索半边的查询向量
                    query_vector,
                    # 返回条数上限
                    top_k,
                    # 扩大候选池倍率：初检 2*top_k 条再精排，提升召回质量
                    effective_search_ratio=2,
                )
            )
        # 并发执行所有检索任务，并等待全部结果返回。
        results = await asyncio.gather(*tasks)

        # 处理检索结果
        # pairs 与 results 按提交顺序一一对应，配对遍历归并
        for (label, _), result in zip(pairs, results):  # 遍历每一对标签和对应的检索结果
            # 根据标签类型构建结果格式，提取节点名称/值和得分，添加到retrieved_nodes字典中
            # setdefault 归并到标签键下；条件表达式区分属性节点与常规节点
            retrieved_nodes.setdefault(label, []).extend(
                # 对于非"Attr"标签，使用{标签名}_name作为键
                [
                    {
                        # 提取节点名称属性（如 spu_name）作为展示键
                        f"{label.lower()}_name": i["node"][f"{label.lower()}_name"],
                        # 相似度得分（混合检索归一得分）
                        "score": i["score"],
                    }
                    # 遍历该标签的检索记录
                    for i in result.records
                ]
                # 判定逻辑：Attr 节点的文本属性名是 attr_value 而非 attr_name，需分支处理
                if label != "Attr"
                # 对于"Attr"标签，使用{标签名}_value作为键
                else [
                    {
                        # 属性节点提取 attr_value 作为展示键
                        f"{label.lower()}_value": i["node"][f"{label.lower()}_value"],
                        # 相似度得分
                        "score": i["score"],
                    }
                    # 遍历该标签的检索记录
                    for i in result.records
                ]
            )
        # 日志留痕：入口节点检索结果全景
        logger.info("入口节点:%s", retrieved_nodes)
        # 返回标签 → 节点信息列表 的映射
        return retrieved_nodes

    async def generate_cypher(self, query, entry_nodes):
        """
        Cypher语句生成：生成 Cypher 语句
        用LLM根据入口节点和用户查询生成Cypher查询语句

        核心功能：以 schema + 入口节点为上下文，让 LLM 产出针对本次查询的 Cypher。

        业务流程：1. 填充生成 prompt（schema/query/entry_nodes）；
              2. 调用 LLM；3. _clean_cypher 从回复中清洗出纯语句。

        副作用：调用 DashScope LLM API（消耗 tokens）。

        参数：
            query: 用户查询文本，字符串。
            entry_nodes: node_retrieval 的输出字典（标签 → 节点信息）。

        返回：Cypher 语句字符串；抽取不到语句时返回空串（由验证环节兜底）。
        """

        # 1、填充prompt中的变量
        # 注入 schema、用户输入与入口节点信息
        prompt = self.generate_cypher_prompt.format_prompt(
            schema=self.neo4j_schema, query=query, entry_nodes=entry_nodes
        )

        # 2、调用LLM
        # 异步调用：返回原始文本消息
        llm_output = await self.llm.ainvoke(prompt)

        # 3、从回复中清洗出 Cypher 语句
        # qwen3-coder-flash 思考模式输出含 <think> 块与围栏残渣，三方提取器
        # extract_cypher 会取到思考块示例或泄漏围栏（实测语句尾部混入 ```），
        # 改用自有清洗：删思考块 → 成对围栏提取/孤立残渣剥离
        # isinstance 收窄：LangChain content 类型为 str | list[...]（list 仅出现在
        # 多模态/工具块场景，本分支普通文本调用恒为 str），非 str 兜底空串——
        # 空串经清洗返回空结果，由下游 validate_cypher 的空语句处理承接
        content = llm_output.content
        cypher = self._clean_cypher(content if isinstance(content, str) else "")

        # 日志留痕：生成的语句，便于排查生成质量
        logger.info("Cypher生成:%s", cypher)
        # 返回语句供验证环节使用
        return cypher

    async def validate_cypher(self, query, entry_nodes, cypher):
        """
        Cypher语句验证：验证 Cypher 语句
        使用 LLM 验证生成的Cypher语句的语法和逻辑正确性

        核心功能：双重验证——数据库 explain 语法检查 + LLM 对照 schema 的逻辑审查。

        业务流程：1. 以 "explain {cypher}" 前缀执行语法检查（不真正执行查询），
              捕获 CypherSyntaxError 收集进错误列表；
              2. 填充验证 prompt 调 LLM，解析其 JSON 错误列表并入；
              3. JSON 解析失败静默忽略（bare except），仅保留语法层错误。

        副作用：对 Neo4j 执行 explain（只读、无结果副作用）；调用 LLM API。

        参数：
            query: 用户查询文本，字符串。
            entry_nodes: 入口节点字典。
            cypher: 待验证的 Cypher 语句字符串。

        返回：错误信息字符串列表（无错误时为空列表）。
        """

        # 1、验证 Cypher 语法
        # 错误列表，用于收集验证过程中发现的错误
        errors = []  # 错误列表，用于收集验证过程中发现的错误
        # 语法检查可能抛 CypherSyntaxError，需捕获收集而非中断流程
        try:
            # 通过explain关键字只检查语法而不实际执行
            # 驱动注解误报：query 形参类型为 LiteralString | Query，要求语句是
            # 字面量以便静态检查；本模块语句由 LLM 生成→校正→验证链路动态构造，
            # 属设计内形态，故压制本行重载匹配与下一行实参类型误报
            self.driver.execute_query(  # pyright: ignore[reportCallIssue]
                # explain 前缀：仅做查询计划解析，不执行查询本体
                f"explain {cypher}"  # pyright: ignore[reportArgumentType]
            )  # 通过explain关键字只检查语法而不实际执行
        except CypherSyntaxError as e:  # 捕获语法错误并添加到错误列表中
            # 语法错误信息（含行/位置）加入错误列表，供校正环节引用
            errors.append(str(e))

        # 2、验证 Cypher 逻辑是否符合用户查询意图
        # 注入 schema、查询、语句与入口节点，请 LLM 逻辑审查
        prompt = self.validate_cypher_prompt.format_prompt(
            schema=self.neo4j_schema,
            query=query,
            cypher=cypher,
            entry_nodes=entry_nodes,
        )
        # 异步调用 LLM 获取错误清单
        llm_output = await self.llm.ainvoke(prompt)

        # 解析可能失败的 JSON：失败时静默忽略，不覆盖已有语法错误
        try:
            # isinstance 收窄：LangChain content 类型为 str | list[...]（list 仅出现
            # 在多模态/工具块场景，本分支普通文本调用恒为 str），非 str 兜底空串——
            # json.loads 抛错由下方 bare except 静默忽略，与既有降级语义一致
            content = llm_output.content
            # 解析 LLM 输出为 JSON（预期为字符串列表）
            llm_errors = json.loads(content if isinstance(content, str) else "")
            # 判定逻辑：仅接受列表类型，其他形态（对象/字符串）一律忽略
            if isinstance(llm_errors, list):
                # 逻辑错误并入总错误列表
                errors.extend(llm_errors)
        except:
            # bare except：LLM 输出非 JSON 时静默跳过（保留语法层错误继续流程）
            pass
        # 日志留痕：全部验证错误
        logger.info("Cypher验证:%s", errors)
        # 返回错误列表（空列表表示验证通过）
        return errors

    async def correct_cypher(self, query, entry_nodes, cypher, errors):
        """
        Cypher语句校正：校正 Cypher 语句
        使用 LLM 进行校正：如果验证失败，则根据错误信息校正Cypher语句。

        核心功能：携带验证错误清单，让 LLM 产出修正后的 Cypher 语句。

        业务流程：1. 填充校正 prompt（schema/query/errors/cypher）；
              2. 调用 LLM；3. _clean_cypher 清洗出更正后的纯语句。

        副作用：调用 DashScope LLM API（消耗 tokens）。

        参数：
            query: 用户查询文本，字符串。
            entry_nodes: 入口节点字典。
            cypher: 待更正的 Cypher 语句字符串。
            errors: validate_cypher 返回的错误信息列表。

        返回：更正后的 Cypher 语句字符串。
        """

        # 1、填充prompt中的变量
        # 注入 schema、查询、错误清单与待更正语句
        prompt = self.correct_cypher_prompt.format_prompt(
            schema=self.neo4j_schema,
            query=query,
            cypher=cypher,
            entry_nodes=entry_nodes,
            errors=errors,
        )

        # 2、调用LLM
        # 异步调用获取更正后的回复
        llm_output = await self.llm.ainvoke(prompt)

        # 3、从回复中清洗出更正后的 Cypher 语句
        # 与 generate_cypher 同理改用自有清洗（三方提取器在思考模式输出下
        # 会取错围栏或泄漏残渣，实测校正步曾因此拿到空语句）
        # isinstance 收窄：LangChain content 类型为 str | list[...]（list 仅出现在
        # 多模态/工具块场景，本分支普通文本调用恒为 str），非 str 兜底空串——
        # 空串经清洗返回空结果，由上游对空语句的兜底判断承接
        content = llm_output.content
        cypher = self._clean_cypher(content if isinstance(content, str) else "")

        # 返回更正后的语句
        return cypher

    def _format_record_as_text(self, record: dict[str, Any]) -> str:
        """将 Cypher 查询结果记录格式化为自然语言文本。

        将形如 {'s.sku_name': 'xxx', 'tm.trademark_name': 'yyy'} 的结果
        转换为易于 LLM 理解的文本格式。

        核心功能：把 Cypher 记录字典翻译为 "中文字段名: 值" 的逗号连接文本。

        业务流程：1. 先按完整键（含变量前缀）查友好名映射；
              2. 未命中则剥掉 "变量." 前缀再查；仍未命中用原始键；
              3. 跳过 None/空值字段；4. 用中文逗号连接。

        副作用：无——纯格式化。

        参数：
            record: Cypher 记录转成的字典（dict(rec) 产物），键为返回别名。

        返回：格式化文本；无有效字段时返回 str(record) 原样兜底。
        """
        # 字段名映射：将 Cypher 返回的别名映射为友好的中文名
        # 同时覆盖"裸键"与"变量.键"两种形态（LLM 生成的返回别名不固定）
        field_name_map = {
            # SKU 相关
            "sku_name": "商品名称",
            "s.sku_name": "商品名称",
            "sku_price": "价格",
            "s.sku_price": "价格",
            "sku_desc": "商品描述",
            "s.sku_desc": "商品描述",
            # SPU 相关
            "spu_name": "SPU名称",
            "sp.spu_name": "SPU名称",
            # 品牌相关
            "trademark_name": "品牌",
            "tm.trademark_name": "品牌",
            # 分类相关
            "category1_name": "一级分类",
            "c1.category1_name": "一级分类",
            "category2_name": "二级分类",
            "c2.category2_name": "二级分类",
            "category3_name": "三级分类",
            "c3.category3_name": "三级分类",
            # 属性相关
            "attr_name": "属性名",
            "a.attr_name": "属性名",
            "attr_value": "属性值",
            "av.attr_value": "属性值",
            # 用户相关
            "user_id": "用户ID",
            "u.user_id": "用户ID",
        }

        # 累积 "友好名: 值" 片段的列表
        parts = []
        # 逐字段遍历记录
        for key, value in record.items():
            # 尝试获取友好的字段名
            # 第一优先：完整键精确命中（含变量前缀形态）
            friendly_name = field_name_map.get(key)
            if not friendly_name:
                # 如果没有预定义映射，尝试从键名提取
                # 例如 's.sku_name' -> 'sku_name' -> '商品名称'
                # 判定逻辑：剥掉 "变量." 前缀取末段，再查一次映射
                clean_key = key.split(".")[-1] if "." in key else key
                # 第二优先：裸键命中；两轮未命中则用原始键兜底展示
                friendly_name = field_name_map.get(clean_key, clean_key)

            # 只添加非空值
            # 空值兜底：None 或空白字符串字段不产出片段，保持文本干净
            if value is not None and str(value).strip():
                # 拼接 "友好名: 值" 片段
                parts.append(f"{friendly_name}: {value}")

        # 有片段则中文逗号连接；无有效字段时原样返回字典字符串
        return "，".join(parts) if parts else str(record)

    def _determine_source(self, record: dict[str, Any]) -> str:
        """根据查询结果字段确定数据来源类型。

        核心功能：从记录键名推断命中的知识域，作为 SearchResult.metadata.source。

        业务流程：把全部键名小写串接后按优先级匹配 sku/spu/trademark/
              category/attr/user 关键词，逐一未命中则归入"知识库"。

        副作用：无——纯字符串判定。

        参数：
            record: Cypher 记录字典（键为返回别名）。

        返回：来源类型中文名（如 "商品信息"），兜底返回 "知识库"。
        """
        # 收集全部键名集合
        keys = set(record.keys())
        # 集合转小写字符串：便于一次 contains 匹配多别名（含变量前缀键）
        key_str = str(keys).lower()

        # 判定链：按业务粒度从细到粗依次匹配关键词
        if "sku" in key_str:
            # 命中 SKU 字段：商品级信息
            return "商品信息"
        elif "spu" in key_str:
            # 命中 SPU 字段：产品级信息
            return "产品信息"
        elif "trademark" in key_str:
            # 命中品牌字段
            return "品牌信息"
        elif "category" in key_str:
            # 命中分类字段
            return "分类信息"
        elif "attr" in key_str:
            # 命中属性字段
            return "属性信息"
        elif "user" in key_str:
            # 命中用户字段
            return "用户信息"
        else:
            # 全部未命中：归入通用知识库来源
            return "知识库"

    async def search(
        self,
        query: str,
        top_k: int = 5,
        tracker_state: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        执行查询
        整个检索流程的主入口，按顺序执行各步骤并返回结果。

        核心功能：串起六步 Text2Cypher 流程，把图谱查询结果格式化为 SearchResult 列表。

        业务流程：1. 空查询兜底返回空列表；
              2. 从 tracker 提取 user_id 与聊天历史；
              3. 路由标签 → 检索入口节点（top_k=10）→ 生成 Cypher →
              验证 →（有错时）校正 → CypherQueryCorrector 校正关系方向；
              4. 执行最终语句，逐记录格式化为 SearchResult（score 固定 1.0，
              metadata 含来源与原始数据）；执行异常仅告警不中断，返回已收集结果。

        副作用：调用 LLM API 多次；对 Neo4j 只读；不修改库数据。

        参数：
            query: 用户查询文本，字符串；空/纯空白返回空列表。
            top_k: 入口节点检索数量（本方法固定以 10 调 node_retrieval，
                   此参数为基类签名保留），正整数。
            tracker_state: 对话状态字典（slots/events），允许 None。

        返回：SearchResult 列表；查询失败/无结果时返回空列表，不抛异常。
        """

        # 如果查询为空，则返回空列表
        # 空值兜底：None 转空串并去首尾空白
        query = (query or "").strip()
        # 纯空查询无检索意义，直接返回空结果
        if not query:
            # 空列表短路返回
            return []

        # 获取用户ID
        user_id = None
        # 仅在有对话状态时尝试提取 user_id 槽值
        if tracker_state:
            # 从 slots 字典取 user_id（切换用户后路由 prompt 需要它标注身份）
            user_id = tracker_state.get("slots", {}).get("user_id")
        # 获取聊天历史
        # 有状态时提取最近聊天；无状态时历史退化为当前查询本身
        chat_history = (
            get_chat_history(tracker_state, user_id) if tracker_state else query
        )
        # 获取入口节点标签
        # 第一步：LLM 路由出 (标签, 实体) 对
        route_res = await self.route_label(chat_history)
        # 检索入口节点
        # 第二步：混合检索定位入口节点（固定取 10 条候选）
        entry_nodes = await self.node_retrieval(route_res, 10)
        # 生成 Cypher 语句
        # 第三步：LLM 依 schema 与入口节点生成语句
        cypher = await self.generate_cypher(query, entry_nodes)
        # 验证 Cypher 语句
        # 第四步：explain 语法检查 + LLM 逻辑审查
        errors = await self.validate_cypher(query, entry_nodes, cypher)
        # 校正 Cypher 语句
        # 判定逻辑：仅在有错误时才多花一次 LLM 调用做校正
        if errors:
            # 第五步：按错误清单校正语句
            cypher = await self.correct_cypher(query, entry_nodes, cypher, errors)
        # 校正关系方向。如果某个关系和其反向关系都不合法，会返回空字符串
        # 第六步前置：基于库 schema 的确定性关系方向校正（langchain API）
        cypher = self.cypher_corrector(cypher)
        # 日志留痕：校正后待执行的最终语句
        logger.info("Cypher校正:%s", cypher)
        # 执行 Cypher 语句
        # 结果累积列表
        results = []
        # 执行可能因语句被校正为空串/库异常而失败——兜底不中断服务
        try:
            # 执行最终语句并取全部记录
            # 驱动注解误报：语句经 LLM 生成→校正→验证链路动态构造（同上 explain
            # 检查），非字面量触发 LiteralString 参数类型误报，按惯例压制
            records = self.driver.execute_query(cypher).records  # pyright: ignore[reportCallIssue, reportArgumentType]
            # 逐条记录格式化
            for rec in records:
                # neo4j Record 转纯字典（键为返回别名）
                record_dict = dict(rec)
                # 格式化为自然语言文本
                # 翻译为 "中文字段名: 值" 文本，供下游 LLM/用户阅读
                text = self._format_record_as_text(record_dict)
                # 确定数据来源
                # 依据键名推断知识域
                source = self._determine_source(record_dict)
                # 构建 SearchResult，设置友好的 text 和 metadata
                # 分数固定 1.0：Text2Cypher 结果为精确查询命中，不做相似度排序
                results.append(
                    SearchResult(
                        # 友好文本
                        text=text,
                        # 元数据：来源类型 + 原始记录（供下游追溯）
                        metadata={"source": source, "raw_data": record_dict},
                        # 固定得分
                        score=1.0,
                    )
                )
        except Exception as e:
            # 异常兜底：记录告警后返回已收集结果，保证检索失败不拖垮对话服务
            logger.warning("执行Cypher语句异常: %s", e)
        # 日志留痕：最终检索结果
        logger.info("检索结果: %s", results)
        # 返回结果列表（可能为空）
        return results

    def close(self):
        """关闭连接

        核心功能：释放 Neo4j 驱动连接。

        业务流程：driver 存在时调用 close() 并记录日志。

        副作用：关闭 Neo4j 网络连接。

        参数：无。

        返回：无返回值（None）。
        """
        # 判定逻辑：driver 仅在 connect() 后才存在，未连接时无需关闭
        if self.driver:
            # 关闭驱动及其连接池
            self.driver.close()
            # 日志留痕：连接已释放
            logger.info("Neo4j 连接已关闭")


if __name__ == "__main__":
    # 检索测试
    # 测试块专用导入：异步运行器与 .env 加载（避免污染模块级导入）
    import asyncio

    from dotenv import load_dotenv

    # 加载 .env：为 LLM 读取 DASHSCOPE_API_KEY（须在 ChatTongyi 调用前生效）
    load_dotenv()

    # 本地 Neo4j 连接地址（联调专用）
    neo4j_url = "bolt://127.0.0.1:7687"
    # 本地默认用户名
    neo4j_user = "neo4j"
    # 本地开发库密码（联调专用，硬编码）
    neo4j_password = "12345678"

    async def test_retrieval(query):
        """测试检索过程

        业务流程：创建 GraphRAG → 连接本地 Neo4j → 以模拟 tracker_state
              执行一次 search → 打印结果数与前 200 字符 → 关闭连接。

        副作用：调用 LLM API；连接本地 Neo4j（只读）。

        参数：
            query: 测试查询文本，字符串。

        返回：无返回值（None），结果打印到控制台。
        """
        # 构造检索器（不注入 embeddings，走 _init_embeddings 延迟加载本地模型）
        graphrag = GraphRAG()
        # 以硬编码本地凭据连接并完成全部装配
        graphrag.connect(
            {
                # 连接地址
                "uri": neo4j_url,
                # 用户名
                "user": neo4j_user,
                # 密码
                "password": neo4j_password,
            }
        )
        # 执行完整检索流程：模拟含 user_id 槽与一条用户事件的最小 tracker_state
        results = await graphrag.search(
            query,
            # 入口节点检索条数（签名保留参数）
            top_k=5,
            # 模拟对话状态：验证用户身份注入与聊天历史链路
            tracker_state={
                # 用户槽位：user_id=25
                "slots": {"user_id": "25"},
                # 事件：当前查询作为用户消息
                "events": [{"event": "user", "text": query}],
            },
        )
        # 打印结果总数
        print(f"检索结果: {len(results)} 条")
        # 逐条打印序号与截断后的文本（超 200 字符截断）
        for i, r in enumerate(results):
            # 三元表达式：长文本截断展示
            print(f"{i + 1}. {r.text[:200] if len(r.text) > 200 else r.text}")
        # 释放 Neo4j 连接
        graphrag.close()

    # 默认测试查询（下方注释为历史测试用例，可替换验证不同场景）
    query = "手机有哪些商品？"
    # query = "白色256GB的手机有哪些？"
    # query = "非有机的大米有哪些，都是什么品牌的？"
    # query = "我想找一款16英寸左右，32G内存2TB硬盘的笔记本，屏幕要求2.5K以上"
    # query = "我之前看到过一款平板电视还不错，我记得是70多寸8K的，能帮我找下是哪个吗"
    # query = "有没有带保湿功能的润唇膏，都是什么品牌的，帮我详细介绍下"
    # query = "帮我推荐oppo的手机"
    # 异步运行测试主流程
    asyncio.run(test_retrieval(query))
