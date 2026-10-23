# neo4j-admin database load --from-path="neo4j.dump文件所在目录" --overwrite-destination=true neo4j --verbose
"""
create_indexing.py - Neo4j 向量/全文索引构建脚本（离线数据准备工具）

核心功能：
    连接本地 Neo4j 图数据库，为电商知识图谱的七类节点（三级分类/商标/
    SPU/SKU/属性）批量构建向量索引（嵌入器统一工厂创建，线上优先/本地兜底）
    与全文索引（jieba 分词）。产物供 addons/information_retrieval.py 的
    HybridRetriever（向量+全文混合检索）查询。
    （上方首行注释为建库提示：图数据需先用 neo4j-admin 从 dump 导入，
    本脚本只负责在导入后的库上建索引。）

执行流程（__main__ 入口）：
    1. 清空处理：删除全部约束（约束自带的索引随约束一并删除），
       再删除独立的普通索引；
    2. 向量索引：清空全部节点 embedding 属性 → 逐标签创建向量索引，
       对 embedding 为空的节点批量计算并写入嵌入向量；
    3. 全文索引：清空全部节点 fulltext 属性 → 逐标签创建全文索引，
       对 fulltext 为空的节点做 jieba 分词并批量写入。

副作用：
    - import 期：模块级经嵌入器统一工厂创建嵌入器——SILICONFLOW_API_KEY
      已配置时创建线上客户端（不联网），并做一次嵌入调用探测向量维度
      （1 次 HTTP 请求）；未配置 Key 时立即加载本地 SentenceTransformer
      模型（数百 MB、有耗时，模型路径经 EMBEDDING_MODEL_PATH 环境变量覆盖）；
    - 运行期：破坏性清库——删除并重建约束/索引，清空并重写节点的
      embedding/fulltext 属性；仅连接本地 Neo4j，不触碰 MySQL；
    - 本脚本不在服务运行链路上（不随 Agent 启动被导入），独立入口在 __main__。

主要依赖：
    - neo4j（官方驱动）、neo4j_graphrag.indexes（向量/全文索引封装）
    - app.ics_ecs.addons.embedding_factory（嵌入器统一工厂，与检索端共用）
    - jieba（中文分词）
"""

# re：正则编译，用于全文索引分词结果的字符过滤
import re

# sys/pathlib：脚本自举——独立运行（python create_indexing.py）时 sys.path[0]
# 为 addons 目录，项目根包 app 不可见；把包根上级目录（src）插入搜索路径，
# 保证下方工厂等项目内部导入在任何工作目录下均可解析
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# jieba：中文分词——节点名称切成词后才能被 Neo4j 全文索引按词命中
import jieba

# logging：标准库日志，输出索引构建进度
import logging

# GraphDatabase：Neo4j 官方驱动工厂，__main__ 入口用它创建数据库连接
from neo4j import GraphDatabase

# neo4j_graphrag.indexes：图检索库的索引构建封装（与检索端 HybridRetriever 同源）
from neo4j_graphrag.indexes import (
    # 在库中创建向量索引
    create_vector_index,
    # 按节点 elementId 批量写入嵌入向量属性
    upsert_vectors,
    # 在库中创建全文索引
    create_fulltext_index,
)

# create_embedder：嵌入器统一工厂（线上硅基流动优先，本地 SentenceTransformer 兜底），
# 与检索端 information_retrieval.py 共用同一嵌入方案，保证写入与查询向量维度一致
from app.ics_ecs.addons.embedding_factory import create_embedder

# 配置控制台日志
# 获取名为 "indexing" 的命名 logger（与根 logger 隔离，便于按名过滤）
logger = logging.getLogger("indexing")
# 设置日志级别为 INFO：进度信息可见，DEBUG 噪音不输出
logger.setLevel(logging.INFO)
# 守卫：仅在未挂载过 handler 时添加——重复导入/重跑不会重复输出日志
if not logger.handlers:
    # 定义日志格式："[级别]时间: 消息"（时间精确到毫秒）
    formatter = logging.Formatter("[%(levelname)s]%(asctime)s: %(message)s")
    # 控制台流 handler：日志输出到标准错误/标准输出
    handler = logging.StreamHandler()
    # 为 handler 应用上述格式
    handler.setFormatter(formatter)
    # 挂载 handler 到 logger
    logger.addHandler(handler)


def drop_constraint(driver):
    """删除所有约束

    核心功能：清空数据库中的全部约束（约束会自动创建索引，删约束同时删除其索引）。

    业务流程：1. 查询库内全部约束；2. 逐条带 IF EXISTS 删除，保证幂等。

    副作用：删除 Neo4j 库内全部约束及其附属索引（破坏性操作）。

    参数：
        driver: 已连接的 Neo4j 驱动实例（GraphDatabase.driver 产物），不为 None。

    返回：无返回值（None）。
    """
    # 查询库内全部约束，取结果记录迭代器（每条含约束名 name 等元数据）
    records = driver.execute_query("show constraints").records
    # 逐条遍历约束记录
    for record in records:
        # 带 IF EXISTS 删除约束：不存在时不报错，保证重复执行幂等
        driver.execute_query(f"drop constraint {record['name']} if exists")


def drop_index_without_constraint(driver):
    """删除所有没有约束的索引

    核心功能：清理独立索引——约束自带的索引随约束删除，这里只删"无约束归属"的索引。

    业务流程：1. 查询库内全部索引；2. 过滤掉有约束归属的；3. 逐条带 IF EXISTS 删除。

    副作用：删除 Neo4j 库内全部独立索引（破坏性操作）。

    参数：
        driver: 已连接的 Neo4j 驱动实例，不为 None。

    返回：无返回值（None）。
    """
    # 查询库内全部索引，取结果记录迭代器（含 owningConstraint 归属信息）
    records = driver.execute_query("show index").records
    # 逐条遍历索引记录
    for record in records:
        # 判定逻辑：owningConstraint 非空说明索引由约束自带（已随约束删除），
        # 跳过；只删除独立创建的普通索引
        if not record["owningConstraint"]:
            # 带 IF EXISTS 删除索引：不存在时不报错，保证幂等
            driver.execute_query(f"drop index {record['name']} if exists")


# --------- 创建向量索引 ---------
# 嵌入器统一工厂：SILICONFLOW_API_KEY 已配置 → 线上 OpenAIEmbeddings（仅创建客户端）；
# 未配置 → 立即加载本地 SentenceTransformer 模型（权重常驻内存）
embed_model = create_embedder()

# 嵌入向量维度：对单条探测文本实测一次嵌入，取返回向量长度作为索引维度——
# 线上（BAAI/bge-m3，1024 维）与本地（bge-m3 快照，1024 维）维度一致，
# 运行时探测仍能保证向量索引 dimensions 与实际写入/查询向量严格一致，避免硬编码漂移；
# 副作用：线上分支为 1 次 HTTP 请求，本地分支为 1 次模型推理
vector_dim = len(embed_model.embed_query("维度探测"))
# 日志留痕：打印实测维度与嵌入器类型，便于核对索引与查询两端一致
logger.info("嵌入器: %s，向量维度: %d", type(embed_model).__name__, vector_dim)


def vector_indexing(driver, label, property="name"):
    """创建向量索引，并添加嵌入向量

    核心功能：为指定标签建向量索引，并给该标签下 embedding 为空的节点计算、写入嵌入向量。

    业务流程：1. 创建 {label 小写}_vector 向量索引（维度取嵌入器实测值，余弦相似度）；
              2. 查询 embedding 为 null 的节点，取 elementId 与文本属性值；
              3. 无待处理节点则直接返回（幂等：重复运行只算增量）；
              4. 批量计算文本嵌入向量（经统一工厂嵌入器）；
              5. 按 elementId 批量写入 embedding 属性。

    副作用：在 Neo4j 创建索引、写入节点 embedding 属性；调用嵌入器（线上 HTTP
            或本地模型推理）。

    参数：
        driver: 已连接的 Neo4j 驱动实例，不为 None。
        label: 节点标签名（如 "SPU"），字符串，决定索引与查询范围。
        property: 计算嵌入所用的节点文本属性名，字符串，缺省 "name"。

    返回：无返回值（None）；无待处理节点时提前返回。
    """

    # 创建向量索引
    create_vector_index(
        # driver：已连接的 Neo4j 驱动实例
        driver,
        name=f"{label.lower()}_vector",  # 索引的唯一名称（标签小写 + _vector 后缀）
        label=label,  # 要索引的节点标签
        embedding_property="embedding",  # 包含嵌入向量值的节点属性键
        dimensions=vector_dim,  # 向量嵌入维度：取嵌入器实测维度，与索引保持一致
        similarity_fn="cosine",  # 向量相似度函数，可选值为 "euclidean"（欧几里得距离）或 "cosine"（余弦相似度）
    )

    # 查询 embedding 为 null 的节点，获取 elementId 和 指定属性
    # 幂等设计：只处理尚未计算向量的节点，重复运行不会重复计算（断点续跑）
    record_list = driver.execute_query(
        # f-string 拼 Cypher：按标签过滤 embedding 为空的节点，取元素 ID 与文本属性
        # （label/property 来自脚本内部调用，非用户输入，无注入风险）
        f"""match (n:{label}) where n.embedding is null
            return elementId(n) as id, n.{property} as text""",
    ).records
    # 整理为 (节点ID, 文本) 元组列表，便于后续转置
    record_tuple_list = [(r["id"], r["text"]) for r in record_list]
    # 空值兜底：无待处理节点说明全部已计算过，记录日志并直接返回
    if not record_tuple_list:
        # 日志留痕：该标签下所有节点均已有嵌入向量
        logger.info(f"{label} 所有节点皆存在嵌入向量")
        # 提前返回，跳过计算与写入
        return
    # [('id1', 'text1'), ('id2', 'text2')] =》 zip(('id1', 'text1'), ('id2', 'text2'))=>[('id1', 'id2'), ('text1', 'text2')]
    # zip(*) 转置：把 (id, text) 对列表拆成独立的 id 序列与 text 序列
    ids, texts = zip(*record_tuple_list)
    # 转为列表：zip 对象不便切片/取长度，转为 list 便于后续批量操作
    ids = list(ids)
    # 同上，文本序列转 list
    texts = list(texts)

    # 计算文本的嵌入向量
    # 日志留痕：待计算节点数，便于观察进度
    logger.info(f"计算 {label} ({len(record_list)}) 的嵌入向量")
    # 批量计算：LangChain Embeddings 接口的批量方法（线上按接口内部分批请求，
    # 本地按适配器内部 batch_size 推理），返回与输入等长的向量嵌套列表
    embeddings = embed_model.embed_documents(texts)

    # 按 elementId 添加嵌入向量属性
    # 日志留痕：写入节点数
    logger.info(f"写入 {label} ({len(record_list)}) 的嵌入向量")
    # 批量写入：按节点 ID 将向量写入 embedding 属性（含属性键与向量数组）
    upsert_vectors(
        # driver：已连接的 Neo4j 驱动实例
        driver,
        ids=ids,  # 待写入的节点 elementId 列表
        embedding_property="embedding",  # 写入的属性键（与索引定义一致）
        # 与 ids 一一对应的嵌入向量数组（embed_documents 返回 list[list[float]]，
        # 与库内注解 List[List[float]] 天然匹配）
        embeddings=embeddings,
    )


# --------- 创建全文索引 ---------
def fulltext_indexing(driver, label, property="name"):
    """创建全文索引，并添加节点属性

    核心功能：为指定标签建全文索引，并对 fulltext 为空的节点做中文分词后写入全文文本。

    业务流程：1. 创建 {label 小写}_fulltext 全文索引（索引 fulltext 属性）；
              2. 查询 fulltext 为 null 的节点，取 elementId 与文本属性值；
              3. 无待处理节点则直接返回（幂等）；
              4. jieba 分词并按正则过滤，把保留词以空格连接为全文文本；
              5. 分批 UNWIND 写入节点 fulltext 属性。

    副作用：在 Neo4j 创建索引、写入节点 fulltext 属性。

    参数：
        driver: 已连接的 Neo4j 驱动实例，不为 None。
        label: 节点标签名（如 "SKU"），字符串，决定索引与查询范围。
        property: 生成分词文本所用的节点源属性名，字符串，缺省 "name"。

    返回：无返回值（None）；无待处理节点时提前返回。
    """

    # 创建全文索引
    create_fulltext_index(
        # driver：已连接的 Neo4j 驱动实例
        driver,
        name=f"{label.lower()}_fulltext",  # 索引的唯一名称（标签小写 + _fulltext 后缀）
        label=label,  # 要创建索引的节点标签
        node_properties=[
            "fulltext"
        ],  # 要创建全文索引的节点属性列表（仅索引 fulltext 属性）
    )

    # 查询 fulltext 为 null 的节点，获取 elementId 和 指定属性
    # 幂等设计：只处理尚未生成全文文本的节点，重复运行只算增量
    record_list = driver.execute_query(
        # f-string 拼 Cypher：按标签过滤 fulltext 为空的节点，取元素 ID 与源文本属性
        f"""match (n:{label}) where n.fulltext is null
            return elementId(n) as id, n.{property} as text""",
    ).records
    # 整理为 (节点ID, 源文本) 元组列表，便于后续转置
    record_tuple_list = [(r["id"], r["text"]) for r in record_list]
    # 空值兜底：无待处理节点说明全部已生成过，记录日志并直接返回
    if not record_tuple_list:
        # 日志留痕：该标签下所有节点均已有全文索引属性
        logger.info(f"{label} 所有节点皆存在全文索引属性")
        # 提前返回，跳过分词与写入
        return

    # 文本分词，作为全文索引属性
    # 日志留痕：待处理节点数
    logger.info(f"计算 {label} ({len(record_list)}) 的全文索引")
    # 正则：仅匹配英文字母、数字和中文字符构成的连续片段，
    # 用于过滤分词结果里的纯符号/标点词汇
    pattern = re.compile(r"[a-zA-Z0-9\u4e00-\u9fa5]+")
    # 列表推导：对每个 (节点ID, 源文本) 生成分词后的 (节点ID, 全文文本) 元组
    fulltext_tuple_list = [
        (
            # 节点 elementId 原样保留
            id_,
            # 把保留的分词用空格连接成一段文本（Neo4j 全文索引按空格分词命中）
            " ".join(
                [
                    # 去除词前后空白后再输出
                    word.strip()
                    # jieba.lcut：把源文本切成词列表（精确模式）
                    for word in jieba.lcut(text)
                    # 判定逻辑：仅保留完整匹配正则的词（纯符号/标点词被过滤）
                    if pattern.fullmatch(word.strip())
                ]
            ),
        )
        # 遍历待处理节点列表
        for id_, text in record_tuple_list
    ]
    # zip(*) 转置：把 (id, 全文文本) 对列表拆成独立的 id 序列与全文序列
    ids, fulltexts = zip(*fulltext_tuple_list)
    # 转为列表：zip 对象不便切片，转为 list 便于分批
    ids = list(ids)
    # 同上，全文文本序列转 list
    fulltexts = list(fulltexts)

    # 按 elementId 添加全文索引属性
    # 日志留痕：写入节点数
    logger.info(f"写入 {label} ({len(fulltexts)}) 的全文索引")
    # 单条 Cypher 的批量行数上限：分批写入避免单次查询体积过大
    insert_batch_size = 1000
    # 按批量大小切片遍历：i 为每批起始下标
    for i in range(0, len(record_tuple_list), insert_batch_size):
        # 组装本批的参数行：每行 {id: 节点ID, fulltext: 全文文本}
        batch_rows = [
            {"id": id_, "fulltext": ft}
            # zip 当前批切片后逐行配对
            for id_, ft in zip(
                ids[i : i + insert_batch_size],
                fulltexts[i : i + insert_batch_size],
            )
        ]

        # UNWIND：将列表数据展开为多行记录
        # 批量写库：把 $rows 参数列表展开为逐行记录，按元素 ID 匹配节点并写入 fulltext 属性
        driver.execute_query(
            # UNWIND 将传入的 rows 列表（通过参数传递）展开，每一项作为一行数据，命名为 row
            "UNWIND $rows AS row "
            # 按元素 ID 定位目标节点
            "MATCH (n) "
            # WHERE 条件：节点元素 ID 与当前行的 id 相等
            "WHERE elementId(n) = row.id "
            # 写入 fulltext 属性为当前行的全文文本
            "SET n.fulltext = row.fulltext ",
            # 参数字典：rows 键对应 Cypher 里的 $rows 占位符
            {"rows": batch_rows},
        )


if __name__ == "__main__":
    # 本地 Neo4j 连接地址（Bolt 协议，离线脚本专用，不读配置）
    neo4j_url = "neo4j://127.0.0.1"
    # 本地开发库认证对（用户名, 密码）——本地默认凭据，仅本脚本使用
    neo4j_auth = ("neo4j", "12345678")
    # 上下文管理器：with 块结束（或异常）时自动关闭驱动连接
    with GraphDatabase.driver(neo4j_url, auth=neo4j_auth) as driver:
        # 1、清空处理：确保所有索引和约束都被清理干净，为后续重新创建索引做好准备
        # 清空所有约束。约束会自动创建索引，删除约束同时会删除对应的索引
        drop_constraint(driver)
        # 清空所有没有约束的索引
        drop_index_without_constraint(driver)

        # 2、清空并创建向量索引
        # 清空所有嵌入向量：全图移除 embedding 属性，强制全量重算
        driver.execute_query("match (n) remove n.embedding")
        # 创建向量索引
        # 一级分类节点：按分类名计算嵌入
        vector_indexing(driver, "Category1", "category1_name")
        # 二级分类节点：按分类名计算嵌入
        vector_indexing(driver, "Category2", "category2_name")
        # 三级分类节点：按分类名计算嵌入
        vector_indexing(driver, "Category3", "category3_name")
        # 商标节点：按商标名计算嵌入
        vector_indexing(driver, "Trademark", "trademark_name")
        # SPU（标准产品单元）节点：按 SPU 名计算嵌入
        vector_indexing(driver, "SPU", "spu_name")
        # SKU（库存量单位）节点：按 SKU 名计算嵌入
        vector_indexing(driver, "SKU", "sku_name")
        # 属性节点：按属性值计算嵌入
        vector_indexing(driver, "Attr", "attr_value")

        # 3、清空并创建全文索引
        # 清空所有节点全文索引属性：全图移除 fulltext，强制全量重建
        driver.execute_query("match (n) remove n.fulltext")
        # 创建全文索引

        # 一级分类节点：按分类名生成分词全文
        fulltext_indexing(driver, "Category1", "category1_name")
        # 二级分类节点：按分类名生成分词全文
        fulltext_indexing(driver, "Category2", "category2_name")
        # 三级分类节点：按分类名生成分词全文
        fulltext_indexing(driver, "Category3", "category3_name")
        # 商标节点：按商标名生成分词全文
        fulltext_indexing(driver, "Trademark", "trademark_name")
        # SPU（标准产品单元）节点：按 SPU 名生成分词全文
        fulltext_indexing(driver, "SPU", "spu_name")
        # SKU（库存量单位）节点：按 SKU 名生成分词全文
        fulltext_indexing(driver, "SKU", "sku_name")
        # 属性节点：按属性值生成分词全文
        fulltext_indexing(driver, "Attr", "attr_value")
