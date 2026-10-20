"""
flow_retriever - Flow 语义检索器模块

核心功能：提供 Flow 维度的向量语义检索——FlowRetriever 将对话理解层
          的 Flow 列表（名称 + 描述 + 触发条件）逐个向量化建立内存
          索引，检索时对查询文本向量化，按余弦相似度过滤并排序，返回
          最相关的 top_k 个 Flow（封装为通用 SearchResult）；另提供
          按 ID 定位 Flow 文档与删除单个 Flow 的辅助能力。

执行流程：1. Agent 装配时创建 FlowRetriever（注入或经工厂默认创建
          Embedder）；2. 首次加载对话配置后调用 index_flows——对每个
          Flow 拼接"名称 + 描述 + 触发条件"文本、向量化、构建
          FlowDocument 内存索引；3. 对话理解节点以用户输入调用 search，
          返回按相似度降序的候选 Flow 列表；4. 下游依据 flow_id 取回
          Flow 执行。

副作用：调用 Embedder 产生嵌入计算（本地模型推理或云 API 网络请求）；
        维护实例级内存索引与构建标记（纯内存，不落盘）；日志记录
        索引完成信息。

主要依赖：numpy（点积相似度计算）、app.ics_ai.retrieval.base_retriever
          （Retriever 基类与 Document/SearchResult 数据结构）、
          app.ics_ai.retrieval.embedder（向量嵌入器及工厂）、
          app.ics_ai.dialogue_understanding.flow（Flow/FlowsList，仅
          类型检查期导入）。
"""

# 延迟注解求值：所有类型注解按字符串处理，规避前向引用与版本差异问题
# （from __future__ 导入必须位于文件最前，仅可在 docstring 之后）。
from __future__ import annotations

# 从标准库 dataclasses 导入数据类工具：FlowDocument 载体自动生成构造/
# 比较方法，field 支持可变默认值（列表）。
from dataclasses import dataclass, field

# 导入 logging 模块：模块级 logger 记录索引完成与未建索引告警。
import logging

# 从标准库 typing 导入 Any 与 TYPE_CHECKING：开放参数类型 + 仅类型
# 检查执行的导入分支标记。
from typing import Any, TYPE_CHECKING

# 导入 numpy：向量点积相似度计算（np.dot）。
import numpy as np

# 从同包基类导入检索抽象接口与数据结构：FlowRetriever 继承 Retriever，
# 检索结果以 Document + SearchResult 封装返回。
from app.ics_ai.retrieval.base_retriever import Retriever, Document, SearchResult

# 从同包嵌入器模块导入抽象接口与工厂：向量化能力来源，未注入时由
# create_embedder 按环境配置默认创建。
from app.ics_ai.retrieval.embedder import Embedder, create_embedder

# 仅类型检查分支：Flow/FlowsList 仅用于注解，运行时以鸭子类型访问其
# 属性（id/name/description/triggers），避免对话理解模块的循环导入。
if TYPE_CHECKING:
    # Flow 定义与集合类型
    from app.ics_ai.dialogue_understanding.flow import Flow, FlowsList

# 模块级日志器：以模块路径命名，继承全局日志配置
logger = logging.getLogger(__name__)


@dataclass
class FlowDocument:
    """Flow 文档数据类

    职责：作为单个 Flow 的检索索引条目，承载其标识信息与向量表示。

    设计模式：数据传输对象（DTO）——仅承载数据不含业务逻辑。

    生命周期：index_flows 时构造并加入内存索引，随索引重建清除或经
          delete_document 移除。

    Attributes:
        flow_id: Flow ID
        flow_name: Flow名称
        description: Flow描述
        triggers: 触发条件
        embedding: 向量表示
    """

    # Flow 唯一标识：对应 Flow.id，检索结果的 document.id 取自此字段
    flow_id: str

    # Flow 名称：参与检索文本拼接与结果元数据
    flow_name: str

    # Flow 描述：参与检索文本拼接；作为结果 Document 的正文内容
    description: str

    # 触发条件短语列表：参与检索文本拼接；缺省为空列表
    triggers: list[str] = field(default_factory=list)

    # Flow 文本的嵌入向量；索引构建成功后填充，失败保持 None（检索时跳过）
    embedding: np.ndarray | None = None


class FlowRetriever(Retriever):
    """Flow 检索器

    职责：根据用户输入检索最相关的 Flow——维护 Flow 向量索引（构建/
          重建/删除），并以余弦相似度执行阈值过滤 + top_k 截断的语义
          搜索。

    设计模式：策略模式——Retriever 抽象接口的 Flow 专用检索实现；
          嵌入能力经构造注入（依赖注入），可替换为任意 Embedder 实现。

    生命周期：Agent 装配时创建，index_flows 完成后整个运行期常驻复用；
          配置重载时再次调用 index_flows 整体重建索引。

    Attributes:
        embedder: 向量嵌入器
        top_k: 默认返回结果数量
        threshold: 相似度过滤阈值
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        top_k: int = 3,
        threshold: float = 0.5,
    ):
        """初始化 Flow 检索器

        核心功能：装配嵌入器与检索参数，初始化空索引。

        业务流程：1. 调用父类构造；2. 保存注入的嵌入器（未注入时经
                  工厂按环境配置创建默认实现）；3. 保存 top_k 与阈值；
                  4. 初始化空文档列表与未构建标记。

        副作用：embedder 未注入时会触发 create_embedder 的环境配置读取
                （.env/API 密钥）；不建索引、不发起嵌入计算。

        @param embedder: 向量嵌入器实例；允许 None（走默认工厂创建）。
        @param top_k: 默认返回结果数量，正整数，默认 3。
        @param threshold: 相似度过滤阈值，默认 0.5（仅归一化向量下
                          语义为余弦相似度下限）。
        @return: 无返回。
        """
        # 调用抽象基类 Retriever 的构造：完成基类侧的初始化
        super().__init__()

        # 保存嵌入器：未注入时经工厂按环境配置创建（云 API 或本地模型）
        self.embedder = embedder or create_embedder()

        # 保存默认返回数量：search 未显式指定时使用
        self.top_k = top_k

        # 保存相似度阈值：低于该值的结果被过滤
        self.threshold = threshold

        # Flow 文档内存索引：index_flows 填充，初始为空
        self._flow_documents: list[FlowDocument] = []

        # 索引构建标记：False 时 search 直接返回空（防误检）
        self._index_built = False

    def connect(self, config: dict[str, Any] | None = None) -> None:
        """连接/初始化向量索引

        核心功能：满足基类抽象契约的连接方法——本检索器为纯内存索引，
                  无外部连接可建，实现为空操作。

        业务流程：不做任何初始化动作；真正的索引构建由 index_flows
                  承担（Flow 来源是对话配置而非外部向量库），config
                  仅为兼容基类签名而保留。

        副作用：无。

        @param config: 基类约定的连接配置；本实现忽略该参数（允许 None）。
        @return: 无返回。
        """
        # 空操作实现：内存索引无需外部连接，占位满足抽象契约
        return

    async def index_flows(self, flows: list["Flow"]) -> None:
        """索引 Flow 列表

        核心功能：将全部 Flow 向量化后重建内存索引（全量替换语义）。

        业务流程：1. 清空既有索引；2. 逐个 Flow 构造 FlowDocument
                  （triggers 经 getattr 鸭子类型探测，Flow 类型声明中
                  可能无此字段）；3. 拼接"名称 + 描述 + 触发条件"为
                  检索文本（三者从强到弱影响匹配）；4. 异步向量化并
                  回填 embedding；5. 全部完成后置索引标记并记录日志。

        副作用：逐 Flow 调用 embedder.embed（模型推理/网络请求）；清空
                并重建实例级索引状态；写一条 info 日志。

        @param flows: 待索引的 Flow 列表；允许为空列表（等效清空索引）。
        @return: 无返回；嵌入失败沿 embedder 异常向上抛（中断索引）。
        """
        # 第一步：清空既有索引——本方法为全量重建语义，不做增量追加
        self._flow_documents.clear()

        # 第二步：遍历 Flow 逐个构建索引条目
        for flow in flows:
            # 构建索引条目：triggers 为类型声明外的可选属性，
            # 用 getattr 鸭子类型探测，缺失时兜底空列表；
            # name 声明为 str | None（DSL 允许缺省），经 or "" 兜底为
            # 空串以满足 FlowDocument 的必填 str（与 description 同型兜底）
            doc = FlowDocument(
                flow_id=flow.id,
                flow_name=flow.name or "",
                description=flow.description or "",
                triggers=getattr(flow, "triggers", []),
            )

            # 第三步：拼接检索文本——名称必选；描述与触发条件存在才追加
            # （复用 doc.flow_name：已兜底空串，规避 flow.name 的 None 类型）
            text_parts = [doc.flow_name]

            # 条件追加：描述非空才拼入（或语义时已兜底空串，此处二次守卫）
            if flow.description:
                # 追加 Flow 描述，补充语义细节
                text_parts.append(flow.description)

            # 条件追加：触发条件列表非空才拼入
            if doc.triggers:
                # 追加全部触发短语（extend 展开逐个并入）
                text_parts.extend(doc.triggers)

            # 以空格连接各部分为单条检索文本
            text = " ".join(text_parts)

            # 第四步：异步生成嵌入向量并回填到文档条目
            doc.embedding = await self.embedder.embed(text)

            # 将完成的条目加入内存索引
            self._flow_documents.append(doc)

        # 第五步：置位构建标记——后续 search 允许执行
        self._index_built = True

        # 记录索引完成日志：条目数便于核对配置加载结果
        logger.info(f"Indexed {len(self._flow_documents)} flows")

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        tracker_state: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """搜索最相关的 Flow

        核心功能：对查询文本执行向量语义检索，返回按相似度降序、经
                  阈值过滤与数量截断的候选 Flow 列表。

        业务流程：1. 未建索引时告警并返回空列表（防误检）；2. 解析
                  生效的 top_k（未传/传 0 时回退默认值）；3. 向量化
                  查询文本；4. 遍历索引逐条计算点积相似度（embedder
                  默认 L2 归一化，归一化向量的点积即余弦相似度）；5.
                  低于阈值的丢弃；6. 余下的按分数降序排序并截取 top_k。

        副作用：调用 embedder.embed 一次（模型推理/网络请求）；只读
                索引，不修改实例状态。

        @param query: 查询文本（通常为用户原始输入），不能为 None。
        @param top_k: 本次返回数量；None 或 0 时使用构造默认值
                      （int | None 为基类 int 的超类型，override 兼容）。
        @param tracker_state: 对话状态字典——仅为满足基类抽象契约而
                              保留的参数，本检索器不做个性化检索，
                              不消费该值（允许 None）。
        @return: SearchResult 列表（按分数降序，至多 top_k 条）；未建
                 索引或无命中返回空列表。
        """
        # 前置守卫：索引未构建时无法检索，告警并返回空避免误检
        if not self._index_built:
            # 写警告日志提示调用方先执行 index_flows
            logger.warning("Flow index not built")

            # 返回空列表：调用方按"无候选"处理
            return []

        # 解析生效 top_k：未传或传 0（falsy）均回退构造默认值
        top_k = top_k or self.top_k

        # 第三步：生成查询向量（与索引向量同一嵌入空间——同一 embedder）
        query_embedding = await self.embedder.embed(query)

        # 初始化候选结果收集列表
        results = []

        # 第四步：遍历索引逐条计算相似度
        for doc in self._flow_documents:
            # 守卫跳过：嵌入缺失的条目（索引构建中途失败）不参与比对
            if doc.embedding is None:
                # 跳过该条目，继续下一条
                continue

            # 计算查询向量与条目向量的点积：embedder 默认 L2 归一化，
            # 归一化向量的点积即余弦相似度（float 转换脱离 numpy 类型）
            similarity = float(np.dot(query_embedding, doc.embedding))

            # 阈值过滤：低于阈值的候选不进入结果（控制召回质量）
            if similarity >= self.threshold:
                # 封装为通用检索结果：正文取 Flow 描述，标识与触发条件
                # 放入元数据供下游按 flow_id 取回 Flow 执行
                results.append(
                    SearchResult(
                        document=Document(
                            id=doc.flow_id,
                            content=doc.description,
                            metadata={
                                "flow_id": doc.flow_id,
                                "flow_name": doc.flow_name,
                                "triggers": doc.triggers,
                            },
                        ),
                        score=similarity,
                    )
                )

        # 第六步：按分数降序排序（相似度最高者优先）
        results.sort(key=lambda x: x.score, reverse=True)

        # 截取前 top_k 条返回
        return results[:top_k]

    async def add_document(self, document: Document) -> None:
        """添加文档（不支持）。

        核心功能：基类接口的显式拒绝实现——Flow 必须整体经 index_flows
                  索引（需向量化），单个 Document 无法生成嵌入。

        业务流程：直接抛 NotImplementedError 指引调用方使用 index_flows。

        副作用：无。

        @param document: 基类要求的文档参数（本实现不使用）。
        @return: 无返回；调用即抛 NotImplementedError（不返回）。
        """
        # 显式拒绝：给出正确的索引入口指引
        raise NotImplementedError("Use index_flows to add flows")

    async def delete_document(self, doc_id: str) -> bool:
        """删除文档。

        核心功能：从索引中移除指定 flow_id 的条目。

        业务流程：线性遍历索引，首个 flow_id 匹配的条目按位删除并
                  返回 True；遍历结束未命中返回 False。

        副作用：修改实例级索引状态（命中时移除一个条目）。

        @param doc_id: 待删除的 Flow ID（对应 FlowDocument.flow_id）。
        @return: 删除成功返回 True；ID 不存在返回 False。
        """
        # 遍历索引按 ID 线性查找
        for i, doc in enumerate(self._flow_documents):
            # ID 匹配判定：首个命中即删除
            if doc.flow_id == doc_id:
                # 按索引位弹出条目（保持列表其余元素顺序）
                self._flow_documents.pop(i)

                # 返回删除成功
                return True

        # 遍历完成无命中：返回失败标记
        return False

    def get_flow_by_id(self, flow_id: str) -> FlowDocument | None:
        """根据 ID 获取 Flow 文档。

        核心功能：按 flow_id 在索引中线性查找条目（同步方法，纯内存）。

        业务流程：遍历索引，flow_id 匹配即返回该条目；遍历结束返回 None。

        副作用：无（只读查找）。

        @param flow_id: Flow ID。
        @return: 匹配的 FlowDocument；不存在返回 None。
        """
        # 遍历索引按 ID 查找
        for doc in self._flow_documents:
            # ID 匹配即返回条目
            if doc.flow_id == flow_id:
                # 返回命中的文档
                return doc

        # 未命中：返回 None 表示无此 Flow
        return None


# 导出清单：按字母序排列，约束 from ... import * 的导出范围并为 IDE
# 提供准确的补全与跳转依据。
__all__ = [
    # Flow 索引条目数据类
    "FlowDocument",
    # Flow 语义检索器
    "FlowRetriever",
]
