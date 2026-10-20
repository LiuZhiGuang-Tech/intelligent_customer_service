"""
base_retriever - 检索器基类模块

核心功能：定义向量检索的最小化抽象接口——connect（连接/初始化索引）+ search
          （语义搜索）；同时提供 Document 与 SearchResult 两个数据类，
          作为检索结果的统一数据契约，供策略层（如 EnterpriseSearchPolicy）
          与具体检索器实现（如 flow_retriever）共同消费。

执行流程：
    1. 模块导入：仅注册抽象基类与数据类，无 IO；
    2. 自定义检索器继承 InformationRetrieval，实现 connect + search；
    3. 调用方先 connect(config) 初始化索引，再 await search(query, top_k)
       获得 list[SearchResult]；
    4. SearchResult.__post_init__ 自动兼容旧式 document 参数——
       document 提供而 text 为空时，回填 text 与 metadata。

副作用：模块自身无副作用（不读写文件/数据库、不发网络请求）；
        具体检索器子类的 connect/search 可能产生真实 IO（见子类文档）。

主要依赖：
    - abc：ABC/abstractmethod 定义抽象契约。
    - dataclasses：dataclass/field 生成两个数据类。
    - typing：Any 通配符 + TYPE_CHECKING 隔离仅注解导入（Embedder）。
    - app.ics_ai.retrieval.embedder.Embedder：可选的向量化器，
      仅类型注解引用（TYPE_CHECKING 分支）。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，支持前向引用
# （如 "Embedder"）并消除运行时注解求值开销。
from __future__ import annotations

# 导入抽象基类工具：ABC 标记检索器基类，abstractmethod 强制子类实现
# connect / search 两个核心方法。
from abc import ABC, abstractmethod

# 导入 dataclass 工具：两个数据类自动生成 __init__/__repr__；
# field(default_factory=...) 为可变默认值（dict）提供独立工厂。
from dataclasses import dataclass, field

# 导入类型注解工具：Any 类型通配符；TYPE_CHECKING 为"仅类型检查期成立"
# 的常量，隔离只服务于注解的导入。
from typing import TYPE_CHECKING, Any

# 类型检查期专属导入：Embedder 只出现在 __init__ 参数注解中，运行时
# 不需要真实导入（避免潜在的导入开销与循环依赖）。
if TYPE_CHECKING:
    # 嵌入向量化器抽象基类（仅类型注解用）。
    from app.ics_ai.retrieval.embedder import Embedder


@dataclass
class Document:
    """文档类 - 向后兼容。

    职责：表示一份被检索的基础文档（id + 正文 + 元数据）；新代码推荐
          直接使用 SearchResult，本类仅为兼容旧式构造方式保留。
    设计模式：数据传输对象（DTO），无行为逻辑。
    生命周期：由具体检索器或调用方按需创建，随结果传递被消费后即可丢弃。

    Attributes:
        id: 文档ID（唯一标识，如向量库 doc key）；空串表示未指定。
        content: 文档正文内容。
        metadata: 元数据字典（来源、页码等自由扩展键值对）。
    """

    # 文档唯一标识：默认空串（调用方未提供时）。
    id: str = ""
    # 文档正文：默认空串。
    content: str = ""
    # 元数据字典：可变默认值经 field 工厂独立创建，避免实例间共享。
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """搜索结果。

    职责：承载单条检索结果——命中文本、元数据、相似度分数；
          向策略层提供 source / content 两个便捷属性读取接口。
    设计模式：数据传输对象（DTO）+ 兼容适配（__post_init__ 桥接旧式
              document 参数与新式 text/metadata 字段）。
    生命周期：由检索器 search 实现创建并返回，被调用方消费后即可丢弃。

    支持两种构造方式：
    1. 新方式：SearchResult(text="...", score=0.9)
    2. 旧方式：SearchResult(document=Document(...), score=0.9)

    Attributes:
        text: 检索到的文本内容。
        metadata: 元数据（来源、页码等）；source 属性从此读取。
        score: 相似度分数（0.0-1.0，越大越相关）；None 表示检索器未评分。
        document: 向后兼容的 Document 对象；构造时若提供且 text 为空，
                  由 __post_init__ 回填 text / metadata。
    """

    # 检索文本：默认空串；旧式构造时由 document.content 回填。
    text: str = ""
    # 元数据字典：可变默认值经 field 工厂独立创建；旧式构造时由
    # document.metadata 拷贝回填。
    metadata: dict[str, Any] = field(default_factory=dict)
    # 相似度分数：默认 None（未评分）；策略层过滤时 None 视为通过。
    score: float | None = None
    # 向后兼容字段：默认 None；仅旧式构造时提供。
    document: Document | None = None

    def __post_init__(self) -> None:
        """初始化后处理：兼容旧的document参数。

        业务流程：document 提供且 text 为空时，从 document 回填
                  text 与 metadata（metadata 仅在为空时拷贝，避免
                  覆盖调用方显式传入的值）。
        副作用：仅修改本实例字段，无外部副作用。
        """
        # 兼容判定：仅当旧式 document 参数提供且新式 text 未填时回填。
        if self.document is not None and not self.text:
            # 回填正文：document.content → text。
            self.text = self.document.content
            # 元数据仅在未显式传入时回填：避免覆盖调用方给的值。
            if not self.metadata:
                # 拷贝 document 的元数据（copy 防止与 Document 共享引用）。
                self.metadata = self.document.metadata.copy()

    @property
    def source(self) -> str:
        """获取来源。

        返回：
            metadata["source"] 的值；缺失时返回 "unknown" 占位符
            （保证调用方总能拿到字符串，不会 KeyError）。
        """
        # 从元数据读取来源键；dict.get 第二参为缺省兜底。
        return self.metadata.get("source", "unknown")

    # 向后兼容：提供content属性。
    @property
    def content(self) -> str:
        """获取内容（兼容旧接口）。

        返回：
            text 字段值——旧代码习惯读 result.content，
            本属性将其桥接到新式 text 字段。
        """
        # 直接返回 text：新旧接口读写同一份数据。
        return self.text


class InformationRetrieval(ABC):
    """检索器抽象基类。

    职责：定义检索能力的最小契约——connect 负责 heavyweight 初始化
          （连接向量库/加载索引），search 负责异步语义检索并返回
          统一的 SearchResult 列表；embeddings 属性保存可选的
          向量化器（子类按需使用）。自定义检索器只需继承本类并
          实现这两个方法即可。
    设计模式：抽象基类 + 模板方法变体（connect/search 为强制契约，
              __init__ 为子类提供 embeddings 注入点）；模块尾部
              Retriever = InformationRetrieval 为向后兼容别名。
    生命周期：由调用方（Agent/策略装配层）创建 → connect 后进入
              就绪态 → 服务存续期间长期复用；search 每轮调用。

    示例:
        class MyRetriever(InformationRetrieval):
            def connect(self, config):
                self.client = MyVectorDB(**config)

            async def search(self, query, top_k=5):
                results = await self.client.search(query, limit=top_k)
                return [SearchResult(text=r.text, score=r.score) for r in results]
    """

    def __init__(self, embeddings: Embedder | None = None) -> None:
        """初始化检索器。

        业务流程：仅保存可选的向量化器引用——connect 时是否使用
                  embeddings 由子类决定（如需向量化的混合检索器）。

        副作用：仅写入实例属性，无 IO 操作。

        参数：
            embeddings: 嵌入向量化器（可选）；None 表示纯文本检索
                        或子类自行管理向量化。
        """
        # 保存向量化器引用：默认 None，子类在 connect/search 中按需消费。
        self.embeddings = embeddings

    @abstractmethod
    def connect(self, config: dict[str, Any] | None = None) -> None:
        """连接/初始化向量索引。

        业务流程：由子类实现——典型操作为建立数据库/向量库连接、
                  加载索引文件等一次性 heavyweight 初始化。
        副作用：由子类决定（通常涉及网络连接或文件读取）。

        参数：
            config: 连接配置（如索引路径、数据库地址等）；
                    None 时子类使用内部默认配置。

        返回：
            None；子类实现连接失败时应抛异常（而非静默返回）。
        """
        # 兜底保护：子类未覆盖时抛出明确错误。
        raise NotImplementedError()

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 5,
        tracker_state: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """语义搜索。

        业务流程：由子类实现——将 query 向量化（或直接查文本索引），
                  检索 top_k 条候选并封装为 SearchResult 列表返回；
                  tracker_state 可用于个性化检索（用户信息/历史对话）。
        副作用：由子类决定（通常调用向量库等外部服务）。

        参数：
            query: 搜索查询文本；空串由调用方先行拦截。
            top_k: 返回结果数量上限；默认 5。
            tracker_state: 对话状态字典（tracker.to_dict() 的产物），
                           允许为 None——None 表示无上下文可用。

        返回：
            SearchResult 列表（可能为空）；score 字段允许为 None
            （检索器未评分时），调用方需按"None 视为通过"处理。
        """
        # 兜底保护：子类未覆盖时抛出明确错误。
        raise NotImplementedError()


# 向后兼容：保留Retriever别名——旧代码以 Retriever 引用检索器基类，
# 别名指向同一类对象，保证旧导入路径不断裂。
Retriever = InformationRetrieval

# 导出清单：from base_retriever import * 时仅暴露以下名字。
__all__ = [
    # 列表项1：检索器抽象基类（新代码推荐使用此名称）。
    "InformationRetrieval",
    # 列表项2：向后兼容别名（指向 InformationRetrieval 同一对象）。
    "Retriever",
    # 列表项3：基础文档数据类（向后兼容）。
    "Document",
    # 列表项4：搜索结果数据类（检索结果统一契约）。
    "SearchResult",
]
