"""
config - 配置管理模块

【给小白的大白话总结】
这是整个系统的 **"设置中心"**——所有可调参数（LLM 接口地址、API Key、
数据库密码、端口号……）都放在这里统一管理。

继续餐厅比喻（这次是"菜单定价"场景）：
  餐厅要改菜价，总不能冲进厨房把所有菜谱翻出来改吧？
  得有个**统一的菜单文件**，改一处就行。
  
  这个模块就是系统的"统一设置文件"：

  📋 核心设计：**7 套配置类 + 环境变量替换**
    - LLMConfig：LLM 服务连哪个接口、API Key 多少
    - EmbeddingsConfig：向量化模型用哪个
    - TrackerStoreConfig：对话状态存哪（内存？JSON 文件？MySQL？）
    - RetrievalConfig：Flow 检索 + 知识库检索的参数
    - NLGConfig：回复重述（可选）
    - VectorStoreConfig：向量数据库怎么连
    - ICSConfig / EndpointsConfig：最顶层的两个大配置（对应两个 YAML 文件）

  🔐 安全设计：**${VAR} 占位符 + 环境变量注入**
    API Key、数据库密码这些敏感信息，**绝不写死在代码或 YAML 里**！
    配置文件里写 `${MYSQL_PASSWORD}`，运行时从 .env 文件读进来替换。
    支持 `${VAR}` 和 `${VAR:默认值}` 两种语法。

  📦 装配时机：Agent.load() 时读 config.yml → from_dict → 层层嵌套解析。
  后续所有组件（LLM 客户端、检索器、存储）都从这里拿参数。

核心功能：定义系统全量配置体系——ENV_VAR_PATTERN 环境变量替换正则
          与 _resolve_env_vars 递归解析器（${VAR} / ${VAR:default} 两
          种语法），以及七个 dataclass 配置类：LLMConfig（LLM 服务
          连接参数）、EmbeddingsConfig（向量化服务）、TrackerStore
          Config（对话状态存储后端）、RetrievalConfig（Flow/知识库
          双检索模式）、NLGConfig（响应重述）、VectorStoreConfig（
          检索器连接参数）、ICSConfig（config.yml 主配置聚合）与
          EndpointsConfig（endpoints.yml 端点配置聚合）；各配置类均
          提供 from_dict 字典构造（入口统一做环境变量替换）。
执行流程：1. 应用装配时 ICSConfig.load 读 config.yml、Endpoints
          Config.load 读 endpoints.yml（后者可选，缺省回默认配置）；
          2. YAML 解析为字典后逐级 from_dict——先经 _resolve_env_vars
          递归替换字符串中的 ${VAR} 占位（密钥等敏感值经 .env 注入，
          不落明文）；3. 嵌套节（flow_retrieval/rephrase/retriever/
          models/embeddings）分别解析为对应配置对象；4. Agent 经
          to_retriever_config / to_connect_config / get_model_config
          等适配方法消费配置。
副作用：本模块定义期无副作用；load 类方法读 YAML 文件（磁盘 IO）；
        from_dict 触发环境变量读取（os.environ）；不写文件不联网。
主要依赖：os/re（环境变量与正则替换）、dataclasses（配置对象）、
          pathlib（配置文件路径）、app.ics_ai.shared.constants
          （默认路径与默认值常量）、app.ics_ai.shared.exceptions
          （配置异常体系）、app.ics_ai.shared.yaml_loader（YAML
          读取）。
"""
# 导入 os：环境变量读取（os.environ.get 实现 ${VAR} 替换）。
import os
# 导入 re：编译环境变量占位符正则（模块级预编译，替换时复用）。
import re
# 导入 dataclass 工具：配置类声明与可变默认值工厂（字典/列表字段）。
from dataclasses import dataclass, field
# 导入 Path：配置文件路径的统一载体（load 方法接收 str 或 Path）。
from pathlib import Path
# 导入类型注解工具：Any 通配符（配置值开放结构）。
from typing import Any
# 从共享常量模块导入默认值：默认配置文件路径、LLM 采样默认参数、
# 默认服务端口、LLM 类型常量与 Tracker 存储类型常量。
from app.ics_ai.shared.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DOMAIN_PATH,
    DEFAULT_ENDPOINTS_PATH,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_SERVER_PORT,
    LLM_TYPE_OPENAI,
    TRACKER_STORE_JSON,
)
# 从共享异常模块导入配置异常体系：文件缺失/格式错误/非法配置分别
# 抛出，便于调用方精确处理。
from app.ics_ai.shared.exceptions import (
    ConfigurationException,
    InvalidConfigException,
    MissingConfigException,
)
# 从 YAML 加载器导入统一读取函数：load 方法解析配置文件用。
from app.ics_ai.shared.yaml_loader import read_yaml_file
# 环境变量替换正则：匹配 ${VAR_NAME} 或 ${VAR_NAME:default} 两种
# 占位语法——组1 为变量名（非冒号非右括号的连续字符），组2 为可选
# 默认值（冒号后到右括号前的任意内容，允许空串）。
ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")
def _resolve_env_vars(value: Any) -> Any:
    """解析配置值中的环境变量
    核心功能：递归遍历配置结构，将字符串中的 ${VAR} / ${VAR:default}
              占位替换为进程环境变量实际值。
    业务流程：1. 字符串值 → 按预编译正则逐个匹配占位符，经回调函数
              以 os.environ.get(变量名, 默认值) 替换（变量不存在且无
              默认值时替换为空串）；2. 字典值 → 递归解析每个值（键
              不处理）；3. 列表值 → 递归解析每个元素；4. 其他标量
              （int/bool/None 等）原样返回。
    副作用：读取进程环境变量（os.environ）；不修改入参（纯替换，
            返回新结构）。
    参数：
        value: 待解析的配置值——任意 JSON 兼容结构（str/dict/list/
               标量），允许 None。
    返回：
        解析后的配置值——结构与入参一致，字符串中的占位符已替换；
        非字符串标量原样返回。
    """
    # 分支一：字符串值——执行占位符替换
    if isinstance(value, str):
        # 定义单个匹配的替换回调：捕获组转环境变量值。
        def replace_env_var(match: re.Match[str]) -> str:
            # 组1：变量名（如 API_KEY）。
            var_name = match.group(1)
            # 组2：默认值——未提供（None）时兜底空串。
            default_value = match.group(2) if match.group(2) is not None else ""
            # 从环境变量取值：存在用实际值，缺失用默认值（可能为空串）。
            return os.environ.get(var_name, default_value)
        # 正则替换：每个 ${VAR} 占位经回调替换为环境变量实际值。
        return ENV_VAR_PATTERN.sub(replace_env_var, value)
    # 分支二：字典值——递归解析每个值（键名不做替换）。
    elif isinstance(value, dict):
        # 字典推导：键保留原样，值递归解析（嵌套层级不限）。
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    # 分支三：列表值——递归解析每个元素。
    elif isinstance(value, list):
        # 列表推导：逐元素递归（保持顺序）。
        return [_resolve_env_vars(item) for item in value]
    # 分支四：其他标量（int/float/bool/None）——原样返回不处理。
    else:
        return value
@dataclass
class LLMConfig:
    """LLM 配置
    职责：承载单个 LLM 服务的连接与调用参数；endpoints.yml 的
          models 节按名称定义多个本类实例。
    设计模式：数据传输对象（dataclass 参数对象）+ 类级工厂
              （from_dict 内聚解析逻辑）。
    生命周期：from_dict 构造后随配置对象长期只读持有；构造期即完成
              环境变量替换（密钥不保留占位符明文）。
    属性：
        type: LLM类型 (openai/qwen/azure/anthropic)
            - openai: OpenAI API或兼容服务(如vLLM)
            - qwen: 阿里云DashScope通义千问
            - azure: Azure OpenAI
            - anthropic: Anthropic Claude
        model: 模型名称
        api_key: API密钥
        api_base: API基础URL(可选，用于vLLM等自部署服务)
        temperature: 温度参数
        max_tokens: 最大token数
        timeout: 请求超时(秒)
        enable_thinking: 是否启用深度思考模式
            - qwen类型：DashScope原生支持
            - openai类型+api_base：通过chat_template_kwargs传递(vLLM)
        extra: 额外配置参数(如azure_endpoint, api_version等)
    """
    # LLM 类型：提供商标识，默认 openai（与 create_llm_client 工厂对齐）。
    type: str = LLM_TYPE_OPENAI
    # 模型名称：提供商侧模型标识，默认 gpt-3.5-turbo。
    model: str = "gpt-3.5-turbo"
    # API 密钥：鉴权凭据，默认空串（经 ${VAR} 从 .env 注入）。
    api_key: str = ""
    # API 基础 URL：自部署服务（vLLM 等）地址，None 走提供商默认。
    api_base: str | None = None
    # 采样温度：默认取常量（0.7）。
    temperature: float = DEFAULT_LLM_TEMPERATURE
    # 最大生成 token 数：默认取常量。
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS
    # 请求超时（秒）：默认取常量。
    timeout: int = DEFAULT_LLM_TIMEOUT
    # 深度思考开关：qwen 原生支持 / vLLM 经 chat_template_kwargs 透传。
    enable_thinking: bool = False
    # 额外参数：已知字段之外的全部配置（azure_endpoint/api_version 等
    # 提供商特有项），经工厂展开透传给底层客户端。
    extra: dict[str, Any] = field(default_factory=dict)
    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "LLMConfig":
        """从字典创建 LLM 配置
        业务流程：1. 环境变量替换（密钥/地址占位符落实际值）；2.
                  按字段名逐一取值（缺失走 dataclass 默认值同款）；3.
                  已知字段之外的键全部收进 extra（透传底层 SDK）。
        副作用：读取进程环境变量；不修改入参字典。
        参数：
            config: 配置字典——YAML 解析产物，允许空字典（全默认值）。
        返回：
            LLMConfig 实例（字段值已解析）。
        """
        # 第一步：递归替换字典内全部字符串占位符（返回新字典）。
        config = _resolve_env_vars(config)
        # 第二步：按已知字段构造；第三步：未知键收集进 extra。
        return cls(
            type=config.get("type", LLM_TYPE_OPENAI),
            model=config.get("model", "gpt-3.5-turbo"),
            api_key=config.get("api_key", ""),
            api_base=config.get("api_base"),
            temperature=config.get("temperature", DEFAULT_LLM_TEMPERATURE),
            max_tokens=config.get("max_tokens", DEFAULT_LLM_MAX_TOKENS),
            timeout=config.get("timeout", DEFAULT_LLM_TIMEOUT),
            enable_thinking=config.get("enable_thinking", False),
            extra={
                k: v
                for k, v in config.items()
                if k
                   not in [
                       "type",
                       "model",
                       "api_key",
                       "api_base",
                       "temperature",
                       "max_tokens",
                       "timeout",
                       "enable_thinking",
                   ]
            },
        )
@dataclass
class EmbeddingsConfig:
    """向量化配置
    职责：定义单个文本向量化服务的连接参数；endpoints.yml 的
          embeddings 节按名称定义多个本类实例。
    设计模式：数据传输对象 + 类级工厂（同 LLMConfig）。
    生命周期：from_dict 构造后随配置对象长期只读持有。
    属性：
        type: 向量化类型 (openai/qwen/azure)
        model: 模型名称
        api_key: API密钥
        api_base: API基础URL(可选)
        dimensions: 向量维度(可选)
    """
    # 向量化服务类型：默认 openai 兼容协议。
    type: str = LLM_TYPE_OPENAI
    # 嵌入模型名称：默认 OpenAI ada-002。
    model: str = "text-embedding-ada-002"
    # API 密钥：鉴权凭据（经 ${VAR} 注入）。
    api_key: str = ""
    # API 基础 URL：自部署服务地址，None 走默认。
    api_base: str | None = None
    # 向量维度：None 表示由模型默认决定（部分服务要求显式指定）。
    dimensions: int | None = None
    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "EmbeddingsConfig":
        """从字典创建向量化配置
        业务流程：环境变量替换 → 按字段取值构造（缺失走默认）。
        副作用：读取进程环境变量。
        参数：
            config: 配置字典，允许空字典。
        返回：
            EmbeddingsConfig 实例。
        """
        # 环境变量替换：密钥等占位符落实际值。
        config = _resolve_env_vars(config)
        # 按字段构造，缺失项走 dataclass 默认值。
        return cls(
            type=config.get("type", LLM_TYPE_OPENAI),
            model=config.get("model", "text-embedding-ada-002"),
            api_key=config.get("api_key", ""),
            api_base=config.get("api_base"),
            dimensions=config.get("dimensions"),
        )
@dataclass
class TrackerStoreConfig:
    """Tracker 存储配置
    职责：定义对话状态持久化后端的类型与连接参数；工厂函数按 type
          实例化对应存储实现（json/mysql/memory）。
    设计模式：数据传输对象 + 类级工厂。
    生命周期：from_dict 构造后随配置对象长期只读持有。
    属性：
        type: 存储类型 (json/mysql/memory)
        path: JSON存储路径(仅json类型)
        url: 数据库连接URL(仅mysql类型)
        db: 数据库名称
        host: 数据库主机
        port: 数据库端口
        username: 数据库用户名
        password: 数据库密码
    """
    # 存储类型：默认 json 文件存储（开发友好）。
    type: str = TRACKER_STORE_JSON
    # JSON 存储目录：仅 type=json 时消费。
    path: str = "trackers"
    # 数据库完整连接 URL：仅 mysql 时消费，优先于零散参数。
    url: str | None = None
    # 数据库名称：mysql 参数（可空）。
    db: str | None = None
    # 数据库主机：mysql 参数（可空）。
    host: str | None = None
    # 数据库端口：mysql 参数（可空）。
    port: int | None = None
    # 数据库用户名：mysql 参数（可空）。
    username: str | None = None
    # 数据库密码：mysql 参数（可空，经 ${VAR} 注入）。
    password: str | None = None
    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "TrackerStoreConfig":
        """从字典创建 Tracker 存储配置
        业务流程：环境变量替换（密码占位符）→ 按字段取值构造。
        副作用：读取进程环境变量。
        参数：
            config: 配置字典，允许空字典。
        返回：
            TrackerStoreConfig 实例。
        """
        # 环境变量替换：数据库凭据占位符落实际值。
        config = _resolve_env_vars(config)
        # 按字段构造：全部字段允许缺省（工厂侧再按 type 校验）。
        return cls(
            type=config.get("type", TRACKER_STORE_JSON),
            path=config.get("path", "trackers"),
            url=config.get("url"),
            db=config.get("db"),
            host=config.get("host"),
            port=config.get("port"),
            username=config.get("username"),
            password=config.get("password"),
        )
@dataclass
class RetrievalConfig:
    """检索配置
    职责：承载 Flow 检索与知识库检索两种模式的参数——Flow 检索服务
          对话理解（候选 Flow 召回），知识库检索服务 RAG 增强（企业
          知识问答）；另含 retriever 节的自定义检索器选型与嵌入参数。
    设计模式：数据传输对象 + 类级工厂；嵌套节（flow_retrieval/
              knowledge_retrieval/retriever）在 from_dict 内展开。
    生命周期：from_dict 构造后随配置对象长期只读持有。
    属性：
        flow_retrieval_enabled: 是否启用Flow检索
        flow_retrieval_top_k: Flow检索返回数量
        knowledge_retrieval_enabled: 是否启用知识库检索
        knowledge_retrieval_top_k: 知识库检索返回数量
        knowledge_source: 知识库来源路径
        retriever_type: Retriever类型（faiss/knowledge/自定义类路径）
        vector_store_path: 向量存储路径（用于加载已有索引）
        embedding_provider: 嵌入提供商（openai/qwen等）
        embedding_model: 嵌入模型名称
        embedding_dimension: 嵌入向量维度
        similarity_threshold: 相似度阈值
    """
    # Flow 检索开关：默认启用（对话理解的核心召回手段）。
    flow_retrieval_enabled: bool = True
    # Flow 检索返回数量：默认 5。
    flow_retrieval_top_k: int = 5
    # 知识库检索开关：默认关闭（RAG 增强按需开启）。
    knowledge_retrieval_enabled: bool = False
    # 知识库检索返回数量：默认 3。
    knowledge_retrieval_top_k: int = 3
    # 知识库来源路径：目录或文件（可空——未配置时检索降级）。
    knowledge_source: str | None = None
    # 检索器类型：faiss/knowledge/自定义类路径（工厂按此装配）。
    retriever_type: str = "faiss"
    # 向量存储路径：已有索引的加载位置（可空）。
    vector_store_path: str | None = None
    # 嵌入提供商：默认 openai。
    embedding_provider: str = "openai"
    # 嵌入模型名称：默认 OpenAI 3-small。
    embedding_model: str = "text-embedding-3-small"
    # 嵌入向量维度：默认 1536（与 3-small 对齐）。
    embedding_dimension: int = 1536
    # 相似度阈值：低于该值的检索结果被过滤。
    similarity_threshold: float = 0.5
    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "RetrievalConfig":
        """从字典创建检索配置
        业务流程：1. 环境变量替换；2. 拆出三个嵌套节（flow_retrieval/
                  knowledge_retrieval/retriever），各自展开为平铺字段。
        副作用：读取进程环境变量。
        参数：
            config: 配置字典，允许空字典（全默认值）。
        返回：
            RetrievalConfig 实例。
        """
        # 第一步：环境变量替换（返回新字典）。
        config = _resolve_env_vars(config)
        # 第二步：拆出三个嵌套配置节（缺失时兜底空字典）。
        flow_config = config.get("flow_retrieval", {})
        knowledge_config = config.get("knowledge_retrieval", {})
        retriever_config = config.get("retriever", {})
        # 第三步：展开为平铺字段构造（各节缺失键走默认值）。
        return cls(
            flow_retrieval_enabled=flow_config.get("enabled", True),
            flow_retrieval_top_k=flow_config.get("top_k", 5),
            knowledge_retrieval_enabled=knowledge_config.get("enabled", False),
            knowledge_retrieval_top_k=knowledge_config.get("top_k", 3),
            knowledge_source=knowledge_config.get("source"),
            # Retriever配置
            retriever_type=retriever_config.get("type", "faiss"),
            vector_store_path=retriever_config.get("vector_store_path"),
            embedding_provider=retriever_config.get("embedding_provider", "openai"),
            embedding_model=retriever_config.get(
                "embedding_model", "text-embedding-3-small"
            ),
            embedding_dimension=retriever_config.get("embedding_dimension", 1536),
            similarity_threshold=retriever_config.get("similarity_threshold", 0.5),
        )
    def to_retriever_config(self) -> dict[str, Any]:
        """转换为 Retriever 工厂函数所需的配置格式
        业务流程：把平铺的检索器相关字段重新聚合为工厂入参字典
                  （type/vector_store_path/嵌入三元组/top_k/阈值）。
        副作用：无（纯构造新字典，不修改实例状态）。
        返回：
            检索器工厂可消费的配置字典。
        """
        # 聚合工厂入参：知识检索的 top_k 作为检索器默认返回数量。
        return {
            "type": self.retriever_type,
            "vector_store_path": self.vector_store_path,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "top_k": self.knowledge_retrieval_top_k,
            "similarity_threshold": self.similarity_threshold,
        }
@dataclass
class NLGConfig:
    """NLG 配置
    职责：控制响应生成与 LLM 重述功能的参数——重述开关、模型引用、
          风格、触发门槛与槽位占位符保护。
    设计模式：数据传输对象 + 类级工厂；rephrase 嵌套节在 from_dict
              内展开。
    生命周期：from_dict 构造后随配置对象长期只读持有。
    属性：
        rephrase_enabled: 是否启用LLM重述
        rephrase_model: 重述使用的LLM模型名称（引用endpoints.yml中的models）
        rephrase_style: 重述风格 (friendly/professional/casual/empathetic)
        rephrase_threshold: 触发重述的最小文本长度
        preserve_slots: 重述时是否保留槽位占位符
        language: 目标语言
    """
    # LLM 重述开关：默认关闭（纯模板输出）。
    rephrase_enabled: bool = False
    # 重述模型名：引用 endpoints.yml models 节的 key（可空）。
    rephrase_model: str | None = None  # 引用endpoints.yml中的models名称
    # 重述风格：默认 friendly（友好口语化）。
    rephrase_style: str = "friendly"
    # 触发重述的最小文本长度：默认 10 字符（短文本不重述）。
    rephrase_threshold: int = 10
    # 槽位占位符保护：True 时重述不破坏 {slot} 变量。
    preserve_slots: bool = True
    # 目标语言：默认中文。
    language: str = "zh"
    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "NLGConfig":
        """从字典创建 NLG 配置
        业务流程：环境变量替换 → 拆出 rephrase 嵌套节展开为平铺字段。
        副作用：读取进程环境变量。
        参数：
            config: 配置字典，允许空字典。
        返回：
            NLGConfig 实例。
        """
        # 环境变量替换（返回新字典）。
        config = _resolve_env_vars(config)
        # 拆出 rephrase 嵌套节（缺失兜底空字典）。
        rephrase_config = config.get("rephrase", {})
        # 展开为平铺字段构造。
        return cls(
            rephrase_enabled=rephrase_config.get("enabled", False),
            rephrase_model=rephrase_config.get("model"),
            rephrase_style=rephrase_config.get("style", "friendly"),
            rephrase_threshold=rephrase_config.get("threshold", 10),
            preserve_slots=rephrase_config.get("preserve_slots", True),
            language=rephrase_config.get("language", "zh"),
        )
@dataclass
class VectorStoreConfig:
    """向量存储连接配置
    职责：存储传递给 retriever.connect() 的连接参数——本类不解释
          内容（键值结构由具体检索器约定），仅做持有与透传。
    设计模式：数据传输对象（开放结构容器）+ 类级工厂。
    生命周期：from_dict 构造后随配置对象长期只读持有。
    属性：
        config: 连接配置字典（如数据库地址、认证信息等）
    """
    # 连接参数字典：结构由目标检索器定义（Neo4j/FAISS 各不相同）。
    config: dict[str, Any] = field(default_factory=dict)
    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "VectorStoreConfig":
        """从字典创建配置
        业务流程：环境变量替换后整体持有（不拆字段）。
        副作用：读取进程环境变量。
        参数：
            config: 连接配置字典，允许空字典。
        返回：
            VectorStoreConfig 实例。
        """
        # 环境变量替换：连接凭据占位符落实际值。
        config = _resolve_env_vars(config)
        # 整体持有：不解释键结构，透传给检索器。
        return cls(config=config)
    def to_connect_config(self) -> dict[str, Any]:
        """转换为 retriever.connect() 使用的配置字典
        业务流程：返回内部字典的浅拷贝——防止调用方修改污染配置
                  原件（connect 参数的隔离保护）。
        副作用：无（浅拷贝新字典；嵌套可变对象仍共享，约定只读）。
        返回：
            连接配置字典副本。
        """
        # 浅拷贝出参：外层隔离，嵌套值按只读约定使用。
        return self.config.copy()
@dataclass
class ICSConfig:
    """app.ics_ai 主配置类
    职责：聚合 config.yml 的全部配置节——LLM/向量化/存储/检索四个
          子配置对象 + pipeline/policies 组件与策略的原始字典列表
          （由 Agent 装配层按 name 字段分发解析）+ 全局语言与助手 ID。
    设计模式：组合根配置（聚合各子配置对象）+ 类级工厂。
    生命周期：load/from_dict 构造后随应用长期持有；to_dict 支持调试
              序列化。
    属性：
        llm: LLM配置
        embeddings: 向量化配置
        tracker_store: Tracker存储配置
        retrieval: 检索配置
        pipeline: Pipeline组件配置列表
        policies: 策略配置列表
    """
    # LLM 服务配置：config.yml llm 节。
    llm: LLMConfig = field(default_factory=LLMConfig)
    # 向量化配置：config.yml embeddings 节。
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    # Tracker 存储配置：config.yml tracker_store 节。
    tracker_store: TrackerStoreConfig = field(default_factory=TrackerStoreConfig)
    # 检索配置：config.yml retrieval 节。
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    # Pipeline 组件原始配置列表：每项含 name 字段，由 Agent 装配解析。
    pipeline: list[dict[str, Any]] = field(default_factory=list)
    # 策略原始配置列表：每项含 name 字段，由 PolicyEnsemble 装配解析。
    policies: list[dict[str, Any]] = field(default_factory=list)
    # 全局对话语言：模板选择与 LLM 提示语言。
    language: str = "zh"
    # 助手标识：会话元数据与日志区分用。
    assistant_id: str = "ics_assistant"
    @classmethod
    def load(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "ICSConfig":
        """从配置文件加载配置
        业务流程：1. 路径归一化为 Path；2. 文件不存在抛
                  MissingConfigException（config.yml 是必需文件）；
                  3. YAML 解析失败包装为 ConfigurationException；
                  4. 解析成功交 from_dict 构造。
        副作用：读取 YAML 文件（磁盘 IO）。
        参数：
            config_path: 配置文件路径，默认取常量（config.yml）。
        返回：
            ICSConfig 实例。
        异常：
            MissingConfigException: 配置文件不存在；
            ConfigurationException: 配置文件格式错误（YAML 解析失败）。
        """
        # 第一步：路径归一化（str/Path 统一为 Path）。
        path = Path(config_path)
        # 第二步：存在性校验——主配置必需，缺失即失败。
        if not path.exists():
            raise MissingConfigException(f"配置文件不存在: {path}")
        try:
            # 第三步：读取并解析 YAML（统一加载器）。
            config_dict = read_yaml_file(path)
        except Exception as e:
            # 解析失败包装为配置异常（保留原始错误信息）。
            raise ConfigurationException(f"配置文件格式错误: {e}")
        # 第四步：字典转配置对象；read_yaml_file 对空 YAML 文件返回
        # None，经 or {} 兜底为空字典（走全默认值构造，与
        # EndpointsConfig.load 的空文件语义对齐）。
        return cls.from_dict(config_dict or {})
    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ICSConfig":
        """从字典创建配置
        业务流程：按节分发——四个子配置各走自身 from_dict（内含环境
                  变量替换），pipeline/policies 保留原始字典由装配层
                  消费。
        副作用：经子配置 from_dict 读取进程环境变量。
        参数：
            config: 配置字典（YAML 解析产物），允许空字典。
        返回：
            ICSConfig 实例。
        """
        # 按节分发构造：子配置缺失时各 from_dict 以空字典兜底（全默认）。
        return cls(
            llm=LLMConfig.from_dict(config.get("llm", {})),
            embeddings=EmbeddingsConfig.from_dict(config.get("embeddings", {})),
            tracker_store=TrackerStoreConfig.from_dict(config.get("tracker_store", {})),
            retrieval=RetrievalConfig.from_dict(config.get("../retrieval", {})),
            pipeline=config.get("pipeline", []),
            policies=config.get("../policies", []),
            language=config.get("language", "zh"),
            assistant_id=config.get("assistant_id", "ics_assistant"),
        )
    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式
        业务流程：dataclasses.asdict 递归展开——子配置对象转嵌套
                  字典（调试/日志友好）。
        副作用：无（纯序列化）。
        返回：
            全量配置字典（结构与 config.yml 对齐）。
        """
        # 局部导入：asdict 仅此方法使用（延迟加载）。
        from dataclasses import asdict
        # 递归展开为字典。
        return asdict(self)
@dataclass
class EndpointsConfig:
    """端点配置类
    职责：聚合 endpoints.yml 的外部服务端点配置——存储/向量库/NLG/
          按名称索引的 LLM 模型表与嵌入模型表；与 ICSConfig 平行
          （前者管运行参数，本类管外部服务端点）。
    设计模式：组合根配置 + 注册表（models/embeddings 以名称为 key
              的字典表，供 rephrase_model 等按名引用）。
    生命周期：load/from_dict 构造后随应用长期持有。
    属性：
        tracker_store: Tracker存储配置
        vector_store: 向量存储配置（知识检索后端）
        nlg: NLG配置（响应重述等）
        models: LLM模型配置字典，key为模型名称
        embeddings: 嵌入模型配置字典，key为模型名称
    """
    # Tracker 存储端点：endpoints.yml tracker_store 节。
    tracker_store: TrackerStoreConfig = field(default_factory=TrackerStoreConfig)
    # 向量存储端点：知识检索后端连接参数。
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    # NLG 配置：响应重述参数（可空——未配置则无重述）。
    nlg: NLGConfig | None = None
    # LLM 模型表：按名称索引的多模型配置。
    models: dict[str, LLMConfig] = field(default_factory=dict)
    # 嵌入模型表：按名称索引的多嵌入配置。
    embeddings: dict[str, EmbeddingsConfig] = field(default_factory=dict)
    @classmethod
    def load(
            cls, endpoints_path: str | Path = DEFAULT_ENDPOINTS_PATH
    ) -> "EndpointsConfig":
        """从端点配置文件加载配置
        业务流程：1. 路径归一化；2. 文件不存在返回默认实例——端点
                  配置是可选文件（全默认值降级）；3. YAML 解析失败
                  包装为 ConfigurationException；4. 成功交 from_dict。
        副作用：读取 YAML 文件（磁盘 IO）。
        参数：
            endpoints_path: 端点配置文件路径，默认取常量。
        返回：
            EndpointsConfig 实例（文件缺失时为全默认值实例）。
        异常：
            ConfigurationException: 配置文件格式错误。
        """
        # 第一步：路径归一化。
        path = Path(endpoints_path)
        if not path.exists():
            # 端点配置文件是可选的，不存在时返回默认配置
            return cls()
        try:
            # 第三步：读取并解析 YAML。
            config_dict = read_yaml_file(path)
        except Exception as e:
            # 解析失败包装为配置异常。
            raise ConfigurationException(f"端点配置文件格式错误: {e}")
        # 第四步：字典转配置对象（YAML 空文件兜底空字典）。
        return cls.from_dict(config_dict or {})
    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "EndpointsConfig":
        """从字典创建配置
        业务流程：1. 遍历 models 节逐项转 LLMConfig（名称→配置表）；
                  2. 遍历 embeddings 节逐项转 EmbeddingsConfig；3.
                  nlg 节存在才解析（可空语义）；4. vector_store/
                  tracker_store 节各自转对应配置对象。
        副作用：经子配置 from_dict 读取进程环境变量。
        参数：
            config: 端点配置字典，允许空字典。
        返回：
            EndpointsConfig 实例。
        """
        # 解析models配置
        models_dict = {}
        models_config = config.get("models", {})
        # 逐项转换：每个模型名映射到独立的 LLMConfig 实例。
        for model_name, model_config in models_config.items():
            models_dict[model_name] = LLMConfig.from_dict(model_config)
        # 解析embeddings配置
        embeddings_dict = {}
        embeddings_config = config.get("embeddings", {})
        # 逐项转换：每个嵌入名映射到独立的 EmbeddingsConfig 实例。
        for embed_name, embed_config in embeddings_config.items():
            embeddings_dict[embed_name] = EmbeddingsConfig.from_dict(embed_config)
        # 解析NLG配置
        nlg_config = None
        # nlg 节存在才解析——缺失保持 None（可空语义：无重述配置）。
        if "nlg" in config:
            nlg_config = NLGConfig.from_dict(config.get("../nlg", {}))
        # 解析vector_store配置
        vector_store_config = VectorStoreConfig.from_dict(
            config.get("vector_store", {})
        )
        # 聚合构造：全部子节就位。
        return cls(
            tracker_store=TrackerStoreConfig.from_dict(config.get("tracker_store", {})),
            vector_store=vector_store_config,
            nlg=nlg_config,
            models=models_dict,
            embeddings=embeddings_dict,
        )
    def get_model_config(self, model_name: str) -> LLMConfig | None:
        """根据模型名称获取 LLM 配置
        业务流程：按名称查 models 注册表（字典 get，不抛异常）。
        副作用：无（只读查找）。
        参数：
            model_name: 模型名称（在endpoints.yml中定义的key）。
        返回：
            LLMConfig 实例，如果不存在则返回None。
        """
        # 按名称查表：未注册返回 None 由调用方降级。
        return self.models.get(model_name)
    def get_embeddings_config(self, embeddings_name: str) -> EmbeddingsConfig | None:
        """根据名称获取嵌入模型配置
        业务流程：按名称查 embeddings 注册表（字典 get，不抛异常）。
        副作用：无（只读查找）。
        参数：
            embeddings_name: 嵌入模型名称（在endpoints.yml中定义的key）。
        返回：
            EmbeddingsConfig 实例，如果不存在则返回None。
        """
        # 按名称查表：未注册返回 None 由调用方降级。
        return self.embeddings.get(embeddings_name)