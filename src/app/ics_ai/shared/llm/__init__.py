"""
llm - 统一 LLM 客户端包入口模块

核心功能：聚合 LLM 抽象契约与统一工厂——LLMClient 抽象基类与
          LLMResponse 响应 DTO（base_client）、LangChainClient 唯一
          实现（langchain_client，经 LangChain 适配 openai/qwen/
          azure/anthropic 四类后端）、create_llm_client 工厂函数
          （业务层获取客户端的唯一入口）。

执行流程：
    1. 导入期：按序执行 base_client 与 langchain_client 两个子模块
       的模块体初始化（langchain 的重依赖包在客户端构造期才延迟
       加载，导入期轻量）；
    2. 组装 __all__ 公开名单（基类 + DTO + 实现 + 工厂）；
    3. 调用期：业务层经 create_llm_client(type=..., **kwargs) 创建
       客户端实例——内部直通 LangChainClient 构造（类型校验在此
       发生，不支持的 type 抛 ValueError）。

副作用：仅导入期的普通模块导入，不读写文件/数据库、不发起网络
        请求、不修改全局状态（网络请求发生在客户端 complete/
        complete_sync 调用期）。

主要依赖：app.ics_ai.shared.llm.base_client、
          app.ics_ai.shared.llm.langchain_client。
"""

# 导入类型注解工具：Any 类型通配符（工厂 **kwargs 的透传注解）。
from typing import Any

# 从基类子模块导入：LLMClient 抽象基类（所有客户端的共同契约）与
# LLMResponse 响应 DTO（补全产出的载体），供业务层做类型注解。
from app.ics_ai.shared.llm.base_client import (
    LLMClient,
    LLMResponse,
)

# 从实现子模块导入：LangChainClient——LLMClient 的唯一实现，
# 四类后端（openai/qwen/azure/anthropic）的统一适配点。
from app.ics_ai.shared.llm.langchain_client import LangChainClient

# 包级公开 API 白名单：约束 `from app.ics_ai.shared.llm import *`
# 的导出范围；按字母序排列（大写开头的类名先于小写函数名）。
__all__ = [
    "LLMClient",  # 抽象基类：complete/complete_sync 契约
    "LLMResponse",  # 响应 DTO：文本/用量/延迟/思考内容
    "LangChainClient",  # 唯一实现：LangChain 四后端适配
    "create_llm_client",  # 工厂函数：业务层获取客户端的入口
]


def create_llm_client(
    type: str = "openai",
    **kwargs: Any,
) -> LLMClient:
    """创建 LLM 客户端实例（统一工厂）。

    业务流程：校验参数合法性（type 是否在支持列表——校验发生在
              LangChainClient 构造内）→ 以 type 与全部关键字参数
              直通构造 LangChainClient → 返回基类类型的实例
              （调用方按 LLMClient 契约使用，不感知具体实现）。

    副作用：构造期不发网络请求（LangChain 客户端实例在首次
            complete/complete_sync 时才真正建连）。

    参数：
        type: LLM 类型（openai/qwen/azure/anthropic）——
              openai: OpenAI API 或兼容服务（如 vLLM）；
              qwen: 阿里云 DashScope 通义千问；
              azure: Azure OpenAI；
              anthropic: Anthropic Claude。
              注：参数名与 LangChainClient.__init__ 的 type 形参
              对齐（遮蔽内建 type 名，仅作关键字用）。
        **kwargs: 客户端配置参数，透传 LangChainClient——
              model: 模型名称；
              api_key: API 密钥；
              api_base: 自定义 API 基础 URL（用于 vLLM 等自部署）；
              temperature: 温度参数；
              max_tokens: 最大生成 Token 数；
              timeout: 请求超时（单位秒）；
              enable_thinking: 启用深度思考模式；
              其余键进入 extra_config（如 Azure 的
              azure_endpoint/api_version）。

    返回：
        LLMClient 实例（LangChainClient 对象）——type 不受支持时
        构造内抛 ValueError，不返回。

    示例：
        # OpenAI API
        >>> client = create_llm_client(
        ...     type="openai",
        ...     model="gpt-4",
        ...     api_key="sk-xxx",
        ... )

        # 阿里云 DashScope（支持 thinking）
        >>> client = create_llm_client(
        ...     type="qwen",
        ...     model="qwen-plus",
        ...     api_key="sk-xxx",
        ...     enable_thinking=True,
        ... )

        # vLLM 自部署（OpenAI 兼容接口）
        >>> client = create_llm_client(
        ...     type="openai",
        ...     model="Qwen/Qwen3-8B",
        ...     api_base="http://localhost:8000/v1",
        ...     api_key="EMPTY",
        ...     enable_thinking=True,
        ... )

        # 使用
        >>> response = await client.complete([
        ...     {"role": "user", "content": "你好"}
        ... ])
    """
    # 直通构造：type 与全部配置键原样透传（类型校验与后端分发在
    # LangChainClient 内完成，工厂不重复校验）；返回基类类型，
    # 调用方仅依赖抽象契约。
    return LangChainClient(type=type, **kwargs)
