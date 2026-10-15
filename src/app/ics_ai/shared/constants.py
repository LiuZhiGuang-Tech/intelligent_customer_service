"""
constants - 全局常量定义模块

核心功能：定义 ics_ai 系统中使用的所有常量，作为全系统统一的"魔法值"来源，
          包括默认路径与端口、企业搜索降级原因、命令类型、槽位类型、
          Action 名称、LLM 参数、Tracker Store 类型与通道名称等。

执行流程：本模块被导入时执行全部常量赋值；配置加载、命令解析、策略决策、
          Action 服务等业务模块从本模块导入所需常量，保证全系统取值一致。

副作用：仅定义常量与一个常量容器类，不读写文件/数据库、不发起网络请求、
        不修改全局状态；导入本模块本身无任何副作用。

主要依赖：typing.Final（静态类型检查用，无运行时依赖）。
"""

# 导入 Final 类型注解：向类型检查器声明"该名称不可重新赋值"，
# 运行时不改变赋值行为，仅提供静态约束。
from typing import Final

# =============================================================================
# 服务器配置
# =============================================================================

# 默认服务器端口：int，端口号；取 5005 与 Rasa 框架默认端口一致，便于对照学习。
DEFAULT_SERVER_PORT: Final[int] = 5005

# 默认服务器主机地址：str；"0.0.0.0" 表示监听本机所有网卡，本机与外部均可访问。
DEFAULT_SERVER_HOST: Final[str] = "0.0.0.0"

# 服务器 URL 格式模板：str；含 {0}协议、{1}端口 两个占位符，
# 经 str.format() 填入协议与端口后拼出完整访问地址。
DEFAULT_SERVER_FORMAT: Final[str] = "{0}://localhost:{1}"

# =============================================================================
# 默认路径
# =============================================================================

# 默认模型存储路径：str，相对工作目录的目录名，存放训练产物（模型包）。
DEFAULT_MODELS_PATH: Final[str] = "models"

# 默认配置文件路径：str，主配置文件名（pipeline/policies 定义所在）。
DEFAULT_CONFIG_PATH: Final[str] = "config.yml"

# 默认 Domain 文件路径：str，领域定义文件名（槽位、回复模板、Action 声明）。
DEFAULT_DOMAIN_PATH: Final[str] = "domain.yml"

# 默认端点配置文件路径：str，外部端点配置文件名（LLM/向量库/数据库连接）。
DEFAULT_ENDPOINTS_PATH: Final[str] = "endpoints.yml"

# 默认训练数据路径：str，目录名，存放训练语料等数据文件。
DEFAULT_DATA_PATH: Final[str] = "data"

# 默认 Flow 定义路径：str，目录名，业务流程 YAML 所在（FlowPolicy 加载）。
DEFAULT_FLOWS_PATH: Final[str] = "flows"

# 默认 Action 模块路径：str，目录名，自定义 Action 代码所在。
DEFAULT_ACTIONS_PATH: Final[str] = "actions"

# =============================================================================
# 降级原因常量
# =============================================================================


class DegradationReason:
    """降级原因常量类。

    职责：封装企业搜索策略中的降级原因常量。
    设计模式：字符串常量容器（命名空间类，仅类属性、无实例状态，
              是枚举的轻量替代方案）。
    生命周期：随模块导入创建，进程内全局唯一，全程序只读复用。

    降级链: Flow -> RAG -> Chitchat -> CannotHandle
    """

    # 默认值：未指定具体降级原因时的兜底取值。
    DEFAULT: str = "default"

    # 转入闲聊处理：业务接不住时降级为闲聊应答。
    CHITCHAT: str = "chitchat"

    # 功能或渠道不支持当前请求。
    NOT_SUPPORTED: str = "not_supported"

    # 用户输入无法映射到有效命令/意图。
    INVALID_INTENT: str = "invalid_intent"

    # 检索知识库但未命中相关答案。
    NO_RELEVANT_ANSWER: str = "no_relevant_answer"

    # 系统内部错误。
    INTERNAL_ERROR: str = "internal_error"

    # 彻底无法处理用户请求，降级链终点。
    CANNOT_HANDLE: str = "cannot_handle"


# =============================================================================
# 命令类型常量
# =============================================================================

# 启动Flow命令：请求启动指定业务流程（由对话栈压入对应流程帧执行）。
COMMAND_START_FLOW: Final[str] = "start_flow"

# 取消Flow命令：请求取消当前正在执行的流程。
COMMAND_CANCEL_FLOW: Final[str] = "cancel_flow"

# 设置槽位命令：请求为 Tracker 中指定槽位赋值。
COMMAND_SET_SLOT: Final[str] = "set_slot"

# 闲聊命令：转入闲聊应答，不推进业务流程。
COMMAND_CHITCHAT: Final[str] = "chitchat"

# 知识库回答命令：用知识库检索结果回答用户。
COMMAND_KNOWLEDGE_ANSWER: Final[str] = "knowledge_answer"

# 澄清命令：向用户发起澄清追问以补全缺失信息。
COMMAND_CLARIFY: Final[str] = "clarify"

# 无法处理命令：声明无法处理用户请求（降级终点）。
COMMAND_CANNOT_HANDLE: Final[str] = "cannot_handle"

# 会话开始命令：标记新会话开始，触发会话初始化。
COMMAND_SESSION_START: Final[str] = "session_start"

# =============================================================================
# 槽位类型常量
# =============================================================================

# 文本槽位类型：槽位值为字符串。
SLOT_TYPE_TEXT: Final[str] = "text"

# 布尔槽位类型：槽位值为 True/False。
SLOT_TYPE_BOOL: Final[str] = "bool"

# 浮点槽位类型：槽位值为浮点数。
SLOT_TYPE_FLOAT: Final[str] = "float"

# 列表槽位类型：槽位值为列表（可容纳多个值）。
SLOT_TYPE_LIST: Final[str] = "list"

# 任意类型槽位：不限制槽位值类型。
SLOT_TYPE_ANY: Final[str] = "any"

# 分类槽位类型：槽位值限定在预定义的枚举集合内。
SLOT_TYPE_CATEGORICAL: Final[str] = "categorical"

# =============================================================================
# Action名称常量
# =============================================================================

# 监听用户输入动作：等待下一轮用户输入，对话在此暂停。
ACTION_LISTEN: Final[str] = "action_listen"

# 重启会话动作：重置会话与 Tracker，重新开始对话。
ACTION_RESTART: Final[str] = "action_restart"

# 会话开始动作：开启新会话（初始化 Tracker、触发会话开始事件）。
ACTION_SESSION_START: Final[str] = "action_session_start"

# 默认降级动作：所有策略均无法决策时执行的兜底动作。
ACTION_DEFAULT_FALLBACK: Final[str] = "action_default_fallback"

# 停用循环动作：结束/停用当前循环（表单）流程。
ACTION_DEACTIVATE_LOOP: Final[str] = "action_deactivate_loop"

# 返回上一步动作：回退到对话的上一步。
ACTION_BACK: Final[str] = "action_back"

# =============================================================================
# LLM相关常量
# =============================================================================

# OpenAI类型：走 OpenAI API 协议，兼容其兼容服务（如 vLLM）。
LLM_TYPE_OPENAI: Final[str] = "openai"

# 通义千问类型：走阿里云 DashScope 接口协议。
LLM_TYPE_QWEN: Final[str] = "qwen"

# Azure OpenAI类型：走微软 Azure 托管的 OpenAI 服务。
LLM_TYPE_AZURE: Final[str] = "azure"

# Anthropic Claude类型：走 Anthropic Claude 接口协议。
LLM_TYPE_ANTHROPIC: Final[str] = "anthropic"

# 默认LLM温度参数：float，采样温度；0.0 为确定性输出，保证命令解析稳定可复现。
DEFAULT_LLM_TEMPERATURE: Final[float] = 0.0

# 默认LLM最大token数：int，单次生成最大 token 数；1024 兼顾成本与响应延迟。
DEFAULT_LLM_MAX_TOKENS: Final[int] = 1024

# 默认LLM请求超时：int，单位秒；超时后按 LLM 超时异常处理。
DEFAULT_LLM_TIMEOUT: Final[int] = 30

# =============================================================================
# Tracker Store类型常量
# =============================================================================

# JSON本地存储类型：以本地 JSON 文件持久化对话状态。
TRACKER_STORE_JSON: Final[str] = "json"

# MySQL存储类型：将对话状态存入 MySQL 数据库，可跨重启保留。
TRACKER_STORE_MYSQL: Final[str] = "mysql"

# 内存存储类型：对话状态仅存进程内存，重启即丢失，适合开发调试。
TRACKER_STORE_MEMORY: Final[str] = "memory"

# =============================================================================
# 通道名称常量
# =============================================================================

# REST API通道：基于 HTTP 请求-响应的用户接入通道。
CHANNEL_REST: Final[str] = "rest"

# SocketIO通道：基于 WebSocket 长连接的实时双向通道。
CHANNEL_SOCKETIO: Final[str] = "socketio"

# 控制台通道：命令行交互通道，用于本地调试。
CHANNEL_CONSOLE: Final[str] = "console"

# =============================================================================
# 其他常量
# =============================================================================

# 默认发送者ID：str，未指定会话方时的兜底会话标识（用于区分不同用户的
# Tracker），主要用于单机调试场景。
DEFAULT_SENDER_ID: Final[str] = "default"

# 默认文件编码：str，读写 yaml/json 等项目文件时统一使用的字符编码。
DEFAULT_ENCODING: Final[str] = "utf-8"

# 会话开始元数据槽位：str，会话开始事件触发时在 Tracker 中存放
# 会话元数据的专用槽位名。
SESSION_START_METADATA_SLOT: Final[str] = "session_started_metadata"
