"""
exceptions - 异常类定义模块

核心功能：定义 ics_ai 系统中使用的所有自定义异常类，形成统一的异常层次结构。
          层次共三层：ICSException（系统总基类） -> 领域基类（如 LLMException、
          ActionException、ChannelException 等） -> 具体异常（如 LLMTimeoutError）。
          调用方既可用领域基类做宽泛捕获，也可用具体异常做精确处理。
          所有异常都继承自 ICSException 基类。

执行流程：本模块被导入时按顺序执行各 class 语句完成异常类定义；运行时由各
          业务模块在出错位置 raise 对应异常，在上层（如 API 错误处理器、
          策略降级逻辑）按需 except 捕获处理。

副作用：仅定义类，不读写文件/数据库、不发起网络请求、不修改全局状态；
        导入本模块本身无任何副作用。

主要依赖：typing（Optional、Text 类型注解）。

设计模式：异常层次分类——按业务领域分组的继承树，兼顾捕获粒度与语义清晰度。
"""

# 导入类型注解工具：Optional 用于标注"可为 None"的参数类型；
# Text 是 str 的别名（typing 提供，现已标记弃用，等价于 str），
# 该写法常见于 Rasa 源码，此处沿用其风格表示"文本"语义。
from typing import Optional, Text


class ICSException(Exception):
    """ics_ai系统的基础异常类

    职责：作为所有自定义异常的统一根，负责携带 message 并提供稳定的字符串表示。
    设计模式：模板式基类——统一 __init__/__str__ 行为，子类只负责语义分类。
    生命周期：随 raise 语句创建实例，被 except 捕获处理后由垃圾回收销毁。

    所有自定义异常都应继承此类，便于统一捕获和处理。

    属性：
        message: 异常描述信息（str；构造时未传则回退为类名，永不为 None）
    """

    def __init__(self, message: Optional[Text] = None) -> None:
        """初始化异常

        业务流程：1. 若未传 message，用当前类名作为兜底描述；
                  2. 将 message 存入实例属性 self.message；
                  3. 调用 Exception 基类构造函数，使消息进入 args，
                     保证 traceback 与 repr 能正常展示消息内容。

        副作用：仅设置实例属性，无其他副作用。

        参数：
            message: 异常描述信息，str；允许为 None（缺省），
                     为 None 或空串时自动回退为类名。

        返回：无（构造函数不返回值；本方法自身不抛出异常）。
        """
        # or 短路兜底：message 为 None 或空串时取"类名"作为描述，
        # 设计原因：保证任何异常实例 str() 后都有可读内容，便于日志排查。
        self.message = message or self.__class__.__name__
        # 调用父类构造：把消息传入 Exception.__init__（写入 args 元组），
        # 设计原因：未重写 __str__ 的子类也能通过默认机制正确显示消息。
        super().__init__(self.message)

    def __str__(self) -> str:
        # 直接返回 message：使 print/日志输出简洁描述，
        # 而非 Exception 默认的"类名(参数)"包裹形式。
        return self.message


# =============================================================================
# 配置相关异常
# =============================================================================


class ConfigurationException(ICSException):
    """配置异常

    职责：作为配置类异常的领域基类，聚合所有配置加载/校验错误。

    当配置文件格式错误、缺少必要配置项或配置值无效时抛出。
    """

    # 空类体：__init__/__str__ 行为完全继承基类，本类仅提供语义分类。
    pass


class InvalidConfigException(ConfigurationException):
    """无效配置异常

    当配置值不符合预期格式或范围时抛出。
    """

    # 空类体：行为完全继承基类。
    pass


class MissingConfigException(ConfigurationException):
    """缺少配置异常

    当必需的配置项未提供时抛出。
    """

    # 空类体：行为完全继承基类。
    pass


# =============================================================================
# 模型相关异常
# =============================================================================


class ModelNotFound(ICSException):
    """模型未找到异常

    当指定路径没有找到模型文件时抛出。
    """

    # 空类体：行为完全继承基类。
    pass


class ModelLoadException(ICSException):
    """模型加载异常

    当模型文件损坏或格式不正确导致加载失败时抛出。
    """

    # 空类体：行为完全继承基类。
    pass


class ModelTrainingException(ICSException):
    """模型训练异常

    当训练过程中出现错误时抛出。
    """

    # 空类体：行为完全继承基类。
    pass


# =============================================================================
# LLM相关异常
# =============================================================================


class LLMException(ICSException):
    """LLM异常基类

    职责：聚合所有大模型调用相关异常，便于对话组件整体捕获 LLM 故障。
    """

    # 空类体：行为完全继承基类。
    pass


class LLMConnectionError(LLMException):
    """LLM连接错误

    当无法连接到LLM API服务时抛出。
    """

    # 空类体：行为完全继承基类。
    pass


class LLMTimeoutError(LLMException):
    """LLM超时错误

    当LLM API请求超时时抛出。
    """

    # 空类体：行为完全继承基类。
    pass


class LLMResponseError(LLMException):
    """LLM响应错误

    当LLM返回无效或无法解析的响应时抛出。
    """

    # 空类体：行为完全继承基类。
    pass


class LLMAuthenticationError(LLMException):
    """LLM认证错误

    当API密钥无效或认证失败时抛出。
    """

    # 空类体：行为完全继承基类。
    pass


class LLMRateLimitError(LLMException):
    """LLM速率限制错误

    当超过API调用速率限制时抛出。
    """

    # 空类体：行为完全继承基类。
    pass


# =============================================================================
# Tracker Store相关异常
# =============================================================================


class TrackerStoreException(ICSException):
    """Tracker存储异常基类

    职责：聚合所有对话状态存储相关异常。
    """

    # 空类体：行为完全继承基类。
    pass


class TrackerStoreConnectionError(TrackerStoreException):
    """Tracker存储连接错误

    当无法连接到存储后端时抛出。
    """

    # 空类体：行为完全继承基类。
    pass


class TrackerSerializationError(TrackerStoreException):
    """Tracker序列化错误

    当Tracker对象无法正确序列化或反序列化时抛出。
    """

    # 空类体：行为完全继承基类。
    pass


# =============================================================================
# 对话理解相关异常
# =============================================================================


class DialogueUnderstandingException(ICSException):
    """对话理解异常基类

    职责：聚合对话理解模块（命令生成、命令处理、Flow 执行）相关异常。
    """

    # 空类体：行为完全继承基类。
    pass


class CommandParseError(DialogueUnderstandingException):
    """命令解析错误

    当LLM输出无法解析为有效命令时抛出。
    """

    # 空类体：行为完全继承基类。
    pass


class FlowNotFoundError(DialogueUnderstandingException):
    """Flow未找到错误

    当尝试启动不存在的Flow时抛出。
    """

    # 空类体：行为完全继承基类。
    pass


class FlowExecutionError(DialogueUnderstandingException):
    """Flow执行错误

    当Flow执行过程中出现错误时抛出。
    """

    # 空类体：行为完全继承基类。
    pass


class InvalidSlotValueError(DialogueUnderstandingException):
    """无效槽位值错误

    当槽位值不符合定义的类型或约束时抛出。
    """

    # 空类体：行为完全继承基类。
    pass


# =============================================================================
# Action相关异常
# =============================================================================


class ActionException(ICSException):
    """Action异常基类

    职责：聚合所有自定义 Action 执行相关异常。
    """

    # 空类体：行为完全继承基类。
    pass


class ActionNotFoundError(ActionException):
    """Action未找到错误

    当尝试执行不存在的Action时抛出。
    """

    # 空类体：行为完全继承基类。
    pass


class ActionExecutionError(ActionException):
    """Action执行错误

    当Action执行过程中出现错误时抛出。
    """

    # 空类体：行为完全继承基类。
    pass


class ActionServerConnectionError(ActionException):
    """Action服务器连接错误

    当无法连接到远程Action服务器时抛出。
    """

    # 空类体：行为完全继承基类。
    pass


# =============================================================================
# 通道相关异常
# =============================================================================


class ChannelException(ICSException):
    """通道异常基类

    职责：聚合所有对话通道（REST、SocketIO、Console）相关异常。
    """

    # 空类体：行为完全继承基类。
    pass


class ChannelNotFoundError(ChannelException):
    """通道未找到错误

    当尝试使用不存在的通道时抛出。
    """

    # 空类体：行为完全继承基类。
    pass


class ChannelConnectionError(ChannelException):
    """通道连接错误

    当通道连接失败时抛出。
    """

    # 空类体：行为完全继承基类。
    pass


# =============================================================================
# 图引擎相关异常
# =============================================================================


class GraphException(ICSException):
    """图引擎异常基类

    职责：聚合图引擎（LangGraph 消息处理图）相关异常。
    """

    # 空类体：行为完全继承基类。
    pass


class GraphValidationError(GraphException):
    """图验证错误

    当图结构无效(如存在循环依赖)时抛出。
    """

    # 空类体：行为完全继承基类。
    pass


class GraphExecutionError(GraphException):
    """图执行错误

    当图执行过程中出现错误时抛出。
    """

    # 空类体：行为完全继承基类。
    pass


# =============================================================================
# 检索相关异常
# =============================================================================


class RetrievalException(ICSException):
    """检索异常基类

    职责：聚合检索增强（向量检索）相关异常。
    """

    # 空类体：行为完全继承基类。
    pass


class VectorStoreError(RetrievalException):
    """向量存储错误

    当向量存储操作失败时抛出。
    """

    # 空类体：行为完全继承基类。
    pass


class EmbeddingError(RetrievalException):
    """向量化错误

    当文本向量化失败时抛出。
    """

    # 空类体：行为完全继承基类。
    pass


# =============================================================================
# 数据验证相关异常
# =============================================================================


class ValidationException(ICSException):
    """验证异常基类

    职责：聚合所有数据定义校验相关异常。
    """

    # 空类体：行为完全继承基类。
    pass


class DomainValidationError(ValidationException):
    """Domain验证错误

    当Domain定义不符合规范时抛出。
    """

    # 空类体：行为完全继承基类。
    pass


class FlowValidationError(ValidationException):
    """Flow验证错误

    当Flow定义不符合规范时抛出。
    """

    # 空类体：行为完全继承基类。
    pass


class DataValidationError(ValidationException):
    """数据验证错误

    当训练数据不符合规范时抛出。
    """

    # 空类体：行为完全继承基类。
    pass
