"""
shared - 共享基础组件包入口模块
功能：
ics_ai 的共享基础组件包（shared 包），定义跨模块复用的基础定义——公共常量与异常层次，为业务模块提供统一的基础组件导入入口。
"""
# 从常量子模块导入最常用的公共常量：默认端口、默认路径（模型/配置/Domain/端点/训练数据）与降级原因容器类；
# 各常量的含义与取值原因见 constants.py 内的逐项注释，此处统一提升到包级，供业务模块经本包（shared 包）直接引用
from app.ics_ai.shared.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_DOMAIN_PATH,
    DEFAULT_ENDPOINTS_PATH,
    DEFAULT_MODELS_PATH,
    DEFAULT_SERVER_PORT,
    DegradationReason,
)
from app.ics_ai.shared.exceptions import (
    ConfigurationException,
    ICSException,
    InvalidConfigException,
    LLMException,
    ModelNotFound,
    TrackerStoreException,
)
# 包级公开 API 白名单：约束“from app.ics_ai.shared import *”的导出范围，
# 未列入的名称需要按完整路径显示导入
__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_DATA_PATH",
    "DEFAULT_DOMAIN_PATH",
    "DEFAULT_ENDPOINTS_PATH",
    "DEFAULT_MODELS_PATH",
    "DEFAULT_SERVER_PORT",
    "ConfigurationException",
    "DegradationReason",
    "ICSException",
    "InvalidConfigException",
    "LLMException",
    "ModelNotFound",
    "TrackerStoreException",
]