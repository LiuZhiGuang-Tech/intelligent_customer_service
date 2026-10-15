"""
ics_ai - 智能客服系统包入口模块
功能：
ics_ai 是智能客服系统包（ics_ai 包），定义智能客服系统的所有功能模块、组件与异常层次，为业务模块提供统一的入口。
"""
# 包版本号：语义化版本（主版本.次版本.修订号），当前处于初始开发阶段
__version__ = "0.1.0"
# 作者：标注本项目的作为 ityanglaoshi
__author__ = "ityanglaoshi"
# 从共享常量子模块导入两个最常用的默认配置常量，提升到包级别是为了让调用方无需关心 constants 的内部路径即可使用。
from app.ics_ai.shared.constants import (
    # 列表项1：默认模型存储目录名
    DEFAULT_MODELS_PATH,
    # 列表项2：默认服务端口
    DEFAULT_SERVER_PORT,
)
# 包级公开 API 白名单：约束“from app.ics_ai import *”时的导出范围，
# 未列入的名称需要按完整路径显示导入
__all__ = [
    "DEFAULT_MODELS_PATH",
    "DEFAULT_SERVER_PORT",
    "__version__",
]