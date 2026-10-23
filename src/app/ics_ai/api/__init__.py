"""
ICS AI API 入口模块
核心功能：聚合导出 FastAPI 应用服务器（server.py），提供基于
FastAPI 的 Web 服务接口。
执行流程：包被导入时执行本文件，从 server 子模块引入服务器类与
应用工厂，经 __all__ 控制对外可见面。
副作用：无（仅转发导入，不触发服务器启动）。
主要依赖：app.ics_ai.api.server（服务器实现）。
"""
# 导入服务器实现：ICSServer 为服务器类，create_app 为应用工厂。
from app.ics_ai.api.server import ICSServer, create_app
# 对外导出面：按字母序排列。
__all__ = [
    # 服务器类。
    "ICSServer",
    # 应用工厂函数。
    "create_app",
]