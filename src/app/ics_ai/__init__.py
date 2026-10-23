"""
ics_ai - 智能客服对话系统包入口
核心功能：模拟 Rasa Pro CALM 架构的电商智能客服对话系统（教学版）。运行时
          接收用户消息，先由 LLM 将自然语言翻译为结构化命令（如启动流程、
          设置槽位），再由对话策略推进业务流程、调用 Action 完成查订单等
          业务逻辑并生成回复；业务流程无法处理的问题由企业搜索策略检索
          知识库兜底回答。
教学定位：用精简代码复现上述核心机制（LLM 命令生成、Flow 执行、策略决策、
          槽位管理），适合教学和理解对话系统核心原理。
执行流程：本文件是 ics_ai 包的入口/门面模块，仅在包被导入时执行——
          1. 声明包的版本号与作者元数据；
          2. 从 shared.constants 子模块导入两个最常用的默认常量，提升到包级；
          3. 通过 __all__ 将版本号与常量纳入包级公开 API，
             使外部可直接 `from app.ics_ai import DEFAULT_SERVER_PORT` 使用。
副作用：仅执行普通模块导入（加载 shared.constants），不读写文件/数据库、
        不发起网络请求、不修改任何全局状态。
主要依赖：app.ics_ai.shared.constants（全局常量定义）。
核心模块：
- agent: 对话代理，基于 LangGraph 实现图式消息处理流程
- dialogue_understanding: 对话理解模块(DU)，包含命令生成、处理、Flow执行
- core: 核心对话管理，包含 Tracker、Domain、Slot、Store
- policies: 对话策略，包含 FlowPolicy、EnterpriseSearchPolicy
- nlg: 自然语言生成（模板渲染 + LLM 重述）
- retrieval: 检索增强（嵌入器 + FAISS 流程检索）
- api: Web服务(FastAPI)
- channels: 对话通道（REST、SocketIO、Console）
- shared: 共享基础组件包（常量、异常层次）
"""
# 包版本号：语义化版本（主版本.次版本.修订号），当前处于初始开发阶段。
__version__ = "0.1.0"
# 包作者标识：标注本教学项目的作者为 harryliu888。
__author__ = "harryliu888"
# 从共享常量子模块导入两个最常用的默认配置常量，提升到包级别是为了
# 让调用方无需关心 constants 的内部路径即可使用。
from app.ics_ai.shared.constants import (
    # 列表项1：默认模型存储目录名，字符串 "models"，
    # 作为模型包的加载/保存基准路径。
    DEFAULT_MODELS_PATH,
    # 列表项2：默认服务端口，int 5005，与 Rasa 框架默认端口一致，
    # 作为服务启动时的监听端口。
    DEFAULT_SERVER_PORT,
)
# 包级公开 API 白名单：约束 `from app.ics_ai import *` 时导出的名称，
# 未列入的名称需按完整路径显式导入。
__all__ = [
    # 列表项1：默认模型存储目录名（与上方导入同名，纳入 `import *` 导出）。
    "DEFAULT_MODELS_PATH",
    # 列表项2：默认服务端口（与上方导入同名，纳入 `import *` 导出）。
    "DEFAULT_SERVER_PORT",
    # 列表项3：包版本号（供外部做版本探测与日志展示）。
    "__version__",
]