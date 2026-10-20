"""
core - 核心对话管理包入口模块

核心功能：ics_ai 的对话系统核心包（对标 Rasa 的 rasa.core），聚合 Domain 领域配置、
          Slots 槽位系统、Tracker 对话状态追踪三大子系统，为上层对话理解与流程执行
          模块提供统一的核心组件包入口。

执行流程：本包被导入时按顺序执行——
          1. domain 子模块——加载 YAML 领域配置，聚合槽位/动作/响应/Flow/表单定义；
          2. slots 子模块——定义槽位数据类与槽位工厂函数；
          3. tracker 子模块——管理对话状态的持久化存储与会话历史追踪；
          4. 组装 __all__ 公开名单，仅暴露三个子模块名。

副作用：仅执行普通模块导入（连带执行各子模块的模块体初始化），
        不读写文件/数据库、不发起网络请求、不修改全局状态。

主要依赖：app.ics_ai.core.domain、app.ics_ai.core.slots、app.ics_ai.core.tracker。
"""

# 包级公开 API 白名单：约束 `from app.ics_ai.core import *` 的导出范围，
# 当前仅暴露三个子模块名（低频类型需按完整路径显式导入）：
# domain——领域配置聚合根（Domain/ResponseTemplate）；
# slots——槽位数据类与工厂（Slot/create_slot）；
# tracker——对话状态追踪（Tracker）。
__all__ = [
    "slots",  # 槽位系统子模块
    "domain",  # 领域定义子模块
    "tracker",  # 状态追踪子模块
]
