"""
flow - 对话流程包入口

核心功能：聚合"流程定义 → 加载 → 执行"三层能力，对外统一暴露对话流程体系
          的公共 API——Flow/FlowStep/FlowsList（数据结构）、FlowLoader
          （从 YAML 加载）、FlowExecutor（执行步骤并推进流程）。

执行流程：导入本包即执行下列 import 语句，把三个子模块的类提升到包级命名空间
          （如 from app.ics_ai.dialogue_understanding.flow import Flow 即可），
          使调用方无需关心类定义在哪个文件；__all__ 声明通配导入的导出清单。

副作用：仅导入模块、注册类与函数等内存对象，不读写文件/数据库、不发起网络
        请求、不修改全局状态（子模块内的模块级 logger 除外）。

主要依赖：本包内的 flow（数据结构）、flow_loader（加载器）、flow_executor
          （执行器）三个子模块；三个子模块本身会间接依赖 core（tracker/domain）
          与 dialogue_understanding.stack（流程栈帧）。
"""

# 导入流程数据结构：Flow（单个流程）、FlowStep（流程步骤）、FlowsList（流程集合），
# 三者是子模块 flow 对外提供的核心数据模型。
from app.ics_ai.dialogue_understanding.flow.flow import Flow, FlowsList, FlowStep

# 导入流程执行器：负责按当前步骤执行流程并推进流程状态。
from app.ics_ai.dialogue_understanding.flow.flow_executor import FlowExecutor

# 导入流程加载器：负责从 YAML 文件/目录/字符串读取并解析为 FlowsList。
from app.ics_ai.dialogue_understanding.flow.flow_loader import FlowLoader

# 导出清单：from ...flow import * 时仅暴露以下名字，同时供 IDE 静态识别包级 API；
# 顺序按"数据结构 → 加载 → 执行"分层排列，便于按职责分组阅读。
__all__ = [
    # 列表项1：流程对象——承载一个完整业务流程的步骤序列。
    "Flow",
    # 列表项5：流程执行器——运行时驱动流程推进的最后环。
    "FlowExecutor",
    # 列表项4：流程加载器——外部配置（YAML）进入本体系的第一环。
    "FlowLoader",
    # 列表项2：流程步骤——流程内的单个执行单元（动作/收集/条件等）。
    "FlowStep",
    # 列表项3：流程集合——管理多个 Flow 并提供按 ID 查找。
    "FlowsList",
]
