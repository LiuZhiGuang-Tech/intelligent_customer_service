"""
processor - 命令处理入口模块（执行命令并更新对话状态）

核心功能：聚合对话理解阶段"命令处理"环节的子模块——以 CommandProcessor
          为核心：接收命令生成阶段产出的 Command 列表，逐个执行并落地为
          对话状态变更（Tracker 槽位/命令历史 + DialogueStack 栈帧），
          归类响应类型并推断下一步动作提示，产出统一的 ProcessResult。

执行流程：
    1. 导入期：本包被导入时加载 command_processor 子模块，配置/
       结果数据类与处理器类随之定义就绪，无其他初始化动作；
    2. 使用期：message_processor 在消息处理期构造 CommandProcessor，
       graph understand 节点调用 process 执行命令，policy 节点仅在
       第一轮优先采用结果中的 next_action（未设置时由 Policy 检测
       栈帧决定动作）。

副作用：仅导入子模块并绑定符号名；子模块导入期只做类定义与模块级
        日志器初始化，不读写文件/数据库、不发起网络请求、不修改
        全局状态。

主要依赖：包内唯一子模块 command_processor（命令处理器/配置/
          结果 DTO，以及便捷函数 process_commands）。
"""

# 命令处理器：CommandProcessor 为核心执行器（命令模式消费者——调度、
# force_slot_filling 过滤、响应归类、下一步动作推断）；ProcessorConfig
# 为行为开关配置（并行预留/流程与槽位校验）；ProcessResult 为结果信封
# （事件/执行计数/错误/响应类型/next_action）。
from app.ics_ai.dialogue_understanding.processor.command_processor import (
    CommandProcessor,
    ProcessorConfig,
    ProcessResult,
)

# 导出：本包对外暴露的公共 API（按字母序，便于 IDE 提示与 diff 整洁）。
__all__ = [
    "CommandProcessor",
    "ProcessorConfig",
    "ProcessResult",
]
