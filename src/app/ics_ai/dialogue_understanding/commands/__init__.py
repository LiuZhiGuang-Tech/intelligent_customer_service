"""
commands - 命令系统入口模块

核心功能：聚合对话系统的全部命令类型并统一导出——命令是本架构中
          的核心概念，表示对话系统可以执行的原子操作（启动/取消
          Flow、设置槽位、闲聊回答、澄清追问、人工转接、错误降级
          等），本模块把基类与五个命令族子模块汇聚为单一导入入口。

执行流程：
    1. 导入期：本模块被导入时依次加载 base 与五个命令族子模块，各
       命令类经 @register_command 装饰器以 command_name() 为键登记
       进 base 模块的 _COMMAND_REGISTRY（共 17 个具体命令类）；
    2. 解析期：命令解析器按注册表把 LLM 输出的命令文本/字典按名
       还原为命令实例（注册的完整性由本模块的聚合导入保证）；
    3. 使用期：上层包 dialogue_understanding 从本模块导入 10 个
       常用命令与基类 Command 再导出；其余模块（CommandProcessor、
       understand 节点等）按需直接导入具体子模块。

副作用：
    - 导入期：向 base 模块全局注册表 _COMMAND_REGISTRY 写入全部
      命令类（全局可变状态，跨会话持续生效）——这是本模块存在的
      核心价值：聚合导入保证注册表完整；
    - 无其他副作用（不读写文件、不调用网络）。

主要依赖：
    - base：命令基类 Command、注册装饰器 register_command、全局
      注册表 _COMMAND_REGISTRY；
    - answer_commands / error_commands / flow_commands /
      session_commands / slot_commands：五个命令族子模块。

导出说明：
    - __all__ 按字母序排列；未纳入导出的命令（FreeFormAnswerCommand、
      InternalErrorCommand、ParseErrorCommand、ChangeFlowCommand、
      RestartCommand、NoopCommand、ResetSlotCommand）仍随模块导入
      完成注册，可经注册表按名解析，或按需直接从子模块导入。
"""

# 导入命令基类：统一命令契约（command_name/from_dict/run/to_dsl），
# 供类型注解与 isinstance 判断使用。
from app.ics_ai.dialogue_understanding.commands.base import Command

# 导入回答族命令：闲聊回复、无法处理降级、知识库检索回答
# （另含 FreeFormAnswerCommand 未纳入包导出）。
from app.ics_ai.dialogue_understanding.commands.answer_commands import (
    ChitChatAnswerCommand,
    CannotHandleCommand,
    KnowledgeAnswerCommand,
)

# 导入错误族命令：LLM 输出无法解析时的降级命令
# （另含 InternalErrorCommand/ParseErrorCommand 未纳入包导出）。
from app.ics_ai.dialogue_understanding.commands.error_commands import (
    ErrorCommand,
)

# 导入 Flow 族命令：启动/取消对话流程
# （另含 ChangeFlowCommand 未纳入包导出）。
from app.ics_ai.dialogue_understanding.commands.flow_commands import (
    CancelFlowCommand,
    StartFlowCommand,
)

# 导入会话族命令：会话开始、澄清追问、人工转接
# （另含 RestartCommand/NoopCommand 未纳入包导出）。
from app.ics_ai.dialogue_understanding.commands.session_commands import (
    ClarifyCommand,
    HumanHandoffCommand,
    SessionStartCommand,
)

# 导入槽位族命令：设置槽位值
# （另含 ResetSlotCommand 未纳入包导出）。
from app.ics_ai.dialogue_understanding.commands.slot_commands import (
    SetSlotCommand,
)

# 导出：包对外暴露的公共 API（按字母序，便于 IDE 提示与 diff 整洁）。
__all__ = [
    "CancelFlowCommand",
    "CannotHandleCommand",
    "ChitChatAnswerCommand",
    "ClarifyCommand",
    "Command",
    "ErrorCommand",
    "HumanHandoffCommand",
    "KnowledgeAnswerCommand",
    "SessionStartCommand",
    "SetSlotCommand",
    "StartFlowCommand",
]
