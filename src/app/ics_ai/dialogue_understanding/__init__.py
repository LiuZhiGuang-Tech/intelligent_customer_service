"""
dialogue_understanding - 对话理解入口模块

核心功能：对话理解（DU）子包的入口——当前聚合导出 commands 包的
          命令家族（Command 抽象基类与九个具体命令）。命令是对话
          系统的原子操作：LLM 生成器把用户输入翻译为命令序列，
          命令处理器执行命令并更新对话状态，构成
          "生成 → 处理 → 状态更新"的理解主链路。

执行流程：1. 模块导入：聚合导出 commands 子模块的公开命令符号
          （无 IO）；2. 生成侧（LLMCommandGenerator）按名称查找命令
          类输出 DSL 文本；3. 处理侧（CommandProcessor）经
          register_command 注册表按名字取回命令类实例化执行。

副作用：模块导入无 IO（仅类定义与聚合导出）；命令执行的状态更新、
        Flow 启停等副作用见各命令类文档。

主要依赖：app.ics_ai.dialogue_understanding.commands（命令抽象与
          全部具体命令实现）。
"""

# 从命令包导入命令家族：抽象基类与九个具体命令——DU 子包的
# 核心操作集，全部经 register_command 装饰器注册进命令注册表。
from app.ics_ai.dialogue_understanding.commands import (
    # 取消流程命令：终止当前正在执行的 Flow
    CancelFlowCommand,
    # 无法处理命令：兜底响应，表示当前请求超出系统能力
    CannotHandleCommand,
    # 闲聊应答命令：非业务闲聊输入的应答路径
    ChitChatAnswerCommand,
    # 澄清命令：向用户追问以消歧后继续
    ClarifyCommand,
    # 命令抽象基类：定义 execute 接口与 DSL 序列化契约
    Command,
    # 错误命令：内部异常的统一上报路径
    ErrorCommand,
    # 人工介入命令：将会话转交人工客服
    HumanHandoffCommand,
    # 会话开始命令：标记新会话起点（重置计数器等）
    SessionStartCommand,
    # 设置槽位命令：写入对话槽位值（填槽核心操作）
    SetSlotCommand,
    # 启动流程命令：开启指定 Flow 并压入对话栈
    StartFlowCommand,
)

# 导出清单：from dialogue_understanding import * 时仅暴露以下名字，
# 按字母序排列为 IDE 提供准确的补全与跳转依据。
__all__ = [
    # 取消流程命令
    "CancelFlowCommand",
    # 无法处理命令（兜底）
    "CannotHandleCommand",
    # 闲聊应答命令
    "ChitChatAnswerCommand",
    # 澄清命令
    "ClarifyCommand",
    # 命令抽象基类
    "Command",
    # 错误命令
    "ErrorCommand",
    # 人工介入命令
    "HumanHandoffCommand",
    # 会话开始命令
    "SessionStartCommand",
    # 设置槽位命令
    "SetSlotCommand",
    # 启动流程命令
    "StartFlowCommand",
]
