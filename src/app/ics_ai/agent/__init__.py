"""
agent - Agent 入口模块

核心功能：聚合导出对话系统主入口的三大组成部分——Agent（对话系统
          主类，装配 NLU/policy/Action 执行器等组件并驱动整轮处理）、
          AgentConfig（Agent 装配配置）、MessageProcessor（消息处理
          器，将用户消息编排为理解→决策→执行的管线）、Action 体系
          （动作抽象基类、结果封装与内置动作实现）。

执行流程：
    1. 模块导入：聚合导出 actions / agent / message_processor 三个
       子模块的公开符号（无 IO）；
    2. 使用方经 Agent.create(config) 装配完整 Agent 实例；
    3. 每条用户消息经 MessageProcessor 管线处理，产生 BotResponse。

副作用：
    - 模块导入：无 IO（仅类定义与聚合导出）。
    - 后续使用由各子模块决定（Agent 装配读配置/文件、消息处理触发
      LLM 调用与 Tracker 存储读写等，见对应模块文档）。

主要依赖：
    - app.ics_ai.agent.actions：Action 抽象基类、ActionResult 结果
      封装与内置动作实现（Listen/Restart/DefaultFallback/SendText）。
    - app.ics_ai.agent.agent：Agent 主类与 AgentConfig 装配配置。
    - app.ics_ai.agent.message_processor：MessageProcessor 消息
      处理管线。
"""

# 从动作体系模块导入 Action 抽象基类、结果封装与四个内置动作：
# Action 为所有自定义/内置动作的统一契约（execute 异步接口），
# ActionResult 为动作执行结果封装，Listen/Restart/DefaultFallback/
# SendText 为系统默认注册的基础动作实现。
from app.ics_ai.agent.actions import (
    Action,
    ActionDefaultFallback,
    ActionListen,
    ActionRestart,
    ActionResult,
    ActionSendText,
)

# 从主类模块导入 Agent 与装配配置：Agent 是对话系统的组合根，
# 负责 create() 工厂装配与 handle_message 主流程调度；
# AgentConfig 承载装配参数（组件路径、策略配置等）。
from app.ics_ai.agent.agent import Agent, AgentConfig

# 从消息处理器模块导入 MessageProcessor：将单条用户消息编排为
# "理解 → 决策 → 执行" 的处理管线并产出响应。
from app.ics_ai.agent.message_processor import MessageProcessor

# 导出清单：from agent import * 时仅暴露以下名字，按字母序排列
# （便于 IDE 补全与 diff 干净）。
__all__ = [
    # 动作抽象基类（execute 统一契约）
    "Action",
    # 默认兜底动作（未匹配任何策略时执行）
    "ActionDefaultFallback",
    # 监听动作（等待用户输入，对话轮次让出）
    "ActionListen",
    # 会话重启动作（清空对话栈重新开始）
    "ActionRestart",
    # 动作执行结果封装（事件列表载体）
    "ActionResult",
    # 文本发送动作（按模板渲染回复）
    "ActionSendText",
    # 对话系统主类（组合根与流程调度）
    "Agent",
    # Agent 装配配置
    "AgentConfig",
    # 消息处理管线
    "MessageProcessor",
]
