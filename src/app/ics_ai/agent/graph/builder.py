"""
graph_builder - 消息处理图构建器模块

核心功能：构建 LangGraph 消息处理图的两种入口——
          build_message_processing_graph（装配五节点两条件边的
          StateGraph 并编译）与 get_message_processing_graph
          （全局单例惰性获取）；另提供 reset_graph_instance
          清空单例缓存强制重建（测试辅助）。

执行流程：
    1. 装配期：Agent 初始化（agent.py）调用 get_message_processing_graph
       → 首次调用触发 build_message_processing_graph：
       以 MessageProcessingState 为通道骨架创建 StateGraph →
       注册 understand/policy/action/guard/response 五个节点 →
       连接入口边、两条条件边（policy 与 guard 之后按状态路由）
       与终止边 → compile() 产出可执行图；
    2. 执行期：agent.py 对每轮用户消息构造初始状态后 ainvoke 本图
       （异步调用），节点读写的组件依赖（生成器/处理器/策略集成）
       不经构建期注入，而是由 agent.py 经状态通道（_command_* 字段）
       传入，图本身无参数、无业务状态、可全局复用。

副作用：
    - get_message_processing_graph：读写模块级全局 _graph_instance
      （惰性单例缓存，跨会话持续生效）；
    - build_message_processing_graph：图构建为纯内存操作，但会
      经 logger 写两条构建日志；
    - 不读写文件、不调用网络。

主要依赖：
    - logging：构建期日志；
    - langgraph.graph：StateGraph/START/END 图构建原语；
    - langgraph.graph.state：CompiledStateGraph 编译产物类型；
    - app.ics_ai.agent.graph.state / nodes / edges：状态骨架、
      节点函数与路由函数（图结构的三大素材）。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，PEP 604 联合
# 类型注解（X | None）不受运行时版本限制。
from __future__ import annotations

# 导入日志器工厂：构建期记录"开始构建/构建完成"两条日志。
import logging

# 导入类型注解工具：TYPE_CHECKING 隔离仅注解导入。
from typing import TYPE_CHECKING

# 导入 LangGraph 图构建三原语：StateGraph 状态图类、START/END
# 图的入口/出口哨兵节点。
from langgraph.graph import StateGraph, START, END

# 导入编译产物类型：graph.compile() 的返回类型（仅注解用）。
from langgraph.graph.state import CompiledStateGraph

# 导入状态定义：图的通道骨架（TypedDict 键集合即状态通道）。
from app.ics_ai.agent.graph.state import MessageProcessingState

# 导入五个节点函数：图执行的最小工作单元（understand 命令生成/
# policy 策略预测/action 动作执行/guard 循环守卫/response 响应
# 汇总）。
from app.ics_ai.agent.graph.nodes import (
    understand_node,
    policy_node,
    action_node,
    response_node,
    guard_node,
)

# 导入两个路由函数：条件边的判定依据（返回目标分支键）。
from app.ics_ai.agent.graph.edges import (
    should_execute_action,
    should_continue,
)

# 类型检查期专属导入：组件类型只出现在注解中，运行时不导入。
if TYPE_CHECKING:
    # 命令生成器：由 understand 节点经状态通道使用（图构建期不触碰）。
    from app.ics_ai.dialogue_understanding.generator import LLMCommandGenerator

    # 命令处理器：由 understand 节点经状态通道使用。
    from app.ics_ai.dialogue_understanding.processor import CommandProcessor

    # 策略集成器：由 policy 节点经状态通道使用。
    from app.ics_ai.policies import PolicyEnsemble

# 模块级日志器：以模块路径为名，供构建函数记录日志。
logger = logging.getLogger(__name__)


def build_message_processing_graph() -> CompiledStateGraph:
    """构建消息处理图。

    业务流程：1. 以 MessageProcessingState 为骨架创建 StateGraph；
              2. 注册五个节点函数；3. 连接入口边与固定边
              （understand→policy、action→guard、response→END）；
              4. 连接两条条件边（policy/guard 之后按状态路由）；
              5. compile 编译为可执行图。

    图结构（箭头方向即数据流）：

        START → understand → policy → [route] → action → guard → [route] → ...
                                        ↓                           ↓
                                     response ← ← ← ← ← ← ← ← ← ← ←
                                        ↓
                                       END

    两条条件边的路由规则：
        - policy 后（should_execute_action）→ response 的三种情形：
          本轮已完成（is_finished）、无预测结果、预测动作为
          action_listen（等待用户输入）或为 None；其余情形 → action；
        - guard 后（should_continue）→ response 的两种情形：本轮
          已完成（is_finished）、动作数达上限（action_count >=
          max_actions）；其余情形 → policy（继续循环）。

    副作用：经 logger 写两条构建日志（纯内存构建，无其他副作用）。

    返回：
        编译后的可执行图（CompiledStateGraph），可经 invoke/ainvoke
        以 MessageProcessingState 字典驱动执行。
    """
    logger.info("构建消息处理图...")

    # 创建状态图：MessageProcessingState 的全部键即图的状态通道，
    # 节点返回的部分更新 dict 由 LangGraph 按通道合并。
    graph = StateGraph(MessageProcessingState)

    # 添加节点：节点名（字符串键）→ 节点函数；边与条件边均以
    # 节点名引用。
    graph.add_node("understand", understand_node)
    graph.add_node("policy", policy_node)
    graph.add_node("action", action_node)
    graph.add_node("guard", guard_node)
    graph.add_node("response", response_node)

    # 设置入口边：每轮图执行从 understand 节点开始（命令生成是
    # 消息处理的第一步）。
    graph.add_edge(START, "understand")

    # understand → policy：命令生成后固定进入策略预测。
    graph.add_edge("understand", "policy")

    # policy → [条件边] → action 或 response：should_execute_action
    # 返回的键经映射字典翻译为目标节点名（"action"→action 节点，
    # "response"→response 节点）。
    graph.add_conditional_edges(
        "policy",
        should_execute_action,
        {
            "action": "action",
            "response": "response",
        },
    )

    # action → guard：动作执行后固定进入守卫（判断是否继续循环）。
    graph.add_edge("action", "guard")

    # guard → [条件边] → policy 或 response：should_continue 返回
    # "policy" 时回到策略节点形成"策略→动作→守卫"执行循环，
    # 返回 "response" 时结束循环。
    graph.add_conditional_edges(
        "guard",
        should_continue,
        {
            "policy": "policy",
            "response": "response",
        },
    )

    # response → END：响应汇总后图执行终止。
    graph.add_edge("response", END)

    # 编译图：校验图结构（入口存在、节点可达性）并产出可执行实例。
    compiled_graph = graph.compile()

    logger.info("消息处理图构建完成")

    return compiled_graph


# 全局图实例（惰性初始化）：首次调用 get_message_processing_graph
# 时才构建并缓存——编译有开销，单例复用避免重复构建；None 表示
# 尚未构建。
_graph_instance: CompiledStateGraph | None = None


def get_message_processing_graph() -> CompiledStateGraph:
    """获取消息处理图单例。

    业务流程：检查全局缓存 _graph_instance——为 None（首次调用）
              时调用 build_message_processing_graph 构建并缓存，
              之后所有调用方共享同一图实例。

    副作用：首次调用时写入模块级全局 _graph_instance（跨会话
            持续生效），并触发 build 的构建日志。

    返回：
        编译后的图（全局唯一实例）——agent.py 在 Agent 初始化时
        调用本函数持有图引用。
    """
    # 声明写模块级全局变量：赋值语句修改的是全局缓存而非局部名。
    global _graph_instance

    # 惰性构建：仅在缓存为 None（首次调用或被 reset）时构建。
    if _graph_instance is None:
        _graph_instance = build_message_processing_graph()

    # 返回缓存实例：后续调用零构建开销。
    return _graph_instance


def reset_graph_instance() -> None:
    """重置图实例。

    业务流程：把全局缓存置回 None，使下次
              get_message_processing_graph 重新构建。

    副作用：写模块级全局 _graph_instance（清空缓存）。

    使用现状：项目内无调用方（docstring 声明的测试场景辅助——
              测试中替换节点/路由后强制重建图）。
    """
    # 声明写模块级全局变量：清空单例缓存。
    global _graph_instance
    # 置 None：下次 get 走惰性构建分支。
    _graph_instance = None


# 导出：模块对外暴露的公共 API（按字母序，便于 IDE 提示与 diff 整洁）。
__all__ = [
    "build_message_processing_graph",
    "get_message_processing_graph",
    "reset_graph_instance",
]
