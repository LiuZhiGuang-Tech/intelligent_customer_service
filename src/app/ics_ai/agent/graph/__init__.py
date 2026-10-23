"""
graph - 消息处理图包入口模块

核心功能：聚合 LangGraph 消息处理图的全部构件——状态定义
          （MessageProcessingState / create_initial_state）、图
          构建三函数（build / get / reset）、条件边路由（
          should_execute_action / should_continue）与五个节点
          函数（understand / policy / action / guard / response），
          供 agent.py 一站式装配与驱动图执行。

执行流程：本包被导入时按顺序执行——
          1. builder 子模块——定义图的构建/单例获取/缓存重置
             三函数；
          2. edges 子模块——定义两条条件边路由函数；
          3. nodes 子包——聚合导入五个节点函数（连带执行其
             五个节点子模块）；
          4. state 子模块——定义图状态 TypedDict 与每轮消息的
             初始状态工厂；
          5. 组装 __all__ 公开名单，暴露全部 12 个公共名称。

副作用：仅执行普通模块导入（连带执行各子模块的模块体初始化），
        不读写文件/数据库、不发起网络请求、不修改全局状态。
        （图构建三函数的运行时副作用——logger 输出与模块级单例
        缓存 _graph_instance 的写入——发生在调用期，不在本包
        导入期。）

主要依赖：app.ics_ai.agent.graph 下的 builder、edges、nodes、
          state 四个子模块。
"""

# 从图构建子模块导入三函数：build 构建并编译消息处理图（装配期
# 一次性调用）、get 惰性构建并缓存图单例（agent.py 初始化时调用）、
# reset 清空单例缓存（测试场景辅助——替换组件后强制重建）。
from app.ics_ai.agent.graph.builder import (
    build_message_processing_graph,
    get_message_processing_graph,
    reset_graph_instance,
)

# 从条件边子模块导入两路由函数：should_execute_action（policy 后
# 判定执行动作还是收尾）与 should_continue（guard 后判定继续循环
# 还是结束），二者共同构成图的循环控制。
from app.ics_ai.agent.graph.edges import (
    should_execute_action,
    should_continue,
)

# 从节点子包导入五节点函数：understand（命令理解）→ policy（策略
# 预测）→ action（动作执行）→ guard（循环守卫）→ response（响应
# 汇总终点），即图执行期按序驱动的全部节点。
from app.ics_ai.agent.graph.nodes import (
    understand_node,
    policy_node,
    action_node,
    response_node,
    guard_node,
)

# 从状态子模块导入：MessageProcessingState 声明图的全部状态通道
# （TypedDict 骨架），create_initial_state 为每轮消息构造初始
# 状态字典（agent.py 收到消息时调用）。
from app.ics_ai.agent.graph.state import (
    MessageProcessingState,
    create_initial_state,
)

# 包级公开 API 白名单：约束 `from app.ics_ai.agent.graph import *`
# 的导出范围，覆盖图的四类构件（状态/构建器/条件边/节点）共 12 个
# 名称（按字母序，便于 IDE 提示与 diff 整洁）。
__all__ = [
    "MessageProcessingState",  # 图状态定义：TypedDict 通道骨架
    "action_node",  # 节点：执行策略预测的动作
    "build_message_processing_graph",  # 构建器：装配并编译图
    "create_initial_state",  # 状态工厂：每轮消息的初始状态
    "get_message_processing_graph",  # 构建器：图单例惰性获取
    "guard_node",  # 节点：动作计数熔断守卫
    "policy_node",  # 节点：多策略置信度裁决
    "reset_graph_instance",  # 构建器：清空单例缓存
    "response_node",  # 节点：响应汇总终点
    "should_continue",  # 条件边：guard 后循环判定
    "should_execute_action",  # 条件边：policy 后动作判定
    "understand_node",  # 节点：命令理解与执行
]
