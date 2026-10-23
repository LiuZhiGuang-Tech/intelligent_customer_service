"""
nodes - 图节点包入口模块

核心功能：聚合 LangGraph 消息处理图的五个节点函数——understand
          （命令理解）、policy（策略预测）、action（动作执行）、
          guard（循环守卫）、response（响应汇总终点），供
          builder.py 经 add_node 注册进 StateGraph。

执行流程：本包被导入时按顺序执行——
          1. action 子模块——定义动作执行节点（解析并执行动作、
             累积响应、回退与等待判定）；
          2. guard 子模块——定义循环守卫节点（动作计数熔断）；
          3. policy 子模块——定义策略预测节点（多策略置信度
             裁决下一动作）；
          4. response 子模块——定义响应汇总终点节点（收尾留痕
             并标记完结）；
          5. understand 子模块——定义命令理解节点（LLM 生成
             命令并执行、产出处理结果）；
          6. 组装 __all__ 公开名单，暴露五个节点函数。

副作用：仅执行普通模块导入（连带执行五个子模块的模块体初始化——
        各自创建模块级 logger），不读写文件/数据库、不发起网络
        请求、不修改全局状态。（各节点运行时的真实副作用——
        tracker 状态变更/LLM 调用等——发生在图执行阶段，不在
        本包导入期。）

主要依赖：app.ics_ai.agent.graph.nodes 下的五个节点子模块
          （action / guard / policy / response / understand）。
"""

# 从动作执行子模块导入：action_node 执行策略预测出的动作——经
# get_action 解析实例并异步执行，响应累积进 final_responses，
# 内含 utter 空响应回退与等待用户输入判定。
from app.ics_ai.agent.graph.nodes.action_node import (
    action_node,
)

# 从循环守卫子模块导入：guard_node 在动作执行后检查计数是否达到
# 单轮上限，超限置 is_finished 强制终止（防无限循环的第一道防线）。
from app.ics_ai.agent.graph.nodes.guard_node import (
    guard_node,
)

# 从策略预测子模块导入：policy_node 经 PolicyEnsemble 收集各策略
# 预测并按置信度裁决下一动作；第一轮优先采用命令处理结果推导的
# next_action（直通预测，豁免 FlowPolicy 的流程内前缀）。
from app.ics_ai.agent.graph.nodes.policy_node import (
    policy_node,
)

# 从响应汇总子模块导入：response_node 为图的终点汇合节点——汇总
# 响应数与动作数记日志收尾、置 is_finished 标记本轮完结，图经
# response → END 终止。
from app.ics_ai.agent.graph.nodes.response_node import (
    response_node,
)

# 从命令理解子模块导入：understand_node 调用命令生成器产出命令、
# 经 CommandProcessor 执行并写入 ProcessResult（命令执行统计 +
# 事件 + next_action）。
from app.ics_ai.agent.graph.nodes.understand_node import (
    understand_node,
)

# 包级公开 API 白名单：约束 `from app.ics_ai.agent.graph.nodes
# import *` 的导出范围，列出全部五个节点函数（按字母序，与上方
# 导入顺序一致，便于 IDE 提示与 diff 整洁）。
__all__ = [
    "action_node",
    "guard_node",
    "policy_node",
    "response_node",
    "understand_node",
]
