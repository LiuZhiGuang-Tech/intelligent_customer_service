"""
guard - 动作循环保护节点模块

核心功能：定义 LangGraph 消息处理图的守卫节点 guard_node——在每次
          动作执行后检查动作计数是否达到单轮上限，达到则置
          is_finished 强制终止本轮循环（防止策略-动作循环失控的
          第一道防线），并把节点名追加进执行轨迹。

执行流程：
    1. 装配期：builder.py 注册本节点并把 action → guard 接为固定
       边（guard 之后接 should_continue 条件边，路由回 policy 或
       到 response）；
    2. 执行期：每次 action_node 执行完毕进入本节点——读取
       action_count 与 max_actions 比较；未超限仅记 debug 日志
       放行，超限则记 warning 并返回 is_finished=True + error
       描述；
    3. 两种路径均把 "guard" 追加进 node_history；随后
       should_continue 按最新状态路由（is_finished=True →
       response 结束本轮；否则回 policy 继续）。

副作用：无外部副作用——不读写文件/数据库、不调用网络；仅经返回的
        部分更新 dict 修改图状态（超限写 is_finished/error/
        node_history 三个通道，未超限只写 node_history 一个通道）。

主要依赖：
    - logging：节点级日志留痕（超限 warning / 正常进度 debug）；
    - app.ics_ai.agent.graph.state：MessageProcessingState（仅注解
      用，TYPE_CHECKING 隔离）。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，TYPE_CHECKING
# 分支中的状态类型前向引用无需引号即可书写，PEP 604 联合类型注解
# （X | None）也不受运行时版本限制。
from __future__ import annotations

# 导入日志器工厂：本节点仅用日志留痕，无其他第三方依赖。
import logging

# 导入类型注解工具：Any 类型通配符（部分更新 dict 的值类型）；
# TYPE_CHECKING 隔离仅注解导入（PEP 585 新式注解下 Dict 已不需要）。
from typing import Any, TYPE_CHECKING

# 类型检查期专属导入：这些类型只出现在注解中，运行时不导入。
if TYPE_CHECKING:
    # 图状态：guard_node 的入参类型（读取动作计数/上限/轨迹通道）。
    from app.ics_ai.agent.graph.state import MessageProcessingState

# 节点级日志器：以模块路径命名，运行日志带 [guard_node] 前缀。
logger = logging.getLogger(__name__)


async def guard_node(state: "MessageProcessingState") -> dict[str, Any]:
    """保护节点：检查动作循环次数，超限强制终止。

    核心功能：动作执行后的循环熔断检查——计数达到 max_actions 上限
              时置 is_finished 并写 error 描述，防止 policy →
              action 循环失控。与 should_continue 条件边构成双保险：
              本节点负责"标记 + 错误留痕"，条件边负责"路由"（其
              自带同款计数判断，即使本节点未置标志也能兜底）。

    业务流程：
        1. 读取动作计数与上限（均带默认值兜底：计数缺省 0、上限
           缺省 10，保证防线不因键缺失失效）；
        2. 计数 ≥ 上限：记 warning 日志，返回 is_finished=True +
           error 描述 + 追加 "guard" 的执行轨迹（should_continue
           随后据 is_finished 路由到 response 结束本轮）；
        3. 未超限：仅记 debug 日志并追加轨迹放行，是否继续循环由
           should_continue 决定。

    副作用：无——不修改任何外部状态；返回的部分更新 dict 经
            LangGraph 按通道合并进图状态。

    参数：
        state: 当前图状态（MessageProcessingState），读取
               action_count / max_actions / node_history 三个通道。

    返回：
        部分更新字典（dict[str, Any]）——超限时含 is_finished
        (bool, True)/error(str)/node_history(list) 三键；未超限时
        仅含 node_history 一键。本函数不抛异常（全部读取带默认值
        兜底）。

    说明：函数体无 await 挂起点（纯同步逻辑），async 签名仅为与
          图节点的统一异步驱动方式（ainvoke）对齐。
    """
    # 读取已执行的动作计数：total=False 状态下键可能缺失，用默认
    # 值 0 兜底（本轮尚未执行动作时计数为 0）。
    action_count = state.get("action_count", 0)
    # 读取单轮动作上限：同样默认值兜底（正常由 create_initial_state
    # 注入 10；缺失时按 10 处理，防线不失效）。
    max_actions = state.get("max_actions", 10)

    # 熔断判定：计数达到上限即强制终止——此处只置标志不直接路由，
    # 路由决策统一交给 should_continue 条件边（节点与路由职责分离）。
    if action_count >= max_actions:
        # 记 warning 级日志：超限属异常情形（通常是策略反复预测
        # 动作不停），需在日志留痕供排查。
        logger.warning(
            f"[guard_node] 达到最大动作数限制 ({action_count}/{max_actions})，强制终止"
        )
        # 返回熔断更新：置完成标志（should_continue 据此路由到
        # response）、写错误描述（调试可查）、追加节点轨迹。
        return {
            "is_finished": True,
            "error": f"达到最大动作数限制: {max_actions}",
            "node_history": state.get("node_history", []) + ["guard"],
        }

    # 未超限：记 debug 级日志留痕当前计数进度。
    logger.debug(f"[guard_node] 动作计数: {action_count}/{max_actions}")

    # 正常放行：仅追加节点轨迹（新列表拼接而非原地 append，符合图
    # 状态不可变更新惯例），循环是否继续交 should_continue 判定。
    return {
        "node_history": state.get("node_history", []) + ["guard"],
    }


# 导出：模块对外暴露的公共 API（节点函数供 builder.py 注册引用）。
__all__ = ["guard_node"]
