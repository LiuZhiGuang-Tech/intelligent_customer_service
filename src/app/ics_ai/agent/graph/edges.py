"""
edges - 条件边路由函数模块

核心功能：定义 LangGraph 消息处理图的两条条件边路由函数——
          should_execute_action（policy 后：决定执行动作还是
          收尾）与 should_continue（guard 后：决定继续循环还是
          结束），二者共同构成图的循环控制逻辑。

执行流程：
    1. 装配期：builder.py 经 add_conditional_edges 把两个函数
       分别挂在 policy 与 guard 节点之后——返回的路由键经映射
       字典翻译为目标节点名（"action"/"policy"/"response"）；
    2. 运行期：policy_node 产出 current_prediction 后进入
       should_execute_action——已完成/无预测/动作为 action_listen
       或 None 时路由到 response，否则路由到 action 执行动作；
    3. action_node 执行后经固定边进入 guard_node，guard 更新
       is_finished 后进入 should_continue——已完成/计数达上限时
       路由到 response 结束本轮，否则回 policy 形成
       "policy → action → guard" 决策循环。

副作用：无——纯路由判定（读状态、返回节点名字符串），不修改
        任何图状态、不读写文件/数据库、不调用网络；仅经 logger
        写 debug 级日志留痕路由决策。

主要依赖：
    - typing：Any/Literal（状态值通配与路由键字面量约束）；
    - app.ics_ai.shared.constants：ACTION_LISTEN（监听动作名，
      "把控制权交还用户"的约定标识）。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，PEP 604 联合
# 类型注解（X | None）不受运行时版本限制。
from __future__ import annotations

# 导入日志器工厂：路由决策经 debug 日志留痕，无其他第三方依赖。
import logging

# 导入类型注解工具：Any 类型通配符（状态字典的值类型）、Literal
# 约束路由键只能取映射字典中声明的节点名字符串。
from typing import Any, Literal

# 导入监听动作常量：与 action.py/state.py 共用的动作名约定——
# should_execute_action 据此识别"等待用户输入"类预测。
from app.ics_ai.shared.constants import ACTION_LISTEN

# 模块级日志器：以模块路径命名，运行日志带 [should_*] 前缀。
logger = logging.getLogger(__name__)


def should_execute_action(state: dict[str, Any]) -> Literal["action", "response"]:
    """policy 后的条件边：决定执行动作还是收尾。

    核心功能：检查策略预测结果，判断预测出的动作是否需要真正
              执行——"等用户说话"类动作（action_listen）与异常
              空预测直接路由到 response 结束本轮。

    业务流程：
        1. 读取完成标志与预测结果（均带默认值兜底：未完成/无
           预测）；
        2. 本轮已完成 → response（guard 熔断等情形已置标志）；
        3. 无预测结果（current_prediction 为 None）→ response
           （异常兜底：无可执行内容，避免空转）；
        4. 有预测但动作为 action_listen（控制权交还用户）或
           None（无效动作名）→ response；
        5. 其余 → action（预测出待执行的有效动作）。

    副作用：无（纯判定函数，仅写 debug 日志留痕路由决策）。

    参数：
        state: 当前图状态，读取 is_finished / current_prediction
               两个通道。

    返回：
        路由键字符串——"response"（收尾）或 "action"（执行动作）；
        builder.py 经映射字典翻译为目标节点名。本函数不抛异常
        （全部读取带默认值兜底）。
    """
    # 读取完成标志：total=False 下缺键按 False 兜底（未完成）。
    is_finished = state.get("is_finished", False)
    # 读取策略预测结果：缺键按 None 兜底（policy 异常路径未产出）。
    current_prediction = state.get("current_prediction")

    # 完成判定：guard 熔断等情形已把 is_finished 置 True——直接
    # 收尾，不再执行任何动作。
    if is_finished:
        # debug 留痕：路由到 response 的原因（本轮已完成）。
        logger.debug("[should_execute_action] 已完成，跳转到 response")
        # 返回路由键：经 builder 映射字典翻译为 response 节点。
        return "response"

    # 预测结果分派：有无预测决定走正常判定还是异常兜底。
    if current_prediction:
        # 取出预测的动作名（PolicyPrediction 的 action 字段）。
        action = current_prediction.action
        # 监听判定：action_listen 表示"把控制权交还用户、等待下一
        # 轮输入"——无需执行动作；None 表示无效动作名——同样直接
        # 收尾（二者都不会产生新的机器人响应）。
        if action == ACTION_LISTEN or action is None:
            # debug 留痕：路由到 response 的原因（监听/空动作）。
            logger.debug(f"[should_execute_action] 动作为 {action}，跳转到 response")
            # 返回路由键：结束本轮处理。
            return "response"
    else:
        # 无预测结果：policy 未产出任何预测（异常路径）——无可执行
        # 内容，直接收尾避免空转。
        # debug 留痕：路由到 response 的原因（无预测）。
        logger.debug("[should_execute_action] 没有预测结果，跳转到 response")
        # 返回路由键：结束本轮处理。
        return "response"

    # 其余情形：预测出了待执行的有效动作——路由到 action 节点。
    logger.debug("[should_execute_action] 跳转到 action 执行动作")
    # 返回路由键：经映射字典翻译为 action 节点。
    return "action"


def should_continue(state: dict[str, Any]) -> Literal["policy", "response"]:
    """guard 后的条件边：决定继续循环还是结束。

    核心功能：动作执行 + 守卫检查后判断"策略→动作"循环是否继续
              ——与 guard_node 构成双保险的第二道防线（guard 负责
              置标志，本函数负责路由且自带计数判断兜底）。

    业务流程：
        1. 读取完成标志与动作计数/上限（均带默认值兜底）；
        2. 记 debug 日志留痕当前循环进度；
        3. 本轮已完成（含 guard 熔断置标志）→ response；
        4. 计数达上限（action_count >= max_actions）→ response
           （guard 未置标志时的兜底拦截）；
        5. 其余 → policy（回策略节点继续下一轮动作决策）。

    副作用：无（纯判定函数，仅写 debug 日志留痕路由决策）。

    参数：
        state: 当前图状态，读取 is_finished / action_count /
               max_actions 三个通道。

    返回：
        路由键字符串——"policy"（继续循环）或 "response"（结束
        本轮）；builder.py 经映射字典翻译为目标节点名。本函数不
        抛异常（全部读取带默认值兜底）。
    """
    # 读取完成标志：缺键按 False 兜底（未完成）。
    is_finished = state.get("is_finished", False)
    # 读取已执行动作数：缺键按 0 兜底（本轮尚未执行动作）。
    action_count = state.get("action_count", 0)
    # 读取单轮上限：缺键按 10 兜底（与 guard_node 的兜底值一致，
    # 保证两道防线判断口径相同）。
    max_actions = state.get("max_actions", 10)

    # debug 留痕：循环进度（完成标志 + 计数进度）——排查循环行为
    # 的入口日志，每次路由决策都会输出一行。
    logger.debug(
        f"[should_continue] is_finished={is_finished}, "
        f"action_count={action_count}/{max_actions}"
    )

    # 完成判定：guard 熔断已置 is_finished=True——结束循环。
    if is_finished:
        # debug 留痕：路由到 response 的原因（本轮已完成）。
        logger.debug("[should_continue] 已完成，跳转到 response")
        # 返回路由键：结束本轮处理。
        return "response"

    # 计数兜底判定：计数达到上限即结束——即使 guard 未置标志（如
    # 状态被外部改写）也能在此拦截，构成双保险的第二道防线。
    if action_count >= max_actions:
        # debug 留痕：路由到 response 的原因（计数超限）。
        logger.debug(
            f"[should_continue] 达到最大动作数 ({action_count}/{max_actions})，"
            "跳转到 response"
        )
        # 返回路由键：结束本轮处理。
        return "response"

    # 其余情形：继续循环——回到 policy 决策下一动作，形成
    # "policy → action → guard" 执行循环直至收敛或熔断。
    logger.debug("[should_continue] 继续循环，跳转到 policy")
    # 返回路由键：经映射字典翻译为 policy 节点。
    return "policy"


# 导出：模块对外暴露的公共 API（路由函数供 builder.py 注册条件边）。
__all__ = [
    "should_execute_action",
    "should_continue",
]
