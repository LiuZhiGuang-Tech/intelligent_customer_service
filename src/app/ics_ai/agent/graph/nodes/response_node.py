"""
response - 响应汇总终点节点模块

核心功能：定义 LangGraph 消息处理图的终点节点 response_node——
          汇总本轮处理的响应数与动作数并记日志留痕、对过程中的
          错误告警，置 is_finished 标记本轮完结；图走到本节点后
          经固定边进入 END 终止执行。

执行流程：
    1. 装配期：builder.py 注册本节点，接两条条件边入路（policy
       后 should_execute_action 判定跳过动作时、guard 后
       should_continue 判定终止循环时）与一条固定边出路
       （response → END）；
    2. 执行期：读取 final_responses / action_count / error 做
       终点留痕（info 汇总 + 错误 warning），返回 is_finished=True
       与追加 "response" 的执行轨迹；
    3. 图执行终止后，agent.py 从最终状态提取 final_responses
       作为本轮回复消息（响应内容本身由 action 节点累积写入
       final_responses 通道，本节点不产出、不修改）。

副作用：无外部副作用——不读写文件/数据库、不调用网络；仅经返回
        的部分更新 dict 修改图状态（is_finished / node_history
        两个通道）。

主要依赖：
    - logging：终点日志留痕（汇总 info / 过程错误 warning）；
    - app.ics_ai.agent.graph.state：MessageProcessingState（仅
      注解用，TYPE_CHECKING 隔离）。
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
    # 图状态：response_node 的入参类型（读取响应/计数/错误/轨迹通道）。
    from app.ics_ai.agent.graph.state import MessageProcessingState

# 节点级日志器：以模块路径命名，运行日志带 [response_node] 前缀。
logger = logging.getLogger(__name__)


async def response_node(state: "MessageProcessingState") -> dict[str, Any]:
    """响应节点：汇总收尾并标记本轮处理完结。

    核心功能：消息处理流程的最后一个节点——确认处理完结、记录
              最终统计（响应数/动作数）与过程错误，随后图经固定
              边进入 END。

    业务流程：
        1. 读取累积响应、动作计数、错误描述（均带默认值兜底，
           读取仅用于日志统计，本节点不修改任何业务数据）；
        2. 记 info 日志汇总本轮产出规模（响应数 + 动作数）；
        3. 过程中存在 error 时记 warning 告警（error 本身留在
           状态里，由 agent.py 从最终状态提取，此处不吞不改）；
        4. 返回终点更新：置 is_finished=True（终点语义保证）并
           追加 "response" 轨迹。

    副作用：无——不修改任何外部状态；返回的部分更新 dict 经
            LangGraph 按通道合并进图状态。

    参数：
        state: 当前图状态（MessageProcessingState），读取
               final_responses / action_count / error /
               node_history 四个通道。

    返回：
        部分更新字典（dict[str, Any]），恒含两键——is_finished
        (bool, True) 与 node_history(list, 追加 "response")。
        本函数不抛异常（全部读取带默认值兜底）。

    说明：函数体无 await 挂起点（纯同步逻辑），async 签名仅为与
          图节点的统一异步驱动方式（ainvoke）对齐。is_finished
          在多数入路时已为 True（guard 熔断路径），但 policy 后
          直进的路径（如动作为 action_listen）可能仍为 False，
          此处统一置 True 保证终点语义。
    """
    # 读取累积响应列表：action 节点逐步追加的机器人回复——此处仅
    # 取长度做日志统计，内容留给 agent.py 提取。
    final_responses = state.get("final_responses", [])
    # 读取已执行动作数：同样仅用于日志统计（便于核对单轮执行规模）。
    action_count = state.get("action_count", 0)
    # 读取错误描述：允许 None（无错误）；total=False 下缺键返回 None。
    error = state.get("error")

    # 记 info 级汇总日志：本轮处理的最终产出规模（响应条数 + 动作
    # 执行次数），是排查"本轮为什么回这些内容"的入口日志。
    logger.info(
        f"[response_node] 处理完成, "
        f"共 {len(final_responses)} 个响应, "
        f"执行了 {action_count} 个动作"
    )

    # 错误告警：None/空串均为假——仅在有错误时 warning 留痕；
    # 不清除也不改写 error（它保留在状态中供 agent.py 提取上报）。
    if error:
        logger.warning(f"[response_node] 处理过程中出现错误: {error}")

    # 返回终点更新：置完成标志（统一终点语义，见 docstring 说明）、
    # 追加节点轨迹（新列表拼接而非原地 append，符合图状态不可变
    # 更新惯例）；随后图经固定边 response → END 终止。
    return {
        "is_finished": True,
        "node_history": state.get("node_history", []) + ["response"],
    }


# 导出：模块对外暴露的公共 API（节点函数供 builder.py 注册引用）。
__all__ = ["response_node"]
