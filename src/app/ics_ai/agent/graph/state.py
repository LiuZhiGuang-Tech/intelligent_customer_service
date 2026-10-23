"""
graph_state - 对话图状态定义模块

核心功能：定义 LangGraph 消息处理图的统一状态结构——
          MessageProcessingState（TypedDict 声明图的全部通道：
          核心对话状态、输入输出、流程控制、中间结果、组件引用）
          与 create_initial_state（每轮消息的初始状态工厂）。

执行流程：
    1. 装配期：builder.py 以 MessageProcessingState 为骨架创建
       StateGraph（TypedDict 的每个键即一个状态通道）；
    2. 创建期：agent.py 每收到一条用户消息调用
       create_initial_state，携带 tracker、输入文本与三个组件
       引用（生成器/处理器/策略集成）构造完整状态字典；
    3. 执行期：guard/understand/policy/action/response 五个节点
       读写字段并返回部分更新 dict（total=False 允许只含变更键），
       由 LangGraph 按通道合并；edges.py 按 is_finished/
       action_count/current_prediction 做节点间路由。

副作用：无——本模块仅声明类型与构造字典，不修改任何全局状态、
        不读写文件、不调用网络（字段所引用对象的操作发生在
        节点内部）。

主要依赖：
    - typing_extensions：TypedDict（跨版本行为一致的实现）；
    - typing：Any（其余注解工具经 PEP 585/604 新式写法不再需要）。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，PEP 604 联合
# 类型注解（X | None）不受运行时版本限制。
from __future__ import annotations

# 导入类型注解工具：Any 类型通配符（复杂对象在 LangGraph 运行时
# 类型解析下统一注解为 Any，Dict/List/Optional 已按新式写法移除）。
from typing import Any

# 导入 TypedDict：声明"带固定键集合的字典类型"——LangGraph
# StateGraph 以其键集合为状态通道骨架；选自 typing_extensions
# 以获得跨 Python 版本一致的 TypedDict 行为。
from typing_extensions import TypedDict


class MessageProcessingState(TypedDict, total=False):
    """消息处理图的状态定义。

    职责：声明 LangGraph 图的全部状态通道及其类型——节点经
          state["key"]/state.get("key") 读取、经返回的部分更新
          dict 写入，是所有节点/边共享的数据契约。
    设计模式：数据传输对象（DTO）——TypedDict 是结构化字典而非
              类，无方法无行为；total=False 使全部键可选，支撑
              LangGraph 的部分更新合并语义。
    生命周期：每轮用户消息由 agent.py 经 create_initial_state
              创建一个实例，作为图执行的初始状态；图执行结束
              （agent.py 取 final_state 提取响应）后随请求销毁。

    注意：为了兼容 LangGraph 的运行时类型解析，复杂对象类型
    （tracker/flows/命令结果/组件引用等）统一使用 Any 注解，
    真实类型以行内注释标明。
    """

    # ----------------------------- 核心状态 -----------------------------
    # （使用 Any 以兼容 LangGraph 运行时类型解析）

    # 对话状态追踪器：DialogueStateTracker 实例——整轮对话的运行时
    # 状态载体（槽位/对话栈/消息历史）；understand 节点写入用户
    # 消息并执行命令，policy/action 节点读取。
    tracker: Any
    # 业务域定义：DialogueStateTracker 所属的 Domain（槽位定义/
    # 响应模板的静态来源），允许 None；understand 节点读取。
    domain: Any
    # 可用 Flow 列表：FlowsList 实例（流程定义集合），允许 None；
    # understand 节点解包后传给命令生成器。
    flows: Any

    # ----------------------------- 输入输出 -----------------------------
    # 用户输入消息：本轮待处理的原始文本（图执行的触发输入）。
    input_message: str
    # 消息元数据：随消息透传的附加信息（渠道/时间戳等），初始为空
    # 字典（create_initial_state 对 None 兜底）。
    metadata: dict[str, Any]
    # 累积的响应列表：action 节点逐步追加机器人回复（消息字典），
    # 图执行结束后由 agent.py 提取作为本轮回复。
    final_responses: list[dict[str, Any]]

    # ----------------------------- 流程控制 -----------------------------
    # 是否已完成处理：action 节点在等待用户输入/流程完成时置 True，
    # edges.py 据此路由到 response 节点结束本轮。
    is_finished: bool
    # 已执行的动作计数：action 节点每执行一个动作 +1（初始 0）。
    action_count: int
    # 最大动作数限制：单轮图执行的动作上限（edges.py 超限即路由
    # 到 response），由 create_initial_state 注入（agent.py 传 10）。
    max_actions: int

    # ----------------------------- 中间结果 -----------------------------
    # 当前生成的命令结果：understand 节点写入的 GenerationResult
    # （命令列表 + 解析诊断），允许 None（payload 直构路径无此值）。
    current_commands: Any
    # 当前命令处理结果：understand 节点写入的 ProcessResult（命令
    # 执行统计 + 事件 + next_action）——policy 节点第一轮优先采用其
    # next_action（policy.py），允许 None（无命令或未执行处理时）。
    # 注意：该通道为 2026-10-16 补充——此前 schema 未声明它，LangGraph
    # 会静默丢弃节点返回的同名键，导致 policy 的第一轮优先逻辑失效。
    process_result: Any
    # 当前策略预测结果：policy 节点写入的 PolicyPrediction（动作名
    # + 元数据），edges.py 据此路由、action.py 据此执行；允许 None。
    current_prediction: Any
    # 当前动作执行结果：action 节点写入的 ActionResult（成功/失败
    # + 响应），允许 None。
    current_action_result: Any

    # ----------------------------- 调试信息 -----------------------------
    # 执行过的节点历史：各节点执行时把自己的名字追加进来（如
    # ["understand", "policy"]），用于调试与执行轨迹追踪。
    node_history: list[str]
    # 错误信息：节点捕获异常时写入的描述文本，允许 None（无错误）。
    error: str | None

    # ----------------- 组件引用（内部使用，以 _ 开头） -----------------
    # 命令生成器引用：LLMCommandGenerator 实例——agent.py 装配期经
    # create_initial_state 注入，understand 节点经 state.get 读取
    # （下划线前缀标记内部组件通道，区别于业务数据字段）。
    _command_generator: Any
    # 命令处理器引用：CommandProcessor 实例——注入方式同上，
    # understand 节点读取。
    _command_processor: Any
    # 策略集成器引用：PolicyEnsemble 实例——注入方式同上，
    # policy 节点读取。
    _policy_ensemble: Any


def create_initial_state(
    tracker: Any,
    input_message: str,
    domain: Any = None,
    flows: Any = None,
    metadata: dict[str, Any] | None = None,
    max_actions: int = 10,
    command_generator: Any = None,
    command_processor: Any = None,
    policy_ensemble: Any = None,
) -> MessageProcessingState:
    """创建初始状态。

    业务流程：把本轮消息的运行时对象（tracker）、输入数据（消息
              文本/元数据）、静态业务定义（domain/flows）、流程
              控制（max_actions）与三个组件引用组装为完整状态
              字典——全部 18 键显式赋初值，中间结果统一为 None。

    副作用：无（纯构造，不修改任何状态）。

    参数：
        tracker: 对话状态追踪器（DialogueStateTracker 实例，唯一
                 必传的运行时对象）。
        input_message: 用户输入消息文本。
        domain: 业务域定义，允许 None。
        flows: 可用的Flow列表（FlowsList 实例），允许 None。
        metadata: 消息元数据，允许 None（None 时兜底为空字典）。
        max_actions: 最大动作数限制，默认 10（agent.py 按此值传入）。
        command_generator: 命令生成器（LLMCommandGenerator 实例），
                           允许 None（未装配时 understand 节点跳过
                           命令生成）。
        command_processor: 命令处理器（CommandProcessor 实例），
                           允许 None。
        policy_ensemble: 策略集成器（PolicyEnsemble 实例），允许
                         None。

    返回：
        初始化的状态字典（MessageProcessingState，全部键就绪，
        可直接作为 LangGraph 图的初始输入）。
    """
    # 组装完整状态字典：TypedDict 调用即普通字典构造（运行时无类型
    # 校验，键名拼写错误不会在此暴露，靠静态检查器把关）。
    return MessageProcessingState(
        # 核心状态：运行时对象与业务定义。
        tracker=tracker,
        domain=domain,
        flows=flows,
        # 输入输出：消息文本、元数据（None 兜底空字典）、响应列表
        # 置空待 action 节点累积。
        input_message=input_message,
        metadata=metadata or {},
        final_responses=[],
        # 流程控制：未完成、动作计数清零、注入上限。
        is_finished=False,
        action_count=0,
        max_actions=max_actions,
        # 中间结果：统一置 None，由各节点在执行期写入。
        current_commands=None,
        process_result=None,
        current_prediction=None,
        current_action_result=None,
        # 调试信息：轨迹与错误从空白开始。
        node_history=[],
        error=None,
        # 组件引用：装配期注入的三个组件（供节点经 state.get 读取）。
        _command_generator=command_generator,
        _command_processor=command_processor,
        _policy_ensemble=policy_ensemble,
    )


# 导出：模块对外暴露的公共 API（按字母序，便于 IDE 提示与 diff 整洁）。
__all__ = [
    "MessageProcessingState",
    "create_initial_state",
]
