"""
flow_policy - Flow 策略实现

核心功能：基于 YAML Flow 定义推进对话流程的具体策略——当对话栈中有活动的
          FlowStackFrame 时，此策略通过 FlowExecutor 逐步执行 Flow 步骤
          （action → collect → condition → end 等），最终决定应执行的动作。
          继承基类 Policy，拥有最高优先级（默认 priority=100）。

执行流程：每轮对话由上层（PolicyEnsemble）调用——
          1. should_predict 检查对话栈中是否有活动 Flow；
          2. 有则 predict 按 max_steps_per_turn 上限循环执行 FlowExecutor；
          3. 每步结果分三类：
             - collect 步骤：返回 utter_ask_{slot}，预设 slot_to_collect 供下一轮使用；
             - action 步骤：执行动作并推进步骤，若为最后一步则标记 completing；
             - flow_completed：重置 scoped slots 并触发 action_flow_completed；
          4. 超过迭代上限或无后续步骤时返回 ACTION_LISTEN 等待用户输入；
          5. 任何异常降级为 action_default_fallback（低置信度）。

副作用：会读写 tracker.slots（重置 scoped slots）、操作 tracker.dialogue_stack
        （读写 FlowStackFrame 的 completing/slot_to_collect 等属性）；
        异常路径记录 logger.error。

主要依赖：
    - app.ics_ai.policies.base_policy：Policy 基类、PolicyConfig / PolicyPrediction DTO
    - app.ics_ai.dialogue_understanding.flow.FlowExecutor / FlowsList：流程执行器与集合
    - app.ics_ai.dialogue_understanding.flow.flow.StepType：步骤类型枚举
    - app.ics_ai.dialogue_understanding.stack.stack_frame.FlowStackFrame：流程栈帧
    - app.ics_ai.shared.constants.ACTION_LISTEN：监听动作常量
    - TYPE_CHECKING 分支：DialogueStateTracker、Domain（仅类型注解用）
"""

from __future__ import annotations  # 延迟求值所有类型注解，支持前向引用

import logging  # Python 标准库日志模块，用于本策略的 debug/error 输出
from dataclasses import (
    dataclass,
)  # dataclass 装饰器，FlowPolicyConfig 自动生成 __init__
from typing import (
    TYPE_CHECKING,
    Any,
)  # 仅保留 Any 与 TYPE_CHECKING；旧式 Dict/List/Optional 已移除

from app.ics_ai.dialogue_understanding.flow import (
    FlowExecutor,
    FlowsList,
)  # 流程执行器 + 流程集合
from app.ics_ai.dialogue_understanding.stack.stack_frame import (
    FlowStackFrame,
)  # 流程栈帧（栈中区分普通帧/Flow 帧）
from app.ics_ai.policies.base_policy import (
    Policy,
    PolicyConfig,
    PolicyPrediction,
)  # 策略基类及 DTO
from app.ics_ai.shared.constants import ACTION_LISTEN  # 系统默认"监听用户输入"动作常量

# 类型检查期专属导入：这两个类型只出现在方法注解中，运行时导入会形成循环依赖
# （core/tracker → dialogue_understanding → policies 反向依赖 core），
# 故放入 TYPE_CHECKING 分支，让类型检查器能看到但运行时不执行。
if TYPE_CHECKING:
    from app.ics_ai.core.domain import Domain  # Domain 领域定义（仅类型注解用）
    from app.ics_ai.core.tracker import (
        DialogueStateTracker,
    )  # 对话状态追踪器（仅类型注解用）

# 模块级 logger：所有本模块的 debug/warning/error 日志均由此 logger 输出，
# 名称沿用 __name__ 即 "app.ics_ai.policies.flow_policy"，便于日志过滤。
logger = logging.getLogger(__name__)


@dataclass
class FlowPolicyConfig(PolicyConfig):
    """Flow 策略配置

    职责：在 PolicyConfig 基类之上追加 Flow 策略的专属配置项（优先级固定最高、
          单轮最大步数防死循环）。
    设计模式：继承 + 扩展——复用基类的 priority 字段，覆盖默认值 1 → 100；
              新增 max_steps_per_turn 字段。
    生命周期：FlowPolicy 实例创建时随 Policy.__init__ 一同构造，与策略实例同生命周期。

    Attributes:
        priority: 策略优先级；FlowPolicy 声明最高优先级 100（基类默认 1）
        max_steps_per_turn: 每轮对话中 Flow 执行的最大步数上限；防止 Flow 死循环
    """

    priority: int = 100  # 高优先级：Flow 策略应优先于其他策略执行
    max_steps_per_turn: int = 50  # 单轮最大执行步数：防止死循环导致单轮卡死


class FlowPolicy(Policy):
    """Flow 策略

    职责：基于 Flow 定义推进对话流程——当前对话栈中有活动 Flow 时，按步骤
          类型（action / collect / condition / set_slot / call / link / end）
          分发执行，每轮循环直到遇到 collect/action/flow_completed 等
          需要返回动作的节点。
    设计模式：继承基类 Policy + 模板方法——基类定义 predict 抽象方法，
              本类实现具体决策逻辑；should_predict 被覆盖为"仅在有活动
              Flow 时参与决策"。
    生命周期：框架启动时创建并注入 FlowsList 配置，随会话长期复用；
              set_flows() 可在运行时替换流程集合。

    Attributes:
        DEFAULT_PRIORITY: 类级默认优先级常量 = 100（与 FlowPolicyConfig 默认值一致）
    """

    DEFAULT_PRIORITY = (
        100  # 类级默认优先级常量；供未显式配置时与 FlowPolicyConfig 默认值保持一致
    )

    def __init__(
        self,
        config: FlowPolicyConfig
        | None = None,  # 策略专属配置；None 时兜底默认 FlowPolicyConfig()
        flows: FlowsList | None = None,  # 流程集合；None 时兜底空 FlowsList()
        **kwargs: Any,  # 额外配置透传给基类 Policy.__init__（基类忽略多余键）
    ):
        """初始化 Flow 策略

        业务流程：1. 调用基类 __init__ 处理 config 兜底；2. 兜底 flows 为空集合；
                 3. 构造 FlowExecutor（后续 predict 循环的实际执行引擎）。
        副作用：写入 self.config / self.flows / self.executor 三个实例属性（纯内存）。

        @param config: FlowPolicyConfig 配置 DTO；None 时使用默认 FlowPolicyConfig()（priority=100）。
        @param flows: FlowsList 流程集合；None 时兜底为空 FlowsList()（此时 predict 会直接 abstain）。
        @param **kwargs: 额外参数，透传给 Policy.__init__；本基类不解析、不报错。
        @return: None
        """
        # 调用基类 __init__：处理 config 兜底 + 设置 self._name = 类名
        super().__init__(config or FlowPolicyConfig(), **kwargs)
        # 兜底 flows 为空 FlowsList（避免后续 executor 因 None 报错）
        self.flows = flows or FlowsList()
        # 构造 FlowExecutor：将 FlowsList 注入执行器，后续 predict 循环反复调用 execute_next_step
        self.executor = FlowExecutor(flows=self.flows)

    def should_predict(
        self,
        tracker: DialogueStateTracker,  # 对话状态追踪器，用于判断是否有活动 Flow
    ) -> bool:
        """检查策略是否应该进行预测

        核心功能：覆盖基类默认的"恒 True"行为——FlowPolicy 只有在对话栈中存在
                 活动 Flow（tracker.active_flow 非 None）时才参与决策。
        业务流程：直接返回 tracker.active_flow is not None 的布尔结果。
        副作用：无，只读 tracker.active_flow。

        @param tracker: 对话状态追踪器；不能为空。
        @return: True 表示当前有活动 Flow，本策略应参与决策；False 表示放弃。
        """
        return tracker.active_flow is not None  # 仅在有活动 Flow 时参与决策

    async def predict(
        self,
        tracker: DialogueStateTracker,  # 对话状态追踪器；predict 全程的状态读取来源
        domain: Domain | None = None,  # Domain 定义；本类未使用但保留基类签名兼容
        flows: FlowsList | None = None,  # 外部传入的流程集合（覆盖初始化时的 flows）
        **kwargs: Any,  # 额外参数透传；本类不解析
    ) -> PolicyPrediction:
        """预测下一步动作（Flow 策略核心方法）

        业务流程：
            1. 无活动 Flow → 直接 abstain
            2. 处理 completing 标志（上一轮 action 是最后一步 → 触发 action_flow_completed）
            3. 循环执行 FlowExecutor（上限 max_steps_per_turn）：
               a. collect 步骤 → 返回 utter_ask_{slot} + 预设 slot_to_collect
               b. action 步骤 → 执行 + 判断是否最后一步（设置 completing）
               c. flow_completed → 重置 scoped slots + end_flow + 返回 action_flow_completed
               d. next_step_id 存在但无动作 → 推进步骤 continue
               e. 其他情况 → break
            4. 循环正常结束（无动作无下一步）→ 返回 ACTION_LISTEN
            5. 任何异常 → 降级为 action_default_fallback（置信度 0.5）

        副作用：
            - 写入 FlowStackFrame.completing / slot_to_collect 属性
            - 重置 tracker.slots 中 scoped slots 的值
            - 调用 tracker.end_flow() / tracker.dialogue_stack.top_flow_frame()
            - 调用 self.executor.execute_next_step() / advance_flow() / set_flows()
            - 异常路径 logger.error

        @param tracker: 对话状态追踪器，必需。
        @param domain: Domain 定义，允许 None；本类未使用。
        @param flows: 外部传入的 FlowsList，允许 None；非 None 时覆盖 self.flows 注入 executor。
        @param **kwargs: 额外参数透传；本类忽略。
        @return: PolicyPrediction；成功时 confidence=1.0，异常时 confidence=0.5。
        """
        # 如果没有活动 Flow，直接放弃预测 → 上层 ensemble 继续尝试其他策略
        if not self.should_predict(tracker):
            logger.debug("No active flow, abstaining")  # debug 日志：记录放弃原因
            return PolicyPrediction.abstain(
                self.name
            )  # abstain 工厂方法：action=None, confidence=0

        # 外部传入 flows 时覆盖 executor 中的流程集合（支持动态切换流程）
        if flows:  # 非 None 且非空（FlowsList 布尔值取决于内部 flows 长度）
            self.executor.set_flows(flows)  # 更新 executor 的流程引用

        # --- 处理 completing 标志 ---
        # 检查是否有"正在完成"的 flow（上一轮执行了最后一个 action 步骤，设置了 completing=True）
        flow_frame = (
            tracker.dialogue_stack.top_flow_frame()
        )  # 取栈顶 FlowStackFrame（可能为 None）
        if flow_frame and flow_frame.completing:  # 两者均非空 → 触发完成收尾逻辑
            completed_flow = (
                flow_frame.flow_id
            )  # 记录结束的 flow ID，用于日志和 metadata
            logger.debug(  # debug 日志：触发 action_flow_completed 收尾动作
                f"Flow {completed_flow} completing, triggering action_flow_completed"
            )
            # 重置该 flow 作用域内的槽位（保证不同 flow 之间的槽位隔离）
            self._reset_scoped_slots(tracker, completed_flow)
            # 从对话栈中结束该 flow（pop FlowStackFrame 等清理动作）
            tracker.end_flow()
            # 返回 action_flow_completed 动作，metadata 携带完成标记
            return PolicyPrediction(
                action="action_flow_completed",  # 自定义收尾动作名
                confidence=1.0,  # 确定完成，置信度满分
                events=[],  # 无额外事件
                policy_name=self.name,  # 标记来源策略
                metadata={  # metadata 携带 flow_completed 标记 + 完成的 flow ID
                    "flow_completed": True,
                    "completed_flow": completed_flow,
                },
            )

        # --- 主循环：执行 Flow 步骤 ---
        try:
            max_iterations = (
                self.config.max_steps_per_turn  # pyright: ignore[reportAttributeAccessIssue]
            )  # 取迭代上限（配置化，默认 50）
            all_events: list[
                Any
            ] = []  # 累积本轮执行产生的所有事件（供最终 prediction 返回）

            for _ in range(max_iterations):  # 迭代循环：最多执行 max_iterations 步
                # 执行当前步骤，得到 FlowExecutionResult（含 step_id / events / action / ...）
                result = self.executor.execute_next_step(tracker)
                # 将本步产生的事件追加到累积列表
                all_events.extend(result.events)

                # --- 分支 A：collect 步骤（优先处理，因为 collect 会同时设置 slot_to_collect 和 action）---
                # 必须先处理 slot_to_collect，否则会走 action 分支而丢失 metadata 信息
                if result.slot_to_collect:  # 步骤指定了要收集的槽位名
                    logger.debug(
                        f"Collecting slot: {result.slot_to_collect}"
                    )  # debug：记录收集目标

                    # 把 slot_to_collect 写入 FlowStackFrame → 下一轮 understand_node 能读取到
                    flow_frame = (
                        tracker.dialogue_stack.top_flow_frame()
                    )  # 取当前 FlowStackFrame
                    if flow_frame:  # 栈中确实有 FlowStackFrame 才写入（防御性判空）
                        flow_frame.slot_to_collect = (
                            result.slot_to_collect
                        )  # 预设槽位名

                    # 使用 collect 步骤指定的 action，或按默认规则 utter_ask_{slot} 生成
                    action = result.action or f"utter_ask_{result.slot_to_collect}"

                    # 构建 metadata：携带 slot_to_collect（供上层 prompt 注入）+ next_step_id
                    prediction_metadata: dict[str, Any] = {
                        "slot_to_collect": result.slot_to_collect,  # 当前要收集的槽位名
                        "next_step_id": result.next_step_id,  # 下一步骤 ID（供下次 predict 推进）
                    }
                    # 如果 collect 步骤配置了 fallback_action（槽位收集失败时的降级动作）一并传递
                    if result.metadata.get(
                        "fallback_action"
                    ):  # 检查内部 metadata 是否携带 fallback
                        prediction_metadata["fallback_action"] = result.metadata[
                            "fallback_action"
                        ]

                    # 返回 collect 步骤的 Prediction：confidence=1.0（确定需要用户输入）
                    return PolicyPrediction(
                        action=action,  # utter_ask_xxx 或自定义 ask 动作
                        confidence=1.0,  # 满置信度（确定要让用户填写）
                        events=all_events,  # 累积事件
                        policy_name=self.name,  # 策略来源标记
                        metadata=prediction_metadata,  # 携带收集上下文
                    )

                # --- 分支 B：action 步骤（非 collect 的动作）---
                if result.action:  # 步骤指定了要执行的动作名
                    logger.debug(
                        f"Flow action: {result.action}"
                    )  # debug：记录执行的动作

                    # 判断是否是最后一个动作（两种情形：显式 flow_completed 或 next_step_id 为 END）
                    is_final_action = (
                        result.flow_completed  # FlowExecutor 标记了"流程已完成"
                        or (
                            isinstance(
                                result.next_step_id, str
                            )  # 类型守卫：next_step_id 必须是 str
                            and result.next_step_id.upper()
                            == "END"  # 兼容 end/End/END 大小写
                        )
                    )

                    if is_final_action:  # 最后一个动作：标记 completing，让下一轮触发 action_flow_completed
                        completed_flow = tracker.active_flow  # 记录当前活动 flow ID
                        logger.debug(
                            f"Flow {completed_flow} will complete after action"
                        )  # debug 日志
                        # 获取 FlowStackFrame 并设置 completing=True（下一轮 predict 开头触发收尾）
                        flow_frame = (
                            tracker.dialogue_stack.top_flow_frame()
                        )  # 取栈顶 FlowStackFrame
                        if flow_frame:  # 防御性判空
                            flow_frame.completing = True  # 标记为"待完成"
                            flow_frame.slot_to_collect = (
                                None  # 清除 slot_to_collect（流程即将结束）
                            )
                    elif (
                        result.next_step_id
                    ):  # 非最后一步且有下一步 → 立即推进 Flow（继续循环执行下一步）
                        # 更新步骤到 next_step_id
                        self.executor.advance_flow(tracker, result.next_step_id)
                        # 如果下一步是 collect 步骤，预先写入 slot_to_collect → 下一轮 understand_node 就能感知
                        self._preset_slot_to_collect_if_needed(
                            tracker,
                            result.next_step_id,  # 把下一步骤 ID 传给辅助方法判断
                        )

                    # 返回 action 步骤的 Prediction
                    return PolicyPrediction(
                        action=result.action,  # 执行的动作名
                        confidence=1.0,  # 满置信度
                        events=all_events,  # 累积事件
                        policy_name=self.name,  # 策略来源
                        # metadata 由 _build_action_metadata 构建（含 next_step_id + flow_completed 标记）
                        metadata=self._build_action_metadata(
                            tracker, result.next_step_id, is_final_action
                        ),
                    )

                # --- 分支 C：Flow 已完成且没有 action（end 步骤显式标记完成）---
                if result.flow_completed:  # executor 标记了流程结束
                    completed_flow = tracker.active_flow  # 记录当前活动 flow
                    logger.debug(f"Flow {completed_flow} completed")  # debug 日志
                    # 结束 flow 前先重置 scoped slots（保证槽位隔离）
                    if completed_flow:  # completed_flow 非空才执行（防御性判空）
                        self._reset_scoped_slots(tracker, completed_flow)
                    tracker.end_flow()  # 从对话栈中结束该 flow
                    # 返回 action_flow_completed 动作（与 completing 分支统一）
                    return PolicyPrediction(
                        action="action_flow_completed",  # 收尾动作名
                        confidence=1.0,  # 满置信度
                        events=all_events,  # 累积事件
                        policy_name=self.name,  # 策略来源
                        metadata={  # metadata 携带完成标记
                            "flow_completed": True,
                            "completed_flow": completed_flow,
                        },
                    )

                # --- 分支 D：有下一步但没有动作（典型场景：slot 已填充，等待推进到下一个 collect/action）---
                if result.next_step_id:  # executor 返回了下一步 ID（说明还有后续步骤）
                    logger.debug(
                        f"Slot filled, advancing to step: {result.next_step_id}"
                    )  # debug 日志
                    # 清除 slot_to_collect（因为槽位已填充，不再需要收集提示）
                    flow_frame = (
                        tracker.dialogue_stack.top_flow_frame()
                    )  # 取当前 FlowStackFrame
                    if flow_frame:  # 防御性判空
                        flow_frame.slot_to_collect = None  # 清零，避免残留的收集提示
                    # 推进 Flow 到下一步骤 → 下一轮循环 execute_next_step 就处理它
                    self.executor.advance_flow(tracker, result.next_step_id)
                    continue  # continue → 进入下次循环迭代

                # --- 分支 E：没有动作也没有下一步 → 流程到此暂停，等待用户输入 ---
                break  # 跳出 for 循环

            # 循环正常退出（执行了若干步但还没到 action/collect/flow_completed）→ 返回 ACTION_LISTEN
            return PolicyPrediction(
                action=ACTION_LISTEN,  # "action_listen"：系统监听用户输入
                confidence=1.0,  # 满置信度（明确需要等用户）
                events=all_events,  # 累积事件
                policy_name=self.name,  # 策略来源标记
            )

        # 异常兜底：任何未知异常降级 → action_default_fallback，置信度 0.5（低置信度让 ensemble 可能选其他策略）
        except Exception as e:
            logger.error(f"Error executing flow: {e}")  # 记录完整异常信息便于排查
            return PolicyPrediction(
                action="action_default_fallback",  # 降级兜底动作（不是系统常量，是自定义 fallback 名）
                confidence=0.5,  # 低置信度：告知 ensemble "我不太确定，请考虑其他策略"
                policy_name=self.name,  # 策略来源
                metadata={"error": str(e)},  # metadata 携带异常消息，方便上层/用户查看
            )

    def _build_action_metadata(
        self,
        tracker: DialogueStateTracker,  # 对话状态追踪器（入参，未使用但保留基类签名兼容 + 可读性）
        next_step_id: str | None,  # 下一步骤 ID；None 表示流程已到达终点
        flow_completed: bool,  # Flow 是否已完成（由 is_final_action 传入）
    ) -> dict[str, Any]:
        """构建 action 步骤的 metadata（私有辅助方法）

        核心功能：为 action 步骤返回的 PolicyPrediction 构造 metadata 字典，
                 包含 next_step_id 和 flow_completed 标记，供上层 action_node
                 或 understand_node 消费。
        业务流程：简单字典构造，两个 key 直接对应入参。
        副作用：无（纯函数）。

        @param tracker: DialogueStateTracker；当前未使用，但保留签名与调用意图（后续扩展可读取更多上下文）。
        @param next_step_id: 下一步骤 ID 字符串；None 表示流程无后续步骤。
        @param flow_completed: True 表示当前 action 后 Flow 将结束（或已标记 completing）。
        @return: dict[str, Any]，包含 "next_step_id" 与 "flow_completed" 两个 key。

        说明：注意本方法**不**在此处添加 slot_to_collect。slot_to_collect 应仅在
             collect 步骤返回时添加到 metadata，若 action 步骤也添加会导致
             action_node 误以为当前是 collect 步骤而提前终止，属于难以排查的逻辑错误。
        """
        # 构建 metadata 字典：next_step_id 供后续 Flow 推进、flow_completed 供上层感知结束
        metadata: dict[str, Any] = {
            "next_step_id": next_step_id,  # 下一步骤 ID；None 表示终点
            "flow_completed": flow_completed,  # 是否流程完成
        }
        # 注意：不在这里添加 slot_to_collect——详见方法 docstring 说明
        return metadata  # 返回构造好的 metadata 字典

    def _preset_slot_to_collect_if_needed(
        self,
        tracker: DialogueStateTracker,  # 对话状态追踪器
        next_step_id: str,  # 下一步骤 ID（已确认非 None）
    ) -> None:
        """预设 slot_to_collect（如果下一步是 collect 步骤）（私有辅助方法）

        核心功能：当执行完 action 步骤后推进到 collect 步骤时，预先将 collect 步骤
                 指定的槽位名写入 FlowStackFrame.slot_to_collect。这样下一轮
                 understand_node 在构造 prompt 时，就能告诉 LLM "当前正在收集 X 槽位"。
        业务流程：
            1. 取当前活动 flow_id → 2. 从 FlowsList 中取 Flow → 3. 取 next_step 对应的 FlowStep
            4. 判断 step_type 是否为 COLLECT 且有 collect 槽位名 → 5. 写入 FlowStackFrame.slot_to_collect
        副作用：写入 tracker.dialogue_stack.top_flow_frame().slot_to_collect 属性。

        @param tracker: 对话状态追踪器；不能为空。
        @param next_step_id: 下一步骤 ID 字符串；不能为空（由调用方保证非 None 时才调用）。
        @return: None（任何一步判空失败都会静默返回，不抛异常）。
        """
        flow_id = tracker.active_flow  # 取当前活动 flow ID
        if not flow_id:  # 无活动 flow → 无法预设 → 静默返回（防御性判空）
            return

        flow = self.executor.flows.get_flow(
            flow_id
        )  # 从 FlowsList 中按 ID 取 Flow 对象
        if not flow:  # FlowsList 中没有该 flow → 静默返回（防御性判空）
            return

        next_step = flow.get_step(next_step_id)  # 按 next_step_id 取 FlowStep
        if not next_step:  # Flow 中没有该步骤 → 静默返回（防御性判空）
            return

        # 局部延迟导入：StepType 只在本方法用到，局部导入避免模块级提前加载
        from app.ics_ai.dialogue_understanding.flow.flow import StepType

        # 判断下一步骤是否是 COLLECT 类型且指定了要收集的槽位
        if (
            next_step.step_type == StepType.COLLECT and next_step.collect
        ):  # 双重条件：类型对 + 槽位名非空
            flow_frame = (
                tracker.dialogue_stack.top_flow_frame()
            )  # 取栈顶 FlowStackFrame
            if flow_frame:  # 防御性判空
                flow_frame.slot_to_collect = next_step.collect  # 预设槽位名到栈帧
                logger.debug(  # debug 日志：记录预设动作及下一步信息
                    f"预设 slot_to_collect: {next_step.collect} "
                    f"（下一步: {next_step_id}）"
                )

    def _reset_scoped_slots(
        self,
        tracker: DialogueStateTracker,  # 对话状态追踪器
        flow_id: str,  # 结束的 flow ID（用于取 Flow 定义的 collect/set_slot 步骤）
    ) -> None:
        """重置 Flow 作用域内的槽位（私有辅助方法）

        核心功能：当 Flow 结束时，重置该 flow 中通过 collect 和 set_slot 步骤
                 设置的槽位值，**保证不同 Flow 之间的槽位隔离**，避免一个 flow
                 设置的槽位被下一个 flow 错误复用。
        业务流程：
            1. 从 FlowsList 中取 Flow → 取 persisted_slots（配置中声明的"持久槽位"，跨 flow 保留）
            2. 遍历 collect 步骤：若 reset_after_flow_ends=True 且槽位不在 persisted_slots → 重置
            3. 遍历 set_slot 步骤：槽位不在 not_resettable_slots 和 persisted_slots → 重置
        副作用：修改 tracker.slots 中对应 Slot 实例的 value（调用 slot.reset()）。

        @param tracker: 对话状态追踪器；不能为空。
        @param flow_id: 结束的 flow ID 字符串；不能为空。
        @return: None

        重置规则说明：
            - 默认行为：collect/set_slot 步骤设置的槽位在 Flow 结束后重置为 None
            - 例外1：step.reset_after_flow_ends=False → 该步骤的槽位不重置
            - 例外2：slot 名在 flow.persisted_slots 列表中 → 跨 Flow 保留（不重置）
        """
        flow = self.executor.flows.get_flow(flow_id)  # 按 ID 取 Flow 对象
        if not flow:  # FlowsList 中没有该 flow → 记 warning 后静默返回
            logger.warning(f"[FlowPolicy] 未找到 flow: {flow_id}，跳过槽位重置")
            return

        logger.debug(
            f"[FlowPolicy] Flow {flow_id} 结束，开始重置 scoped slots"
        )  # debug 日志

        persisted_slots = set(
            flow.persisted_slots
        )  # 转为 set 以便 O(1) 查找；声明为"持久"的槽位
        not_resettable_slots: set[str] = (
            set()
        )  # 暂存"不可重置"槽位名（后续 set_slot 检查用）

        # --- 遍历 collect 步骤：重置声明了 reset_after_flow_ends 的槽位 ---
        # 局部延迟导入：StepType 只在本方法用到
        from app.ics_ai.dialogue_understanding.flow.flow import StepType

        for step in flow.steps:  # 遍历 Flow 的所有步骤
            if (
                step.step_type == StepType.COLLECT and step.collect
            ):  # 仅处理 COLLECT 类型且有 collect 槽位名
                slot_name = step.collect  # 取出此 collect 步骤指定的槽位名
                # 三重条件：该步骤声明了需要重置 + 槽位不在持久列表 → 执行重置
                if step.reset_after_flow_ends and slot_name not in persisted_slots:
                    # 在 tracker.slots 字典中查找该槽位（Slot 实例）
                    if slot_name in tracker.slots:  # 槽位确实存在才重置
                        old_value = tracker.slots[slot_name].value  # 记录旧值用于 debug
                        tracker.slots[
                            slot_name
                        ].reset()  # 调用 Slot.reset() 将值清空为 None
                        logger.debug(  # debug 日志：记录重置动作及旧值
                            f"[FlowPolicy] 重置槽位 {slot_name}: {old_value} -> None"
                        )
                    else:  # 槽位不存在（可能该 flow 中从未收集过此槽位）→ 记 debug 并跳过
                        logger.debug(f"[FlowPolicy] 槽位 {slot_name} 不存在，跳过重置")
                else:  # 槽位被配置为不重置 → 加入 not_resettable_slots 集合
                    not_resettable_slots.add(slot_name)  # 收集"不需要重置"的槽位名
                    logger.debug(  # debug：记录不重置原因
                        f"[FlowPolicy] 槽位 {slot_name} 不需要重置 "
                        f"(reset_after_flow_ends={step.reset_after_flow_ends}, "
                        f"persisted={slot_name in persisted_slots})"
                    )

        # --- 遍历 set_slot 步骤：重置显式设置的槽位（排除不可重置和持久槽位）---
        # 注：set_slot 步骤可能在 collector 之外直接写入槽位，这些也需要在 Flow 结束后清理
        for step in flow.steps:  # 再次遍历所有步骤（这次筛 SET_SLOT 类型）
            if step.step_type == StepType.SET_SLOT:  # 仅处理 SET_SLOT 类型步骤
                # step.slots 是 dict[slot_name, value]，取出所有被设置的槽位名
                slot_names = (
                    list(step.slots.keys()) if step.slots else []
                )  # slots 为 None 时空列表

                for slot_name in slot_names:  # 逐个检查
                    # 槽位既不在不可重置集合，也不在持久集合 → 执行重置
                    if (
                        slot_name not in not_resettable_slots
                        and slot_name not in persisted_slots
                    ):
                        if slot_name in tracker.slots:  # 槽位存在才重置
                            old_value = tracker.slots[slot_name].value  # 记录旧值
                            tracker.slots[slot_name].reset()  # 清空槽位值
                            logger.debug(  # debug 日志：set_slot 重置
                                f"[FlowPolicy] 重置槽位 {slot_name} (set_slot): {old_value} -> None"
                            )

    def set_flows(self, flows: FlowsList) -> None:
        """运行时替换 Flow 列表

        核心功能：在不重建策略实例的情况下，替换当前使用的 FlowsList。
                 同时更新 self.flows 和 self.executor.flows（两者保持一致）。
        业务流程：直接赋值两个引用。
        副作用：写入 self.flows 和 self.executor.flows。

        @param flows: 新的 FlowsList 实例；不能为空（与 __init__ 不同，此处不兜底）。
        @return: None
        """
        self.flows = flows  # 更新策略实例的 flows 引用
        self.executor.set_flows(flows)  # 同步更新执行器的 flows 引用


# 包级公开 API 白名单：约束 `from flow_policy import *` 的导出范围，
# 仅暴露本模块对外的两个核心类型（策略类 + 配置 DTO）。
__all__ = [
    "FlowPolicy",  # 具体策略实现类：基于 Flow 定义推进对话
    "FlowPolicyConfig",  # 策略专属配置 DTO：priority + max_steps_per_turn
]
