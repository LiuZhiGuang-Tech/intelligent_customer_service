"""
flow_executor - Flow 执行器

核心功能：按"当前激活流程 + 当前步骤"解释执行 Flow 中的单个步骤并给出下一步，
          是静态流程图（flows 配置）与运行时对话状态（DialogueStateTracker）
          之间的执行引擎：先按步骤类型分发（action/collect/set_slot/condition/
          link/call/end），再执行相应副作用（写槽位、切换流程），最后返回
          ExecutionResult 供策略层决定要执行的动作与下一步骤。

执行流程：策略层每轮调用 execute_next_step(tracker)——1. 从 tracker.active_flow
          取当前流程 ID 并据其在 flows 中查表拿到 Flow 对象；2. 经
          _get_current_step_id 读取栈顶 FlowStackFrame 的 step_id 定位当前步骤
          （step_id 为 "start" 时视为首步）；3. 按 StepType 分发到对应
          _execute_*_step 方法；4. 各分支通过 _resolve_next_step /
          _evaluate_condition 解析 next 字符串或 if-then-else 条件列表，最终产出
          action（要执行的动作）/ slot_to_collect（待收集槽位）/
          next_step_id（下一步）/ flow_completed（流程是否结束）四个结果字段。

副作用：1. 经 tracker 写槽位（set_slot）、启动或结束流程（start_flow/end_flow）；
        2. 直接修改栈顶 FlowStackFrame 的 step_id（advance_flow）；
        3. 输出 debug/warning 日志；4. 不读写文件与数据库、不发起网络请求。

主要依赖：app.ics_ai.dialogue_understanding.flow.flow（Flow/FlowStep/FlowsList/
          StepType）、app.ics_ai.dialogue_understanding.stack.stack_frame
          （FlowStackFrame）；DialogueStateTracker 与 Domain 仅用于类型注解，
          置于 TYPE_CHECKING 分支下导入以避免循环导入。
"""

# 从 __future__ 导入 annotations：启用 PEP 563 延迟注解求值，注解以字符串形式
# 保存，既支持前向引用（如 "DialogueStateTracker"），也避免运行时求值开销。
from __future__ import annotations

# 导入 logging：模块级日志器，记录步骤执行轨迹与异常情况。
import logging

# 导入 dataclass 与 field：dataclass 自动生成 __init__/__repr__ 等，
# field(default_factory=...) 为可变默认值（list/dict）提供独立工厂，
# 避免所有实例共享同一个默认对象。
from dataclasses import dataclass, field

# 导入类型注解工具：Any 任意类型；Dict/List/Optional 为旧式 typing 写法
# （本文件尚未迁移 PEP 604/585）；TYPE_CHECKING 为"仅类型检查期成立"的常量，
# 用于隔离只服务于类型注解的导入；Union 全文件未使用。
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# 导入流程数据结构：Flow（单个流程）、FlowStep（步骤）、FlowsList（流程集合）、
# StepType（步骤类型枚举，用于分发执行逻辑）。
from app.ics_ai.dialogue_understanding.flow.flow import (
    Flow,
    FlowsList,
    FlowStep,
    StepType,
)

# 导入 Flow 栈帧：advance_flow 通过它把"当前步骤"写回对话栈，驱动流程推进。
from app.ics_ai.dialogue_understanding.stack.stack_frame import FlowStackFrame

# 类型检查期专属导入：这两个类型只出现在注解中，运行时导入会形成循环依赖
# （tracker/domain 反向依赖 flow 体系），故放入 TYPE_CHECKING 分支。
if TYPE_CHECKING:
    # 对话状态追踪器：槽位读写与流程启停的载体（仅类型注解用）。
    from app.ics_ai.core.tracker import DialogueStateTracker

    # 领域定义：槽位/动作/响应的注册表（仅类型注解用）。
    from app.ics_ai.core.domain import Domain

# 模块级日志器：按当前模块名命名，日志来源可精确定位到本文件。
logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """执行结果。

    职责：承载"执行一个步骤后"的全部输出信息，是 FlowExecutor 与调用方
          （策略层）之间的数据传输对象——调用方据 action/slot_to_collect
          决定下一轮要做什么，据 next_step_id/flow_completed 决定流程走向。
    设计模式：数据传输对象（DTO），无可变实例状态以外的行为。
    生命周期：每次执行步骤时由 FlowExecutor 临时创建，用完即弃（不跨轮持久化）。

    Attributes:
        action: 要执行的动作
        slot_to_collect: 要收集的槽位
        events: 产生的事件
        flow_completed: Flow是否已完成
        next_step_id: 下一个步骤ID
        metadata: 额外元数据（如 collect 步骤的降级动作、call 步骤的返回步骤）
    """

    # 要执行的动作名（utter_xxx 或 action_xxx）；无需执行动作时为 None。
    action: Optional[str] = None
    # 待收集的槽位名：仅 collect 步骤需要询问用户时设置；否则为 None。
    slot_to_collect: Optional[str] = None
    # 执行期间产生的事件列表（如 slot_set/flow_switched），每项为事件字典；
    # 可变默认值经 field 工厂独立创建。
    events: List[Dict[str, Any]] = field(default_factory=list)
    # 流程是否已结束：True 表示调用方可结束当前流程（不再推进步骤）。
    flow_completed: bool = False
    # 下一个步骤 ID：None 表示无后续步骤（流程到末尾）。
    next_step_id: Optional[str] = None
    # 额外元数据：扩展信息字典（fallback_action/return_step 等）；
    # 可变默认值经 field 工厂独立创建。
    metadata: Dict[str, Any] = field(default_factory=dict)


class FlowExecutor:
    """Flow执行器。

    负责执行Flow中的步骤，管理Flow的状态转换。

    工作流程：
    1. 获取当前Flow和步骤
    2. 根据步骤类型执行相应操作
    3. 确定下一个步骤
    4. 更新状态

    职责：把静态流程定义"跑起来"——定位当前步骤、按类型分发执行、
          解析条件跳转、产出下一步动作与流程状态，并对缺失流程/步骤等异常
          情况做安全降级（标记流程结束而非抛异常）。
    设计模式：策略分发——execute_next_step 按 StepType 把工作委派给不同的
              _execute_*_step 方法，各步骤类型互不感知；条件解析辅助方法
              （_evaluate_condition/_resolve_next_step）为共享工具。
    生命周期：由策略层/框架在初始化时创建并注入 FlowsList 与 Domain，
              随会话存续复用；新配置加载后可用 set_flows 更新流程集合。
    """

    def __init__(
        self,
        flows: Optional[FlowsList] = None,
        domain: Optional["Domain"] = None,
    ):
        """初始化执行器。

        业务流程：保存流程集合与领域定义；flows 为 None 时兜底为空集合，
        保证后续 get_flow 调用无需再判空。

        副作用：仅写入实例属性，无 IO 操作。

        Args:
            flows: Flow列表，None 时按空 FlowsList 处理（此时执行任何步骤都会
                   因查不到流程而直接判定流程结束）。
            domain: Domain定义，允许为 None（当前实现未直接使用，仅作扩展保留）。
        """
        # 流程集合：None 兜底为空 FlowsList，避免各处重复判空。
        self.flows = flows or FlowsList()
        # 领域定义：暂时仅持有，供后续扩展（如校验动作是否存在）。
        self.domain = domain

    def execute_next_step(
        self,
        tracker: "DialogueStateTracker",
    ) -> ExecutionResult:
        """执行下一个步骤。

        业务流程：1. 从 tracker.active_flow 取当前流程 ID，无则直接返回
        "流程已完成"（没有流程可执行）；2. 按 ID 在 flows 中查流程，查不到
        记 warning 并同样按完成处理；3. 定位当前步骤 ID 并取步骤对象，
        步骤不存在（如 step_id 指向已删除的步骤）按完成处理；4. 按步骤类型
        分发到对应执行方法；5. 返回执行结果。

        副作用：会经 tracker 读写槽位、启停流程（由各 _execute_*_step 内部
                触发）；输出 debug/warning 日志。

        参数：
            tracker: 对话状态追踪器，必须提供 active_flow 与 dialogue_stack
                     （本方法直接访问其属性，不做 None 校验）。

        返回：
            ExecutionResult 实例；分三种"无步骤可执行"的异常情况返回
            已标记 flow_completed=True 的空结果（不抛异常）。
        """
        # 预置空结果：仅在下方"无流程/无步骤"的提前返回分支中被使用；
        # 正常执行路径会被对应 _execute_*_step 的返回值整体覆盖。
        result = ExecutionResult()

        # 获取当前Flow：激活流程 ID 由 tracker（对话栈）维护。
        flow_id = tracker.active_flow
        # 无激活流程：没有流程可执行，按"流程已完成"返回，
        # 调用方可安全地认为当前流程结束。
        if not flow_id:
            logger.debug("No active flow")
            result.flow_completed = True
            return result

        # 按 ID 取流程对象：流程定义可能未加载或已被移除。
        flow = self.flows.get_flow(flow_id)
        # 流程不存在：记 warning 提示配置/状态不一致，按完成处理避免卡死。
        if not flow:
            logger.warning(f"Flow not found: {flow_id}")
            result.flow_completed = True
            return result

        # 获取当前步骤：从对话栈顶 FlowStackFrame 读取 step_id 并解析。
        current_step_id = self._get_current_step_id(tracker, flow)
        # 按 ID 取步骤对象（Flow 内部已建 ID 索引，O(1) 查找）。
        current_step = flow.get_step(current_step_id)

        # 步骤不存在：通常意味着 step_id 指向已删除/写错的步骤，
        # 记 debug 日志后按完成处理（防御性降级，不抛异常）。
        if not current_step:
            logger.debug(f"Step not found: {current_step_id}, flow completed")
            result.flow_completed = True
            return result

        # 记录即将执行的步骤与所属流程，便于复现执行轨迹。
        logger.debug(f"Executing step: {current_step_id} in flow {flow_id}")

        # 根据步骤类型执行：每个分支把工作委派给专门的执行方法，
        # 结果整体替换 result（覆盖上方的空结果）。
        if current_step.step_type == StepType.ACTION:
            result = self._execute_action_step(current_step, tracker, flow)

        elif current_step.step_type == StepType.COLLECT:
            result = self._execute_collect_step(current_step, tracker, flow)

        elif current_step.step_type == StepType.SET_SLOT:
            result = self._execute_set_slot_step(current_step, tracker, flow)

        elif current_step.step_type == StepType.CONDITION:
            result = self._execute_condition_step(current_step, tracker, flow)

        elif current_step.step_type == StepType.LINK:
            result = self._execute_link_step(current_step, tracker, flow)

        elif current_step.step_type == StepType.CALL:
            result = self._execute_call_step(current_step, tracker, flow)

        # 显式结束步骤：不执行任何动作，仅在空结果上标记流程已完成。
        elif current_step.step_type == StepType.END:
            result.flow_completed = True

        return result

    def _get_current_step_id(
        self,
        tracker: "DialogueStateTracker",
        flow: Flow,
    ) -> str:
        """获取当前步骤ID。

        业务流程：1. 在对话栈中查找属于该流程的 FlowStackFrame；
        2. 找到则取其 step_id，"start/START"（流程刚启动的哨兵值）替换为
        流程首步 ID；3. 未找到栈帧时兜底返回首步 ID。

        副作用：只读（读取对话栈与流程定义，不修改状态）。

        参数：
            tracker: 对话状态追踪器，需提供 dialogue_stack。
            flow: 当前流程对象，用于查 FlowStackFrame 与取首步。

        返回：
            当前步骤 ID 字符串；流程无任何步骤时返回哨兵值 "START"
            （后续 get_step 查不到会按"流程结束"降级处理）。
        """
        # 从dialogue_stack获取当前步骤：按流程 ID 找到对应的流程栈帧。
        flow_frame = tracker.dialogue_stack.find_flow_frame(flow.id)
        # 命中栈帧：以栈帧记录的 step_id 作为当前步骤（栈是运行时状态源）。
        if flow_frame:
            step_id = flow_frame.step_id
            # 如果 step_id 是 "start" 或 "START"，返回 flow 的第一个步骤
            # 说明流程刚启动、尚未推进过，故换算成流程定义中的首步。
            if step_id.lower() == "start":
                first_step = flow.get_first_step()
                return first_step.id if first_step else step_id
            return step_id

        # 默认返回第一个步骤：栈中无该流程的栈帧（异常状态）时兜底为首步，
        # 保证流程能从开头正常跑起来。
        first_step = flow.get_first_step()
        return first_step.id if first_step else "START"

    def _execute_action_step(
        self,
        step: FlowStep,
        tracker: "DialogueStateTracker",
        flow: Flow,
    ) -> ExecutionResult:
        """执行动作步骤。

        业务流程：1. 把 step.action 作为待执行动作写入结果；2. 把 step.next
        写入 next_step_id（可能为 None 或条件列表，由调用方/解析器处理）；
        3. next 为空或为 "end" 时标记流程已完成。

        副作用：无（本方法只组装结果，不触碰 tracker；动作的实际执行由
                调用方按 result.action 触发）。

        参数：
            step: 当前步骤，需为 ACTION 类型（读取其 action/next 字段）。
            tracker: 对话状态追踪器，本方法未使用（为与其他执行方法保持
                     统一签名而保留）。
            flow: 当前流程，本方法未使用（同上）。

        返回：
            ExecutionResult 实例，action 可能为 None（步骤未配置动作）。
        """
        # 结果容器：本步骤的输出。
        result = ExecutionResult()
        # 待执行动作：直接取步骤配置的 action（None 表示无动作）。
        result.action = step.action
        # 下一步骤：原样透传 step.next（可能是字符串、条件列表或 None）。
        result.next_step_id = step.next

        # 检查是否到达结束：next 未指定（流程末尾）或显式写 "end"，
        # 都视为该动作执行完即流程结束。
        if step.next is None or step.next == "end":
            result.flow_completed = True

        return result

    def _execute_collect_step(
        self,
        step: FlowStep,
        tracker: "DialogueStateTracker",
        flow: Flow,
    ) -> ExecutionResult:
        """执行收集步骤。

        collect步骤行为：
        - 不指定action：默认先调用 utter_ask_{slot_name}，找不到再调用 action_ask_{slot_name}
        - 显式指定action：直接使用指定的动作（可以是utter_xxx或action_xxx）
        - ask_before_filling: 是否在LLM预填充后仍询问用户确认
          - 语义：第一次进入此步骤时清空槽位并询问，用户填充后继续执行
          - 避免死循环：通过检查 slot_to_collect 判断是否已经询问过

        业务流程：1. 未配置 collect 槽位时直接跳到下一步；2. 读取槽位当前值
        与栈顶帧的"正在收集"标记，判定是否需要询问：槽位为空→询问；
        ask_before_filling 且尚未在收集该槽位→清空槽位后询问；
        ask_before_filling 且正在收集该槽位→说明用户刚填充，继续往下走；
        3. 需要询问时产出询问动作（显式 action 或 utter/action 降级策略）；
        4. 不需要询问时解析 next（含嵌套在 then 中的动作）得到下一步；
        5. 最后统一检查 next 是否到达 END，仅在无待执行动作时标记流程完成。

        副作用：ask_before_filling 首次进入时经 tracker 清空槽位（写状态）；
                输出 debug 日志。

        参数：
            step: 当前步骤，需为 COLLECT 类型（读取 collect/action/
                  ask_before_filling/next 字段）。
            tracker: 对话状态追踪器，用于读写槽位与读取对话栈顶帧。
            flow: 当前流程，本方法未直接使用（保持统一签名）。

        返回：
            ExecutionResult 实例；需要询问时 action/slot_to_collect 均有值，
            且 flow_completed 保持 False（等用户答复后再推进）。
        """
        # 结果容器：本步骤的输出。
        result = ExecutionResult()

        # 要收集的槽位名：collect 字段。
        slot_name = step.collect
        # 未配置槽位名：该步骤退化为"跳转"——仅解析 next 后返回，
        # 避免后续用 None 去读写槽位。
        if not slot_name:
            result.next_step_id = self._resolve_next_step(step.next, tracker)
            return result

        # 检查槽位是否已填充：None 表示从未填过（get_slot 未设置时返回 None）。
        current_value = tracker.get_slot(slot_name)

        # 获取 flow_frame 以检查是否正在收集此槽位：栈顶帧的 slot_to_collect
        # 是"上一轮已经向用户发起过询问"的标记，用于区分首次进入与用户已回答。
        flow_frame = tracker.dialogue_stack.top_flow_frame()
        currently_collecting = flow_frame.slot_to_collect if flow_frame else None

        # 判断是否需要询问用户：默认不询问（走"继续下一步"路径）。
        need_ask = False
        # 分支1：槽位为空——无论配置如何都必须询问用户。
        if current_value is None:
            # 槽位未填充，需要询问
            need_ask = True
        # 分支2：要求填充前询问，且当前并未处于"正在收集该槽位"状态——
        # 说明是首次进入本步骤（可能有 LLM 预填充或上一轮残留值），
        # 需清空后重新询问，保证用户确实确认过。
        elif step.ask_before_filling and currently_collecting != slot_name:
            # ask_before_filling 为 True，且我们还没开始收集这个槽位
            # 说明是第一次进入此步骤，需要清空槽位并询问
            need_ask = True
            # 清空槽位，让用户重新输入
            tracker.set_slot(slot_name, None)
            logger.debug(
                f"Slot {slot_name} has value but ask_before_filling=True and "
                f"not currently collecting, clearing slot and asking user"
            )
        # 分支3：要求填充前询问，且正好在收集该槽位——说明上一轮已问过、
        # 用户已作答，本轮应继续推进，不能重复询问（否则死循环）。
        elif step.ask_before_filling and currently_collecting == slot_name:
            # 我们已经在收集这个槽位，用户刚刚填充了值
            # 不需要再次询问，应该继续执行
            logger.debug(
                f"Slot {slot_name} filled by user (was collecting), proceeding"
            )

        # 需要询问：产出询问动作与待收集槽位，本轮到此为止。
        if need_ask:
            # 需要收集槽位：告知调用方"正在等用户提供该槽位"。
            result.slot_to_collect = slot_name

            # 确定询问动作
            if step.action:
                # 显式指定了action，直接使用（可绕过默认命名约定）
                result.action = step.action
            else:
                # 未指定action，使用默认降级策略：
                # 先尝试 utter_ask_{slot_name}，找不到再调用 action_ask_{slot_name}
                # 约定优先用静态话术，缺失时由 metadata 提示调用方回退到查库动作。
                result.action = f"utter_ask_{slot_name}"
                result.metadata["fallback_action"] = f"action_ask_{slot_name}"
        # 无需询问：槽位已就绪，解析下一步（含条件分支）并继续执行。
        else:
            # 槽位已填充且不需要确认，处理条件分支并跳到下一步
            # step.next 可能是字符串或条件列表
            resolved_next = self._resolve_next_step(step.next, tracker)

            # 检查是否有嵌套的动作需要执行：当 next 的条件分支里 then 直接
            # 写了步骤列表（如 then: [- action: ...]）时，本轮需要立刻执行
            # 该动作，而不是只跳转。
            nested_action = self._get_nested_action(step.next, tracker)
            if nested_action:
                # 使用嵌套步骤里的动作作为本轮动作。
                result.action = nested_action["action"]
                # 嵌套动作自身可能带 next，先取其值。
                result.next_step_id = nested_action.get("next")
                # 嵌套动作的 next 也需经解析（可能仍是条件列表）。
                if result.next_step_id:
                    result.next_step_id = self._resolve_next_step(
                        result.next_step_id, tracker
                    )
                logger.debug(
                    f"Slot {slot_name} filled, executing nested action: {result.action}"
                )
            else:
                # 普通情况：直接采用解析后的下一步。
                result.next_step_id = resolved_next
                logger.debug(
                    f"Slot {slot_name} already filled, moving to: {resolved_next}"
                )

        # 检查是否流程结束：再次解析 next 判断是否已到末尾/END
        # （上面只看分支，这里做统一的收口判断）。
        next_val = self._resolve_next_step(step.next, tracker)
        # next 无值或解析为 END：流程已到末尾。
        if next_val is None or (
            isinstance(next_val, str) and next_val.upper() == "END"
        ):
            # 只有在没有动作要执行时才标记完成：无动作才是真正的"无事可做"，
            # 否则会把本轮该执行的动作一起丢掉。
            if not result.action:  # 只有在没有动作要执行时才标记完成
                result.flow_completed = True

        return result

    def _execute_set_slot_step(
        self,
        step: FlowStep,
        tracker: "DialogueStateTracker",
        flow: Flow,
    ) -> ExecutionResult:
        """执行设置槽位步骤。

        业务流程：1. 遍历 step.slots 预置槽位；2. 跳过值为 None 的项
        （None 语义是"清空"，预置场景不适用）；3. 每个写入同时追加
        slot_set 事件便于回溯；4. 记录下一步；5. next 为空或 "end" 时
        标记流程完成。

        副作用：直接经 tracker 写入槽位（修改对话状态）；填充 result.events。

        参数：
            step: 当前步骤，需为 SET_SLOT 类型（读取 slots/next 字段）。
            tracker: 对话状态追踪器，槽位写入的目标。
            flow: 当前流程，本方法未使用（保持统一签名）。

        返回：
            ExecutionResult 实例，events 中每项为
            {"event": "slot_set", "name": 槽位名, "value": 值}。
        """
        # 结果容器：本步骤的输出。
        result = ExecutionResult()

        # 使用 slots 字典设置所有槽位：字典已在解析阶段由 set_slots 列表归并而来。
        if step.slots:
            # 逐项写入：slot_name 为槽位名，slot_value 为预置值。
            for slot_name, slot_value in step.slots.items():
                # 跳过 None 值：预置步骤意在"赋确定值"，None 会造成误解
                # （且 set_slot 步骤不应承担清空语义）。
                if slot_value is not None:
                    # 写入槽位（触发 tracker 的槽位校验与状态更新）。
                    tracker.set_slot(slot_name, slot_value)
                    # 记录事件：供上层生成对话事件/调试追踪。
                    result.events.append(
                        {
                            "event": "slot_set",
                            "name": slot_name,
                            "value": slot_value,
                        }
                    )

        # 下一步骤：原样透传（可能为 None/字符串）。
        result.next_step_id = step.next

        # 到达末尾或显式结束：标记流程完成。
        if step.next is None or step.next == "end":
            result.flow_completed = True

        return result

    def _execute_condition_step(
        self,
        step: FlowStep,
        tracker: "DialogueStateTracker",
        flow: Flow,
    ) -> ExecutionResult:
        """执行条件步骤。

        业务流程：1. 用 _evaluate_condition 求值 step.condition；2. 条件成立
        取 then、否则取 else_ 作为下一步；3. 下一步为空或 "end" 时标记流程完成。

        副作用：无（条件求值只读槽位，不修改状态）。

        参数：
            step: 当前步骤，需为 CONDITION 类型（读取 condition/then/else_ 字段）。
            tracker: 对话状态追踪器，条件求值的数据来源。
            flow: 当前流程，本方法未使用（保持统一签名）。

        返回：
            ExecutionResult 实例，next_step_id 为选中的分支目标（可能为 None）。
        """
        # 结果容器：本步骤的输出。
        result = ExecutionResult()

        # 评估条件：解析槽位比较/真值判断，返回布尔结果。
        condition_met = self._evaluate_condition(step.condition, tracker)

        # 条件成立：走 then 分支。
        if condition_met:
            result.next_step_id = step.then
        # 条件不成立：走 else_ 分支（属性名带下划线，对应 YAML 的 else 键）。
        else:
            result.next_step_id = step.else_

        # 分支目标缺失或为 "end"：流程结束。
        if result.next_step_id is None or result.next_step_id == "end":
            result.flow_completed = True

        return result

    def _execute_link_step(
        self,
        step: FlowStep,
        tracker: "DialogueStateTracker",
        flow: Flow,
    ) -> ExecutionResult:
        """执行链接步骤（切换到另一个Flow）。

        业务流程：1. 步骤配置了目标流程时，先结束当前流程再启动目标流程
        （link 语义为"切换"，不返回原流程）；2. 追加 flow_switched 事件记录
        切换关系；3. 无条件标记当前流程完成（控制权已交给新流程）。

        副作用：经 tracker 结束当前流程并启动新流程（修改对话栈与激活流程）；
                填充 result.events。

        参数：
            step: 当前步骤，需为 LINK 类型，flow_id 为空时为空操作。
            tracker: 对话状态追踪器，流程切换的执行者。
            flow: 当前流程，用于在事件中记录来源流程 ID。

        返回：
            ExecutionResult 实例，flow_completed 恒为 True，events 中每项为
            {"event": "flow_switched", "from_flow": ..., "to_flow": ...}。
        """
        # 结果容器：本步骤的输出。
        result = ExecutionResult()

        # 有目标流程才执行切换；flow_id 为空时仅做"流程结束"处理。
        if step.flow_id:
            # 结束当前Flow：避免新旧流程栈帧并存导致的状态混乱。
            tracker.end_flow()
            # 启动新Flow：新流程从其首步开始执行。
            tracker.start_flow(step.flow_id)
            # 记录切换事件，便于回溯"流程为何突然变了"。
            result.events.append(
                {
                    "event": "flow_switched",
                    "from_flow": flow.id,
                    "to_flow": step.flow_id,
                }
            )

        # link 语义为切换而非调用：无论是否切换成功，当前流程都到此结束。
        result.flow_completed = True
        return result

    def _execute_call_step(
        self,
        step: FlowStep,
        tracker: "DialogueStateTracker",
        flow: Flow,
    ) -> ExecutionResult:
        """执行调用步骤（调用子Flow，完成后返回）。

        业务流程：1. 步骤配置了目标流程时启动子流程；2. 追加 subflow_called
        事件记录父子关系；3. 把 step.next 记入 metadata["return_step"]，
        作为子流程结束后返回的步骤；4. 不标记 flow_completed（父流程只是
        暂时让出控制权，并未结束）。

        副作用：经 tracker 启动子流程（向对话栈压入新的流程帧）；
                填充 result.events 与 result.metadata。

        参数：
            step: 当前步骤，需为 CALL 类型，flow_id 为空时为空操作。
            tracker: 对话状态追踪器，子流程启动的执行者。
            flow: 当前流程，用于在事件中记录父流程 ID。

        返回：
            ExecutionResult 实例，flow_completed 保持 False；
            events 中每项为
            {"event": "subflow_called", "parent_flow": ..., "child_flow": ...}。
        """
        # 结果容器：本步骤的输出。
        result = ExecutionResult()

        # 有目标流程才执行调用；flow_id 为空时本步骤为空操作（结果保持默认值）。
        if step.flow_id:
            # 压入当前Flow状态：start_flow 会把子流程帧压栈，
            # 父流程帧保留在栈中，子流程结束后可回到父流程继续执行。
            tracker.start_flow(step.flow_id)
            # 记录子流程调用事件，便于回溯流程嵌套关系。
            result.events.append(
                {
                    "event": "subflow_called",
                    "parent_flow": flow.id,
                    "child_flow": step.flow_id,
                }
            )
            # 记录返回步骤：子流程结束后应回到父流程的这一步。
            result.metadata["return_step"] = step.next

        return result

    def _evaluate_condition(
        self,
        condition: Optional[str],
        tracker: "DialogueStateTracker",
    ) -> bool:
        """评估条件表达式。

        支持简单的槽位检查条件。

        业务流程：1. 条件为空时直接视为成立（无条件即放行）；
        2. 去首尾空白并剥离 "slots." 前缀（兼容 Rasa 风格的 slots.xxx 写法）；
        3. 含 "==" 按相等比较（右值去引号，左值与实际值都转字符串比较）；
        4. 含 "!=" 按不等比较（同上）；
        5. 否则按"槽位名"处理——取槽位值做真值判断，其中字符串 "true"/"false"
        特判为布尔，其余交给 bool()。

        副作用：无（只读 tracker 的槽位值）。

        参数：
            condition: 条件表达式，允许为 None（视为无条件成立）；
                       支持 "slot == value"、"slot != value"、"slot"/
                       "slots.slot" 三种形式。
            tracker: 对话状态追踪器，槽位取值来源。

        返回：
            条件是否成立；格式无法识别时（如 "==" 出现多次导致 split 结果
            不是两段）会继续向后走真值判断，最终返回该槽位的真值。
        """
        # 无条件：不配置即视为通过，避免上层到处补判空。
        if not condition:
            return True

        # 简单的槽位检查
        # 格式: slot_name == value 或 slot_name != value 或 slot_name
        # 也支持 slots.slot_name 格式
        # 去掉首尾空白，避免 YAML 缩进/换行带来的比较失败。
        condition = condition.strip()

        # 移除 "slots." 前缀（如果有）：统一成裸槽位名，两种写法共用后续逻辑。
        condition = condition.replace("slots.", "")

        # 检查相等：形如 slot == value。
        if "==" in condition:
            parts = condition.split("==")
            # 恰好两段才合法（多于两段属异常表达式，落到下方真值判断兜底）。
            if len(parts) == 2:
                # 左侧为槽位名。
                slot_name = parts[0].strip()
                # 右侧为期望值：去掉引号（YAML 里常写成 "false" 这类字符串）。
                expected_value = parts[1].strip().strip("\"'")
                # 取槽位实际值。
                actual_value = tracker.get_slot(slot_name)
                # 双方都转字符串比较：兼容槽位值为 bool/None 而表达式右侧是字符串的场景。
                return str(actual_value) == expected_value

        # 检查不相等：形如 slot != value。
        if "!=" in condition:
            parts = condition.split("!=")
            # 同样要求恰好两段。
            if len(parts) == 2:
                # 左侧槽位名。
                slot_name = parts[0].strip()
                # 右侧期望值（去引号）。
                expected_value = parts[1].strip().strip("\"'")
                # 槽位实际值。
                actual_value = tracker.get_slot(slot_name)
                # 字符串化后比较不相等。
                return str(actual_value) != expected_value

        # 检查槽位值是否为真（truthy）
        # 当条件只是槽位名时（如 "if: slots.set_receive_info"），检查槽位值是否为真
        # 此时 condition 已被剥掉 slots. 前缀，故直接当槽位名使用。
        slot_value = tracker.get_slot(condition)
        # 处理字符串形式的布尔值：用户/数据库写入的 "true"/"false" 字符串
        # 若不特判，"false" 会因非空字符串被判为真（bool("false") is True）。
        if isinstance(slot_value, str):
            if slot_value.lower() == "true":
                return True
            elif slot_value.lower() == "false":
                return False
        # 返回布尔值（None、False、空字符串等都返回 False）
        return bool(slot_value)

    def _resolve_next_step(
        self,
        next_value: Any,
        tracker: "DialogueStateTracker",
    ) -> Optional[str]:
        """解析下一步骤ID。

        处理 next 字段可能是字符串或条件列表的情况。

        业务流程：1. None 直接返回 None；2. 字符串原样返回；
        3. 列表则按顺序逐项评估：命中 "if" 且条件成立时取 then——
        then 为字符串则直接返回，then 为步骤列表时返回一个嵌套标记
        "__nested__{对象id}" 交由 _get_nested_action 提取实际动作；
        4. 命中 "else" 分支时取其值（非字符串则返回 None）；
        5. 所有分支都不匹配时返回 None。

        副作用：无（评估条件只读槽位）。

        参数：
            next_value: next 字段的值，任意类型——None/字符串/条件分支列表。
            tracker: 对话状态追踪器，供条件求值取槽位。

        返回：
            解析后的下一步骤 ID 字符串；无法解析（无匹配分支、else 值非字符串、
            嵌套 then 为空等）时返回 None。
        """
        # 未配置 next：无后续步骤。
        if next_value is None:
            return None

        # 如果是字符串，直接返回：普通"下一步骤 ID"写法。
        if isinstance(next_value, str):
            return next_value

        # 如果是列表，评估条件找到正确的下一步：按书写顺序取第一个成立的分支
        # （与 if/elif/else 的语义一致）。
        if isinstance(next_value, list):
            # 逐项评估分支。
            for branch in next_value:
                # 非字典项无法解释，跳过（防御配置写错类型）。
                if not isinstance(branch, dict):
                    continue

                # 处理 if-then 分支
                if "if" in branch:
                    # 条件表达式（如 slots.order_id != "false"）。
                    condition = branch["if"]
                    # 条件成立：使用该分支的 then。
                    if self._evaluate_condition(condition, tracker):
                        # then 可能是字符串或嵌套的步骤列表
                        then_value = branch.get("then")
                        # then 为字符串：即目标步骤 ID（或 END），直接返回。
                        if isinstance(then_value, str):
                            return then_value
                        # then 为嵌套步骤列表：本方法无法表达"要执行的动作"，
                        # 故返回以对象 id 编码的嵌套标记，由调用方配合
                        # _get_nested_action 还原（同一轮的 next 会再解析一次）。
                        elif isinstance(then_value, list) and then_value:
                            # 如果是步骤列表，返回第一个步骤的 action 或 id
                            first_step = then_value[0]
                            # 首项为字典时用标记表示"存在嵌套步骤"。
                            if isinstance(first_step, dict):
                                # 返回一个标记，表示需要执行嵌套步骤
                                return f"__nested__{id(then_value)}"
                        # 空列表等其他情况：原样返回 then（如 None）。
                        return then_value  # pyright: ignore[reportReturnType]

                # 处理 else 分支：兜底分支，命中即返回其目标。
                if "else" in branch:
                    else_value = branch["else"]
                    # 仅字符串可作为步骤 ID；否则视为无后续步骤。
                    return else_value if isinstance(else_value, str) else None

        # 类型不可识别（如字典/数字等非法配置）：视为无后续步骤。
        return None

    def _get_nested_action(
        self,
        next_value: Any,
        tracker: "DialogueStateTracker",
    ) -> Optional[Dict[str, Any]]:
        """获取嵌套步骤中的动作。

        当 next 是条件列表且 then 包含嵌套步骤时，返回第一个动作。

        业务流程：1. 非列表直接返回 None；2. 逐项评估分支：命中 "if" 且条件
        成立时，若 then 是非空列表且首项为含 "action" 的字典，则返回
        {"action": ..., "next": ...}；否则返回 None（该轮无嵌套动作）；
        3. 命中 "else" 分支时同样返回 None（else 分支不承载嵌套动作）；
        4. 全部不匹配返回 None。

        副作用：无（条件求值只读槽位）。

        参数：
            next_value: next 字段的值，任意类型（通常为条件分支列表）。
            tracker: 对话状态追踪器，供条件求值取槽位。

        返回：
            嵌套动作信息字典，含 "action" 与 "next" 两个键；
            无嵌套动作时返回 None。
        """
        # 非列表：结构上不含分支，必然无嵌套动作。
        if not isinstance(next_value, list):
            return None

        # 逐项评估分支（与 _resolve_next_step 保持同样的遍历顺序，
        # 保证"解析出的下一步"与"取出的动作"来自同一分支）。
        for branch in next_value:
            # 跳过非字典项。
            if not isinstance(branch, dict):
                continue

            # 处理 if-then 分支
            if "if" in branch:
                # 条件表达式。
                condition = branch["if"]
                # 仅条件成立的分支才取其嵌套动作。
                if self._evaluate_condition(condition, tracker):
                    # 该分支的 then。
                    then_value = branch.get("then")
                    # then 为非空列表才可能含嵌套步骤。
                    if isinstance(then_value, list) and then_value:
                        # 取首项（本实现只执行第一个嵌套步骤）。
                        first_step = then_value[0]
                        # 首项必须是含 action 键的字典才构成"可执行动作"。
                        if isinstance(first_step, dict) and "action" in first_step:
                            return {
                                # 动作名。
                                "action": first_step["action"],
                                # 动作自身的下一步（可能为 None）。
                                "next": first_step.get("next"),
                            }
                    # 条件成立但 then 中无动作：本轮无嵌套动作可执行。
                    return None

            # 处理 else 分支：按约定 else 不含嵌套步骤列表，
            # 命中即表示本轮无动作，直接返回 None。
            if "else" in branch:
                return None

        # 无任何匹配分支。
        return None

    def advance_flow(
        self,
        tracker: "DialogueStateTracker",
        next_step_id: str,
    ) -> None:
        """推进Flow到下一步。

        业务流程：取对话栈顶的流程栈帧，把其 step_id 更新为 next_step_id，
        使下一轮 execute_next_step 定位到新步骤。

        副作用：修改栈顶 FlowStackFrame 的 step_id（原地变更对话栈状态）。

        参数：
            tracker: 对话状态追踪器，需提供 dialogue_stack。
            next_step_id: 下一步 ID，字符串（可为步骤 ID 或 "END" 等哨兵值）。

        返回：
            无返回值；栈中无流程栈帧（异常状态）时静默跳过，不抛异常。
        """
        # 更新栈顶FlowStackFrame的step_id：栈帧是"当前步骤"的唯一状态源。
        flow_frame = tracker.dialogue_stack.top_flow_frame()
        # 有栈帧才更新；无栈帧说明当前没有流程在跑，保持不变即可。
        if flow_frame:
            flow_frame.step_id = next_step_id

    def set_flows(self, flows: FlowsList) -> None:
        """设置Flow列表。

        业务流程：直接用入参替换执行器持有的流程集合，使后续执行基于新配置。

        副作用：修改实例属性 self.flows（影响该实例后续所有执行）。

        参数：
            flows: Flow列表（FlowsList），不允许为 None（与构造函数不同，
                   此方法不做空值兜底）。

        返回：
            无返回值。
        """
        # 替换流程集合：用于热更新场景（如重新加载 flows 配置后）。
        self.flows = flows


# 导出清单：from flow_executor import * 时仅暴露以下两个名字。
__all__ = [
    # 列表项1：Flow 执行器类。
    "FlowExecutor",
    # 列表项2：执行结果数据传输对象。
    "ExecutionResult",
]
