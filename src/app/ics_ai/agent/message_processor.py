"""
message_processor - 消息处理器模块（图通道备用实现）

核心功能：定义不经 LangGraph 的消息处理完整流程——MessageProcessor
          自含"命令生成 → 命令执行 → 策略预测 → 动作执行循环"四段
          逻辑，是 agent.py 保留的向后兼容备用通道（正常路径走
          agent/graph 图执行）；另提供 MessageResponse 统一响应
          DTO 与 ProcessorConfig 处理器配置。

执行流程：
    1. 装配期：Agent.__init__ 构造本处理器（携带与图通道同一套
       组件引用：domain/flows/policy_ensemble/command_generator）；
    2. 运行期：process_message 依次执行——UserMessage 写入 tracker
       →（开关开启且有生成器时）LLM 生成命令并经 CommandProcessor
       执行 → PolicyEnsemble 预测首动作 → while 循环执行动作
       （终止条件：无动作/action_listen/达每轮上限）→ 汇总响应
       与事件；
    3. 异常期：处理全程被 try 包裹，异常时返回道歉消息 + error
       元数据（不向调用方抛出）。

副作用：
    - 修改 tracker（用户消息写入、机器人消息追加、
      latest_action_name 更新）；
    - 闲聊动作经 kwargs 注入 LLM 配置（调用外部 LLM API 的实际
      请求发生在 Action.run 内部）；
    - 无文件/数据库读写（tracker 持久化由调用方经 TrackerStore
      负责）。

主要依赖：
    - app.ics_ai.agent.actions：get_action 解析与 ActionResult；
    - app.ics_ai.dialogue_understanding：LLMCommandGenerator/
      CommandProcessor；
    - app.ics_ai.policies：PolicyEnsemble/FlowPolicy/
      EnterpriseSearchPolicy；
    - app.ics_ai.core.tracker：DialogueStateTracker/UserMessage/
      BotMessage；
    - app.ics_ai.shared.constants：ACTION_LISTEN。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，TYPE_CHECKING
# 分支中的类型前向引用无需引号即可书写，PEP 604 联合类型注解
# （X | None）不受运行时版本限制。
from __future__ import annotations

# 导入日志器工厂：处理流程各阶段留痕。
import logging

# 导入 dataclass 工具：两个 DTO 类与可变默认值工厂。
from dataclasses import dataclass, field

# 导入类型注解工具：Any 类型通配符 + TYPE_CHECKING 隔离仅注解导入
# （PEP 585/604 新式注解下 Dict/List/Optional 已不需要）。
from typing import Any, TYPE_CHECKING

# 从动作子模块导入：get_action 按名解析动作实例、ActionResult 动作
# 执行结果 DTO（responses/events/success 三字段）。
from app.ics_ai.agent.actions import get_action, ActionResult

# 从追踪器子模块导入：对话状态追踪器与两类消息 DTO（用户消息写入/
# 机器人消息追加）。
from app.ics_ai.core.tracker import DialogueStateTracker, UserMessage, BotMessage

# 从生成器子模块导入：LLM 命令生成器（命令生成段的执行者）。
from app.ics_ai.dialogue_understanding.generator import LLMCommandGenerator

# 从命令处理器子模块导入：命令执行器（命令执行段的载体）。
from app.ics_ai.dialogue_understanding.processor import CommandProcessor

# 从策略包入口导入：策略集成器与两类具体策略（默认组合构造用）。
from app.ics_ai.policies import PolicyEnsemble, FlowPolicy, EnterpriseSearchPolicy

# 从共享常量导入：监听动作名——动作循环的终止标识之一。
from app.ics_ai.shared.constants import ACTION_LISTEN

# 类型检查期专属导入：这些类型只出现在注解中，运行时不导入。
if TYPE_CHECKING:
    # 业务域定义（槽位/响应模板来源，仅注解用）。
    from app.ics_ai.core.domain import Domain

    # 流程集合（构造器注解用，实例由调用方传入）。
    from app.ics_ai.dialogue_understanding.flow import FlowsList

# 模块级日志器：以模块路径命名，日志带处理各阶段前缀。
logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    """处理器配置。

    职责：承载消息处理器的两项可调参数——每轮动作上限与命令
          生成开关。
    设计模式：数据传输对象（dataclass 参数对象）。
    生命周期：未显式传入时由构造器创建默认实例，随处理器长期
              只读持有。

    Attributes:
        max_actions_per_turn: 每轮最大动作数（动作循环的熔断
                              上限，默认 10，与图通道 max_actions
                              的默认值一致）。
        enable_command_generation: 是否启用命令生成（False 时
                                   跳过 LLM 生成段，仅走策略
                                   预测 + 动作循环）。
    """

    # 每轮最大动作数：超过即中断循环并告警（防策略死循环）。
    max_actions_per_turn: int = 10
    # 命令生成开关：与 command_generator 非空双条件共同决定是否
    # 执行 LLM 生成段。
    enable_command_generation: bool = True


@dataclass
class MessageResponse:
    """消息响应 DTO。

    职责：汇总单条消息处理的产出——机器人回复消息、产生的事件、
          处理元数据（命令明细/策略信息/错误），是 process_message
          的统一返回载体（图通道经 agent.py 组装同构字典）。
    设计模式：数据传输对象（dataclass，可变默认值经 default_factory
              保证实例隔离）。
    生命周期：每次 process_message 调用创建一个实例，填充后随
              返回值交付调用方。

    Attributes:
        messages: 机器人回复消息列表（每项为 {"text": ...} 及
                  附加键如 buttons 的字典）。
        events: 产生的事件列表（每项为 {"event": 事件类型} 及
                附加键的字典）。
        metadata: 处理元数据（commands/policy/action/confidence/
                  error 等键）。
    """

    # 机器人回复消息列表：动作响应逐条追加（add_message 或循环
    # 内直接 append）。
    messages: list[dict[str, Any]] = field(default_factory=list)
    # 事件列表：命令执行事件与动作事件逐段 extend。
    events: list[dict[str, Any]] = field(default_factory=list)
    # 处理元数据：命令明细/策略裁决信息/错误描述等。
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, text: str, **kwargs: Any) -> None:
        """添加回复消息。

        业务流程：构造 {"text": text} 基底 → 合并附加键（buttons
                  等）→ 追加进 messages。

        副作用：修改本实例的 messages 列表。

        参数：
            text: 消息文本（messages 项的 text 键）。
            **kwargs: 消息附加键（如 buttons/图片引用等结构化
                      数据，直接并入消息字典）。
        """
        # 构造基底消息字典。
        message = {"text": text}
        # 合并附加键（kwargs 展开覆盖同名键）。
        message.update(kwargs)
        # 追加进响应消息列表。
        self.messages.append(message)


class MessageProcessor:
    """消息处理器（图通道备用实现）。

    职责：自含处理用户消息的完整流程——命令生成、命令执行、
          策略预测、动作执行循环，产出 MessageResponse；与
          LangGraph 图通道（agent/graph）行为对齐但实现独立，
          供 agent.py 向后兼容保留。
    设计模式：门面（单入口 process_message 编排四段逻辑）+
              策略（动作执行委托给注册表解析的 Action 实例）。
    生命周期：由 Agent.__init__ 创建并长期持有；每条消息调用
              process_message 一次，处理器本身跨消息复用。

    流程步骤：
        1. 接收用户消息（写入 tracker）；
        2. 使用命令生成器生成命令（开关开启且生成器就位时）；
        3. 使用命令处理器处理命令（收集事件与命令明细）；
        4. 使用策略集成器确定下一步动作；
        5. 执行动作并循环（直到无动作/action_listen/达上限），
           收集响应返回。
    """

    def __init__(
        self,
        domain: Domain | None = None,
        flows: FlowsList | None = None,
        policy_ensemble: PolicyEnsemble | None = None,
        command_generator: LLMCommandGenerator | None = None,
        config: ProcessorConfig | None = None,
    ):
        """初始化消息处理器。

        业务流程：持有 domain/flows/config → 策略集成器（注入
                  优先，否则默认组合 FlowPolicy + EnterpriseSearchPolicy）
                  → 命令生成器（纯注入，允许 None）→ 命令处理器
                  （必建，接收解包后的流程列表）。

        副作用：创建 PolicyEnsemble 与 CommandProcessor 实例
                （组件构造期无外部请求）。

        参数：
            domain: 业务域定义，允许 None（预测/生成以 None 透传）。
            flows: 流程集合，允许 None（空流程能力运行）。
            policy_ensemble: 策略集成器，允许 None（按默认组合创建）。
            command_generator: 命令生成器，允许 None（无 LLM
                               生成能力，配合开关跳过生成段）。
            config: 处理器配置，允许 None（默认 ProcessorConfig）。
        """
        # 持有业务域定义（预测与生成的透传参数，可后续经
        # set_domain 替换）。
        self.domain = domain
        # 持有流程集合（可后续经 set_flows 替换）。
        self.flows = flows
        # 配置兜底：未传入时用默认值（上限 10 + 生成开启）。
        self.config = config or ProcessorConfig()

        # 初始化策略集成器：外部注入优先——与图通道共享同一集成器
        # 实例（Agent 场景），保证两通道裁决行为一致。
        if policy_ensemble:
            # 直接持有注入的集成器。
            self.policy_ensemble = policy_ensemble
        else:
            # 默认组合：流程策略（最高优先级）+ 企业搜索策略。
            self.policy_ensemble = PolicyEnsemble(
                policies=[
                    FlowPolicy(flows=flows),
                    EnterpriseSearchPolicy(),
                ]
            )

        # 初始化命令生成器：纯注入持有（None 表示无 LLM 生成能力，
        # 配合 enable_command_generation 开关跳过生成段）。
        self.command_generator = command_generator

        # 初始化命令处理器：必建组件——接收解包后的流程列表
        # （容器注解跟随运行时事实；flows 已可能为 None，条件
        # 兜底保证空列表传入）。
        self.command_processor = CommandProcessor(
            domain=domain,
            flows=flows.flows if flows else [],
        )

    async def process_message(
        self,
        message: str,
        tracker: DialogueStateTracker,
        metadata: dict[str, Any] | None = None,
    ) -> MessageResponse:
        """处理一条用户消息（核心入口）。

        业务流程：
            1. 创建 UserMessage 写入 tracker（本轮对话历史更新）；
            2.（命令生成段）开关开启且生成器就位时：LLM 生成命令
               → CommandProcessor 执行 → 事件并入响应、命令明细
               记入元数据；
            3.（策略预测段）PolicyEnsemble 预测首动作，策略名/
               动作名/置信度记入元数据；
            4.（动作循环段）while 循环执行动作：响应并入 messages
               并同步为 tracker 的 BotMessage、事件并入响应；每轮
               结束后重新预测下一动作；终止条件——无动作、
               action_listen（交还控制权）、达每轮上限；
            5.（异常兜底）全程 try 包裹：异常时记 error 日志、
               追加道歉消息与 error 元数据（不抛出）。

        副作用：修改 tracker（用户消息、机器人消息、
                latest_action_name）；命令生成段调用外部 LLM。

        参数：
            message: 用户消息文本。
            tracker: 该用户的对话状态追踪器（由调用方从存储取/
                     建，本方法不负责持久化）。
            metadata: 消息元数据（渠道/时间戳等），允许 None
                      （None 时兜底为空字典）。

        返回：
            MessageResponse——messages 为机器人回复，events 为
            命令/动作事件，metadata 携带命令明细、策略裁决信息
            与可能的 error。本方法不抛异常（内部兜底）。
        """
        # 创建空响应载体：全程向其累积产出。
        response = MessageResponse()

        # 1. 创建用户消息并更新tracker：构造消息 DTO（发送者取自
        # tracker，元数据 None 兜底空字典）。
        user_message = UserMessage(
            text=message,
            sender_id=tracker.sender_id,
            metadata=metadata or {},
        )
        # 写入 tracker（对话历史/状态联动更新）。
        tracker.update_with_message(user_message)

        # info 留痕：消息处理起点（文本全量留痕便于排查）。
        logger.info(f"Processing message from {tracker.sender_id}: {message}")

        try:
            # 2. 使用命令生成器生成命令（如果启用）：开关与生成器
            # 双条件——任一不满足即跳过生成段。
            if self.config.enable_command_generation and self.command_generator:
                # LLM 生成命令（传入 tracker/域定义/解包后的流程
                # 列表——生成链按运行时事实接收列表而非容器）。
                generation_result = await self.command_generator.generate(
                    tracker, self.domain, self.flows.flows if self.flows else []
                )

                # 有命令才进入执行段（空结果即纯闲聊/静默拒绝）。
                if generation_result.commands:
                    # 3. 使用命令处理器处理命令：执行命令并收集
                    # 产出（槽位写入/流程切换等状态变更在其内部）。
                    process_result = self.command_processor.process(
                        generation_result.commands, tracker
                    )
                    # 命令事件并入响应（slot_set 等审计轨迹）。
                    response.events.extend(process_result.events)
                    # 命令明细（DSL 序列化形式）记入元数据供观测。
                    response.metadata["commands"] = [
                        cmd.as_dict() for cmd in generation_result.commands
                    ]

            # 4. 使用策略确定下一步动作：多策略按置信度裁决。
            prediction = await self.policy_ensemble.predict(
                tracker, self.domain, self.flows
            )

            # 策略裁决信息记入元数据（动作来源可观测）。
            response.metadata["policy"] = prediction.policy_name
            response.metadata["action"] = prediction.action
            response.metadata["confidence"] = prediction.confidence

            # 5. 执行动作循环：计数清零，以首预测动作起步。
            action_count = 0
            current_action = prediction.action

            # 循环终止条件：动作耗尽（None）或为 action_listen
            # （控制权交还用户）——与图通道 should_execute_action
            # 的收尾判定对应。
            while current_action and current_action != ACTION_LISTEN:
                # 熔断判定：达每轮上限即中断（与图通道 guard/
                # should_continue 的双保险对应，此处为单点判定）。
                if action_count >= self.config.max_actions_per_turn:
                    # warning 留痕：循环被熔断。
                    logger.warning("Max actions per turn reached")
                    # 跳出循环（已产出的响应保留）。
                    break

                # 执行动作（解析/闲聊配置注入/异常兜底见
                # _execute_action）。
                action_result = await self._execute_action(
                    current_action, tracker, prediction.metadata
                )

                # 收集响应：动作产出的每条响应双写——进响应列表 +
                # 同步为 tracker 的机器人消息（对话历史完整）。
                for resp in action_result.responses:
                    # 并入响应列表。
                    response.messages.append(resp)
                    # 添加机器人消息到tracker（文本缺省空串，原始
                    # 字典存 data 键保留按钮等结构化数据）。
                    bot_message = BotMessage(
                        text=resp.get("text", ""),
                        data=resp,
                    )
                    # 写入 tracker 对话历史。
                    tracker.add_bot_message(bot_message)

                # 动作事件并入响应。
                response.events.extend(action_result.events)

                # 动作计数递增。
                action_count += 1

                # 获取下一个动作：每轮执行后重新预测（策略据最新
                # tracker 状态推进对话）。
                prediction = await self.policy_ensemble.predict(
                    tracker, self.domain, self.flows
                )
                current_action = prediction.action

            # info 留痕：处理完成的响应规模。
            logger.info(
                f"Processed message, generated {len(response.messages)} responses"
            )

        except Exception as e:
            # 异常兜底：error 留痕 + 道歉消息 + 错误元数据——
            # 保证调用方总能拿到可展示的响应（不向上抛出）。
            logger.error(f"Error processing message: {e}")
            # 追加用户可见的失败话术。
            response.add_message("抱歉，处理您的消息时出现了问题。")
            # 错误描述记入元数据（调用方可观测失败原因）。
            response.metadata["error"] = str(e)

        # 返回汇总后的响应 DTO。
        return response

    async def _execute_action(
        self,
        action_name: str,
        tracker: DialogueStateTracker,
        metadata: dict[str, Any] | None = None,
    ) -> ActionResult:
        """执行单个动作。

        业务流程：
            1. get_action 按名解析动作实例（自定义表 → 内置表 →
               utter_ 约定 → None）；
            2. 未解析到：warning 留痕并返回失败结果（success=
               False，responses 为空）；
            3. 闲聊动作（action_chitchat_response）且有生成器时：
               从生成器配置组装 llm_config 注入 kwargs（与图通道
               action_node 的注入逻辑同构）；
            4. 异步执行动作，成功则更新 tracker 的
               latest_action_name（下一轮 FlowPolicy 据此推进）；
            5. 执行异常：error 留痕并返回失败结果（不抛出）。

        副作用：动作内部可能修改 tracker 状态/调用外部服务；本
                方法自身更新 tracker.latest_action_name。

        参数：
            action_name: 动作名称（策略预测产出）。
            tracker: 对话状态追踪器（动作的操作对象）。
            metadata: 动作元数据（预测附带的 kwargs 透传，如
                      slot_to_collect/fallback_action），允许 None
                      （None 时按空 kwargs 处理）。

        返回：
            ActionResult——成功时为动作产出；未注册或执行异常时
            为 success=False 的空结果（responses 空、不抛异常）。
        """
        # debug 留痕：动作执行起点。
        logger.debug(f"Executing action: {action_name}")

        # 获取动作：按名从注册表解析（未注册返回 None）。
        action = get_action(action_name)

        # 未注册动作：warning 留痕并返回失败空结果——调用方循环
        # 继续预测下一动作（不中断消息处理）。
        if action is None:
            # warning 留痕。
            logger.warning(f"Action not found: {action_name}")
            # 失败结果（responses 空、success=False）。
            return ActionResult(success=False)

        # 合并元数据到kwargs：预测附带的元数据直接作为动作参数。
        kwargs = metadata or {}

        # 如果是闲聊动作，传递LLM配置：与图通道 action_node 同构
        # 的注入逻辑（闲聊 Action 需要 LLM 生成话术）。
        if action_name == "action_chitchat_response" and self.command_generator:
            # 取生成器的 LLM 配置。
            config = self.command_generator.config
            # 组装 llm_config 字典注入 kwargs（五要素对齐图通道）。
            kwargs["llm_config"] = {
                "type": config.type,
                "model": config.model,
                "api_key": config.api_key,
                "api_base": config.api_base,
                "enable_thinking": config.enable_thinking,
            }

        # 执行动作：异常兜底为失败结果（不中断消息处理循环）。
        try:
            # 异步执行（tracker/域定义/展开的参数）。
            result = await action.run(tracker, self.domain, **kwargs)
            # 更新 tracker 的最新动作名（下一轮策略推进依据）。
            tracker.latest_action_name = action_name
            # 返回动作产出。
            return result
        except Exception as e:
            # 执行异常：error 留痕 + 失败结果（不向上抛出）。
            logger.error(f"Action {action_name} failed: {e}")
            # 失败空结果。
            return ActionResult(success=False)

    def set_domain(self, domain: "Domain") -> None:
        """设置业务域定义（运行期替换）。

        业务流程：更新自身引用 → 同步联动命令处理器（其槽位
                  校验等依赖 Domain）。

        副作用：修改本实例与命令处理器的 domain 引用。

        参数：
            domain: 新的业务域定义（必传，不允许 None）。
        """
        # 更新自身引用。
        self.domain = domain
        # 联动命令处理器（保持两处引用一致）。
        self.command_processor.set_domain(domain)

    def set_flows(self, flows: "FlowsList") -> None:
        """设置流程集合（运行期替换）。

        业务流程：更新自身引用 → 联动命令处理器（传解包后的流程
                  列表）→ 联动策略集成器中的 FlowPolicy（传整个
                  FlowsList——其 set_flows 签名接收容器本身）。

        副作用：修改本实例、命令处理器与 FlowPolicy 的 flows 引用。

        参数：
            flows: 新的流程集合（必传）。
        """
        # 更新自身引用。
        self.flows = flows
        # 联动命令处理器（其内部接收解包后的列表）。
        self.command_processor.set_flows(flows.flows if flows else [])

        # 更新FlowPolicy：遍历策略清单，仅流程策略需要换流程引用
        # （企业搜索策略不依赖 FlowsList）。
        for policy in self.policy_ensemble.policies:
            # 类型判定：命中流程策略才联动。
            if isinstance(policy, FlowPolicy):
                # 传整个 FlowsList（与其 set_flows 签名对齐）。
                policy.set_flows(flows)


# 导出：模块对外暴露的公共 API（处理器/配置/响应 DTO）。
__all__ = [
    "MessageProcessor",
    "ProcessorConfig",
    "MessageResponse",
]
