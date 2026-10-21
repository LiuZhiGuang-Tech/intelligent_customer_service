"""
flow_commands - Flow 生命周期命令模块

核心功能：定义围绕对话流程（Flow）生命周期的三个命令类——
          StartFlowCommand（启动 Flow，"即时数据命令"）、
          CancelFlowCommand（取消 Flow，"动作触发命令"）、
          ChangeFlowCommand（切换 Flow，"动作触发命令"）；
          三者均继承 Command 基类并经 @register_command 注册，
          支持字典反序列化（from_dict）与 DSL 文本正则解析
          （regex_pattern + _from_regex_match）两条构造路径。

执行流程：
    1. 装配期：模块被导入时，三个命令类经 @register_command
       装饰器以 command_name() 为键登记进 base 模块的
       _COMMAND_REGISTRY（如 "start_flow" → StartFlowCommand）；
    2. 反序列化：LLM 输出的命令字典经各自的 from_dict 构造
       （必填字段缺失时抛 ValueError），或命令 DSL 文本经
       from_dsl 正则匹配构造（覆盖 "start flow xxx"、
       "CancelFlow(xxx)" 等多种写法）；
    3. 执行期：CommandProcessor 调用 run()——即时数据命令直接
       修改 tracker 状态（StartFlowCommand 调用
       tracker.start_flow），以便 FlowPolicy 立即接管执行 Flow
       步骤；动作触发命令仅返回事件标记
       （cancel_flow_requested / change_flow_requested），由对应
       Action（ActionCancelFlow / ActionChangeFlow）执行实际
       操作，实现"命令声明意图、动作执行操作"的解耦。

副作用：
    - 导入期：向 base 模块的全局注册表 _COMMAND_REGISTRY 写入
      三个命令类（全局可变状态，跨会话持续生效）；
    - run 的具体实现：StartFlowCommand 会修改 tracker 的活动
      Flow 状态；本模块自身无 IO、不调用网络。

主要依赖：
    - re：from_dsl 的 DSL 文本正则解析。
    - dataclasses：@dataclass 自动生成命令类的数据类方法。
    - typing：Any + TYPE_CHECKING（注解已按 PEP 585/604 新式写法）。
    - app.ics_ai.dialogue_understanding.commands.base：命令基类
      Command 与注册装饰器 register_command。
    - app.ics_ai.core.tracker：命令 run 的操作对象（仅注解用）。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，支持前向引用
# （如 TYPE_CHECKING 分支中的 tracker 类型与各命令类自引用）。
from __future__ import annotations

# 导入正则引擎：from_dsl 的模式匹配（regex_pattern + _from_regex_match）依赖它。
import re

# 导入 dataclass 装饰器：三个命令类靠它自动生成 __init__ / __repr__ 等数据类方法。
from dataclasses import dataclass

# 导入类型注解工具：Any 类型通配符；TYPE_CHECKING 隔离仅注解导入
# （PEP 585/604 新式注解下 Dict/List/Optional 已不需要）。
from typing import TYPE_CHECKING, Any

# 导入命令基类与注册装饰器：三个命令类继承 Command 并经 register_command 注册。
from app.ics_ai.dialogue_understanding.commands.base import Command, register_command

# 类型检查期专属导入：tracker 类型只出现在方法注解中，运行时不
# 导入（避免与 core 包形成循环依赖）。
if TYPE_CHECKING:
    # 对话状态追踪器：命令 run 的操作对象（仅注解用）。
    from app.ics_ai.core.tracker import DialogueStateTracker


@register_command
@dataclass
class StartFlowCommand(Command):
    """启动Flow命令。

    职责：封装"启动指定对话流程"的意图——当 LLM 识别到用户意图
          与某个 Flow 匹配时生成此命令，run() 阶段校验 Flow
          存在性并在 tracker 上启动它。
    设计模式：命令模式 + 注册表工厂（继承 Command 并经
              @register_command 注册）；同时是"即时数据命令"——
              run() 直接操作 tracker 设置 active_flow，以便
              FlowPolicy 能立即接管执行 Flow 步骤（与"动作触发
              命令"如 CancelFlowCommand 形成对照）。
    生命周期：LLM 生成器产出命令文本/字典 → from_dict 或
              from_dsl 构造实例 → CommandProcessor 执行 run()
              后随单轮消息处理流程销毁。

    字段说明：
        flow: 要启动的Flow ID。
    """

    # 要启动的 Flow ID（必填；由 LLM 按用户意图从可用 Flow 中选择）。
    flow: str

    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。

        返回：
            "start_flow"：本命令在注册表中的唯一键。
        """
        # 注册表键：LLM 字典输出中 "command" 字段的取值即此名。
        return "start_flow"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StartFlowCommand:
        """从字典创建命令。

        业务流程：1. 读取 data["flow"] 构造实例；2. 字段缺失时
                  捕获 KeyError 并转抛 ValueError（携带缺失字段
                  说明，保留原始异常链）。

        副作用：无（纯构造）。

        参数：
            data: LLM 输出的命令字典，须含 "flow" 键（Flow ID 字符串）。

        返回：
            StartFlowCommand 实例；缺少 "flow" 字段时抛 ValueError。
        """
        # 读取必填字段 flow 构造实例；缺失时走异常分支。
        try:
            return StartFlowCommand(flow=data["flow"])
        # KeyError 说明 LLM 输出缺少 flow 字段：转抛 ValueError 快速失败。
        except KeyError as e:
            raise ValueError(
                "Missing required field 'flow' for StartFlowCommand"
            ) from e

    def run(
        self,
        tracker: DialogueStateTracker,
        flows: Any | None = None,
    ) -> list[dict[str, Any]]:
        """执行启动Flow命令。

        业务流程：1. 若调用方提供了 flows，则校验目标 Flow 是否
                  存在于可用列表（不存在则静默拒绝，返回空事件）；
                  2. 调用 tracker.start_flow 设置活动 Flow；
                  3. 返回 flow_started 事件字典。

        副作用：修改 tracker 状态（活动 Flow 被设置为本命令的
                flow 字段）。

        参数：
            tracker: 对话状态追踪器，必需。
            flows: 可用的 Flow 列表（或带 flow_ids 属性的对象），
                   允许 None——为 None 时跳过存在性校验直接启动。

        返回：
            单元素事件列表（event=flow_started，含 flow_id 与
            时间戳占位）；Flow 不存在时返回空列表。
        """
        # 检查Flow是否存在：flows 为 None 表示调用方未提供可用列表，跳过校验。
        if flows is not None:
            # 第一步兜底：尝试从 flows 对象读取 flow_ids 属性，取不到则以空列表占位。
            flow_ids = getattr(flows, "flow_ids", [])
            # 第二步：flows 本身可迭代（且不是字符串）时，以迭代内容为准重新收集。
            if hasattr(flows, "__iter__") and not isinstance(flows, str):
                # 逐个元素取 id 属性；元素无 id 属性时退化为字符串形式参与比对。
                flow_ids = [f.id if hasattr(f, "id") else str(f) for f in flows]
            # 目标 Flow 不在可用列表中：静默拒绝启动（返回空事件）。
            if self.flow not in flow_ids:
                # Flow不存在，返回空事件列表
                return []

        # 启动Flow：直接在 tracker 上设置活动 Flow（即时数据命令的核心副作用）。
        tracker.start_flow(self.flow)

        # 返回 flow_started 事件列表：
        return [
            {
                # 事件类型标记：由下游 Policy/Action 识别。
                "event": "flow_started",
                # 事件负载：被启动的 Flow ID。
                "flow_id": self.flow,
                # 事件时间戳占位：由 tracker 后续补齐。
                "timestamp": None,
            }
        ]

    def to_dsl(self) -> str:
        """转换为DSL字符串。

        业务流程：输出 "start flow <flow_id>" 格式的 DSL 文本
                  （与 regex_pattern 的第一种格式对应，可往返解析）。

        副作用：无。

        返回：
            DSL 格式的命令字符串。
        """
        # f-string 拼接：命令名 + Flow ID，空格分隔。
        return f"start flow {self.flow}"

    @classmethod
    def regex_pattern(cls) -> str:
        """返回用于解析此命令的正则表达式模式。

        业务流程：返回支持两种格式的模式串——
                  1. "start flow <flow_name>"（Flow 名入捕获组 1）；
                  2. "StartFlow(<flow_name>)"（Flow 名入捕获组 2）。

        副作用：无。

        返回：
            正则表达式字符串（含两个互斥的可选捕获组）。
        """
        # 支持多种格式:
        # - start flow <flow_name>
        # - StartFlow(<flow_name>)
        # 外层 (?:...) 非捕获组包裹两种写法的分支；引号与反引号均可选包裹 Flow 名。
        return r"""(?:start\s+flow\s+['"`]?([a-zA-Z0-9_-]+)['"`]?|StartFlow\(['"]?([a-zA-Z0-9_-]+)['"]?\))"""

    @classmethod
    def _from_regex_match(cls, match: re.Match[str]) -> StartFlowCommand | None:
        """从正则匹配结果创建命令。

        业务流程：1. 取第一个捕获组（start flow 格式的 Flow 名），
                  为空则取第二个捕获组（StartFlow() 格式）；
                  2. 命中任一分组则 strip 去空白后构造实例；
                  3. 两组均空返回 None。

        副作用：无（纯构造）。

        参数：
            match: from_dsl 中 re.match 的匹配结果。

        返回：
            StartFlowCommand 实例；两个捕获组均为空时返回 None。
        """
        # 第一个分组是 start flow 格式，第二个是 StartFlow() 格式：优先取前者。
        flow = match.group(1) or match.group(2)
        # 命中任一分组：去除首尾空白后构造实例。
        if flow:
            return StartFlowCommand(flow=flow.strip())
        # 两组均空：无法确定 Flow 名，返回 None 由调用方降级处理。
        return None

    def __hash__(self) -> int:
        """计算命令的哈希值。

        说明：覆盖基类"按命令名哈希"的口径，改为仅按 flow 字段
              哈希——同 Flow 的启动命令共享哈希口径，与 __eq__
              的字段级等价口径配套。
        """
        # 以 flow 为哈希依据：等价命令（同 flow）必同哈希。
        return hash(self.flow)

    def __eq__(self, other: object) -> bool:
        """判断两个命令是否相等。

        业务流程：1. 非 StartFlowCommand 实例直接判不等；
                  2. 同类型命令按 flow 字段比较（覆盖基类的
                  as_dict 全字段比较，收窄为同类型 + 同 Flow）。

        参数：
            other: 待比较对象（任意类型）。

        返回：
            True 表示同为 StartFlowCommand 且 flow 相等；否则 False。
        """
        # 类型守卫：非本类实例一律不等（避免属性访问错误）。
        if not isinstance(other, StartFlowCommand):
            return False
        # 字段级比较：仅 flow 参与等价判定。
        return other.flow == self.flow


@register_command
@dataclass
class CancelFlowCommand(Command):
    """取消Flow命令。

    职责：封装"取消当前正在执行的 Flow"的意图；run() 只返回
          cancel_flow_requested 事件标记，实际的取消操作由
          ActionCancelFlow 执行。
    设计模式：命令模式 + 注册表工厂；动作触发模式——命令仅声明
              意图，动作执行实际操作（与 StartFlowCommand 的
              "即时数据命令"形成对照）。
    生命周期：同 StartFlowCommand——LLM 产出 → from_dict 或
              from_dsl 构造实例 → CommandProcessor 执行 run()
              后随单轮消息处理流程销毁。

    字段说明：
        flow: 要取消的Flow ID，如果为None则取消当前活动的Flow。
    """

    # 要取消的 Flow ID（可选；默认 None 表示取消当前活动 Flow）。
    flow: str | None = None

    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。

        返回：
            "cancel_flow"：本命令在注册表中的唯一键。
        """
        # 注册表键：LLM 字典输出中 "command" 字段的取值即此名。
        return "cancel_flow"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CancelFlowCommand:
        """从字典创建命令。

        业务流程：读取 data["flow"]（可缺省，缺省得 None）构造实例。

        副作用：无（纯构造）。

        参数：
            data: LLM 输出的命令字典；"flow" 键可缺失。

        返回：
            CancelFlowCommand 实例（flow 可能为 None）。
        """
        # flow 为可选字段：dict.get 未命中时得 None（表示取消当前活动 Flow）。
        return CancelFlowCommand(flow=data.get("flow"))

    def run(
        self,
        tracker: DialogueStateTracker,
        flows: Any | None = None,
    ) -> list[dict[str, Any]]:
        """执行取消Flow命令。

        业务流程：1. 以 self.flow 为准、tracker.active_flow 兜底
                  确定待取消 Flow；2. 有 Flow 可取消则返回
                  cancel_flow_requested 事件标记（不直接取消）；
                  3. 无 Flow 可取消返回空列表。

        副作用：无（不修改 tracker；实际取消由 ActionCancelFlow 执行）。

        参数：
            tracker: 对话状态追踪器，必需（用于读取当前活动 Flow）。
            flows: 可用的 Flow 列表，本命令不使用，允许 None。

        返回：
            单元素事件列表（event=cancel_flow_requested，含
            flow_id 与时间戳占位）；无 Flow 可取消时返回空列表。
        """
        # 兜底链：优先取消命令显式指定的 Flow；未指定时取消当前活动 Flow。
        flow_id = self.flow or tracker.active_flow
        # 存在可取消的 Flow：返回请求事件，由 ActionCancelFlow 执行实际取消。
        if flow_id:
            # 不直接取消，让 ActionCancelFlow 来执行
            return [
                {
                    # 事件类型标记：声明"请求取消"意图。
                    "event": "cancel_flow_requested",
                    # 事件负载：待取消的 Flow ID。
                    "flow_id": flow_id,
                    # 事件时间戳占位：由 tracker 后续补齐。
                    "timestamp": None,
                }
            ]
        # 无活动 Flow 也未指定 Flow：无可取消对象，返回空事件列表。
        return []

    def to_dsl(self) -> str:
        """转换为DSL字符串。

        业务流程：flow 非空输出 "cancel flow <flow_id>"；为 None
                  输出 "cancel flow"（两种写法均被 regex_pattern
                  覆盖，可往返解析）。

        副作用：无。

        返回：
            DSL 格式的命令字符串。
        """
        # 带 Flow ID 的完整写法。
        if self.flow:
            return f"cancel flow {self.flow}"
        # 不带 Flow ID 的省略写法：表示取消当前活动 Flow。
        return "cancel flow"

    @classmethod
    def regex_pattern(cls) -> str:
        """返回用于解析此命令的正则表达式模式。

        业务流程：返回支持两种格式的模式串——
                  1. "cancel flow [flow_name]"（Flow 名可选，入捕获组 1）；
                  2. "CancelFlow([flow_name])"（Flow 名可选，入捕获组 2）。

        副作用：无。

        返回：
            正则表达式字符串（两个捕获组均可选，组内允许空匹配）。
        """
        # 支持多种格式（Flow 名均可省略）:
        # - cancel flow <flow_name>
        # - CancelFlow(<flow_name>)
        # 组 2 使用 * 量词允许空匹配：CancelFlow() 也算合法（取消当前活动 Flow）。
        return r"""(?:cancel\s+flow(?:\s+['"`]?([a-zA-Z0-9_-]+)['"`]?)?|CancelFlow\((?:['"]?([a-zA-Z0-9_-]*)['"]?)?\))"""

    @classmethod
    def _from_regex_match(cls, match: re.Match[str]) -> CancelFlowCommand | None:
        """从正则匹配结果创建命令。

        业务流程：1. 取第一个捕获组（cancel flow 格式的 Flow 名），
                  为空（含空串）则取第二个捕获组（CancelFlow()
                  格式）；2. 命中 Flow 名则 strip 去空白后构造；
                  3. 两组均空构造 flow=None 实例（取消当前活动 Flow）。

        副作用：无（纯构造）。

        参数：
            match: from_dsl 中 re.match 的匹配结果。

        返回：
            CancelFlowCommand 实例（flow 可能为 None，本实现不返回 None）。
        """
        # 第一个分组是 cancel flow 格式的 Flow 名，第二个是 CancelFlow() 格式：优先取前者。
        flow = match.group(1) or match.group(2)
        # 命中 Flow 名：strip 去空白后构造；否则构造 flow=None 实例。
        return CancelFlowCommand(flow=flow.strip() if flow else None)


@register_command
@dataclass
class ChangeFlowCommand(Command):
    """切换Flow命令。

    职责：封装"从当前 Flow 切换到另一个 Flow"的意图；run() 只
          返回 change_flow_requested 事件标记（携带新旧 Flow
          ID），实际的切换操作由 ActionChangeFlow 执行。
    设计模式：命令模式 + 注册表工厂；动作触发模式——与
              CancelFlowCommand 同属"动作触发命令"，与
              StartFlowCommand 的"即时数据命令"形成对照，确保
              "命令只声明意图、动作执行实际操作"的原则。
    生命周期：同其余两个命令——LLM 产出 → from_dict 或 from_dsl
              构造实例 → CommandProcessor 执行 run() 后随单轮
              消息处理流程销毁。

    字段说明：
        flow: 要切换到的Flow ID。
    """

    # 要切换到的目标 Flow ID（必填；切换后 tracker 的活动 Flow 将变更为它）。
    flow: str

    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。

        返回：
            "change_flow"：本命令在注册表中的唯一键。
        """
        # 注册表键：LLM 字典输出中 "command" 字段的取值即此名。
        return "change_flow"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChangeFlowCommand:
        """从字典创建命令。

        业务流程：1. 读取 data["flow"] 构造实例；2. 字段缺失时
                  捕获 KeyError 并转抛 ValueError（携带缺失字段
                  说明，保留原始异常链）。

        副作用：无（纯构造）。

        参数：
            data: LLM 输出的命令字典，须含 "flow" 键（Flow ID 字符串）。

        返回：
            ChangeFlowCommand 实例；缺少 "flow" 字段时抛 ValueError。
        """
        # 读取必填字段 flow 构造实例；缺失时走异常分支。
        try:
            return ChangeFlowCommand(flow=data["flow"])
        # KeyError 说明 LLM 输出缺少 flow 字段：转抛 ValueError 快速失败。
        except KeyError as e:
            raise ValueError(
                "Missing required field 'flow' for ChangeFlowCommand"
            ) from e

    def run(
        self,
        tracker: DialogueStateTracker,
        flows: Any | None = None,
    ) -> list[dict[str, Any]]:
        """执行切换Flow命令。

        业务流程：1. 读取 tracker.active_flow 作为被切换的原
                  Flow；2. 返回 change_flow_requested 事件（携带
                  新旧 Flow ID），不直接修改 tracker。

        副作用：无（不修改 tracker；实际切换由 ActionChangeFlow 执行）。

        参数：
            tracker: 对话状态追踪器，必需（用于读取当前活动 Flow）。
            flows: 可用的 Flow 列表，本命令不使用，允许 None。

        返回：
            单元素事件列表（event=change_flow_requested，含
            old_flow_id、new_flow_id 与时间戳占位）。
        """
        # 记录当前活动 Flow 作为切换起点（可能为 None：当前无活动 Flow）。
        old_flow = tracker.active_flow
        # 返回切换请求事件：由 ActionChangeFlow 执行实际切换。
        return [
            {
                # 事件类型标记：声明"请求切换"意图。
                "event": "change_flow_requested",
                # 事件负载：被切换的原 Flow ID（可能为 None）。
                "old_flow_id": old_flow,
                # 事件负载：切换目标 Flow ID。
                "new_flow_id": self.flow,
                # 事件时间戳占位：由 tracker 后续补齐。
                "timestamp": None,
            }
        ]

    def to_dsl(self) -> str:
        """转换为DSL字符串。

        业务流程：输出 "change flow <flow_id>" 格式的 DSL 文本
                  （与 regex_pattern 的格式对应，可往返解析）。

        副作用：无。

        返回：
            DSL 格式的命令字符串。
        """
        # f-string 拼接：命令名 + 目标 Flow ID，空格分隔。
        return f"change flow {self.flow}"

    @classmethod
    def regex_pattern(cls) -> str:
        """返回用于解析此命令的正则表达式模式。

        业务流程：返回单一格式 "change flow <flow_name>" 的模式串
                  （Flow 名必填，入捕获组 1）。

        副作用：无。

        返回：
            正则表达式字符串（含单个捕获组）。
        """
        # 仅支持 "change flow <flow_name>" 一种格式；Flow 名必填，无省略写法。
        return r"""change\s+flow\s+['"`]?([a-zA-Z0-9_-]+)['"`]?"""

    @classmethod
    def _from_regex_match(cls, match: re.Match[str]) -> ChangeFlowCommand | None:
        """从正则匹配结果创建命令。

        业务流程：1. 取第一个捕获组（目标 Flow 名）；2. 命中则
                  strip 去空白后构造实例；3. 捕获组为空返回 None。

        副作用：无（纯构造）。

        参数：
            match: from_dsl 中 re.match 的匹配结果。

        返回：
            ChangeFlowCommand 实例；捕获组为空时返回 None。
        """
        # 取唯一捕获组：目标 Flow 名。
        flow = match.group(1)
        # 命中 Flow 名：去除首尾空白后构造实例。
        if flow:
            return ChangeFlowCommand(flow=flow.strip())
        # 捕获组为空：无法确定目标 Flow，返回 None 由调用方降级处理。
        return None


# 导出
__all__ = [
    "StartFlowCommand",
    "CancelFlowCommand",
    "ChangeFlowCommand",
]
