"""
prompt_builder - 命令生成提示词构建器模块

核心功能：将对话状态（Tracker）、业务域（Domain）与可用流程（Flows）
          组装为 Jinja2 模板上下文，渲染 command_prompt.jinja2 主模板，
          产出命令生成阶段发送给 LLM 的完整提示词（单条 user 消息）。

执行流程：
    1. 装配期：PromptBuilder 实例化时 __post_init__ 创建 Jinja2
       Environment（trim_blocks + lstrip_blocks 修剪块标签空白）并
       预加载主模板（默认 command_prompt.jinja2）；
    2. 每轮消息：LLMCommandGenerator 调用 build_messages →
       build_prompt → _build_template_context 从 tracker/domain/
       flows 收集 8 个模板变量（用户消息、可用 Flows、槽位元信息、
       当前活动 Flow、正在收集的槽位、Flow 槽位已填状态、对话历史）；
    3. 渲染：模板实例 render(**context) 产出提示词文本，交由上游
       生成器解析为 Command 列表。

副作用：无——仅内存中组装与渲染，不读写文件、不调用网络。

主要依赖：
    - jinja2：Environment/FileSystemLoader/select_autoescape 模板引擎；
    - pathlib.Path：模板目录定位（本文件同级的 templates/ 目录）；
    - dataclasses：@dataclass 生成 PromptBuilder 的数据类方法；
    - typing：Any + TYPE_CHECKING（注解已按 PEP 585/604 新式写法）；
    - app.ics_ai.core.tracker / core.domain / dialogue_understanding.flow：
      上下文数据来源（仅注解用）。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，TYPE_CHECKING 分支
# 中的 tracker/domain/flows 前向引用无需引号即可书写。
from __future__ import annotations

# 导入 dataclass 装饰器：PromptBuilder 靠它自动生成 __init__/__repr__。
from dataclasses import dataclass

# 导入 Path：定位模板目录（本文件同级的 templates/）。
from pathlib import Path

# 导入类型注解工具：Any 类型通配符；TYPE_CHECKING 隔离仅注解导入
# （PEP 585/604 新式注解下 Dict/List/Optional 已不需要）。
from typing import TYPE_CHECKING, Any

# 导入 Jinja2 模板引擎三件套：Environment 管理渲染配置，FileSystemLoader
# 从目录加载模板，select_autoescape 按扩展名决定是否自动转义。
from jinja2 import Environment, FileSystemLoader, select_autoescape

# 类型检查期专属导入：这些类型只出现在方法注解中，运行时不导入
# （避免与 core/dialogue_understanding 包形成循环依赖）。
if TYPE_CHECKING:
    # 业务域定义：提供槽位元信息（类型/描述/允许值）。
    from app.ics_ai.core.domain import Domain

    # 对话状态追踪器：上下文组装的数据源头。
    from app.ics_ai.core.tracker import DialogueStateTracker

    # 可用 Flow 列表：提供 Flow 元数据与待收集槽位。
    from app.ics_ai.dialogue_understanding.flow import FlowsList

# 模板目录：PromptBuilder 同级的 templates/ 目录（存放 command_prompt.jinja2
# 等提示词模板）。
TEMPLATES_DIR = Path(__file__).parent / "templates"


@dataclass
class PromptBuilder:
    """Prompt构建器。

    职责：把 Tracker 的运行时对话状态与 Domain/Flows 的静态业务定义
          组装成模板上下文，用 Jinja2 渲染出命令生成阶段的完整提示词。
    设计模式：构建器（Builder）+ 模板方法——对外仅暴露 build_prompt/
              build_messages 两个构建入口，上下文组装拆分为多个
              _prepare_*/_get_* 私有钩子，各司其职。
    生命周期：随生成器（LLMCommandGenerator）创建并长期复用；
              Jinja2 模板在 __post_init__ 中一次性加载缓存。

    Attributes:
        template_name: 主模板文件名（默认 command_prompt.jinja2）
        max_history_turns: 包含的最大历史轮数
        include_slots: 是否包含槽位信息
        include_flows: 是否包含Flow信息
    """

    # 主模板文件名：默认命令生成模板；可替换为 TEMPLATES_DIR 下其他 .jinja2。
    template_name: str = "command_prompt.jinja2"
    # 对话历史最大轮数：传入 tracker.get_messages_for_llm 控制上下文长度。
    max_history_turns: int = 5
    # 是否向模板注入槽位元信息（slot_info）；关闭可缩短提示词。
    include_slots: bool = True
    # 是否向模板注入可用 Flow 清单（flows）；关闭可缩短提示词。
    include_flows: bool = True

    def __post_init__(self):
        """初始化Jinja2环境。

        业务流程：1. 创建 Environment（文件系统加载器指向 TEMPLATES_DIR，
                  按扩展名自动转义、开启块标签空白修剪）；2. 预加载
                  主模板并缓存——后续每次渲染零查找开销。

        副作用：创建 self._env / self._template 两个实例属性（dataclass
                未声明的内部字段，仅本类内部使用）。

        异常：
            TemplateNotFound：template_name 在模板目录中不存在时抛出
            （实例化期即失败，快速暴露配置错误）。
        """
        # 创建 Jinja2 环境：
        # - FileSystemLoader：从模板目录按文件名加载模板；
        # - select_autoescape(['html', 'xml'])：仅 HTML/XML 模板自动转义，
        #   .jinja2 纯文本模板不转义（输出是提示词文本，无需 HTML 转义）；
        # - trim_blocks=True：移除块标签后的首个换行；
        # - lstrip_blocks=True：移除行首到块标签间的空白；
        #   两者叠加使模板中独立成行的标签/注释不留空白痕迹。
        self._env = Environment(
            loader=FileSystemLoader(str(TEMPLATES_DIR)),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        # 预加载并缓存主模板：模板缺失时在实例化期立即报错。
        self._template = self._env.get_template(self.template_name)

    def build_prompt(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        flows: FlowsList | None = None,
    ) -> str:
        """构建完整的提示词。

        业务流程：1. 调用 _build_template_context 组装模板上下文；
                  2. 用预加载的模板实例渲染，返回提示词文本。

        副作用：无（渲染为纯内存操作）。

        参数：
            tracker: 对话状态追踪器（用户消息/活动 Flow/槽位/历史的来源）。
            domain: 业务域定义，允许 None（缺省时槽位元信息为空）。
            flows: 可用的 Flow 列表，允许 None（缺省时 Flow 清单为空）。

        返回：
            渲染后的完整提示词字符串。
        """
        # 组装模板上下文：从 tracker/domain/flows 收集全部模板变量。
        context = self._build_template_context(tracker, domain, flows)
        # 渲染模板：上下文以关键字参数展开注入。
        return self._template.render(**context)

    def build_messages(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        flows: FlowsList | None = None,
    ) -> list[dict[str, str]]:
        """构建消息列表格式的提示词。

        业务流程：1. 复用 build_prompt 得到完整提示词文本；2. 包装为
                  单条 user 消息——提示词本身已含系统指令与上下文，
                  无需拆分为 system/user 多条消息。

        副作用：无。

        参数：
            tracker: 对话状态追踪器。
            domain: 业务域定义，允许 None。
            flows: 可用的 Flow 列表，允许 None。

        返回：
            消息列表，形如 [{"role": "user", "content": "..."}]，
            可直接传给 LLM API 的 messages 参数。
        """
        # 复用 build_prompt：保证两种构建入口产出一致的提示词。
        prompt = self.build_prompt(tracker, domain, flows)

        # 使用单个用户消息包含完整提示词
        # 这样设计是因为提示词已经包含了系统指令和上下文
        return [{"role": "user", "content": prompt}]

    def _build_template_context(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        flows: FlowsList | None = None,
    ) -> dict[str, Any]:
        """构建模板渲染上下文。

        业务流程：1. 提取用户最新消息；2. 按开关注入 Flows 清单与
                  槽位元信息；3. 填充当前活动 Flow/正在收集的槽位/
                  Flow 槽位已填状态；4. 提取对话历史。

        副作用：无（只读 tracker/domain/flows，不修改任何状态）。

        参数：
            tracker: 对话状态追踪器。
            domain: 业务域定义，允许 None。
            flows: 可用的 Flow 列表，允许 None。

        返回：
            与 command_prompt.jinja2 变量一一对应的上下文字典
            （user_message/flows/slot_info/current_flow/current_slot/
            current_slot_description/flow_slots/conversation_history）。
        """
        # 初始化空上下文：后续按区块逐项填充。
        context: dict[str, Any] = {}

        # 用户消息：取最新一条的文本；无消息时以空串占位（模板分支兼容）。
        if tracker.latest_message:
            context["user_message"] = tracker.latest_message.text
        else:
            context["user_message"] = ""

        # Flows信息：开关开启且有 Flow 列表时，准备模板用的 Flow 数据；
        # 否则以空列表占位（模板走"当前没有定义 Flows"分支）。
        if self.include_flows and flows:
            context["flows"] = self._prepare_flows_data(flows, domain)
        else:
            context["flows"] = []

        # 槽位信息：开关开启时从 Domain 提取槽位元信息；否则空字典。
        if self.include_slots:
            context["slot_info"] = self._prepare_slot_info(domain)
        else:
            context["slot_info"] = {}

        # 当前Flow状态：活动 Flow ID、正在收集的槽位及其描述、各槽位已填值。
        context["current_flow"] = tracker.active_flow
        context["current_slot"] = self._get_current_slot(tracker)
        context["current_slot_description"] = self._get_current_slot_description(
            tracker, domain
        )
        context["flow_slots"] = self._get_flow_slots(tracker, domain, flows)

        # 对话历史：角色汉化后的最近若干轮消息列表。
        context["conversation_history"] = self._get_conversation_history(tracker)

        return context

    def _prepare_flows_data(
        self,
        flows: FlowsList,
        domain: Domain | None = None,
    ) -> list[dict[str, Any]]:
        """准备Flow数据用于模板渲染。

        业务流程：遍历 Flow 列表，逐个提取 id、描述（三级兜底：
                  description → name → id）与待收集槽位名列表。

        副作用：无。

        参数：
            flows: 可用的 Flow 列表（须可迭代，元素含 id/description 等）。
            domain: 业务域定义，允许 None——目前未参与组装，保留为
                    接口占位以便未来按 Domain 补充 Flow 元数据。

        返回：
            Flow 数据列表，每项含 id/description/slots_to_collect 三键，
            与模板第 12 行手工 JSON 的字段约定对应。
        """
        # 初始化结果列表：逐个 Flow 收集展示数据。
        flows_data = []
        # 遍历可用 Flow：提取模板所需的最小字段集。
        for flow in flows:
            flow_data = {
                # Flow 唯一标识：模板中作为 "name" 输出，也是 start flow 的目标。
                "id": flow.id,
                # 描述三级兜底：无描述时退化为名称，再退化为 ID，保证非空可读。
                "description": flow.description or flow.name or flow.id,
                # 待收集槽位名列表：模板据此展示每个 Flow 需要收集的字段。
                "slots_to_collect": flow.get_slots_to_collect(),
            }
            flows_data.append(flow_data)
        return flows_data

    def _prepare_slot_info(
        self,
        domain: Domain | None = None,
    ) -> dict[str, dict[str, Any]]:
        """准备槽位信息用于模板渲染。

        业务流程：遍历 Domain 的全部槽位定义，提取类型与描述；
                  分类槽位额外提取允许值列表。

        副作用：无。

        参数：
            domain: 业务域定义，允许 None（None 或无槽位时返回空字典）。

        返回：
            槽位名 → 元信息字典（type/description，可选 allowed_values）
            的映射，供模板按槽位名反查展示。
        """
        # 初始化结果字典：槽位名 → 元信息。
        slot_info = {}
        # Domain 缺省或未定义槽位：直接返回空字典。
        if domain and domain.slots:
            # 遍历全部槽位定义：提取模板所需元信息。
            for name, slot in domain.slots.items():
                info = {
                    # 槽位类型：getattr 兜底 "text"（兼容无 slot_type 属性的对象）。
                    "type": getattr(slot, "slot_type", "text"),
                    # 槽位描述：getattr 兜底空串。
                    "description": getattr(slot, "description", ""),
                }
                # 获取允许的值（如果是分类槽位）
                allowed_values = getattr(slot, "allowed_values", None)
                # 仅有允许值的槽位才写入：非分类槽位不占模板字段。
                if allowed_values:
                    info["allowed_values"] = allowed_values
                # 以槽位名为键登记：模板经 slot_info[slot] 反查。
                slot_info[name] = info
        return slot_info

    def _get_current_slot(
        self,
        tracker: DialogueStateTracker,
    ) -> str | None:
        """获取当前正在收集的槽位。

        业务流程：取对话栈顶帧，若为 FlowStackFrame 则返回其
                  slot_to_collect；其余帧类型或空栈返回 None。

        副作用：无。

        参数：
            tracker: 对话状态追踪器。

        返回：
            当前正在收集的槽位名；栈顶非 Flow 帧或无待收集槽位时返回 None。
        """
        # 从对话栈获取当前需要收集的槽位
        top_frame = tracker.dialogue_stack.top()
        # 栈非空才进一步判断帧类型。
        if top_frame:
            # 局部导入 FlowStackFrame：延迟到方法内，保持模块加载期轻量。
            from app.ics_ai.dialogue_understanding.stack.stack_frame import (
                FlowStackFrame,
            )

            # 仅 Flow 帧携带槽位收集状态；getattr 兜底兼容无该属性的帧。
            if isinstance(top_frame, FlowStackFrame):
                return getattr(top_frame, "slot_to_collect", None)
        # 空栈或栈顶非 Flow 帧：当前不处于槽位收集状态。
        return None

    def _get_current_slot_description(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
    ) -> str | None:
        """获取当前槽位的描述。

        业务流程：1. 复用 _get_current_slot 拿到当前槽位名；2. 从
                  Domain 槽位定义中查描述文本。

        副作用：无。

        参数：
            tracker: 对话状态追踪器。
            domain: 业务域定义，允许 None（缺省或无该槽位时得 None）。

        返回：
            当前槽位的描述文本；不在收集槽位状态或 Domain 缺失时返回 None。
        """
        # 复用槽位查询：与 current_slot 保持同一事实来源。
        current_slot = self._get_current_slot(tracker)
        # 有槽位名且 Domain 可用：按名查槽位定义取描述。
        if current_slot and domain and domain.slots:
            slot = domain.slots.get(current_slot)
            # 槽位定义存在：getattr 兜底 None（兼容无 description 属性）。
            if slot:
                return getattr(slot, "description", None)
        # 不在收集状态或查不到定义：无描述可提供。
        return None

    def _get_flow_slots(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        flows: FlowsList | None = None,
    ) -> list[dict[str, Any]]:
        """获取当前Flow的槽位状态。

        业务流程：1. 无活动 Flow 或无 Flow 列表时返回空；2. 两步兼容
                  查找当前 Flow（get_flow 方法优先，普通列表按 id 扫描
                  兜底）；3. 遍历待收集槽位，从 tracker 取已填值并按
                  Domain 补充类型/描述元信息。

        副作用：无。

        参数：
            tracker: 对话状态追踪器（槽位已填值的来源）。
            domain: 业务域定义，允许 None（缺省时仅输出名称与值）。
            flows: 可用的 Flow 列表，允许 None。

        返回：
            槽位状态列表，每项含 name/value（可选 type/description），
            供模板展示"已填/未填"进度；无活动 Flow 时为空列表。
        """
        # 初始化结果列表。
        flow_slots = []
        # 读取当前活动 Flow ID：无活动 Flow 时无从谈起槽位状态。
        active_flow = tracker.active_flow

        # 无活动 Flow 或未提供 Flow 列表：返回空列表（模板跳过该 JSON 字段）。
        if not active_flow or not flows:
            return flow_slots

        # 获取当前Flow（兼容FlowsList和普通列表）
        flow = None
        # 第一步：对象带 get_flow 方法时直接按名查找（FlowsList 正常路径）。
        if hasattr(flows, "get_flow"):
            flow = flows.get_flow(active_flow)
        # 第二步兜底：普通列表按 id 逐个比对。
        elif isinstance(flows, list):
            for f in flows:
                # 命中活动 Flow：记录并提前结束扫描。
                if f.id == active_flow:
                    flow = f
                    break

        # 仍未找到 Flow（如活动 Flow 不在列表中）：返回空列表。
        if not flow:
            return flow_slots

        # 获取Flow需要收集的槽位：待收集槽位名列表。
        slots_to_collect = flow.get_slots_to_collect()

        # 遍历待收集槽位：组装"名称 + 已填值 + 元信息"。
        for slot_name in slots_to_collect:
            slot_data = {
                # 槽位名。
                "name": slot_name,
                # 已填值：从 tracker 实时读取；未填为 None（模板显示 null/None）。
                "value": tracker.get_slot(slot_name),
            }

            # 从domain获取槽位元信息：Domain 可用时补充类型与描述。
            if domain and domain.slots:
                slot_def = domain.slots.get(slot_name)
                # 槽位定义存在：getattr 兜底默认类型与空描述。
                if slot_def:
                    slot_data["type"] = getattr(slot_def, "slot_type", "text")
                    slot_data["description"] = getattr(slot_def, "description", "")

            flow_slots.append(slot_data)

        return flow_slots

    def _get_conversation_history(
        self,
        tracker: DialogueStateTracker,
    ) -> list[dict[str, str]]:
        """获取对话历史。

        业务流程：1. 从 tracker 取最近 max_history_turns 轮消息；
                  2. 将角色映射为中文（user→用户，assistant→助手），
                  其余角色（如 system）丢弃不进入提示词。

        副作用：无。

        参数：
            tracker: 对话状态追踪器。

        返回：
            格式化的历史消息列表，每项含 role（用户/助手）与 content 两键，
            与模板 "{{ msg.role }}: {{ msg.content }}" 的渲染约定对应。
        """
        # 取最近若干轮历史：轮数由 max_history_turns 控制。
        history = tracker.get_messages_for_llm(max_turns=self.max_history_turns)

        # 初始化格式化结果列表。
        formatted_history = []
        # 逐条映射角色：只保留用户/助手两类（模板叙述视角为中文对话）。
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # 用户消息 → 角色"用户"。
            if role == "user":
                formatted_history.append({"role": "用户", "content": content})
            # 助手消息 → 角色"助手"；其余角色（system 等）静默丢弃。
            elif role == "assistant":
                formatted_history.append({"role": "助手", "content": content})

        return formatted_history


# 导出
__all__ = [
    # 提示词构建器。
    "PromptBuilder",
    # 模板目录常量。
    "TEMPLATES_DIR",
]
