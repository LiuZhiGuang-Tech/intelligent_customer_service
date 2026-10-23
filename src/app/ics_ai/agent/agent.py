"""
agent - 对话系统主类模块

核心功能：定义对话系统的装配与驱动入口——AgentConfig（装配参数
          DTO）、_load_custom_actions（自定义动作自动发现注册）、
          Agent 主类（组件装配 + LangGraph 图驱动消息处理 + 项目
          加载工厂 load/create）。

执行流程：
    1. 装配期：Agent.load(project_path) 从项目目录或模型压缩包
       加载 domain/flows/endpoints/config 四类配置，构造命令
       生成器、策略集成器、TrackerStore 等组件后经 __init__
       组装；Agent() 直构则接收现成组件（默认补齐空实现）；
    2. 运行期：每条用户消息经 handle_message——取/建对话
       tracker → create_initial_state 打包组件引用与输入 →
       graph.ainvoke 驱动 understand → policy → action → guard
       → response 图循环 → 从最终状态提取响应并回存 tracker；
    3. 扩展期：actions/ 目录下的 Action 子类经 _load_custom_actions
       动态导入并注册进全局动作注册表。

副作用：
    - __init__：创建 TrackerStore 并绑定 Domain；经
      get_message_processing_graph 触发图单例构建（首次调用时
      写模块级缓存 _graph_instance）；
    - _load_custom_actions：修改 sys.path（插入 actions 父目录）、
      写 sys.modules（动态导入的模块缓存）、向 actions 全局注册表
      注册动作实例；
    - load：写 sys.path（项目/工作目录）、解压模型压缩包到系统
      临时目录（tempfile.mkdtemp）；
    - handle_message：经 tracker_store 读写对话状态（存储介质由
      store 类型决定，默认内存）。

主要依赖：
    - 标准库：importlib/inspect/sys（动态导入与反射）、
      dataclasses/pathlib/typing/logging；
    - app.ics_ai.agent.graph：图单例与初始状态工厂；
    - app.ics_ai.agent.message_processor：消息处理器（备用通道）
      与 MessageResponse 响应 DTO；
    - app.ics_ai.core：Domain/DialogueStateTracker/TrackerStore；
    - app.ics_ai.dialogue_understanding：FlowLoader/FlowsList/
      LLMCommandGenerator/CommandProcessor；
    - app.ics_ai.policies：PolicyEnsemble/FlowPolicy/
      EnterpriseSearchPolicy；
    - app.ics_ai.shared：read_yaml_file/LLMConfig。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，PEP 604 联合
# 类型注解（X | None）不受运行时版本限制。
from __future__ import annotations

# 导入动态导入工具：importlib.util 按文件路径构造模块 spec，
# importlib 提供 module_from_spec 配套工厂。
import importlib

# 导入动态导入子模块：spec_from_file_location 所在命名空间。
import importlib.util

# 导入反射工具：inspect.getmembers/inspect.isclass 扫描模块中的类。
import inspect

# 导入日志器工厂：模块级与装配过程日志。
import logging

# 导入运行时环境修改：sys.path 注入用户自定义模块搜索路径。
import sys

# 导入 dataclass 工具：AgentConfig 配置类与 field 默认值工厂。
from dataclasses import dataclass, field

# 导入路径类型：项目目录/模型压缩包的路径处理。
from pathlib import Path

# 导入类型注解工具：Any 类型通配符（PEP 585/604 新式注解下
# Dict/List/Optional/Union 已不需要，统一用内建泛型与 | None）。
from typing import Any

# 从动作子模块导入：Action 抽象基类（自定义动作的继承判据）与
# register_action 注册函数（写入模块级自定义动作注册表）。
from app.ics_ai.agent.actions import (
    Action,
    register_action,
)

# 从图包入口导入：图单例惰性获取函数与每轮消息的初始状态工厂。
from app.ics_ai.agent.graph import (
    create_initial_state,
    get_message_processing_graph,
)

# 从消息处理器子模块导入：MessageProcessor（图通道的备用实现，
# 向后兼容保留）、ProcessorConfig（处理器配置）、MessageResponse
# （统一响应 DTO：messages/events/metadata 三字段）。
from app.ics_ai.agent.message_processor import (
    MessageProcessor,
    MessageResponse,
)

# 从核心域子模块导入：Domain 业务域定义（槽位/响应/动作的静态来源）。
from app.ics_ai.core.domain import Domain

# 从存储子模块导入：TrackerStore 存储基类与工厂函数（按 type 字符串
# 构造 memory/文件等实现）。
from app.ics_ai.core.stores import (
    TrackerStore,
    create_tracker_store,
)

# 从追踪器子模块导入：DialogueStateTracker 单用户对话状态载体。
from app.ics_ai.core.tracker import DialogueStateTracker

# 从流程子模块导入：FlowsList 流程集合与 FlowLoader YAML 加载器。
from app.ics_ai.dialogue_understanding.flow import (
    FlowLoader,
    FlowsList,
)

# 从生成器子模块导入：LLMCommandGenerator 命令生成器（LLM 驱动）。
from app.ics_ai.dialogue_understanding.generator import LLMCommandGenerator

# 从命令处理器子模块导入：CommandProcessor 命令执行器（图
# understand 节点经状态引用调用）。
from app.ics_ai.dialogue_understanding.processor import CommandProcessor

# 从策略包入口导入：策略集成器与两类具体策略（流程推进/知识检索）。
from app.ics_ai.policies import (
    EnterpriseSearchPolicy,
    FlowPolicy,
    PolicyEnsemble,
)

# 从共享配置子模块导入：LLMConfig 模型配置 DTO（type/model/api_key
# 等，load 期从 endpoints.yml 解析得到）。
from app.ics_ai.shared.config import LLMConfig

# 从 YAML 加载子模块导入：read_yaml_file 通用 YAML 读取（config.yml
# 解析用）。
from app.ics_ai.shared.yaml_loader import read_yaml_file

# 模块级日志器：以模块路径命名，运行日志带 [Agent] 等前缀。
logger = logging.getLogger(__name__)


def _load_custom_actions(actions_path: Path) -> list[str]:
    """从 actions 目录自动发现并注册自定义动作。

    核心功能：扫描目录下全部 .py 文件，动态导入并反射出继承自
              Action 基类的类，实例化后注册进全局动作注册表——
              用户无需手工 import 即可扩展动作。

    业务流程：
        1. 目录不存在或非目录 → 返回空列表（静默跳过，不报错）；
        2. 把 actions 目录的父目录插入 sys.path 首位——动态模块
           以 "actions.<文件名>" 为模块名导入，其内部 import 语句
           需要父目录在搜索路径上才能解析；
        3. 逐个扫描 .py 文件（下划线开头跳过，排除 __init__.py）：
           按路径构造 spec → 建模块对象 → 登记 sys.modules →
           exec_module 执行模块体；
        4. 反射扫描模块命名空间中的类：是 Action 子类、非 Action
           本身、且定义在本模块内（排除误导入的外来类）三个条件
           同时满足才实例化注册；
        5. 单文件/单类失败均仅记 warning 降级继续，不中断整体加载。

    副作用：修改 sys.path（插入父目录，不回滚）；写 sys.modules
            （动态模块缓存）；经 register_action 写全局动作注册表
            （模块级可变状态，跨会话生效）；动态导入会执行用户
            文件的模块体代码。

    参数：
        actions_path: actions 目录路径（通常为工程根下 actions/），
                      Path 类型。

    返回：
        成功注册的动作名列表（list[str]，名称取自 action.name）；
        目录不存在或全部失败时返回空列表（不抛异常）。
    """
    # 防御性入口：路径不存在或不是目录时静默返回（工程无自定义
    # 动作是常态，不应报错）。
    if not actions_path.exists() or not actions_path.is_dir():
        return []

    # 收集成功注册的动作名（作为返回值与日志统计依据）。
    registered_actions: list[str] = []

    # 将 actions 目录的父目录添加到 sys.path，以便正确导入——动态
    # 模块名形如 "actions.xxx"，其相对导入依赖父目录可被搜索到。
    parent_path = str(actions_path.parent)
    # 去重：已在搜索路径上时不重复插入（避免路径列表膨胀）。
    if parent_path not in sys.path:
        # 插入首位：保证优先于同名包被发现。
        sys.path.insert(0, parent_path)

    try:
        # 扫描 actions 目录下的所有 .py 文件（仅顶层，不递归子目录）。
        for py_file in actions_path.glob("*.py"):
            # 跳过下划线开头的文件——__init__.py 等私有/包文件不是
            # 独立动作模块。
            if py_file.name.startswith("_"):
                # 直接进入下一个文件。
                continue

            # 构造导入用模块名：固定 "actions." 前缀 + 去扩展名的
            # 文件名（用户模块内部可用此名 import 自身兄弟模块）。
            module_name = f"actions.{py_file.stem}"

            try:
                # 按文件路径构造模块加载规格（不依赖包结构）。
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                # spec 或 loader 缺失（非法文件等）时跳过该文件。
                if spec is None or spec.loader is None:
                    # 进入下一个文件。
                    continue

                # 由 spec 创建空模块对象。
                module = importlib.util.module_from_spec(spec)
                # 预登记进 sys.modules：保证模块体内 isinstance 等
                # 反射可见、且重复导入时命中缓存。
                sys.modules[module_name] = module
                # 执行模块体：定义类/函数，触发用户自定义代码。
                spec.loader.exec_module(module)

                # 反射扫描模块命名空间中的全部类属性。
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # 三重判据：是 Action 子类（issubclass 要求 obj
                    # 为类，isclass 已保证）、非 Action 基类本身、
                    # 且 __module__ 是本模块（排除经由 import 进入
                    # 本模块命名空间的外来类）。
                    if (
                        issubclass(obj, Action)
                        and obj is not Action
                        and obj.__module__ == module_name
                    ):
                        try:
                            # 无参实例化（自定义动作需提供无参构造）
                            # 并注册进全局注册表。
                            action_instance = obj()
                            # 注册动作（写模块级 _CUSTOM_ACTIONS）。
                            register_action(action_instance)
                            # info 留痕：注册成功的动作名。
                            logger.info(
                                f"Registered custom action: {action_instance.name}"
                            )
                            # 记入返回列表（action.name 为抽象
                            # property，子类必须实现）。
                            registered_actions.append(action_instance.name)
                        except Exception as e:
                            # 单个动作注册失败（如构造需要参数）：
                            # warning 降级，不影响其余动作。
                            logger.warning(f"Failed to register action {name}: {e}")

            except Exception as e:
                # 单个文件加载失败（语法错误/模块体异常等）：warning
                # 降级，继续扫描下一个文件。
                logger.warning(f"Failed to load actions from {py_file}: {e}")

    finally:
        # 清理 sys.path（可选，保留以便后续使用）——当前为空实现，
        # 预留统一清理位（父目录保留在搜索路径上无副作用）。
        pass

    # 返回成功注册的动作名清单。
    return registered_actions


@dataclass
class AgentConfig:
    """Agent 装配配置。

    职责：集中承载 Agent 直构时的路径与组件默认参数——四类配置
          文件的相对路径、TrackerStore 类型与透传参数、LLM 备用
          配置；load 工厂以工程目录为基拼接这些相对路径。
    设计模式：数据传输对象（dataclass 参数对象，避免构造函数
              散装传参）；field(default_factory=dict) 保证可变
              默认值不跨实例共享。
    生命周期：load() 未显式传 config 时创建默认实例；随后随
              Agent 实例长期只读持有。

    Attributes:
        domain_path: Domain 文件相对路径（相对工程根，默认
                     "domain.yml"）。
        flows_path: Flows 文件/目录相对路径（默认 "data/flows"）。
        config_path: 框架配置文件相对路径（默认 "config.yml"，
                     提供 pipeline/policies/retrieval 段）。
        endpoints_path: 端点配置相对路径（默认 "endpoints.yml"，
                        提供模型/嵌入/NLG/存储定义）。
        tracker_store_type: Tracker 存储类型（默认 "memory"，经
                            create_tracker_store 工厂实例化）。
        tracker_store_config: TrackerStore 透传参数字典（如文件
                              存储的 path），默认空字典。
        llm_config: LLM 备用配置（LLMConfig | None），未显式提供
                    时为 None——load 期会从 endpoints.yml 重新
                    解析，此字段主要服务直构场景。
    """

    # Domain 文件相对路径：缺省为工程根下的 domain.yml。
    domain_path: str = "domain.yml"
    # Flows 文件/目录相对路径：缺省为 data/flows（FlowLoader 支持
    # 单文件或目录批量加载）。
    flows_path: str = "data/flows"
    # 框架配置文件相对路径：config.yml（pipeline/policies 配置来源）。
    config_path: str = "config.yml"
    # 端点配置相对路径：endpoints.yml（模型/嵌入/NLG/存储定义来源）。
    endpoints_path: str = "endpoints.yml"
    # Tracker 存储类型：memory 为默认（会话级内存存储，进程重启
    # 即失；生产可换文件/Redis 等实现）。
    tracker_store_type: str = "memory"
    # TrackerStore 透传参数：按存储类型变化（如文件存储的 path），
    # 经 **kwargs 展开传给工厂函数；default_factory 保证各实例
    # 独享字典。
    tracker_store_config: dict[str, Any] = field(default_factory=dict)
    # LLM 备用配置：允许 None（load 期从 endpoints.yml 解析；
    # 直构无配置时命令生成器保持 None，understand 节点跳过
    # LLM 命令生成）。
    llm_config: LLMConfig | None = None


class Agent:
    """对话系统主类。

    职责：对话系统的装配与驱动中枢——持有 Domain/Flows/存储/
          策略/生成器/处理器六大组件，经 LangGraph 消息处理图
          驱动每轮消息的 understand → policy → action → guard
          → response 循环，并提供 tracker 管理与动作注册入口。
    设计模式：门面（Facade）——对上暴露 handle_message 单一
              入口，对下协调全部组件；配合工厂方法 load/create。
    生命周期：经 load（配置装配）或直构（组件注入）创建，长期
              存活并服务多用户多轮会话；图引用为进程级单例。

    使用示例：
    ```python
    agent = Agent.load("./my_bot")
    response = await agent.handle_message("你好", sender_id="user1")
    print(response.messages)
    ```
    """

    def __init__(
        self,
        domain: Domain | None = None,
        flows: FlowsList | None = None,
        tracker_store: TrackerStore | None = None,
        policy_ensemble: PolicyEnsemble | None = None,
        command_generator: LLMCommandGenerator | None = None,
        nlg_generator: Any | None = None,
        config: AgentConfig | None = None,
    ):
        """初始化 Agent：装配六大组件与图引用。

        业务流程：
            1. 兜底核心定义：domain/flows 未提供时以空实现兜底
               （Domain()/FlowsList()），config 未提供时用默认值；
            2. TrackerStore：外部注入优先，否则按 config 类型经
               工厂创建，随后绑定 Domain（响应模板等查询依赖）；
            3. 策略集成器：外部注入优先，否则按默认组合创建
               （FlowPolicy 最高优先级 + EnterpriseSearchPolicy
               知识检索）；
            4. 命令生成器/NLG 生成器：纯注入（无默认构造——LLM
               配置缺失时保持 None，运行期对应能力自动跳过）；
            5. 命令处理器：必建（依赖 domain + flows 槽位定义）；
            6. 获取图单例（首次调用触发图构建并写模块级缓存）；
            7. 构建备用消息处理器（图通道异常时的向后兼容保留）。

        副作用：创建 TrackerStore 并绑定 Domain；首次获取图时写
                模块级 _graph_instance 缓存并输出构建日志。

        参数：
            domain: 业务域定义，允许 None（兜底空 Domain）。
            flows: 流程集合，允许 None（兜底空 FlowsList）。
            tracker_store: 对话状态存储，允许 None（按 config 创建）。
            policy_ensemble: 策略集成器，允许 None（按默认组合创建）。
            command_generator: 命令生成器，允许 None（无 LLM 能力）。
            nlg_generator: NLG 生成器（响应重述用），允许 None。
            config: Agent 配置，允许 None（默认 AgentConfig）。
        """
        # 核心定义兜底：未提供时以空实现创建（保证后续装配不因
        # None 中断，运行期表现为"无槽位/无流程"的降级能力）。
        self.domain = domain or Domain()
        self.flows = flows or FlowsList()
        self.config = config or AgentConfig()

        # 初始化 Tracker 存储：外部注入优先——测试/定制场景直接
        # 传入现成实例。
        if tracker_store:
            # 直接持有注入的存储实例。
            self.tracker_store = tracker_store
        else:
            # 按配置类型经工厂创建（memory/file 等），透传扩展参数。
            self.tracker_store = create_tracker_store(
                self.config.tracker_store_type,
                **self.config.tracker_store_config,
            )
        # 绑定业务域：存储侧的响应模板/槽位定义查询依赖此引用。
        self.tracker_store.set_domain(self.domain)

        # 初始化策略：外部注入优先，否则按默认组合创建——FlowPolicy
        # 持有流程集合（最高优先级，推进 YAML Flow），企业搜索策略
        # 负责知识检索兜底。
        if policy_ensemble:
            # 直接持有注入的集成器。
            self.policy_ensemble = policy_ensemble
        else:
            # 默认组合：流程策略 + 企业搜索策略（无检索器配置，
            # 检索能力由策略内部降级链处理）。
            self.policy_ensemble = PolicyEnsemble(
                policies=[
                    FlowPolicy(flows=self.flows),
                    EnterpriseSearchPolicy(),
                ]
            )

        # 初始化命令生成器：纯注入持有（None 表示无 LLM 命令生成
        # 能力，understand 节点经状态引用读取后自动跳过）。
        self.command_generator = command_generator

        # 初始化 NLG 生成器：纯注入持有（None 表示不做响应重述，
        # 直接输出模板响应）。
        self.nlg_generator = nlg_generator

        # 初始化命令处理器：必建组件——understand 节点执行命令的
        # 载体（槽位校验/force_slot_filling 过滤都在其内部）。
        self.command_processor = CommandProcessor(
            domain=self.domain,
            # 传入解包后的流程列表（容器注解跟随运行时事实）；
            # self.flows 已兜底非 None，此条件为防御性冗余。
            flows=self.flows.flows if self.flows else [],
        )

        # 获取 LangGraph 消息处理图（惰性初始化的单例）——首次
        # 调用触发图构建并写模块级缓存，后续零构建开销。
        self.graph = get_message_processing_graph()

        # 保留消息处理器作为备用（向后兼容）：图通道之外的旧实现，
        # 携带同一套组件引用。
        self.message_processor = MessageProcessor(
            domain=self.domain,
            flows=self.flows,
            policy_ensemble=self.policy_ensemble,
            command_generator=self.command_generator,
        )

    async def handle_message(
        self,
        message: str,
        sender_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> MessageResponse:
        """处理一条用户消息（核心入口）。

        业务流程：
            1. 经存储取/建该用户的对话 tracker（多用户隔离）；
            2. 经 create_initial_state 打包本轮输入与三大组件
               引用（生成器/处理器/策略集成），构造 18 键初始
               状态；
            3. graph.ainvoke 驱动图执行：understand（命令生成
               执行）→ policy（策略预测）→ action（动作执行/
               响应累积）→ guard（循环守卫）→ response（收尾），
               条件边控制循环直至收敛/熔断；
            4. 从最终状态提取 tracker/响应列表/节点轨迹/错误；
            5. 回存更新后的 tracker（对话状态跨轮持久）；
            6. 组装 MessageResponse（响应消息 + 轨迹/错误元数据）
               返回。

        副作用：经 tracker_store 读写对话状态；图执行期各节点的
                业务副作用（LLM 调用/检索等）在节点内部发生。

        参数：
            message: 用户消息文本（本轮输入）。
            sender_id: 发送者 ID（tracker 的隔离键，默认 "default"）。
            metadata: 消息元数据（渠道/时间戳等），允许 None
                      （状态内兜底为空字典）。

        返回：
            MessageResponse——messages 为机器人响应消息列表；
            metadata 携带 node_history（执行轨迹）与 error（过程
            错误，无错时为 None）。图执行异常会向上传播（不在本
            方法内吞异常）。
        """
        # 获取或创建 Tracker：存储按 sender_id 隔离多用户状态，
        # 首次出现的用户自动建空 tracker。
        tracker = await self.tracker_store.get_or_create_tracker(sender_id)

        # 构建初始状态：打包输入消息/元数据/流程上限（单轮 10 个
        # 动作）与三大组件引用——图节点经 state 通道读取这些引用。
        initial_state = create_initial_state(
            tracker=tracker,
            input_message=message,
            domain=self.domain,
            flows=self.flows,
            metadata=metadata,
            max_actions=10,
            command_generator=self.command_generator,
            command_processor=self.command_processor,
            policy_ensemble=self.policy_ensemble,
        )

        # 执行图：异步驱动整条节点链至 response → END 终止
        # （消息文本截断 50 字符留痕，避免日志过长）。
        logger.info(f"[Agent] 使用 LangGraph 处理消息: {message[:50]}...")
        final_state = await self.graph.ainvoke(initial_state)

        # 从最终状态提取结果：tracker 必在（初始状态显式赋值），
        # get 的默认值为防御性兜底；响应/轨迹/错误可能为空。
        updated_tracker = final_state.get("tracker", tracker)
        final_responses = final_state.get("final_responses", [])
        node_history = final_state.get("node_history", [])
        error = final_state.get("error")

        # 保存 Tracker：把本轮变更（槽位/对话栈/消息历史）持久回
        # 存储介质，供下一轮读取。
        await self.tracker_store.save(updated_tracker)

        # 构建响应：messages 即机器人回复；metadata 附带执行轨迹
        # 与过程错误（调用方可观测处理过程）。
        response = MessageResponse(
            messages=final_responses,
            metadata={
                "node_history": node_history,
                "error": error,
            },
        )

        # info 留痕：本轮处理的节点路径与响应规模（排查回复来源
        # 的入口日志）。
        logger.info(
            f"[Agent] 处理完成, 节点路径: {' -> '.join(node_history)}, "
            f"响应数: {len(final_responses)}"
        )

        # 返回统一响应 DTO。
        return response

    def handle_message_sync(
        self,
        message: str,
        sender_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> MessageResponse:
        """同步版本的消息处理。

        业务流程：获取/复用当前线程事件循环 → run_until_complete
                  阻塞驱动异步 handle_message。

        副作用：事件循环缺失时新建并安装到当前线程；驱动的
                handle_message 具有其全部副作用。

        参数/返回：同 handle_message。

        说明：仅适合同步上下文调用；已有运行中循环时会抛
              RuntimeError（异步框架内请直接用 handle_message）。
        """
        # 局部导入：仅同步适配路径需要 asyncio（延迟加载）。
        import asyncio

        # 优先复用当前线程已有的事件循环。
        try:
            loop = asyncio.get_event_loop()
        # 当前线程无循环 → 新建并安装（供本次及后续复用）。
        except RuntimeError:
            # 新建事件循环。
            loop = asyncio.new_event_loop()
            # 安装为当前线程循环。
            asyncio.set_event_loop(loop)

        # 阻塞驱动异步 handle_message 至完成并返回结果。
        return loop.run_until_complete(
            self.handle_message(message, sender_id, metadata)
        )

    async def get_tracker(self, sender_id: str) -> DialogueStateTracker | None:
        """获取指定用户的对话状态。

        业务流程：直接透传存储的 retrieve 查询。

        副作用：无（只读查询）。

        参数：
            sender_id: 发送者 ID（tracker 隔离键）。

        返回：
            该用户的 DialogueStateTracker；不存在时返回 None
            （与 get_or_create_tracker 不同，本方法不创建）。
        """
        # 透传存储查询：命中返回实例，未命中返回 None。
        return await self.tracker_store.retrieve(sender_id)

    async def reset_tracker(self, sender_id: str) -> None:
        """重置指定用户的对话状态。

        业务流程：查询 tracker → 存在则调用 restart() 清空对话
                  栈/槽位等运行时状态 → 回存。

        副作用：修改并持久化该用户的对话状态（重启等价于全新
                会话）；tracker 不存在时静默无操作。

        参数：
            sender_id: 发送者 ID（tracker 隔离键）。
        """
        # 查询该用户 tracker（不创建）。
        tracker = await self.tracker_store.retrieve(sender_id)
        # 存在才重置：restart 清空对话栈/槽位并保留会话骨架。
        if tracker:
            # 执行状态重置。
            tracker.restart()
            # 回存重置后的状态（持久化）。
            await self.tracker_store.save(tracker)

    def register_action(self, action: Action) -> None:
        """注册自定义动作（实例方法入口）。

        业务流程：直接透传模块级 register_action 注册函数。

        副作用：写模块级自定义动作注册表 _CUSTOM_ACTIONS（全局
                可变状态；键与内置动作重复时覆盖内置行为）。

        参数：
            action: 已实例化的 Action 子类对象（注册键取其
                    name property）。
        """
        # 透传注册：实例进入全局注册表，图 action 节点即可按名解析。
        register_action(action)

    @classmethod
    def load(
        cls,
        project_path: str | Path,
        config: AgentConfig | None = None,
    ) -> "Agent":
        """从项目目录加载 Agent（工厂方法）。

        核心功能：读四类配置（domain/flows/endpoints/config）→
                  逐组件构造（生成器/检索器/NLG/策略/存储）→
                  经 cls() 装配返回完整 Agent。

        业务流程：
            1. 校验输入：必须为项目目录，否则抛 FileNotFoundError；
            2. 注入 sys.path（项目目录——用户自定义模块的动态
               导入依赖）；
            3. 加载 Domain：支持 domain.yml 单文件或 domain/ 目录
               两种格式，均缺失时保持 None
               （后续以空 Domain 兜底）；
            4. 加载 Flows：经 FlowLoader 加载（单文件/目录），
               缺失时为空集合；
            5. 扫描注册 actions/ 自定义动作，并把动作名同步进
               Domain（保证 get_action 可解析、domain 校验通过）；
            6. 加载 endpoints.yml（models/embeddings/nlg/
               tracker_store/vector_store 定义），缺失用默认；
            7. 加载 config.yml：从 pipeline 段解析 LLMCommandGenerator
               的 llm 引用 → 经 endpoints 查得 llm_config；从
               policies 段解析 EnterpriseSearchPolicy 的检索器类
               路径与 llm/embeddings 引用；retrieval 段构造检索
               配置；
            8. 逐组件构造：命令生成器（有 llm_config 才建）→
               TrackerStore（endpoints 配置）→ FlowPolicy →
               检索器（config.yml 给类路径 + endpoints 给连接，
               失败仅告警）→ NLG 重述生成器（开关开启才建，失败
               降级 None）→ EnterpriseSearchPolicy（策略级 LLM
               优先、回退主 LLM）→ PolicyEnsemble 组装；
            9. cls(...) 完成装配并返回。

        副作用：写 sys.path（插入项目目录）；动态导入用户 actions
                模块；创建存储/检索器/NLG 等组件（可能建立外部
                连接）。

        参数：
            project_path: 项目目录路径（str | Path）。
            config: Agent 配置，允许 None（默认值；其中的相对
                    路径以 working_path 为基拼接）。

        返回：
            装配完成的 Agent 实例。

        异常：路径不存在时抛 FileNotFoundError；config.yml 结构
              非法时由 read_yaml_file 或字段访问抛出。
        """
        # 统一为 Path 对象（兼容字符串输入）。
        project_path = Path(project_path)

        # 未显式传配置时使用默认装配参数。
        if config is None:
            config = AgentConfig()

        # 输入必须是项目目录：模型压缩包加载能力已随 training 子包
        # 移除（2026-10-20），传文件路径或不存在的路径一律报错。
        if not project_path.is_dir():
            # 抛文件未找到异常（给出完整路径便于排查）。
            raise FileNotFoundError(f"Project directory not found: {project_path}")

        # 工作目录即项目目录：domain/flows/actions/endpoints/config
        # 均按源目录组织加载。
        working_path = project_path
        # info 留痕。
        logger.info(f"Loading agent from project directory: {project_path}")

        # info 留痕：最终生效的工作目录。
        logger.info(f"Working path: {working_path}")

        # 将项目目录添加到 sys.path，以便加载用户自定义模块（如
        # addons/）——动态导入 actions/addons 等依赖此搜索路径。
        project_path_str = str(project_path.absolute())

        # 未收录才插入首位：保证用户模块优先于同名标准库/三方包。
        if project_path_str not in sys.path:
            # 插入首位保证优先级。
            sys.path.insert(0, project_path_str)
            # info 留痕。
            logger.info(f"Added project path to sys.path: {project_path_str}")

        # 加载Domain
        # 支持两种格式: domain.yml 文件或 domain/ 目录
        domain = None
        # 首选：config.domain_path 相对工作目录的拼路径。
        domain_path = working_path / config.domain_path
        # 单文件存在 → 按文件加载。
        if domain_path.exists():
            # 从 YAML 文件解析 Domain 定义。
            domain = Domain.load(str(domain_path))
            # info 留痕。
            logger.info(f"Loaded domain from {domain_path}")
        else:
            # 如果配置的路径不存在，尝试查找 domain 目录（兼容
            # 目录式组织）。
            domain_dir = working_path / "domain"
            # 目录存在且确为目录 → 按目录加载（加载其中全部定义）。
            if domain_dir.exists() and domain_dir.is_dir():
                # 从目录解析 Domain。
                domain = Domain.load(str(domain_dir))
                # info 留痕。
                logger.info(f"Loaded domain from {domain_dir}")

        # 加载Flows：以 config.flows_path 相对工作目录拼接。
        flows_path = working_path / config.flows_path
        # 单文件回退：默认路径为目录形式（data/flows），而 ics init 模板
        # 生成的是 flows.yml 单文件——路径缺失时自动补 .yml/.yaml 扩展名
        # 再探测，避免装配期静默产出空 FlowsList（LLM 将无流程可选）。
        if not flows_path.exists():
            # 依次尝试同名单文件（目录名 + 扩展名）。
            for _ext in (".yml", ".yaml"):
                # 候选路径：工作目录下 flows_path 字符串直接拼扩展名。
                _candidate = working_path / (config.flows_path + _ext)
                # 命中即切换到单文件形态并停止探测。
                if _candidate.exists():
                    # 切换实际加载路径。
                    flows_path = _candidate
                    # 停止扩展名探测。
                    break
        # 兜底空集合（路径缺失时 Agent 以"无流程"能力运行）。
        flows = FlowsList()
        # 路径存在才加载。
        if flows_path.exists():
            # 创建 YAML 流程加载器。
            loader = FlowLoader()
            # 加载（支持单文件或目录批量）。
            flows = loader.load(flows_path)
            # info 留痕：流程条数（FlowsList 支持 len）。
            logger.info(f"Loaded {len(flows)} flows from {flows_path}")

        # 加载用户自定义 Actions：自动发现 actions/ 目录中的
        # Action 类并注册（业务流程见 _load_custom_actions）。
        actions_path = working_path / "actions"
        # 扫描注册，返回成功名单。
        custom_action_names = _load_custom_actions(actions_path)
        # 有注册成功才做同步与日志。
        if custom_action_names:
            # info 留痕：注册数量与目录。
            logger.info(
                f"Loaded {len(custom_action_names)} custom actions from {actions_path}"
            )
            # 将自动发现的 actions 同步到 domain 中——Domain 的
            # 动作清单需包含它们，运行期校验才能通过。
            if domain:
                # 逐个登记动作名。
                for action_name in custom_action_names:
                    # 追加进 Domain 动作清单。
                    domain.add_action(action_name)
                # debug 留痕：同步明细。
                logger.debug(f"Synced custom actions to domain: {custom_action_names}")

        # 加载endpoints配置（包含模型定义）：缺失时用默认配置
        # （全默认组件仍可运行）。
        endpoints_path = working_path / config.endpoints_path
        # 局部导入：端点配置类（延迟加载）。
        from app.ics_ai.shared.config import EndpointsConfig

        # 文件存在则解析，否则构造默认实例。
        endpoints_config = (
            EndpointsConfig.load(endpoints_path)
            if endpoints_path.exists()
            else EndpointsConfig()
        )

        # 加载config配置：四路参数预置 None（未配置即 None）。
        config_path = working_path / config.config_path
        llm_config = None
        retrieval_config = None
        nlg_config = None
        enterprise_llm_config = None
        enterprise_embeddings_config = None
        retriever_class_path = None
        # config.yml 存在才解析。
        if config_path.exists():
            # 读取 YAML 为字典（空文件得 None，需真值判断）。
            config_data = read_yaml_file(str(config_path))
            if config_data:
                # 从 pipeline 配置中获取 LLMCommandGenerator 的 llm 引用
                # ——组件以别名引用 endpoints.yml 中定义的模型。
                pipeline = config_data.get("pipeline", [])
                # 遍历组件清单定位命令生成器。
                for component in pipeline:
                    # 组件名匹配（框架约定名）。
                    if component.get("name") == "LLMCommandGenerator":
                        # 取 llm 引用名（缺省 "default"——与 config.yml
                        # 的 llm: default 写法对应）。
                        llm_ref = component.get("llm", "default")
                        # 经 endpoints 的模型字典解析为完整配置。
                        llm_config = endpoints_config.get_model_config(llm_ref)
                        if llm_config:
                            # info 留痕：解析成功。
                            logger.info(f"从 pipeline 配置加载 LLM '{llm_ref}'")
                        else:
                            # warning：引用名在 endpoints.yml 无定义
                            # （命令生成能力将缺失）。
                            logger.warning(f"endpoints.yml 中未找到模型 '{llm_ref}'")
                        # 定位即退出（只处理第一个匹配组件）。
                        break

                # 从 policies 配置中获取 EnterpriseSearchPolicy 的参数
                # （检索器类路径 + LLM/embeddings 引用）。
                policies = config_data.get("policies", [])
                # 冗余重置（前文已初始化 None，无害保留）。
                retriever_class_path = None
                # 遍历策略清单定位企业搜索策略。
                for policy in policies:
                    # 策略名匹配（框架约定名）。
                    if policy.get("name") == "EnterpriseSearchPolicy":
                        # 获取检索器类路径（vector_store 键存类导入
                        # 路径字符串）。
                        retriever_class_path = policy.get("vector_store")
                        if retriever_class_path:
                            # info 留痕。
                            logger.info(
                                f"从 policies 配置读取检索器类: {retriever_class_path}"
                            )

                        # 获取策略的 llm 引用（缺省 "default"）。
                        policy_llm_ref = policy.get("llm", "default")
                        # 解析为策略级 LLM 配置。
                        enterprise_llm_config = endpoints_config.get_model_config(
                            policy_llm_ref
                        )
                        if enterprise_llm_config:
                            # info 留痕。
                            logger.info(
                                f"从 policies 配置加载 EnterpriseSearchPolicy LLM '{policy_llm_ref}'"
                            )

                        # 获取策略的 embeddings 引用（缺省 "default"）。
                        policy_embeddings_ref = policy.get("embeddings", "default")
                        # 解析为策略级嵌入配置。
                        enterprise_embeddings_config = (
                            endpoints_config.get_embeddings_config(
                                policy_embeddings_ref
                            )
                        )
                        if enterprise_embeddings_config:
                            # info 留痕。
                            logger.info(
                                f"从 policies 配置加载 EnterpriseSearchPolicy embeddings '{policy_embeddings_ref}'"
                            )
                        # 定位即退出。
                        break

                # 加载检索配置（config.yml 的 retrieval 段——流程/
                # 知识检索开关与参数；当前仅解析供扩展，Agent 装配
                # 未直接消费）。
                if "retrieval" in config_data:
                    # 局部导入：检索配置类（延迟加载）。
                    from app.ics_ai.shared.config import RetrievalConfig

                    # 从字典段构造检索配置。
                    retrieval_config = RetrievalConfig.from_dict(
                        config_data.get("retrieval", {})
                    )

        # 从 endpoints.yml 获取 NLG 配置：None 表示未配置（不做
        # 响应重述）。
        nlg_config = endpoints_config.nlg

        # 创建命令生成器：仅在有 LLM 配置时构建（无配置则保持
        # None——运行期 understand 节点跳过 LLM 命令生成）。
        command_generator = None
        if llm_config:
            # 局部导入：生成器与配置类（延迟加载）。
            from app.ics_ai.dialogue_understanding.generator import (
                LLMCommandGenerator,
                LLMGeneratorConfig,
            )

            # 把 endpoints 解析出的 LLMConfig 映射为生成器配置
            # （字段名 type——与 LLMGeneratorConfig 定义对齐）。
            generator_config = LLMGeneratorConfig(
                type=llm_config.type,
                model=llm_config.model,
                api_key=llm_config.api_key,
                api_base=llm_config.api_base,
                temperature=llm_config.temperature,
                enable_thinking=llm_config.enable_thinking,
            )
            # 构造命令生成器实例。
            command_generator = LLMCommandGenerator(config=generator_config)

        # 从 endpoints.yml 获取 Tracker 存储配置（类型 + 路径）。
        tracker_store_config = endpoints_config.tracker_store
        # 经工厂创建存储实例（type 决定实现，path 仅文件类使用）。
        tracker_store = create_tracker_store(
            tracker_store_config.type,
            path=tracker_store_config.path,
        )
        # info 留痕：存储类型与路径。
        logger.info(
            f"创建 TrackerStore: type={tracker_store_config.type}, path={tracker_store_config.path}"
        )

        # 创建策略：流程策略必建（流程推进主力）。
        from app.ics_ai.policies import EnterpriseSearchPolicyConfig

        flow_policy = FlowPolicy(flows=flows)

        # 创建 Retriever（类路径从 config.yml policies 读取，连接
        # 配置从 endpoints.yml 读取——两处配置解耦组合）。
        retriever = None
        # 类路径存在才尝试创建。
        if retriever_class_path:
            try:
                # 局部导入：检索器工厂（retrieval 子包，延迟加载）。
                from app.ics_ai.retrieval import create_retriever

                # 把 endpoints 的向量库配置转换为连接参数字典。
                connect_config = endpoints_config.vector_store.to_connect_config()
                # 按类路径动态实例化检索器。
                retriever = create_retriever(retriever_class_path, connect_config)
                if retriever:
                    # info 留痕：检索器类。
                    logger.info(f"创建检索器: {retriever_class_path}")
            except Exception as e:
                # 创建失败（依赖缺失/连接不可用等）：warning 降级，
                # 策略以无检索器模式运行（内部降级链兜底）。
                logger.warning(f"创建检索器失败: {e}")

        # 创建NLG生成器（如果配置了重述）：nlg 配置缺失或开关关闭
        # 时保持 None（直接输出模板响应）。
        nlg_generator = None
        if nlg_config and nlg_config.rephrase_enabled:
            try:
                # 局部导入：NLG 重述三件套（延迟加载）。
                from app.ics_ai.nlg import (
                    ResponseRephraser,
                    RephraserConfig,
                    TemplateNLG,
                )

                # 获取重述用的LLM配置。
                rephrase_llm_config = None
                # 优先用配置指定的重述模型引用。
                if nlg_config.rephrase_model:
                    # 经 endpoints 解析为完整配置。
                    rephrase_llm_config = endpoints_config.get_model_config(
                        nlg_config.rephrase_model
                    )
                # 未指定或解析失败 → 回退到主 LLM 配置。
                if not rephrase_llm_config and llm_config:
                    # 复用主模型配置。
                    rephrase_llm_config = llm_config  # 回退到主LLM配置

                # 拿到 LLM 配置才继续构造（否则保持 None）。
                if rephrase_llm_config:
                    # 组装重述器参数（风格/阈值/槽位保留/语言）。
                    rephrase_config = RephraserConfig(
                        enabled=True,
                        llm_type=rephrase_llm_config.type,
                        llm_model=rephrase_llm_config.model,
                        style=nlg_config.rephrase_style,
                        rephrase_threshold=nlg_config.rephrase_threshold,
                        preserve_slots=nlg_config.preserve_slots,
                        language=nlg_config.language,
                    )

                    # 创建LLM客户端（重述专用实例——与主链路的
                    # 采样参数隔离）。
                    from app.ics_ai.shared.llm import create_llm_client

                    rephrase_llm = create_llm_client(
                        type=rephrase_llm_config.type,
                        model=rephrase_llm_config.model,
                        api_key=rephrase_llm_config.api_key,
                        api_base=rephrase_llm_config.api_base,
                        temperature=0.7,  # 重述使用较高温度（语言多样性）
                    )

                    # 创建模板NLG作为底层（重述器的兜底生成器）；
                    # TemplateNLG 无状态，domain 在 generate 调用时经
                    # ResponseRephraser 透传，构造期不持有（原写法
                    # TemplateNLG(domain=domain) 会抛 TypeError 被
                    # except 吞掉，导致重述功能静默失效）。
                    template_nlg = TemplateNLG()

                    # 组合：模板生成打底 + LLM 重述修饰。
                    nlg_generator = ResponseRephraser(
                        config=rephrase_config,
                        base_generator=template_nlg,
                        llm_client=rephrase_llm,
                    )
                    # info 留痕：重述风格。
                    logger.info(
                        f"Loaded NLG rephraser with style: {nlg_config.rephrase_style}"
                    )
            except Exception as e:
                # 构造失败（依赖/网络等）：warning 降级为无重述。
                logger.warning(f"Failed to create NLG generator: {e}")

        # 使用 policies 配置中的 LLM 配置创建 EnterpriseSearchPolicy
        # 优先使用 policies 中指定的 llm，否则回退到 pipeline 中的
        # llm（两处引用的回退链）。
        policy_llm_config = enterprise_llm_config or llm_config
        if policy_llm_config:
            # 策略配置仅承载类型与模型名（其余走注入）。
            enterprise_config = EnterpriseSearchPolicyConfig(
                llm_type=policy_llm_config.type,
                llm_model=policy_llm_config.model,
            )
            # 创建策略专用 LLM 客户端（透传全部参数含思考开关）。
            from app.ics_ai.shared.llm import create_llm_client

            llm_client = create_llm_client(
                type=policy_llm_config.type,
                model=policy_llm_config.model,
                api_key=policy_llm_config.api_key,
                api_base=policy_llm_config.api_base,
                temperature=policy_llm_config.temperature,
                enable_thinking=policy_llm_config.enable_thinking,
            )
            # 构造企业搜索策略（注入客户端与检索器）。
            enterprise_policy = EnterpriseSearchPolicy(
                config=enterprise_config,
                llm_client=llm_client,
                retriever=retriever,
            )
            # info 留痕：策略模型名。
            logger.info(f"创建 EnterpriseSearchPolicy: llm={policy_llm_config.model}")
        else:
            # 无任何 LLM 配置：退化构造（仅检索器，LLM 能力由
            # 策略内部降级链处理）。
            enterprise_policy = EnterpriseSearchPolicy(retriever=retriever)

        # 组装策略集成器：FlowPolicy（流程推进，高优先级）在前、
        # 企业搜索（知识问答兜底）在后，按置信度裁决。
        policy_ensemble = PolicyEnsemble(
            policies=[
                flow_policy,
                enterprise_policy,
            ]
        )

        # 经类工厂完成最终装配（组件全部就位后一次性构造 Agent）。
        return cls(
            domain=domain,
            flows=flows,
            tracker_store=tracker_store,
            policy_ensemble=policy_ensemble,
            command_generator=command_generator,
            nlg_generator=nlg_generator,
            config=config,
        )

    @classmethod
    async def create(
        cls,
        domain: Domain | None = None,
        flows: FlowsList | None = None,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        **kwargs: Any,
    ) -> "Agent":
        """创建 Agent 实例的便捷方法。

        业务流程：按传入的提供商/模型名构造 LLMGeneratorConfig →
                  创建命令生成器 → 经 cls() 装配返回（其余组件
                  走 __init__ 默认兜底）。

        副作用：构造生成器（不发起网络请求——连接在使用期建立）。

        参数：
            domain: Domain 定义，允许 None。
            flows: Flow 列表，允许 None。
            llm_provider: LLM 提供商名（映射到配置的 type 字段）。
            llm_model: LLM 模型名。
            **kwargs: 额外配置（透传 cls 构造）。

        返回：
            Agent 实例。

        说明：llm_provider 经参数名 type 传给 LLMGeneratorConfig
        （该配置类字段名为 type，见 llm_generator.py L117）——
        2026-10-16 修复：原以 provider= 构造与字段名不一致，调用
        必抛 TypeError，现已与 load() 的写法对齐。
        """
        # 局部导入：生成器与配置类（延迟加载）。
        from app.ics_ai.dialogue_understanding.generator import (
            LLMCommandGenerator,
            LLMGeneratorConfig,
        )

        # 创建命令生成器：以 type= 传提供商名（与配置类字段名
        # 一致；2026-10-16 修复原 provider= 的键名错误）。
        generator_config = LLMGeneratorConfig(
            type=llm_provider,
            model=llm_model,
        )
        # 构造命令生成器（连接在使用期才建立，构造不发请求）。
        command_generator = LLMCommandGenerator(config=generator_config)

        # 经类工厂装配返回（tracker_store/policy_ensemble 等走
        # __init__ 默认兜底逻辑）。
        return cls(
            domain=domain,
            flows=flows,
            command_generator=command_generator,
        )


# 导出：模块对外暴露的公共 API（Agent 主类与装配配置 DTO）。
__all__ = [
    "Agent",
    "AgentConfig",
]
