"""
base_policy - 策略基类

核心功能：定义"策略（Policy）"的统一抽象接口与通用能力——策略负责依据当前对话
          状态预测下一步应执行的动作；本模块提供配置项 PolicyConfig、预测结果
          PolicyPrediction 两个数据类，以及抽象基类 Policy（子类如 FlowPolicy、
          EnterpriseSearchPolicy 各自实现不同的决策逻辑）。

执行流程：框架在初始化时实例化各策略并注入配置；每轮对话由上层（PolicyEnsemble）
          依次调用 should_predict 判断是否参与决策、does_support_stack_frame
          判断能否处理当前栈帧，再调用抽象方法 predict 异步得到 PolicyPrediction，
          最后按 priority 与 confidence 择优执行动作。子类未覆盖的钩子方法
          （train/persist/load 等）走本模块的默认空实现。

副作用：本模块自身不读写文件/数据库、不发起网络请求；唯一的全局性副作用在
        predict_sync —— 事件循环缺失时会创建并调用 asyncio.set_event_loop
        安装到当前线程。子类的 predict 实现可能产生实际业务副作用（如查库）。

主要依赖：仅标准库——abc（ABC/abstractmethod）、asyncio（predict_sync 内局部
          导入）、dataclasses（dataclass/field）、typing（TYPE_CHECKING 用于
          隔离仅注解用的导入）；DialogueStateTracker、Domain、FlowsList 仅用于
          类型注解，置于 TYPE_CHECKING 分支下导入以避免循环导入。
"""

# 从 __future__ 导入 annotations：启用 PEP 563 延迟注解求值，注解以字符串形式
# 保存，支持前向引用（如 DialogueStateTracker）并避免运行时求值开销。
from __future__ import annotations

# 导入抽象基类工具：ABC 标记策略基类，abstractmethod 强制子类实现 predict。
from abc import ABC, abstractmethod

# 导入 dataclass 与 field：两个数据类靠 dataclass 自动生成 __init__/__repr__，
# field(default_factory=...) 为可变默认值（list/dict）提供独立工厂，
# 避免所有实例共享同一个默认对象。
from dataclasses import dataclass, field

# 导入类型注解工具：Any 任意类型；TYPE_CHECKING 为"仅类型检查期成立"的常量，
# 用于隔离只服务于注解的导入，避免运行时循环依赖。
from typing import TYPE_CHECKING, Any

# 类型检查期专属导入：这三个类型只出现在方法注解中，运行时导入会形成循环依赖
# （core/tracker、core/domain、flow 反向依赖策略层），故放入 TYPE_CHECKING 分支。
if TYPE_CHECKING:
    # 对话状态追踪器：策略做决策时的数据来源（仅类型注解用）。
    from app.ics_ai.core.domain import Domain

    # 领域定义：槽位/动作/响应注册表（仅类型注解用）。
    from app.ics_ai.core.tracker import DialogueStateTracker

    # 流程集合：FlowPolicy 等子类据此推进流程（仅类型注解用）。
    from app.ics_ai.dialogue_understanding.flow import FlowsList


@dataclass
class PolicyConfig:
    """策略配置。

    职责：承载策略的通用配置项，供 Policy 及其子类读取（当前用于优先级与
          历史轮数的集中管理）。
    设计模式：数据传输对象（DTO），无行为逻辑。
    生命周期：随策略创建时构造（Policy.__init__ 未传入时按默认值新建），
              与所属策略实例同生命周期。

    Attributes:
        priority: 策略优先级，数值越大优先级越高
        max_history: 考虑的最大历史轮数
    """

    # 策略优先级：数值越大越优先被采纳；默认 1，子类可通过配置覆盖。
    priority: int = 1
    # 最大历史轮数：None 表示不限制（由策略实现自行决定如何使用）。
    max_history: int | None = None


@dataclass
class PolicyPrediction:
    """策略预测结果。

    职责：承载一次策略预测的输出——要执行的动作、置信度、附带事件与元数据，
          供上层在多个策略之间比较择优。
    设计模式：数据传输对象（DTO），另提供 abstain 工厂方法构造"放弃预测"结果。
    生命周期：每轮预测由策略临时创建并返回，被上层消费后即可丢弃。

    Attributes:
        action: 预测的动作名称
        confidence: 预测置信度 (0.0 - 1.0)
        events: 附带的事件列表
        metadata: 额外的元数据
        policy_name: 产生此预测的策略名称
    """

    # 预测动作名；None 表示未预测出动作（配合 confidence=0 构成"放弃"）。
    action: str | None = None
    # 置信度：取值区间 0.0-1.0；默认 0.0 表示毫无把握（等同于放弃）。
    confidence: float = 0.0
    # 附带事件列表：若该预测被采纳，需要一并写入对话事件流的事件；
    # 可变默认值经 field 工厂独立创建。
    events: list[dict[str, Any]] = field(default_factory=list)
    # 额外元数据：扩展信息字典（如话题 id、引用文档等）；
    # 可变默认值经 field 工厂独立创建。
    metadata: dict[str, Any] = field(default_factory=dict)
    # 产生此预测的策略名称：便于日志与调试时定位"是谁给出的建议"。
    policy_name: str = ""

    # property 装饰器：把方法包装为只读属性，调用方以 prediction.is_abstain 访问。
    @property
    def is_abstain(self) -> bool:
        """策略是否放弃预测。

        返回：
            动作缺失（action is None）或置信度为 0.0 时返回 True；否则 False。
        """
        # 两种放弃情形合并判断：要么没给出动作，要么置信度为 0。
        return self.action is None or self.confidence == 0.0

    # classmethod 装饰器：工厂方法，cls 为被调用的类，
    # 使子类调用 abstain 时仍返回子类实例。
    @classmethod
    def abstain(cls, policy_name: str = "") -> PolicyPrediction:
        """创建一个放弃预测的结果。

        业务流程：以 action=None、confidence=0.0 构造实例——该取值组合使
                  is_abstain 为 True，上层据此忽略本次预测。

        副作用：无（仅创建对象）。

        参数：
            policy_name: 策略名称，用于标记预测来源；默认为空串。

        返回：
            PolicyPrediction 实例（放弃态）。
        """
        # 放弃态的两个关键字段固定为 None 与 0.0，其余走默认值。
        return cls(action=None, confidence=0.0, policy_name=policy_name)


class Policy(ABC):
    """策略基类。

    策略负责根据当前对话状态预测下一步应该执行的动作。
    不同的策略实现不同的决策逻辑：
    - FlowPolicy: 基于Flow定义执行
    - EnterpriseSearchPolicy: 基于知识库检索回答

    职责：定义所有策略的共同契约——抽象方法 predict 强制子类实现决策逻辑，
          should_predict / does_support_stack_frame 作为可覆盖的参与决策钩子，
          train / persist / load 提供默认空实现以减少子类样板代码。
    设计模式：抽象基类 + 策略模式——上层（PolicyEnsemble）面向本基类编程，
              运行时按优先级与置信度在多个具体策略间择优；predict_sync 为
              异步接口提供的同步适配方法。
    生命周期：框架启动时创建并注入配置，随会话/服务存续长期复用；
              persist/load 为可选的跨进程持久化钩子（默认不落盘）。
    """

    # 默认优先级：类属性，供未显式配置时与 PolicyConfig 的默认值保持一致。
    DEFAULT_PRIORITY = 1

    def __init__(
        self,
        config: PolicyConfig | None = None,
        **kwargs: Any,
    ):
        """初始化策略。

        业务流程：1. config 为 None 时兜底新建默认配置，保证 self.config 始终
        可用；2. 以类名作为策略名（日志与预测结果中的标识来源）。

        副作用：仅写入实例属性，无 IO 操作。

        Args:
            config: 策略配置，None 时使用 PolicyConfig() 默认值。
            **kwargs: 额外配置，本基类不解析（接收后忽略），
                      使子类/调用方传入多余键时不会报错。
        """
        # 配置兜底：避免后续访问 self.config.priority 时触发 None 属性错误。
        self.config = config or PolicyConfig()
        # 策略名默认取类名（如 "FlowPolicy"），子类无需重复维护名称常量。
        self._name = self.__class__.__name__

    # property 装饰器：对外只读暴露策略名（外部无法直接赋值 _name）。
    @property
    def name(self) -> str:
        """策略名称。

        返回：
            策略名字符串，默认等于类名。
        """
        # 直接返回初始化时确定的私有属性。
        return self._name

    # property 装饰器：对外只读暴露优先级，屏蔽配置对象的内部结构。
    @property
    def priority(self) -> int:
        """策略优先级。

        返回：
            优先级整数，取自 config.priority（数值越大越优先）。
        """
        # 由配置对象提供实际取值，便于运行期调整配置后即时生效。
        return self.config.priority

    # abstractmethod 装饰器：标记为抽象方法，子类未实现时无法实例化本类。
    @abstractmethod
    async def predict(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        flows: FlowsList | None = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """预测下一步动作。

        业务流程：由子类实现具体决策逻辑（如读栈帧推流程、或检索知识库）；
        本基类不提供默认行为，仅抛出 NotImplementedError 作为兜底。

        副作用：由子类决定（可能读写槽位、调用检索服务等）；基类实现无副作用。

        参数：
            tracker: 对话状态追踪器，必需（子类据此读取槽位与对话栈）。
            domain: Domain定义，允许为 None（子类按需使用）。
            flows: Flow列表，允许为 None（FlowPolicy 类子类需要使用）。
            **kwargs: 额外参数，供上层透传扩展配置。

        返回：
            预测结果 PolicyPrediction；基类实现直接抛 NotImplementedError，
            不返回（正常调用时应由子类覆盖后返回实例，或以 abstain 表示放弃）。

        说明：虽然抽象方法通常无需函数体，这里保留 raise 作为显式保护——
        若子类通过非常规方式绕过抽象检查，调用时仍能得到明确报错。
        """
        # 兜底保护：子类未覆盖时抛出，避免静默返回 None 引发更难排查的错误。
        raise NotImplementedError()

    def predict_sync(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        flows: FlowsList | None = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """同步版本的预测方法。

        业务流程：1. 局部导入 asyncio；2. 尝试获取当前事件循环；3. 获取失败
        （当前线程尚无循环，RuntimeError）时新建循环并安装到当前线程；
        4. 用 run_until_complete 驱动异步 predict 直到返回结果。

        副作用：事件循环缺失时会创建新循环并调用 asyncio.set_event_loop
                安装，改变当前线程的事件循环绑定状态。

        参数：
            tracker: 对话状态追踪器，透传给 predict。
            domain: Domain定义，允许为 None，透传。
            flows: Flow列表，允许为 None，透传。
            **kwargs: 额外参数，透传给 predict。

        返回：
            PolicyPrediction，即异步 predict 的返回值；
            若当前线程已有正在运行的事件循环，run_until_complete 会抛
            RuntimeError（故本方法仅适合在同步上下文中调用）。

        说明：为同步调用方提供的适配层，无需把整个调用链改成 async。
        """
        # 局部导入：仅本方法需要 asyncio，放在函数内可避免模块级依赖。
        import asyncio

        try:
            # 优先复用当前线程已有的事件循环（避免反复创建循环造成资源泄漏）。
            loop = asyncio.get_event_loop()
        # 捕获"当前线程无事件循环"的异常（Python 3.10+ 行为）。
        except RuntimeError:
            # 新建一个事件循环。
            loop = asyncio.new_event_loop()
            # 安装为当前线程的事件循环，使后续调用能复用同一个循环。
            asyncio.set_event_loop(loop)

        # 阻塞等待异步预测完成：把 predict 协程跑在当前事件循环上并取回结果。
        return loop.run_until_complete(self.predict(tracker, domain, flows, **kwargs))

    def should_predict(
        self,
        tracker: DialogueStateTracker,
    ) -> bool:
        """检查策略是否应该进行预测。

        子类可以覆盖此方法以实现条件性预测。

        业务流程：默认恒返回 True（无条件参与决策）；子类可据 tracker 状态
        （如当前栈帧类型、是否存在待处理指令）决定是否跳过本次预测。

        副作用：无（本基类实现不读取、不修改任何状态）。

        参数：
            tracker: 对话状态追踪器，子类实现据此判断条件。

        返回：
            是否应该预测；本基类实现恒为 True。
        """
        # 默认无条件预测：最宽松策略，交由上层用 priority/confidence 择优。
        return True

    def does_support_stack_frame(self, frame: Any | None = None) -> bool:
        """检查策略是否支持处理当前栈帧。

        子类可以覆盖此方法以声明支持特定类型的栈帧。
        PolicyEnsemble可以使用此方法路由请求到合适的策略。

        业务流程：默认恒返回 True（声明支持任何栈帧）；子类可据 frame 的类型
        返回 False，从而在策略集里被排除出本轮的候选。

        副作用：无（仅做类型判断）。

        参数：
            frame: 要检查的栈帧，如果为None则检查是否支持任何栈帧；
                   类型为 Any（栈帧体系有多种子类，此处不做约束）。

        返回：
            是否支持处理该栈帧；本基类实现恒为 True。
        """
        # 默认支持所有栈帧：不限制使用场景，避免子类漏覆盖导致策略不可用。
        return True

    def train(
        self,
        training_data: Any,
        domain: Domain | None = None,
        **kwargs: Any,
    ) -> None:
        """训练策略。

        大多数策略不需要训练，可以保持默认实现。

        业务流程：空实现——本基类策略不涉及训练，子类如需训练可覆盖此方法。

        副作用：无（基类实现不产生任何副作用）。

        参数：
            training_data: 训练数据，任意类型（子类自行解释其结构）。
            domain: Domain定义，允许为 None。
            **kwargs: 额外参数，供子类扩展。

        返回：
            无返回值（None）。
        """
        # 默认不训练：保持空实现，使无训练需求的子类无需编写该方法。
        pass

    def persist(self, path: str) -> None:
        """持久化策略到文件。

        业务流程：空实现——本基类策略无需要保存的状态，子类如需落盘可覆盖。

        副作用：无（基类实现不写任何文件）。

        参数：
            path: 保存路径，字符串（当前实现未使用）。

        返回：
            无返回值（None）。
        """
        # 默认不持久化：策略为无状态时无需保存，子类按需覆盖。
        pass

    # classmethod 装饰器：工厂方法，cls 为被调用的类，保证返回对应子类实例。
    @classmethod
    def load(cls, path: str) -> Policy:
        """从文件加载策略。

        业务流程：忽略 path，直接以默认参数构造 cls 实例——对应"策略无持久化
        状态"的默认设计，子类若实现 persist 应一并覆盖本方法。

        副作用：无（基类实现不读取文件）。

        参数：
            path: 加载路径，字符串（当前实现未使用）。

        返回：
            策略实例（cls() 的默认构造结果）。
        """
        # 直接构造默认实例：与 persist 的空实现配对，保证默认流程可闭环。
        return cls()


# 包级公开 API 白名单：约束 `from base_policy import *` 的导出范围，
# 列出本模块对外暴露的三个核心类型；未列入的名称需按完整路径显式导入。
__all__ = [
    "Policy",  # 列表项1：策略抽象基类
    "PolicyConfig",  # 列表项2：策略配置数据类
    "PolicyPrediction",  # 列表项3：策略预测结果数据类
]
