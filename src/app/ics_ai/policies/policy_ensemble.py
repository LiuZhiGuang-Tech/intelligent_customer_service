"""
policy_ensemble - 策略集成器模块

核心功能：管理多个策略（Policy），按优先级从高到低依次驱动各策略预测，
          在所有非放弃预测中择优（置信度最高者）返回；全部放弃时回退到
          配置的默认动作。提供运行期的策略增删查与批量训练能力。

执行流程：
    1. 构造：收集策略列表 → _sort_policies 按 priority 降序排序；
    2. 每轮对话：predict 遍历策略 → should_predict 门控 → await
       policy.predict → 过滤放弃/低于阈值预测 → 保留最高置信度
       （置信度 1.0 提前短路）→ 返回最佳或 fallback；
    3. create_default_ensemble 工厂：局部导入两类内置策略组装默认集成。

副作用：
    - predict 驱动各策略的 predict（子类可能查库/检索/调用 LLM）；
    - predict_sync 在事件循环缺失时新建并安装到当前线程；
    - 模块自身无文件/数据库/网络 IO。

主要依赖：
    - logging：命名 logger（跳过/错误/选择结果日志）。
    - dataclasses：EnsembleConfig 配置数据类。
    - typing：Any 通配符 + TYPE_CHECKING 隔离仅注解导入。
    - app.ics_ai.policies.base_policy：Policy 抽象基类与
      PolicyPrediction 预测结果契约。
    - app.ics_ai.shared.constants：ACTION_LISTEN 默认回退动作。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，支持前向引用
# （如 TYPE_CHECKING 分支中的 tracker/domain 类型）并消除运行时求值开销。
from __future__ import annotations

# 导入日志器工厂：为本模块创建命名 logger（跳过/异常/择优日志走此通道）。
import logging

# 导入 dataclass：EnsembleConfig 靠它自动生成 __init__/__repr__。
from dataclasses import dataclass

# 导入类型注解工具：Any 类型通配符（kwargs/metadata 等）；
# TYPE_CHECKING 为"仅类型检查期成立"的常量，隔离只服务于注解的导入。
from typing import TYPE_CHECKING, Any

# 导入策略契约：Policy 抽象基类（列表元素类型）、PolicyPrediction
# （predict 的返回值类型）。
from app.ics_ai.policies.base_policy import Policy, PolicyPrediction

# 导入常量：ACTION_LISTEN 作为"所有策略都放弃"时的默认回退动作。
from app.ics_ai.shared.constants import ACTION_LISTEN

# 类型检查期专属导入：这三个类型只出现在方法注解中，运行时不需要
# 真实导入（避免循环依赖与导入开销）。
if TYPE_CHECKING:
    # 领域定义：predict 签名与基类保持一致，透传给各策略（仅注解用）。
    from app.ics_ai.core.domain import Domain

    # 对话状态追踪器：predict 系列方法的数据来源（仅注解用）。
    from app.ics_ai.core.tracker import DialogueStateTracker

    # 流程集合：predict 签名透传参数（仅注解用）。
    from app.ics_ai.dialogue_understanding.flow import FlowsList

# 本模块命名 logger：日志前缀自动带模块路径，便于按集成器过滤日志。
logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """集成器配置。

    职责：承载 PolicyEnsemble 的行为参数——回退动作与最低置信度门槛，
          供 predict 择优逻辑读取。
    设计模式：数据传输对象（DTO），无行为逻辑。
    生命周期：随 PolicyEnsemble 构造时创建（未传入时按默认值新建），
              与集成器实例同生命周期。

    Attributes:
        fallback_action: 所有策略都放弃时的默认动作名。
        min_confidence: 采纳预测的最低置信度阈值（0.0-1.0）。
    """

    # 回退动作：默认 action_listen（交还控制权等待用户输入）。
    fallback_action: str = ACTION_LISTEN
    # 置信度门槛：低于该值的预测不采纳；默认 0.0（不设门槛）。
    min_confidence: float = 0.0


class PolicyEnsemble:
    """策略集成器。

    职责：持有策略列表并维护优先级序；predict 时按优先级从高到低驱动
          各策略、在有效预测中择优返回；提供运行期 add/remove/get
          策略与批量训练（train_all）。
    设计模式：门面（Facade）——对外暴露单一 predict 入口，内部编排
              多个 Policy；择优策略与 Rasa 的 PolicyEnsemble 同构。
    生命周期：应用装配时创建（通常经 create_default_ensemble 工厂），
              服务存续期长期复用；策略列表可运行期动态增删。

    策略选择逻辑：
    1. 按优先级从高到低遍历策略
    2. 选择第一个非放弃且置信度最高的预测
    3. 如果所有策略都放弃，返回默认动作
    """

    def __init__(
        self,
        policies: list[Policy] | None = None,
        config: EnsembleConfig | None = None,
    ):
        """初始化策略集成器。

        业务流程：1. policies 为 None 时兜底空列表（后续可 add_policy）；
                  2. config 为 None 时兜底默认配置；3. 立即按优先级
                  排序，保证后续遍历顺序正确。

        副作用：仅操作实例属性（原地排序），无 IO。

        参数：
            policies: 初始策略列表；None 表示空集成。
            config: 集成器配置；None 时使用 EnsembleConfig() 默认值。
        """
        # 策略列表兜底：避免 None 引发后续 sort/遍历报错。
        self.policies = policies or []
        # 配置兜底：保证 self.config 恒可用（predict 中要读阈值）。
        self.config = config or EnsembleConfig()

        # 按优先级排序（从高到低），构造期即确定遍历顺序。
        self._sort_policies()

    def _sort_policies(self) -> None:
        """按优先级降序原地排序策略列表。

        业务流程：以 policy.priority 为键 reverse 排序——数值越大越
                  靠前，predict 时优先被驱动。
        副作用：原地修改 self.policies 顺序。
        """
        # lambda 取每个策略的 priority；reverse=True 降序（高优先级在前）。
        self.policies.sort(key=lambda p: p.priority, reverse=True)

    def add_policy(self, policy: Policy) -> None:
        """添加策略并维持优先级序。

        业务流程：追加到列表尾 → 重新整体排序（简单且保证顺序正确）。

        副作用：修改 self.policies。

        参数：
            policy: 要添加的策略实例。
        """
        # 追加后再统一排序：免去插入位置的二分比较逻辑。
        self.policies.append(policy)
        # 重排保证优先级序不被破坏。
        self._sort_policies()

    def remove_policy(self, policy_name: str) -> Policy | None:
        """按名称移除策略。

        业务流程：线性查找 name 匹配的策略 → 命中即按索引弹出并返回；
                  未命中返回 None。

        副作用：命中时修改 self.policies。

        参数：
            policy_name: 目标策略名（policy.name）。

        返回：
            被移除的策略实例；不存在时返回 None。
        """
        # enumerate 同时取索引（pop 需要）与实例（名称匹配）。
        for i, policy in enumerate(self.policies):
            # 名称匹配 → 移除并返回。
            if policy.name == policy_name:
                # pop 按索引移除并返回被移除元素。
                return self.policies.pop(i)
        # 遍历完毕未命中 → None。
        return None

    async def predict(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        flows: FlowsList | None = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """使用策略集成进行预测。

        业务流程：
            1. 遍历（已按优先级降序）各策略；
            2. should_predict=False 的策略跳过（门控）；
            3. await policy.predict 收集预测（单策略异常吞掉记日志、
               继续下一个）；
            4. 非放弃且置信度 >= min_confidence 的预测参与择优；
            5. 置信度 1.0 提前短路（高优先级命中即免跑低优先级）；
            6. 有最佳预测返回之；否则回退 fallback_action。

        副作用：驱动各策略 predict（子类可能查库/检索/调 LLM）；写日志。

        参数：
            tracker: 对话状态追踪器，透传给各策略。
            domain: Domain 定义，允许 None，透传。
            flows: Flow 列表，允许 None，透传。
            **kwargs: 额外参数，透传给各策略。

        返回：
            最佳 PolicyPrediction；全部放弃时返回 fallback 预测
            （confidence=0.0、metadata["fallback"]=True）。
        """
        # 当前最佳预测：None 表示尚无有效候选。
        best_prediction: PolicyPrediction | None = None
        # 全量预测收集：供调试/审计（当前未参与决策逻辑）。
        all_predictions: list[PolicyPrediction] = []

        # 主循环：按优先级从高到低驱动各策略。
        for policy in self.policies:
            # 门控检查：策略声明本轮不参与 → 跳过。
            if not policy.should_predict(tracker):
                # 记录跳过原因，便于排查"为何某策略没生效"。
                logger.debug(f"Policy {policy.name} skipped (should_predict=False)")
                continue

            try:
                # 驱动单个策略预测（await 异步接口）。
                prediction = await policy.predict(tracker, domain, flows, **kwargs)
                # 收集到全量列表（调试用途）。
                all_predictions.append(prediction)

                # 记录每个策略的预测结果，便于离线分析。
                logger.debug(
                    f"Policy {policy.name} predicted: "
                    f"action={prediction.action}, confidence={prediction.confidence}"
                )

                # 放弃态预测（无动作或置信度 0）不参与择优。
                if not prediction.is_abstain:
                    # 低于阈值不采纳（config.min_confidence 可调门槛）。
                    if prediction.confidence >= self.config.min_confidence:
                        # 择优：比当前最佳置信度更高才替换（严格大于，
                        # 保证先到的高优先级策略在平分时胜出）。
                        if (
                            best_prediction is None
                            or prediction.confidence > best_prediction.confidence
                        ):
                            # 更新最佳候选。
                            best_prediction = prediction

                        # 满分短路：置信度 1.0 视为确定结论，停止遍历。
                        if prediction.confidence >= 1.0:
                            # 提前退出主循环（节省低优先级策略开销）。
                            break

            # 单个策略异常不拖垮整轮预测：记日志后继续下一个。
            except Exception as e:
                # 错误日志含策略名与异常信息。
                logger.error(f"Policy {policy.name} error: {e}")
                # 继续尝试后续策略。
                continue

        # 存在有效候选 → 返回最佳。
        if best_prediction is not None:
            # info 级记录最终选择，便于线上追踪。
            logger.info(
                f"Selected prediction from {best_prediction.policy_name}: "
                f"action={best_prediction.action}"
            )
            # 返回最佳预测。
            return best_prediction

        # 所有策略都放弃，返回默认动作。
        logger.debug("All policies abstained, using fallback action")
        # fallback 预测：confidence=0.0 + fallback 标记，供下游识别兜底。
        return PolicyPrediction(
            # 动作取配置的回退动作（默认 action_listen）。
            action=self.config.fallback_action,
            # 置信度 0：明确表示"非真实预测"。
            confidence=0.0,
            # 来源标记为集成器自身。
            policy_name="PolicyEnsemble",
            # fallback 标记：下游可据此区分兜底与真实预测。
            metadata={"fallback": True},
        )

    def predict_sync(
        self,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        flows: FlowsList | None = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """同步版本的预测方法。

        业务流程：获取/复用当前线程事件循环 → run_until_complete
                  阻塞驱动异步 predict 至返回。

        副作用：事件循环缺失时新建并安装到当前线程。

        参数/返回：同 predict。

        说明：仅适合同步上下文调用；已有运行中循环时会抛 RuntimeError。
        """
        # 局部导入：仅同步适配路径需要 asyncio。
        import asyncio

        try:
            # 优先复用当前线程已有的事件循环。
            loop = asyncio.get_event_loop()
        # 当前线程无循环（Python 3.10+ 抛 RuntimeError）→ 新建。
        except RuntimeError:
            # 新建事件循环。
            loop = asyncio.new_event_loop()
            # 安装为当前线程循环，供后续复用。
            asyncio.set_event_loop(loop)

        # 阻塞驱动异步 predict 至完成并取回结果。
        return loop.run_until_complete(self.predict(tracker, domain, flows, **kwargs))

    def get_policy(self, name: str) -> Policy | None:
        """根据名称获取策略。

        业务流程：线性查找 name 匹配的策略，命中返回实例。

        副作用：无（只读遍历）。

        参数：
            name: 策略名称（policy.name）。

        返回：
            策略实例；不存在时返回 None。
        """
        # 遍历查找。
        for policy in self.policies:
            # 名称命中 → 返回实例。
            if policy.name == name:
                return policy
        # 未命中 → None。
        return None

    @property
    def policy_names(self) -> list[str]:
        """获取所有策略名称（按当前优先级序）。

        副作用：无。

        返回：
            策略名列表（只读快照，外部修改不影响集成器）。
        """
        # 列表推导提取每个策略的 name。
        return [p.name for p in self.policies]

    def train_all(
        self,
        training_data: Any,
        domain: Domain | None = None,
        **kwargs: Any,
    ) -> None:
        """训练所有策略。

        业务流程：遍历各策略逐一调用 train；单策略异常吞掉记日志，
                  不影响其余策略训练。

        副作用：调用各策略 train（子类可能写磁盘/更新模型状态）；
                写日志。

        参数：
            training_data: 训练数据（结构由各策略自行解释）。
            domain: Domain 定义，允许 None。
            **kwargs: 额外参数，透传。

        返回：
            None。
        """
        # 逐策略训练。
        for policy in self.policies:
            try:
                # 调用策略的 train（默认空实现或子类覆盖版）。
                policy.train(training_data, domain, **kwargs)
                # 成功日志。
                logger.info(f"Trained policy: {policy.name}")
            # 单策略失败不阻断其余训练。
            except Exception as e:
                # 失败日志。
                logger.error(f"Failed to train {policy.name}: {e}")


# 便捷函数：组装生产默认的策略集成器。


def create_default_ensemble() -> PolicyEnsemble:
    """创建默认的策略集成器。

    业务流程：局部导入两类内置策略（延迟到调用时，避免模块级循环
              依赖与不必要的导入开销）→ 以 FlowPolicy（优先级 100，
              先于检索）+ EnterpriseSearchPolicy（优先级 50）组装
              集成器（构造期自动按优先级排序）。

    副作用：实例化策略（构造期仅保存配置，无 IO）。

    返回：
        PolicyEnsemble 实例（含默认两策略，已按优先级排序）。
    """
    # 局部导入 Flow 策略：延迟到工厂调用时执行。
    from app.ics_ai.policies.flow_policy import FlowPolicy

    # 局部导入企业知识检索策略：同上。
    from app.ics_ai.policies.enterprise_search_policy import EnterpriseSearchPolicy

    # 组装集成器：__init__ 内部按优先级排序（Flow 100 > 检索 50）。
    return PolicyEnsemble(
        # 策略列表：Flow 优先接管，检索兜底。
        policies=[
            # 列表项1：Flow 策略（高优先级，有活动 Flow 时接管对话）。
            FlowPolicy(),
            # 列表项2：企业知识检索策略（低优先级，Flow 未命中时兜底）。
            EnterpriseSearchPolicy(),
        ]
    )


# 导出清单：from policy_ensemble import * 时仅暴露以下名字。
__all__ = [
    # 列表项1：策略集成器主类。
    "PolicyEnsemble",
    # 列表项2：集成器配置数据类。
    "EnsembleConfig",
    # 列表项3：默认集成器工厂函数。
    "create_default_ensemble",
]
