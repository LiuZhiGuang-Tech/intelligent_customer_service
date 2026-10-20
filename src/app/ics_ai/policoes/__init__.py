"""
policies - 对话策略包入口模块

核心功能：ics_ai 的策略决策层（对标 Rasa 的 rasa.core.policies），
          聚合策略实现——基类 Policy 定义统一接口、FlowPolicy 基于
          Flow 定义推进对话流程、EnterpriseSearchPolicy 基于知识库
          检索实现 RAG 问答与降级链、PolicyEnsemble 集成多策略按
          置信度裁决。

执行流程：本包被导入时按顺序执行——
          1. base_policy 子模块——定义抽象基类 Policy 及 PolicyConfig / PolicyPrediction DTO；
          2. flow_policy 子模块——实现 Flow 策略（最高优先级，推进 YAML Flow 定义）；
          3. enterprise_search_policy 子模块——实现企业搜索策略（RAG + 四级降级）；
          4. policy_ensemble 子模块——实现策略集成器（遍历策略收集预测并择优）；
          5. 组装 __all__ 公开名单，暴露基类契约 + 三类策略实现 + 集成器。

副作用：仅执行普通模块导入（连带执行四个子模块的模块体初始化），
        不读写文件/数据库、不发起网络请求、不修改全局状态。
        （各策略实例的运行时副作用——LLM 调用/检索器调用——发生在
        predict 阶段，不在本包导入期。）

主要依赖：app.ics_ai.policies.base_policy、flow_policy、
          enterprise_search_policy、policy_ensemble。
"""

# 从策略基类子模块导入核心抽象类型：Policy 基类、配置 DTO、预测结果 DTO，
# 三者是所有策略实现的共同契约，故提升到包级供业务模块直接引用。
from app.ics_ai.policies.base_policy import Policy, PolicyConfig, PolicyPrediction

# 从企业搜索策略子模块导入：EnterpriseSearchPolicy 实现知识库 RAG 检索
# （其依赖 shared.llm 与 retrieval 子包均已就位），EnterpriseSearchPolicyConfig
# 扩展配置（检索参数 + LLM 选型 + 闲聊降级开关）。
# 注：policy_ensemble 因存在未就位依赖暂不导入，待依赖补齐后追加。
from app.ics_ai.policies.enterprise_search_policy import (
    EnterpriseSearchPolicy,
    EnterpriseSearchPolicyConfig,
)

# 从 Flow 策略子模块导入：FlowPolicy 实现基于 YAML Flow 定义推进对话的具体策略，
# FlowPolicyConfig 扩展配置（priority 默认 100 + max_steps_per_turn 上限）。
from app.ics_ai.policies.flow_policy import FlowPolicy, FlowPolicyConfig

# 从策略集成器子模块导入：PolicyEnsemble 遍历已注册策略收集预测并按
# 置信度择优（其依赖 base_policy 与 shared.constants 均已就位）。
from app.ics_ai.policies.policy_ensemble import PolicyEnsemble

# 包级公开 API 白名单：约束 `from app.ics_ai.policies import *` 的导出范围，
# 列出本包对外暴露的所有核心策略类型；未列入的名称需按完整路径显式导入。
__all__ = [
    "Policy",  # 抽象基类：所有策略的共同契约
    "PolicyConfig",  # 策略配置 DTO：优先级、历史轮数等
    "PolicyEnsemble",  # 策略集成器：多策略预测收集与置信度裁决
    "PolicyPrediction",  # 预测结果 DTO：动作、置信度、附带事件等
    "FlowPolicy",  # 具体策略：基于 YAML Flow 定义推进对话（最高优先级）
    "FlowPolicyConfig",  # Flow 策略专属配置 DTO：max_steps_per_turn 等
    "EnterpriseSearchPolicy",  # 具体策略：知识库 RAG 检索 + 降级链
    "EnterpriseSearchPolicyConfig",  # 搜索策略专属配置 DTO：检索/LLM/闲聊开关
]
