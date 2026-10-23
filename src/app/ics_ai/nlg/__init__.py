"""
nlg - 自然语言生成（NLG）包入口模块

核心功能：聚合机器人的回复文本生成能力——抽象契约（NLGGenerator
          基类 + NLGConfig/NLGResponse DTO）、模板实现
          （TemplateNLG：Domain 响应模板 + 槽位变量替换）、LLM
          改写实现（ResponseRephraser：条件触发的重述装饰器）。

执行流程：
    1. 导入期：按序执行三个子模块的模块体初始化（纯定义，无
       重依赖加载——LLM 客户端在重述首次调用时才延迟创建）；
    2. 组装 __all__ 公开名单（基类 + DTO + 两类实现）；
    3. 运行期：agent.py 按 NLG 配置选择模板路径或 LLM 改写路径，
       动作层经 generate 获取回复文本。

副作用：仅导入期的普通模块导入，不读写文件/数据库、不发起网络
        请求、不修改全局状态（网络请求仅发生在 Rephraser 的
        complete 调用期）。

主要依赖：app.ics_ai.nlg.nlg_generator、template_nlg、
          response_rephraser。
"""

# 从基类子模块导入：NLGGenerator 抽象基类（两类实现的共同契约，
# 业务层按此类型注解）与 NLGConfig/NLGResponse 两个 DTO。
from app.ics_ai.nlg.nlg_generator import (
    NLGGenerator,
    NLGConfig,
    NLGResponse,
)

# 从 LLM 改写实现子模块导入：ResponseRephraser——条件触发的
# 重述装饰器（包装其它生成器，LLM 使回复更自然）及其配置 DTO。
from app.ics_ai.nlg.response_rephraser import (
    ResponseRephraser,
    RephraserConfig,
)

# 从模板实现子模块导入：TemplateNLG——纯模板路径生成器（Domain
# 响应模板随机变体 + {slot_name} 槽位替换，无 LLM 依赖）。
from app.ics_ai.nlg.template_nlg import TemplateNLG

# 包级公开 API 白名单：约束 `from app.ics_ai.nlg import *` 的导出
# 范围；按字母序排列，逐项行尾注释角色。
__all__ = [
    "NLGConfig",  # 基类级配置 DTO：use_llm 开关与 LLM 选型
    "NLGGenerator",  # 抽象基类：generate 生成契约
    "NLGResponse",  # 响应 DTO：文本/按钮/图片/自定义/元数据
    "RephraserConfig",  # 重述器配置 DTO：开关/阈值/风格/槽位保持
    "ResponseRephraser",  # LLM 改写实现：条件触发的重述装饰器
    "TemplateNLG",  # 模板实现：Domain 模板 + 槽位变量替换
]
