"""
generator - 命令生成包入口模块（使用 LLM 将用户输入转换为命令）

核心功能：聚合对话理解阶段"命令生成"环节的四个子模块——以抽象基类
          CommandGenerator 定义生成契约，LLMCommandGenerator 为默认
          实现：把对话上下文交给 LLM，将其输出的 DSL 文本解析为
          Command 对象列表，产出统一的 GenerationResult。

执行流程：
    1. 导入期：本包被导入时加载四个子模块（契约 → 解析器 → 生成器 →
       构建器），各模块级类随之定义就绪，无其他初始化动作；
    2. 使用期：agent.py 在装配期按 pipeline 配置构造 LLMCommandGenerator，
       message_processor / graph 流程在消息处理期调用 generate（或基类
       桥接的 generate_sync）产出 GenerationResult（commands 为 Command
       列表），交由下游命令执行流处理。

副作用：仅导入子模块并绑定符号名；子模块导入期只做类定义与模块级
        常量/日志器初始化，不读写文件/数据库、不发起网络请求、
        不修改全局状态。

主要依赖：包内四个子模块——base_generator（生成器契约与结果 DTO）、
          command_parser（DSL 解析）、llm_generator（LLM 生成器实现）、
          prompt_builder（提示词构建）。
"""

# 生成器契约与结果 DTO：CommandGenerator 为抽象基类（generate 异步入口
# 与 generate_sync 同步桥接），子类按策略实现生成逻辑。
from app.ics_ai.dialogue_understanding.generator.base_generator import (
    CommandGenerator,
)

# 命令解析器：把 LLM 输出的 DSL 文本解析为 Command 对象列表
# （供 LLMCommandGenerator 第三步"解析命令"复用）。
from app.ics_ai.dialogue_understanding.generator.command_parser import (
    CommandParser,
)

# LLM 命令生成器：默认生成器实现，编排"提示词构建 → LLM 调用 →
# 结果解析 → 兜底与错误映射"四步；LLMGeneratorConfig 为其配置类。
from app.ics_ai.dialogue_understanding.generator.llm_generator import (
    LLMCommandGenerator,
    LLMGeneratorConfig,
)

# 提示词构建器：把 Tracker/Domain/Flows 组装为 Jinja2 上下文并渲染
# command_prompt.jinja2，产出命令生成阶段的 LLM 提示词消息列表。
from app.ics_ai.dialogue_understanding.generator.prompt_builder import (
    PromptBuilder,
)

# 导出：本包对外暴露的公共 API（按字母序，便于 IDE 提示与 diff 整洁）。
__all__ = [
    "CommandGenerator",
    "CommandParser",
    "LLMCommandGenerator",
    "LLMGeneratorConfig",
    "PromptBuilder",
]
