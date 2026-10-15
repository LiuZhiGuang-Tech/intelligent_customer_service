"""
对话栈包入口（stack）

核心功能：dialogue_understanding 的对话栈子包入口，聚合导出对话栈
          DialogueStack 与常用栈帧类型（基类 StackFrame、Flow 帧、
          搜索帧、闲聊帧），为策略层与命令处理层提供统一的栈上下文
          导入路径。

执行流程：本包被导入时按顺序执行——
          1. 导入 dialogue_stack 子模块，获得对话栈数据类；
          2. 导入 stack_frame 子模块，获得帧基类与常用具体帧；
          3. 组装 __all__ 公开名单，供 `from app.ics_ai.
             dialogue_understanding.stack import *` 使用。

副作用：仅执行普通模块导入（连带执行两个子模块的模块体），不读写
        文件/数据库、不发起网络请求、不修改全局状态。

主要依赖：app.ics_ai.dialogue_understanding.stack.dialogue_stack、
          app.ics_ai.dialogue_understanding.stack.stack_frame。
"""

# 导入对话栈数据类：LIFO 栈管理器，含 Flow 导航与序列化能力
# （职责与逐方法注释见 dialogue_stack.py）。
from app.ics_ai.dialogue_understanding.stack.dialogue_stack import (
    # 列表项1：对话栈类——压/弹/查顶、Flow 导航与序列化的唯一入口类型。
    DialogueStack,
)

# 导入常用栈帧类型：抽象基类与三种核心具体帧；CannotHandleStackFrame、
# CompletedStackFrame、HumanHandoffStackFrame 等低频帧未提升到包级，
# 需要时从 stack_frame 模块按完整路径导入
# （各帧职责与生命周期见 stack_frame.py 的逐类注释）。
from app.ics_ai.dialogue_understanding.stack.stack_frame import (
    # 列表项1：闲聊栈帧——表示正在处理与业务无关的寒暄对话。
    ChitChatStackFrame,
    # 列表项2：Flow栈帧——表示一个正在执行的 Flow（栈内最核心的帧）。
    FlowStackFrame,
    # 列表项3：搜索栈帧——表示正在执行知识库检索（RAG）。
    SearchStackFrame,
    # 列表项4：栈帧抽象基类——所有帧的公共父类，供类型标注与 isinstance。
    StackFrame,
)

# 包级公开 API 白名单：约束 `from ...stack import *` 的导出范围，
# 与上方两个导入块一一对应；未列入的名称需按完整路径显式导入。
__all__ = [
    # 列表项1：闲聊栈帧（与上方导入同名，纳入 `import *` 导出）。
    "ChitChatStackFrame",
    # 列表项2：对话栈类（与上方导入同名，纳入 `import *` 导出）。
    "DialogueStack",
    # 列表项3：Flow栈帧（与上方导入同名，纳入 `import *` 导出）。
    "FlowStackFrame",
    # 列表项4：搜索栈帧（与上方导入同名，纳入 `import *` 导出）。
    "SearchStackFrame",
    # 列表项5：栈帧抽象基类（与上方导入同名，纳入 `import *` 导出）。
    "StackFrame",
]
