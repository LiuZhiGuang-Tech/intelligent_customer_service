"""
init - 项目初始化命令模块

核心功能：提供 `ics init` 命令——在指定目录初始化新的 ICS AI 对话
          机器人项目：创建约定目录骨架（data/actions/models/tests）
          并生成七类模板文件（config.yml、domain.yml、endpoints.yml、
          data/flows.yml、actions/actions.py、actions/__init__.py、
          README.md）与 .gitignore。

执行流程：1. click 解析 --path（目标目录）、--template（模板类型）、
          --no-prompt（跳过交互确认）；2. 目标路径不存在时确认创建、
          非空时确认继续（no_prompt 跳过全部确认）；3.
          create_project_structure 落盘目录树与模板文件（config 模板
          注入随机 assistant_id）；4. 打印项目结构树与下一步指引；
          异常兜底——debug 透传堆栈，普通模式退出码 1。

副作用：向目标目录 mkdir 目录树、写 8 个模板文件；--no-prompt 缺省
        时经 click.confirm 读取 stdin 交互确认。

主要依赖：click（命令框架与交互确认）、pathlib（路径与文件写入）、
          enum（模板类型枚举）、app.ics_ai.shared.constants（框架
          默认路径常量，供模板文案引用）。
"""

# 导入 sys：取消创建时的进程正常退出（sys.exit(0)）与失败退出（sys.exit(1)）。
import sys

# 从标准库 pathlib 导入 Path：目标目录解析、目录创建与文件写入。
from pathlib import Path

# 从标准库 enum 导入 Enum：项目模板类型的枚举基类。
from enum import Enum

# 导入 click：命令装饰器、选项解析、终端输出与交互确认。
import click

# 导入框架默认路径常量：config/domain/data/models/actions/flows 的
# 约定路径名，供模板文案与结构树输出引用（保持与框架查找逻辑一致）。
from app.ics_ai.shared.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_DOMAIN_PATH,
    DEFAULT_MODELS_PATH,
    DEFAULT_ACTIONS_PATH,
    DEFAULT_FLOWS_PATH,
)


class ProjectTemplate(str, Enum):
    """项目模板类型。

    职责：枚举 `ics init --template` 可选的三种项目模板（default/
          basic/tutorial），当前实现共用同一套生成逻辑，枚举值仅作
          前端选项与扩展预留。

    设计模式：str 混入枚举——成员可直接与字符串比较，click.Choice
              直接消费 .value，__str__ 返回原值便于回显。

    生命周期：模块导入时定义；init_command 解析 --template 时按
              字符串值实例化，命令结束即弃。
    """

    # 默认模板：完整项目骨架（当前唯一实现的模板）。
    DEFAULT = "default"

    # 基础模板：最小骨架（预留）。
    BASIC = "basic"

    # 教程模板：配套教学示例（预留）。
    TUTORIAL = "tutorial"

    def __str__(self) -> str:
        """返回枚举原值字符串。

        :return: 成员值（如 "default"），保证 f-string/echo 回显友好。
        """
        # 直接返回成员值：避免默认 Enum 的 "ProjectTemplate.DEFAULT" 长名。
        return self.value


# 默认的config.yml内容：pipeline（LLM 命令生成器）与 policies
# （FlowPolicy + EnterpriseSearchPolicy）的最小可用配置模板；
# assistant_id 经 str.format 注入随机生成的项目标识。
DEFAULT_CONFIG_CONTENT = """# ICS AI 配置文件
# LLM驱动的对话系统配置

assistant_id: "{assistant_id}"

# Pipeline配置 - 对话理解组件
pipeline:
  # LLM命令生成器：将用户输入转换为系统命令
  - name: "LLMCommandGenerator"
    llm: "default"  # 引用endpoints.yml中的模型

# 策略配置
policies:
  # Flow策略：处理对话流程
  - name: "FlowPolicy"
  # 企业搜索策略：处理知识检索和降级
  - name: "EnterpriseSearchPolicy"
    # vector_store: "my_module.MyRetriever"  # 自定义检索器全类名（可选）
    llm: "default"  # 引用endpoints.yml中的模型
    embeddings: "default"  # 引用endpoints.yml中的嵌入模型
"""

# 默认的domain.yml内容：演示三类槽位（text/from_llm、categorical、
# controlled）与四个响应模板、三个动作的完整示例，生成后可直接
# `ics shell` 跑通问候/帮助/告别三条流程。
DEFAULT_DOMAIN_CONTENT = """# Domain配置文件
# 定义对话系统的槽位、动作、响应等

version: "1.0"

# 槽位定义
# 每个槽位必须包含: type（数据类型）和 mappings（映射方式）
slots:
  user_name:
    type: text
    mappings:
      - type: from_llm  # 由LLM从用户输入中提取
    description: "用户姓名"
  
  current_topic:
    type: categorical
    values:
      - greeting
      - help
      - goodbye
    mappings:
      - type: from_llm
    description: "当前对话主题"
  
  order_id:
    type: text
    mappings:
      - type: controlled  # 由Action代码控制填充
    description: "订单编号，由系统生成"

# 响应模板
responses:
  utter_greet:
    - text: "你好！我是ICS AI助手，有什么可以帮助你的吗？"
    - text: "您好！很高兴为您服务。"
  
  utter_goodbye:
    - text: "再见！祝您生活愉快！"
    - text: "感谢使用，再见！"
  
  utter_default:
    - text: "抱歉，我没有理解您的意思，能否换一种方式表达？"
  
  utter_ask_help:
    - text: "请问您需要什么帮助？"

# 动作列表
actions:
  - action_greet
  - action_goodbye
  - action_handle_help

# Forms (表单)
forms: ~
"""

# 默认的endpoints.yml内容：models（qwen LLM）、embeddings、nlg 重述、
# vector_store 与 tracker_store 的示例配置；敏感项（api_key/密码）
# 以注释形式给出 ${ENV_VAR} 占位写法，引导用户从环境变量注入。
DEFAULT_ENDPOINTS_CONTENT = """# Endpoints配置文件
# 定义模型服务和外部端点

# 模型组配置 - 统一管理所有LLM和嵌入模型
models:
  # 默认LLM模型
  default:
    type: "qwen"  # openai, qwen, azure, anthropic
    model: "qwen-plus"
    temperature: 0.0
    max_tokens: 1024
    # api_key: ${DASHSCOPE_API_KEY}  # 从环境变量读取
    # api_base: ""  # 自定义API地址（用于vLLM等自部署服务）
    # enable_thinking: false  # 启用深度思考模式
  
  # 可定义多个模型用于不同用途
  # rephrase:
  #   type: "qwen"
  #   model: "qwen-turbo"
  #   temperature: 0.7

# 嵌入模型配置
embeddings:
  default:
    type: "openai"
    model: "text-embedding-3-small"
    # api_key: ${OPENAI_API_KEY}

# NLG配置 - 响应生成和重述
nlg:
  # 响应重述功能
  rephrase:
    enabled: false
    model: "default"  # 引用上面models中的模型名称
    style: "friendly"  # friendly, professional, casual, empathetic
    language: "zh"

# 向量存储配置 - 传递给 retriever.connect() 的连接参数
# 检索器类名在 config.yml 的 policies.EnterpriseSearchPolicy.vector_store 中指定
vector_store:
  # 示例：Neo4j 连接配置
  # uri: "bolt://localhost:7687"
  # user: "neo4j"
  # password: "${NEO4J_PASSWORD}"
  # 示例：Elasticsearch 连接配置
  # hosts: ["http://localhost:9200"]
  # index: "knowledge_base"

# Tracker Store配置 - 对话状态存储
tracker_store:
  type: json  # json, mysql, memory
  path: "./tracker_store"
"""

# 默认的Flow文件内容：两条最小流程（greet_flow 单步、help_flow 两步
# 跳转），演示 steps 的 id/action/next 约定，与 domain.yml 的响应/
# 动作名互相呼应。
DEFAULT_FLOW_CONTENT = """# Flows定义文件
# 定义对话流程

flows:
  greet_flow:
    description: "问候流程"
    steps:
      - id: "start"
        action: action_greet
        next: "end"
  
  help_flow:
    description: "帮助流程"
    steps:
      - id: "ask_topic"
        action: utter_ask_help
        next: "handle"
      - id: "handle"
        action: action_handle_help
        next: "end"
"""

# 默认的actions.py内容：三个可运行的自定义动作示例（问候个性化/
# 告别/帮助应答），演示 Action 子类约定——name 属性注册名、异步
# run 签名、ActionResult.add_response 回填响应；文件会被 Agent.load
# 的 actions/ 扫描自动发现注册。
DEFAULT_ACTIONS_CONTENT = '''# -*- coding: utf-8 -*-
"""
自定义Action实现

在这里定义你的自定义动作逻辑。
每个自定义动作需要继承Action基类并实现run方法。

注意：
- Action 类会在 Agent.load() 时自动发现和注册
- run 方法必须返回 ActionResult 对象
- 使用 result.add_response(text) 添加响应文本
"""

from typing import Any, Text

from app.ics_ai.agent.actions import Action, ActionResult


class ActionGreet(Action):
    """问候动作。"""
    
    @property
    def name(self) -> Text:
        return "action_greet"
    
    async def run(
        self,
        tracker: Any,
        domain: Any,
        **kwargs: Any,
    ) -> ActionResult:
        """执行问候动作。
        
        根据用户是否已知，返回个性化问候。
        """
        result = ActionResult()
        user_name = tracker.get_slot("user_name") if hasattr(tracker, 'get_slot') else None
        if user_name:
            result.add_response(f"你好，{user_name}！有什么可以帮助你的吗？")
        else:
            result.add_response("你好！我是ICS AI助手，有什么可以帮助你的吗？")
        return result


class ActionGoodbye(Action):
    """告别动作。"""
    
    @property
    def name(self) -> Text:
        return "action_goodbye"
    
    async def run(
        self,
        tracker: Any,
        domain: Any,
        **kwargs: Any,
    ) -> ActionResult:
        """执行告别动作。"""
        result = ActionResult()
        result.add_response("再见！祝您生活愉快！")
        return result


class ActionHandleHelp(Action):
    """处理帮助请求的动作。"""
    
    @property
    def name(self) -> Text:
        return "action_handle_help"
    
    async def run(
        self,
        tracker: Any,
        domain: Any,
        **kwargs: Any,
    ) -> ActionResult:
        """处理帮助请求。
        
        根据用户最新消息提供相应帮助。
        """
        result = ActionResult()
        latest_message = tracker.latest_message if hasattr(tracker, 'latest_message') else None
        if latest_message:
            text = latest_message.text if hasattr(latest_message, 'text') else ""
            if text:
                result.add_response(f"好的，我来帮您处理关于'{text}'的问题。")
                return result
        result.add_response("请告诉我您需要什么帮助。")
        return result
'''

# README内容：面向生成项目的说明文档——项目结构树、快速开始四步
# （安装/环境变量/运行/交互）、endpoints 模型配置与 slot 映射、
# pipeline 的配置要点复述，帮助新手对照修改。
DEFAULT_README_CONTENT = """# ICS AI 项目

LLM驱动的对话系统项目。

## 项目结构

```
.
├── config.yml          # 系统配置（pipeline、policies）
├── domain.yml          # Domain定义（slots、responses、actions）
├── endpoints.yml       # 端点配置（models、nlg、tracker_store）
├── data/
│   └── flows.yml       # Flow定义
├── actions/
│   └── actions.py      # 自定义动作
```

## 快速开始

1. 安装依赖:
   ```bash
   pip install ics-ai
   ```

2. 配置环境变量:
   ```bash
   # 阿里云DashScope
   export DASHSCOPE_API_KEY=your-api-key
   
   # 或 OpenAI
   export OPENAI_API_KEY=your-api-key
   ```

3. 运行服务:
   ```bash
   ics run
   ```

4. 交互测试:
   ```bash
   ics shell
   ```

## 配置说明

### 模型配置 (endpoints.yml)

在 `endpoints.yml` 中配置LLM模型:

```yaml
models:
  default:
    type: "qwen"  # openai, qwen, azure, anthropic
    model: "qwen-plus"
    # api_key: ${DASHSCOPE_API_KEY}
    # api_base: "http://localhost:8000/v1"  # vLLM等自部署服务
    # enable_thinking: false  # 深度思考模式
```

### 槽位配置 (domain.yml)

```yaml
slots:
  user_name:
    type: text
    mappings:
      - type: from_llm  # 由LLM提取
    description: "用户姓名"
  
  order_id:
    type: text
    mappings:
      - type: controlled  # 由Action控制
    description: "订单编号"
```

### Pipeline配置 (config.yml)

```yaml
pipeline:
  - name: "LLMCommandGenerator"
    llm: "default"  # 引用endpoints.yml中的模型

policies:
  - name: "FlowPolicy"
  - name: "EnterpriseSearchPolicy"
    vector_store: "faiss"
    llm: "default"
```

## 更多信息

请参考 ICS AI 文档获取更多信息。
"""


def generate_assistant_id() -> str:
    """生成随机的助手ID。

    核心功能：拼装 "形容词-名词-三位数字" 形式的项目标识（如
              happy-robot-123），写入 config.yml 的 assistant_id。

    业务流程：1. 延迟导入随机模块（仅本函数使用）；2. 从词表随机取
              形容词与名词；3. 随机生成 3 位数字串；4. 以连字符拼接
              三段返回。

    副作用：无（仅内存随机运算，不触碰文件/网络）。

    :return: 形如 "happy-robot-123" 的随机 ID 字符串。
    """
    # 延迟导入 random/string：仅本函数消费，避免模块级多余依赖。
    import random
    import string

    # 形容词词表：ID 第一段的候选池（语义均为拟人化褒义词）。
    adjectives = ["happy", "smart", "quick", "calm", "bright", "wise", "kind"]

    # 名词词表：ID 第二段的候选池（均为助手类称谓）。
    nouns = ["robot", "assistant", "helper", "agent", "bot", "buddy"]

    # 随机取一个形容词作为第一段。
    adj = random.choice(adjectives)

    # 随机取一个名词作为第二段。
    noun = random.choice(nouns)

    # 随机取 3 个数字字符拼接为第三段（可能含前导零）。
    num = "".join(random.choices(string.digits, k=3))

    # 连字符拼装三段，返回最终 ID。
    return f"{adj}-{noun}-{num}"


def create_project_structure(
    path: Path, template: ProjectTemplate = ProjectTemplate.DEFAULT
) -> None:
    """创建项目目录结构和模板文件。

    核心功能：在目标路径落盘完整项目骨架——四个子目录、七类模板
              文件与 .gitignore。

    业务流程：1. mkdir 级联创建 data/actions/models/tests 四目录；
              2. generate_assistant_id 生成项目标识；3. 组装
              "路径 → 模板内容" 映射（config.yml 注入 assistant_id）；
              4. 逐项确保父目录存在后以 UTF-8 写出；5. 写 .gitignore
              （Python/模型/IDE/tracker/日志五组忽略规则）。

    副作用：向磁盘创建目录与写入 8 个文件（已存在内容会被覆盖）。

    :param path: 项目根目录（绝对或相对，父目录不存在则级联创建）。
    :param template: 模板类型，默认 DEFAULT（当前三种模板共用此实现）。
    :return: 无返回；磁盘 IO 异常向上传播由调用方兜底。
    """
    # 第一步：定义并创建四个子目录（parents 级联、exist_ok 幂等）。
    directories = [
        path / "data",
        path / "actions",
        path / "models",
        path / "tests",
    ]

    # 逐个创建：已存在时静默跳过。
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    # 第二步：生成随机 assistant_id（仅 config.yml 消费）。
    assistant_id = generate_assistant_id()

    # 第三步：组装文件映射——键为目标路径，值为模板字符串。
    files = {
        # config 模板经 format 注入 assistant_id 占位符。
        path / "config.yml": DEFAULT_CONFIG_CONTENT.format(assistant_id=assistant_id),
        # 其余模板原样写出。
        path / "domain.yml": DEFAULT_DOMAIN_CONTENT,
        path / "endpoints.yml": DEFAULT_ENDPOINTS_CONTENT,
        path / "data" / "flows.yml": DEFAULT_FLOW_CONTENT,
        path / "actions" / "actions.py": DEFAULT_ACTIONS_CONTENT,
        # 动作包 __init__ 占位：保证 actions/ 可作为包被扫描导入。
        path / "actions" / "__init__.py": "# Custom actions module\n",
        path / "README.md": DEFAULT_README_CONTENT,
    }

    # 第四步：逐文件写出——先确保父目录存在（防御 templates 子目录
    # 未预创建的场景），再以 UTF-8 编码落盘。
    for file_path, content in files.items():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

    # 第五步：写 .gitignore——约定五组忽略项（Python 产物/模型目录/
    # IDE 配置/tracker 存储/日志），models/ 与 tracker_store/ 为运行
    # 期生成物不入库。
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
.env

# Models
models/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Tracker store
tracker_store/

# Logs
*.log
logs/
"""
    # 以 UTF-8 写出 .gitignore（隐藏文件，位于项目根）。
    (path / ".gitignore").write_text(gitignore_content, encoding="utf-8")


@click.command(name="init")
@click.option(
    "--path", "-p", type=click.Path(), default=".", help="项目创建路径，默认为当前目录"
)
@click.option(
    "--template",
    "-t",
    type=click.Choice([t.value for t in ProjectTemplate]),
    default=ProjectTemplate.DEFAULT.value,
    help="项目模板类型",
)
@click.option(
    "--no-prompt", is_flag=True, default=False, help="跳过交互式提示，使用默认值"
)
@click.pass_context
def init_command(ctx: click.Context, path: str, template: str, no_prompt: bool) -> None:
    """初始化新的ICS AI项目。

    核心功能：交互式（或 --no-prompt 静默）创建项目骨架与模板文件，
              收尾打印结构树与下一步操作指引。

    业务流程：1. 打印欢迎横幅；2. 解析目标绝对路径；3. 路径不存在
              时确认/静默创建（拒绝则退出码 0）；4. 目录非空时确认
              继续（拒绝则退出码 0）；5. 调 create_project_structure
              落盘；6. 打印 [OK] 与结构树、三步指引；7. 异常兜底——
              debug 透传堆栈，普通模式退出码 1。

    副作用：随 create_project_structure 写磁盘；非 no-prompt 模式经
            click.confirm 读取 stdin。

    :param ctx: click 上下文（异常兜底读取主组注入的 debug 标志）。
    :param path: 项目创建路径，默认当前目录。
    :param template: 模板类型字符串（default/basic/tutorial 之一）。
    :param no_prompt: True 时跳过全部交互确认，按默认策略执行。
    :return: 无返回；用户取消退出码 0，创建失败退出码 1。
    """
    # 欢迎横幅：空行分隔提升可读性。
    click.echo("欢迎使用 ICS AI!\n")
    click.echo("正在创建新项目...\n")

    # 解析目标路径：resolve 转绝对路径，后续提示与创建均基于此。
    project_path = Path(path).resolve()

    # 第一道检查：路径不存在——确认或静默创建。
    if not project_path.exists():
        # 静默模式：直接创建，不询问。
        if no_prompt:
            click.echo(f"创建目录: {project_path}")
            project_path.mkdir(parents=True, exist_ok=True)
        else:
            # 交互模式：confirm 征询，拒绝则正常退出（退出码 0）。
            if click.confirm(f"目录 '{project_path}' 不存在，是否创建？"):
                project_path.mkdir(parents=True, exist_ok=True)
            else:
                click.echo("已取消。")
                sys.exit(0)

    # 第二道检查：目录非空——防止覆盖用户已有文件（仅交互模式询问，
    # no-prompt 默认继续写入）。
    if project_path.exists() and any(project_path.iterdir()):
        if not no_prompt:
            if not click.confirm(f"目录 '{project_path}' 不为空，是否继续？"):
                click.echo("已取消。")
                sys.exit(0)

    # 第三步：创建项目结构并输出结果报告。
    try:
        # 字符串转枚举成员（click.Choice 已校验取值合法）。
        template_enum = ProjectTemplate(template)

        # 落盘目录树与模板文件（写磁盘的异常在此捕获）。
        create_project_structure(project_path, template_enum)

        # 成功横幅与项目结构树回显。
        click.echo(f"[OK] 项目已创建于: {project_path}")
        click.echo("")
        click.echo("项目结构:")
        click.echo("  +-- config.yml        # 系统配置（pipeline、policies）")
        click.echo("  +-- domain.yml        # Domain定义（slots、responses）")
        click.echo("  +-- endpoints.yml     # 端点配置（models、nlg、tracker）")
        click.echo("  +-- data/")
        click.echo("  |   +-- flows.yml     # Flow定义")
        click.echo("  +-- actions/")
        click.echo("  |   +-- actions.py    # 自定义动作")
        click.echo("")
        click.echo("下一步:")
        click.echo("  1. 编辑 config.yml 配置LLM")
        click.echo("  2. 设置环境变量 OPENAI_API_KEY")
        click.echo("  3. 运行 'ics shell' 测试对话")
        click.echo("")

    except Exception as e:
        # 异常兜底：打印错误；debug 模式透传堆栈，普通模式退出码 1。
        # ctx.obj 由 cli 主组注入（单测直调时可能为 None，真实链路必为 dict）。
        click.echo(f"错误: 创建项目失败 - {e}", err=True)
        if ctx.obj.get("debug"):
            raise
        sys.exit(1)


# 导出清单：cli 聚合层按名导入，ProjectTemplate/create_project_structure
# 同时暴露供单测与扩展复用。
__all__ = ["init_command", "ProjectTemplate", "create_project_structure"]
