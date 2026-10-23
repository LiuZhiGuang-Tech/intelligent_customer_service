"""
ics_ecs 电商客服机器人 - 程序命令行启动入口模块

核心功能：
    将 ics_ecs 包目录按 Agent.load() 的项目目录契约装配为完整机器人，
    并提供两种运行模式：控制台交互模式（默认，ConsoleChannel 问答）与
    服务模式（--serve，ICSServer 暴露 REST/WebSocket 接口）。

执行流程：
    1. 解析命令行参数（运行模式 / host / port / inspect 开关）；
    2. 定位两组路径：包目录（即"项目目录"——domain/、data/flows/、
       actions/ 均在包内）与仓库根 configurations/ 目录（框架层
       config.yml / endpoints.yml 所在，位于包外）；
    3. 构造 AgentConfig：domain/flows 用包内相对路径，config/endpoints
       用绝对路径覆盖——Path / 绝对路径 = 绝对路径本身，借此突破
       "五件套同目录树"的默认布局约束，零文件搬动；
    4. Agent.load() 装配：合并 domain 分片 → 加载 flows → 扫描注册
       actions/ 下的业务动作 → 按 config.yml 组装 LLM/策略/检索器/NLG；
    5. 按模式运行：控制台模式跑 asyncio 交互循环；服务模式起 FastAPI
       监听直至进程退出。

副作用：
    - 装配期：读 domain/flows/config/endpoints 五类配置文件；动态导入
      actions/ 模块（触发动作类注册进全局注册表）；检索器 connect 失败
      时打 warning 降级（不中断装配）；endpoints.yml 展开读取
      DASHSCOPE_API_KEY 等环境变量；
    - 控制台模式：读写 stdin/stdout；对话状态经默认内存 tracker_store
      保存（进程退出即失）；
    - 服务模式：监听 TCP 端口，处理 HTTP/WS 请求。

主要依赖：
    - app.ics_ai.agent：Agent（装配组合根）、AgentConfig（装配参数）；
    - app.ics_ai.api：ICSServer（服务模式入口）；
    - app.ics_ai.channels：ConsoleChannel / ConsoleOutputChannel（控制台模式）；
    - app.ics_ai.channels.base_channel：UserMessage（控制台回调入参 DTO）。
"""

# 从标准库 argparse 导入参数解析器：解析运行模式与 host/port 等启动参数。
import argparse

# 从标准库 asyncio 导入异步运行器：驱动控制台交互循环（run_interactive
# 是协程，需 asyncio.run 启动）。
import asyncio

# 从标准库 pathlib 导入 Path：定位包目录与仓库根 configurations 目录。
from pathlib import Path

# 导入 dotenv：.env 文件加载器（键值对写入 os.environ，供 ${VAR} 替换）。
import dotenv

# 从 agent 包导入装配组合根 Agent 与装配参数对象 AgentConfig：
# Agent.load() 是项目目录装配的唯一入口，AgentConfig 承载五类配置的路径。
from app.ics_ai.agent import Agent, AgentConfig

# 从 channels 包导入控制台输入通道：控制台模式的人机交互载体。
from app.ics_ai.channels import ConsoleChannel

# 从 console_channel 模块导入控制台输出通道：包 __init__ 仅导出输入
# 通道，输出通道需从其定义模块直接导入。
from app.ics_ai.channels.console_channel import ConsoleOutputChannel

# 从 base_channel 导入用户消息 DTO：控制台交互回调的入参类型。
from app.ics_ai.channels.base_channel import UserMessage

# 从 api 包导入服务器：服务模式下暴露 REST/WebSocket/调试页。
from app.ics_ai.api import ICSServer

# 解析包目录：__main__.py 位于 src/app/ics_ecs 下，parent 即包目录——
# 它同时充当 Agent.load 的"项目目录"（domain/ data/flows/ actions/ 均在此）。
PACKAGE_DIR = Path(__file__).resolve().parent

# 解析仓库根目录：包目录向上三级（ics_ecs → app → src → 仓库根），
# configurations/ 目录（框架层 config.yml / endpoints.yml）固定位于此处。
REPO_ROOT = PACKAGE_DIR.parents[2]

# 框架配置目录：Agent.load 默认要求 config.yml 与项目同树，本工程将其
# 集中放在仓库根 configurations/ 下，装配时以绝对路径覆盖。
CONFIG_DIR = REPO_ROOT / "configurations"


def build_agent_config() -> AgentConfig:
    """构造 ics_ecs 的装配参数对象。

    核心功能：生成一份 AgentConfig，把"包内业务资产 + 包外框架配置"
    两处路径拼成 Agent.load 可消费的完整路径集。

    业务流程：
        1. domain/flows 保持包内相对路径（随包携带，属业务资产）；
        2. config/endpoints 用 configurations/ 下的绝对路径覆盖默认值
           （Path / 绝对路径 = 绝对路径本身，Agent.load 的同树约束被
           绕开而无需搬动文件）。

    副作用：无（仅构造数据对象，不做任何 IO）。

    :return: 配置齐备的 AgentConfig 实例。
    """
    # 构造并返回装配配置：四个路径键与 Agent.load 的五件套一一对应
    # （actions 路径未传——使用默认值 "actions"，恰好匹配包内目录名）。
    return AgentConfig(
        # domain_path：包内 domain/ 目录（4 个分片 yml，Domain.load 目录模式自动合并）。
        domain_path="domain",
        # flows_path：包内 data/flows/ 目录（3 个业务流程定义）。
        flows_path="data/flows",
        # config_path：框架配置绝对路径（pipeline/policies/retrieval 定义）。
        config_path=str(CONFIG_DIR / "config.yml"),
        # endpoints_path：端点配置绝对路径（LLM/嵌入/NLG/存储连接定义）。
        endpoints_path=str(CONFIG_DIR / "endpoints.yml"),
    )


def load_agent() -> Agent:
    """装配完整的电商客服 Agent。

    核心功能：以包目录为项目目录执行 Agent.load()，返回可直接对话
    的 Agent 实例。

    业务流程：
        1. build_agent_config() 取装配参数；
        2. Agent.load(PACKAGE_DIR, config=...) 依次完成 domain 合并、
           flows 加载、actions 扫描注册、LLM/策略/检索器/NLG 组装。

    副作用：见模块 docstring 装配期条目（读配置文件/动态导入动作模块/
            检索器连接失败降级）。

    :return: 装配完成的 Agent 实例（装配失败时异常向上传播）。
    """
    # 第一步：构造装配参数对象（包内相对路径 + 包外绝对路径）。
    config = build_agent_config()

    # 第二步：以包目录为"项目目录"执行装配——Agent.load 内部完成
    # domain/flows/actions/组件四段装配并注入构造函数。
    return Agent.load(PACKAGE_DIR, config=config)


async def run_console(agent: Agent) -> None:
    """以控制台交互模式运行机器人。

    核心功能：接通 ConsoleChannel 输入循环与 Agent.handle_message
    处理链，在终端完成多轮问答。

    业务流程：
        1. 创建输入/输出通道实例；
        2. 定义消息回调：UserMessage → Agent.handle_message（文本 +
           发送者 ID 透传，通道名写入 metadata）→ 逐条经输出通道打印；
        3. asyncio.run 驱动交互循环阻塞至用户退出（exit/quit/Ctrl+C）。

    副作用：读写 stdin/stdout；对话状态经内存 tracker_store 保存；
            对话触发的 LLM 调用/检索等在 handle_message 下游发生。

    :param agent: 已装配完成的 Agent 实例。
    :return: 无返回；用户退出后方法结束。
    """
    # 第一步：创建控制台输入通道（读取 stdin、识别退出命令）。
    channel = ConsoleChannel()

    # 第二步：创建配对的输出通道（渲染文本/按钮/图片占位到 stdout）。
    output_channel = ConsoleOutputChannel()

    # 第三步：定义消息处理回调——签名须匹配 run_interactive 的
    # Callable[[UserMessage], Awaitable[Any]] 契约。
    async def on_new_message(message: UserMessage) -> None:
        # 透传文本与发送者 ID 调用 Agent 核心入口；通道名入 metadata
        # 供 tracker 记录消息来源（对应 handle_message 的 metadata 键）。
        response = await agent.handle_message(
            message.text,
            sender_id=message.sender_id,
            metadata={"input_channel": message.input_channel},
        )

        # 遍历响应消息列表逐条渲染：MessageResponse.messages 为消息
        # 字典列表（text/buttons/image 键），单条经输出通道打印。
        for bot_message in response.messages:
            await output_channel.send_response(message.sender_id, bot_message)

    # 第四步：启动交互循环——协程由 asyncio.run 拉起，阻塞至退出命令。
    await channel.run_interactive(on_new_message)


def run_server(agent: Agent, args: argparse.Namespace) -> None:
    """以服务模式运行机器人。

    核心功能：把 Agent 挂到 ICSServer，暴露 REST/WebSocket 接口与
    /inspect 调试页。

    业务流程：构造 ICSServer（inspect 开关来自命令行）→ server.run
    启动 ASGI 服务并阻塞监听。

    副作用：监听 TCP 端口；tracker_store 仍为默认内存存储（进程退出即失）。

    :param agent: 已装配完成的 Agent 实例。
    :param args: 命令行参数（host/port/enable_inspect 取自此）。
    :return: 无返回；服务进程被终止后方法结束。
    """
    # 构造服务器：调试页默认关闭（--inspect 打开时暴露 /inspect 页面）。
    server = ICSServer(agent=agent, enable_inspect=args.enable_inspect)

    # 启动监听：host/port 透传命令行参数，方法内部阻塞直至进程退出。
    server.run(host=args.host, port=args.port)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    核心功能：定义并解析启动参数——运行模式（--console/--serve 互斥，
    默认控制台）、服务监听地址与端口、调试页开关。

    业务流程：构建 ArgumentParser → 添加互斥模式组与选项组 → parse_args。

    副作用：无（仅读 sys.argv，不触碰文件系统/网络）。

    :return: 解析后的参数命名空间（console/serve/host/port/enable_inspect）。
    """
    # 创建参数解析器：描述信息说明两种模式与默认行为。
    parser = argparse.ArgumentParser(
        prog="python -m app.ics_ecs",
        description="ics_ecs 电商客服机器人启动入口（默认控制台模式）",
    )

    # 运行模式互斥组：--console 与 --serve 二选一，dest 统一落两键。
    mode_group = parser.add_mutually_exclusive_group()

    # 控制台模式开关：终端交互问答（默认，靠 default=True 兜底）。
    mode_group.add_argument(
        "--console",
        action="store_true",
        default=True,
        help="控制台交互模式（默认）",
    )

    # 服务模式开关：起 REST/WebSocket 服务替代控制台。
    mode_group.add_argument(
        "--serve",
        action="store_true",
        help="服务模式：暴露 REST/WebSocket 接口",
    )

    # 监听地址选项：仅服务模式生效，默认回环地址。
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="服务监听地址（仅 --serve 生效）",
    )

    # 监听端口选项：仅服务模式生效，默认 5005（与 ics run 端口约定一致）。
    parser.add_argument(
        "--port",
        type=int,
        default=5005,
        help="服务监听端口（仅 --serve 生效）",
    )

    # 调试页开关：服务模式下暴露 /inspect 调试页面。
    parser.add_argument(
        "--inspect",
        dest="enable_inspect",
        action="store_true",
        help="服务模式下启用 /inspect 调试页",
    )

    # 执行解析并返回命名空间。
    return parser.parse_args()


def main() -> None:
    """程序主入口。

    核心功能：串起"解析参数 → 装配 Agent → 按模式运行"完整启动链。

    业务流程：
        1. 加载仓库根 .env 环境变量（DASHSCOPE_API_KEY / MYSQL_PASSWORD
           等敏感配置经此注入，供 endpoints.yml 的 ${VAR} 替换）；
        2. parse_args 取启动参数；--serve 显式选中时关闭 console 默认位；
        3. load_agent() 完成全量装配；
        4. 按模式分派：服务模式进 run_server，否则进 run_console。

    副作用：随装配与所选模式级联（见模块 docstring）。

    :return: 无返回；运行模式退出后进程随之结束。
    """
    # 第零步：加载 .env——优先仓库根的确定位置（cwd 无关），缺失时
    # 交由 dotenv 默认搜索兜底；${VAR} 替换发生在装配期，必须先于
    # load_agent 执行，否则密钥/密码全部解析为空串。
    repo_env = REPO_ROOT / ".env"
    if repo_env.exists():
        # 显式按路径加载（确定性优先，键值写入 os.environ）。
        dotenv.load_dotenv(repo_env)
    else:
        # 兜底：dotenv 从当前目录向上递归搜索 .env。
        dotenv.load_dotenv()

    # 第一步：解析命令行参数。
    args = parse_args()

    # 互斥修正：--serve 选中时同步关闭 console 默认位，保证分派唯一。
    if args.serve:
        args.console = False

    # 第二步：装配完整 Agent（domain/flows/actions/组件四段装配）。
    agent = load_agent()

    # 第三步：按模式分派运行。
    if args.serve:
        # 服务模式：挂载 ICSServer 并阻塞监听。
        run_server(agent, args)
    else:
        # 控制台模式：asyncio 驱动交互循环。
        asyncio.run(run_console(agent))


# 直接运行守卫：仅 python -m app.ics_ecs 时执行主入口（被导入不触发）。
if __name__ == "__main__":
    main()
