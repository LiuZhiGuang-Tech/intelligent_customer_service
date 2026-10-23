"""
run - 服务运行命令模块

核心功能：提供 `ics run` 命令——装配指定项目并启动生产形态的
          API 服务（REST 对话接口 + Socket.IO 实时通道 + 可选
          /inspect 调试页面），是部署运行的标准入口。

执行流程：1. click 解析 --model（项目目录）、--host/--port（监听
          地址）、--cors（跨域白名单，可多值）、--enable-api/
          --enable-inspect 开关与 --channel（通道选择）；2. 延迟
          导入并 Agent.load 装配项目；3. 构造 ICSServer（透传 CORS
          与调试页开关）；4. 回显访问入口后 server.run 阻塞运行
          直至 Ctrl+C；5. 异常分流——KeyboardInterrupt 视为正常
          停止，导入错误/运行异常按兜底策略转退出码（--debug 透传
          堆栈）。

副作用：读项目目录（Agent.load 装配）；监听 TCP 端口起 HTTP/WS
        服务；TrackerStore 随请求读写会话状态。

主要依赖：click（命令框架）、app.ics_ai.agent.agent（Agent 装配，
          延迟导入）、app.ics_ai.api.server（ICSServer 服务，延迟
          导入）、app.ics_ai.shared.constants（默认 host/port）。
"""

# 启用延迟注解求值：类型注解按字符串存储，Agent/ICSServer 等延迟
# 导入的符号无需出现在模块级命名空间。
from __future__ import annotations

# 导入 logging：创建模块级 logger（异常留痕扩展位）。
import logging

# 从标准库 pathlib 导入 Path：项目目录路径的对象化运算。
from pathlib import Path

# 导入 click：命令装饰器、参数选项与终端输出。
import click

# 导入默认监听配置：host/port 形参缺省值来源（与 inspect 命令一致）。
from app.ics_ai.shared.constants import (
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
)

# 创建模块级 logger：供后续细粒度日志留痕扩展。
logger = logging.getLogger(__name__)


@click.command("run", help="运行对话服务")
@click.option(
    "--model",
    "-m",
    type=click.Path(exists=True),
    default=".",
    help="项目目录路径（含 config.yml/domain/flows，模型压缩包模式已移除）",
)
@click.option(
    "--host",
    "-H",
    type=str,
    default=DEFAULT_SERVER_HOST,
    help="服务器监听地址",
)
@click.option(
    "--port",
    "-p",
    type=int,
    default=DEFAULT_SERVER_PORT,
    help="服务器监听端口",
)
@click.option(
    "--cors",
    type=str,
    multiple=True,
    default=["*"],
    help="CORS允许的源 (可多次指定)",
)
@click.option(
    "--enable-api/--disable-api",
    default=True,
    help="启用/禁用REST API",
)
@click.option(
    "--enable-inspect/--disable-inspect",
    default=True,
    help="启用/禁用调试页面",
)
@click.option(
    "--channel",
    type=click.Choice(["rest", "socketio", "all"]),
    default="all",
    help="启用的通道类型",
)
@click.pass_context
def run_command(
    ctx: click.Context,
    model: str,
    host: str,
    port: int,
    cors: tuple[str, ...],
    enable_api: bool,
    enable_inspect: bool,
    channel: str,
) -> None:
    """运行对话服务。

    核心功能：装配项目并阻塞运行 API 服务（REST + WebSocket），
              直至用户 Ctrl+C 中断。

    业务流程：1. 从上下文取 verbose/debug 标志；2. 打印命令横幅、
              解析项目路径并回显关键参数；3. 延迟导入 Agent/ICSServer；
              4. Agent.load 装配（只接受项目目录，非目录由其统一
              抛 FileNotFoundError）；5. 构造 ICSServer（透传 CORS
              与调试页开关）；6. 回显服务入口；7. 阻塞运行服务；
              8. 异常分流兜底。

    副作用：装配读项目目录；监听端口起服务；随请求读写会话状态。

    :param ctx: click 上下文（承载主组的 verbose/debug 标志）。
    :param model: 项目目录路径（exists=True 已由 click 校验存在）。
    :param host: 服务器监听地址（默认取 DEFAULT_SERVER_HOST）。
    :param port: 服务器监听端口（默认取 DEFAULT_SERVER_PORT）。
    :param cors: CORS 允许源元组（multiple=True，默认 ["*"] 全放行）。
    :param enable_api: REST API 开关（True 启用；ICSServer 默认即
                       启用，False 场景经服务端独立配置）。
    :param enable_inspect: 调试页面开关（True 挂载 /inspect）。
    :param channel: 通道类型（rest/socketio/all；当前实现由 ICSServer
                    统一挂载两类通道，此参数作前端选择预留）。
    :return: 无返回；导入/运行失败抛 SystemExit(1)（debug 透传），
             Ctrl+C 视为正常停止。
    """
    # 读取主组注入的调试标志：verbose 留作细粒度输出扩展位。
    verbose = ctx.obj.get("verbose", False)

    # debug 标志控制异常处理策略：False 转为退出码，True 透传堆栈。
    debug = ctx.obj.get("debug", False)

    # 打印命令横幅：分隔线 + 标题。
    click.echo("=" * 50)
    click.echo("ICS AI - 对话服务")
    click.echo("=" * 50)

    # 解析项目路径：字符串转 Path 对象（回显绝对路径用）。
    model_path = Path(model)

    # 回显关键参数：项目路径/服务地址/调试页开关/CORS 白名单。
    click.echo(f"模型路径: {model_path.absolute()}")
    click.echo(f"服务地址: http://{host}:{port}")
    click.echo(f"调试页面: {'启用' if enable_inspect else '禁用'}")
    click.echo(f"CORS来源: {list(cors)}")
    click.echo()

    try:
        # 延迟导入核心构件：仅执行本命令时才加载装配链重依赖。
        from app.ics_ai.agent.agent import Agent
        from app.ics_ai.api.server import ICSServer

        # 装配项目：domain/flows/actions/config/endpoints 全链加载。
        click.echo("加载Agent...")

        # Agent.load 只接受项目目录（模型压缩包加载能力已随 training
        # 子包移除）；非目录/不存在路径由其统一抛 FileNotFoundError，
        # 落入下方通用异常处理退出。
        agent = Agent.load(str(model_path))

        click.echo("Agent加载完成")

        # 创建服务器：CORS 元组转列表传参；调试页开关直接透传
        # （True 挂载 /inspect，False 减少暴露面）。
        click.echo("启动服务器...")
        server = ICSServer(
            agent=agent,
            cors_origins=list(cors),
            enable_inspect=enable_inspect,
        )

        # 回显服务入口与停止方式：调试页仅在启用时展示。
        click.echo()
        click.echo(f"服务已启动: http://{host}:{port}")
        click.echo(f"API文档: http://{host}:{port}/docs")
        if enable_inspect:
            click.echo(f"调试页面: http://{host}:{port}/inspect")
        click.echo("按 Ctrl+C 停止服务")
        click.echo()

        # 阻塞运行服务：uvicorn 事件循环在此持续，直到 Ctrl+C 中断
        # （KeyboardInterrupt 由下方 except 捕获转为正常停止提示）。
        server.run(host=host, port=port)

    except KeyboardInterrupt:
        # 用户主动中断：视为正常停止，礼貌提示后返回（退出码 0）。
        click.echo("\n服务已停止")
    except ImportError as e:
        # 导入错误单列：多为依赖缺失/装配断裂，提示语与运行错误区分。
        click.echo(f"导入错误: {e}", err=True)
        if debug:
            raise
        raise SystemExit(1)
    except Exception as e:
        # 兜底其余异常：打印错误；debug 透传堆栈，普通模式退出码 1。
        click.echo(f"运行失败: {e}", err=True)
        if debug:
            raise
        raise SystemExit(1)


# 导出清单：cli 聚合层按名导入本命令的唯一公开符号。
__all__ = ["run_command"]
