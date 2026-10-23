"""
inspect - 调试命令模块

核心功能：提供 `ics inspect` 命令——加载指定项目并启动带调试页面
          的 API 服务（/inspect 可视化对话调试界面），支持自动拉起
          浏览器访问。

执行流程：1. click 解析 --model（项目目录）、--host/--port（监听
          地址）、--no-browser（禁用自动开浏览器）、--cors（跨域
          白名单，可多值）；2. 延迟导入并 Agent.load 装配项目；3.
          构造 ICSServer（enable_inspect=True 强制挂调试页）；4.
          非 --no-browser 时启动守护线程延迟 1.5 秒打开浏览器（等
          服务器就绪）；5. server.run 阻塞运行直至 Ctrl+C；6. 异常
          分流——KeyboardInterrupt 视为正常停止，导入错误/运行异常
          按兜底策略转退出码（--debug 透传堆栈）。

副作用：读项目目录（Agent.load 装配）；监听 TCP 端口起 HTTP/WS
        服务；无浏览器模式下调用系统默认浏览器；线程仅用于延迟
        开页（守护线程，主线程退出即消亡）。

主要依赖：click（命令框架）、webbrowser/threading（开浏览器）、
          app.ics_ai.agent.agent（Agent 装配，延迟导入）、
          app.ics_ai.api.server（ICSServer 服务，延迟导入）、
          app.ics_ai.shared.constants（默认 host/port）。
"""

# 启用延迟注解求值：类型注解按字符串存储，Agent/ICSServer 等延迟
# 导入的符号无需出现在模块级命名空间。
from __future__ import annotations

# 导入 logging：创建模块级 logger（当前主要依赖异常兜底外的留痕扩展位）。
import logging

# 导入 webbrowser：服务启动后调用系统默认浏览器打开调试页面。
import webbrowser

# 从标准库 pathlib 导入 Path：项目目录路径的对象化运算。
from pathlib import Path

# 导入 click：命令装饰器、参数选项与终端输出。
import click

# 导入默认监听配置：host/port 形参缺省值来源（与 run 命令保持一致）。
from app.ics_ai.shared.constants import (
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
)

# 创建模块级 logger：供后续细粒度日志留痕扩展。
logger = logging.getLogger(__name__)


@click.command("inspect", help="启动调试页面")
@click.option(
    "--model",
    "-m",
    type=click.Path(exists=True),
    default=".",
    help="模型或项目目录路径",
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
    "--no-browser",
    is_flag=True,
    default=False,
    help="不自动打开浏览器",
)
@click.option(
    "--cors",
    type=str,
    multiple=True,
    default=["*"],
    help="CORS允许的源",
)
@click.pass_context
def inspect_command(
    ctx: click.Context,
    model: str,
    host: str,
    port: int,
    no_browser: bool,
    cors: tuple[str, ...],
) -> None:
    """启动Inspect调试页面。

    核心功能：装配指定项目并启动带 /inspect 调试页的 API 服务，
              提供可视化对话调试能力。

    业务流程：1. 从上下文取 verbose/debug 标志；2. 解析项目路径并
              回显参数横幅；3. 延迟导入 Agent/ICSServer 并装配；4.
              构造服务端（强制 enable_inspect=True）；5. 计算页面/
              浏览器 URL（0.0.0.0 特判为 localhost）；6. 非 no-browser
              时起守护线程延迟开浏览器；7. 阻塞运行服务直至 Ctrl+C；
              8. 异常分流兜底。

    副作用：装配读项目目录；监听端口起服务；自动开浏览器（可禁）。

    :param ctx: click 上下文（承载主组的 verbose/debug 标志）。
    :param model: 项目目录路径（exists=True 已由 click 校验存在）。
    :param host: 服务器监听地址（默认取 DEFAULT_SERVER_HOST）。
    :param port: 服务器监听端口（默认取 DEFAULT_SERVER_PORT）。
    :param no_browser: True 时不自动打开浏览器。
    :param cors: CORS 允许源元组（multiple=True，默认 ["*"] 全放行）。
    :return: 无返回；导入/运行失败抛 SystemExit(1)（debug 透传），
             Ctrl+C 视为正常停止。
    """
    # 读取主组注入的调试标志：verbose 留作细粒度输出扩展位。
    verbose = ctx.obj.get("verbose", False)

    # debug 标志控制异常处理策略：False 转为退出码，True 透传堆栈。
    debug = ctx.obj.get("debug", False)

    # 解析项目路径：字符串转 Path 对象（回显绝对路径用）。
    model_path = Path(model)

    # 打印命令横幅：分隔线 + 标题。
    click.echo("=" * 50)
    click.echo("ICS AI - Inspect 调试页面")
    click.echo("=" * 50)

    # 回显关键参数：项目绝对路径与服务访问地址（用户可核对）。
    click.echo(f"模型路径: {model_path.absolute()}")
    click.echo(f"服务地址: http://{host}:{port}")
    click.echo()

    try:
        # 延迟导入核心构件：仅执行本命令时才加载装配链重依赖。
        from app.ics_ai.agent.agent import Agent
        from app.ics_ai.api.server import ICSServer

        # 装配项目：domain/flows/actions/config/endpoints 全链加载。
        click.echo("加载Agent...")
        agent = Agent.load(str(model_path))
        click.echo("Agent加载完成")

        # 构造服务端：cors 元组转列表传参；enable_inspect 强制开启
        # ——调试页是本命令的存在意义，忽略全局默认开关。
        server = ICSServer(
            agent=agent,
            cors_origins=list(cors),
            enable_inspect=True,
        )

        # 服务端实际监听的调试页 URL（回显用原始 host）。
        inspect_url = f"http://{host}:{port}/inspect"

        # 浏览器 URL 使用 localhost（0.0.0.0 在浏览器中无法访问）。
        browser_host = "localhost" if host == "0.0.0.0" else host

        # 拼装浏览器实际访问的调试页地址。
        browser_url = f"http://{browser_host}:{port}/inspect"

        # 回显访问入口与停止方式。
        click.echo()
        click.echo(f"Inspect页面: {inspect_url}")
        click.echo(f"API文档: http://{host}:{port}/docs")
        click.echo("按 Ctrl+C 停止服务")
        click.echo()

        # 自动打开浏览器（--no-browser 跳过）。
        if not no_browser:
            # 延迟导入 threading：仅自动开页模式需要线程。
            import threading

            def open_browser():
                """延迟打开浏览器。

                业务流程：睡眠 1.5 秒等待服务器完成监听绑定，再调
                系统默认浏览器访问调试页。运行于守护线程，服务停止
                时随主线程消亡，无需显式回收。

                副作用：调用操作系统默认浏览器打开 browser_url。
                """
                # 延迟导入 time：仅本闭包使用睡眠计时。
                import time

                # 等待服务器启动：避免浏览器先于监听绑定拿到连接拒绝。
                time.sleep(1.5)  # 等待服务器启动

                # 调系统默认浏览器打开调试页（失败仅静默，不影响服务）。
                webbrowser.open(browser_url)

            # 以守护线程启动：不阻塞主线程的服务运行循环。
            threading.Thread(target=open_browser, daemon=True).start()

        # 阻塞运行服务：uvicorn 事件循环在此持续，直到 Ctrl+C 中断
        # （KeyboardInterrupt 由下方 except 捕获转为正常停止提示）。
        server.run(host=host, port=port)

    except KeyboardInterrupt:
        # 用户主动中断：视为正常停止，礼貌提示后返回（退出码 0）。
        click.echo("\n服务已停止")
    except ImportError as e:
        # 导入错误单列：提示语与运行错误区分（多为依赖缺失/装配断裂）。
        click.echo(f"导入错误: {e}", err=True)
        if debug:
            raise
        raise SystemExit(1)
    except Exception as e:
        # 兜底其余异常：打印错误；debug 透传堆栈，普通模式退出码 1。
        click.echo(f"启动失败: {e}", err=True)
        if debug:
            raise
        raise SystemExit(1)


# 导出清单：cli 聚合层按名导入本命令的唯一公开符号。
__all__ = ["inspect_command"]
