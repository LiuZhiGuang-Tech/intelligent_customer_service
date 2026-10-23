"""
cli - 命令行入口模块

核心功能：聚合 CLI 主组与五大子命令——init（项目脚手架）、run
          （生产服务）、shell（终端对话 REPL）、export（打包导出）、
          inspect（调试页面）；main() 是 `python -m app.ics_ai` 的
          最终汇入点，负责 .env 加载、sys.path 注入与命令注册分发。

执行流程：1. 模块导入：定义 ICS CLI 主组（不注册子命令）；2. main()
          被调用：加载 cwd 的 .env → 注入 sys.path → 延迟导入五个
          子命令并逐个 add_command → cli(args) 进入 click 分发；
          3. 主组回调把 --verbose/--debug 写入 ctx.obj（子命令经
          pass_context 读取）；4. 异常兜底——普通模式打印错误退出
          码 1，命令行带 --debug 时透传原始堆栈。

副作用：读 .env 并写入进程环境变量；修改 sys.path；随子命令产生
        装配/服务/交互等下游副作用；--debug/--verbose 配置根 logger
        级别（进程级）。

主要依赖：click（命令组与分发）、dotenv（.env 加载）、platform/sys
          （--version 环境信息）、app.ics_ai.__version__（版本号）、
          五个子命令模块（延迟导入）。
"""

# 导入 sys：--version 输出解释器路径、main 中注入 sys.path 与退出。
import sys

# 导入 platform：--version 输出 Python 版本与操作系统信息。
import platform

# 导入 click：命令组装饰器、选项解析与终端输出。
import click

# 导入包版本号：--version 的第一行输出。
from app.ics_ai import __version__


# 自定义Click组，用于美化帮助信息
class ICSCLI(click.Group):
    """自定义CLI组。

    职责：覆写帮助排版——在标准命令列表前插入品牌横幅（名称与
          定位语），其余行为完全继承 click.Group。

    设计模式：模板方法——只覆写 format_help 一个钩子，分发逻辑
              零改动。

    生命周期：模块导入时实例化为 cli 主组，进程结束随模块销毁。
    """

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """格式化帮助信息。

        业务流程：先写入品牌横幅段落，再委托父类输出用法/命令清单
                  等标准段落。

        :param ctx: click 上下文（帮助渲染请求来源）。
        :param formatter: click 帮助格式化器（写入目标）。
        :return: 无返回。
        """
        # 空行分隔：让横幅与 Usage 段保持呼吸感。
        formatter.write_paragraph()

        # 品牌横幅：产品名与定位说明（两行文本）。
        formatter.write_text(
            "ICS AI - 教学版对话系统 CLI\nLLM驱动的对话系统，用于学习和教学。"
        )

        # 再空一行，随后交给父类渲染标准帮助内容。
        formatter.write_paragraph()
        super().format_help(ctx, formatter)


def print_version(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    """打印版本信息。

    核心功能：作为 --version 选项的回调，输出框架版本与运行环境
              四元信息后立即退出进程。

    业务流程：1. value 为 False（未传 --version）或 resilient 解析
              模式（补全场景）直接返回不干预；2. 依次打印框架版本/
              Python 版本/操作系统/解释器路径；3. ctx.exit() 以 0
              退出（is_eager 保证先于子命令解析执行）。

    副作用：写 stdout；终止进程。

    :param ctx: click 上下文。
    :param param: 触发本回调的 --version 选项对象（未使用，签名要求）。
    :param value: 选项值（is_flag：传了为 True）。
    :return: 无返回；--version 场景不返回（进程在 ctx.exit 退出）。
    """
    # 未传 --version 或处于补全等宽松解析：静默放行，不做任何事。
    if not value or ctx.resilient_parsing:
        return

    # 四元环境信息：框架版本/Python 版本/操作系统/解释器绝对路径。
    click.echo(f"ICS AI Version : {__version__}")
    click.echo(f"Python Version     : {platform.python_version()}")
    click.echo(f"Operating System   : {platform.platform()}")
    click.echo(f"Python Path        : {sys.executable}")

    # 信息打印完毕立即退出：不进入命令分发。
    ctx.exit()


@click.group(cls=ICSCLI)
@click.option(
    "--version",
    "-V",
    is_flag=True,
    callback=print_version,
    expose_value=False,
    is_eager=True,
    help="显示版本信息",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="启用详细日志输出")
@click.option("--debug", is_flag=True, default=False, help="启用调试模式")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, debug: bool) -> None:
    """ICS AI 命令行工具

    核心功能：CLI 主组回调——把全局开关写入 ctx.obj 供子命令读取，
              并按开关配置根 logger 级别。

    业务流程：1. ensure_object 保证 ctx.obj 为字典（防直调场景
              为 None）；2. 写入 verbose/debug 两键（五个子命令的
              异常兜底策略与输出粒度依赖它们）；3. 按开关配置日志
              ——--debug 设 DEBUG 级，--verbose 设 INFO 级（均延迟
              导入 logging，未开启时不产生日志配置副作用）。

    副作用：修改 ctx.obj（上下文内共享）；--debug/--verbose 时
            basicConfig 配置根 logger（进程级，仅首次生效）。

    :param ctx: click 主组上下文。
    :param verbose: 详细日志开关（--verbose/-v）。
    :param debug: 调试开关（--debug）。
    :return: 无返回。
    """
    # 确保ctx.obj存在：子命令经 pass_context 读取，缺省兜底为空字典。
    ctx.ensure_object(dict)

    # 写入全局开关：子命令读取键约定为 "verbose"/"debug"。
    ctx.obj["verbose"] = verbose
    ctx.obj["debug"] = debug

    # 配置日志级别：仅在实际开启时才触碰 logging（避免默认运行时
    # 改动日志行为）；局部延迟导入保持模块导入轻量。
    if debug:
        # 调试模式：全库 DEBUG 级（LLM 请求、命令解析等全量可见）。
        import logging

        logging.basicConfig(level=logging.DEBUG)
    elif verbose:
        # 详细模式：INFO 级（关键流程可见）。
        import logging

        logging.basicConfig(level=logging.INFO)


def main(args: list[str] | None = None) -> None:
    """CLI主入口函数。

    核心功能：`python -m app.ics_ai` 的最终汇入点——准备运行环境
              （.env/sys.path）、注册五大子命令、进入 click 分发。

    业务流程：1. 加载 cwd 的 .env（显式路径优先，缺失回退 dotenv
              默认搜索）；2. 把 cwd 插入 sys.path[0]（用户项目内
              自定义模块可被导入）；3. 延迟导入五个子命令模块并
              逐个 add_command（CLI 命令集在分发前就绪）；4.
              cli(args) 进入分发；5. 异常兜底——打印错误后按
              "--debug 是否在命令行" 决定透传堆栈或退出码 1。

    副作用：写进程环境变量（.env）；修改 sys.path；注册命令修改
            cli 组；下游子命令的副作用经分发触发。

    :param args: 模拟命令行参数列表（None 时 click 自行读 sys.argv；
                 单测/编程调用可显式传入）。
    :return: 无返回；分发失败退出码 1，--debug 场景透传原异常。
    """
    # 导入 dotenv：.env 文件加载器（键值对写入 os.environ）。
    from pathlib import Path
    import dotenv

    # 自动加载用户工程目录下的 .env 文件
    # 用户可在工程根目录放置 .env 文件配置 API KEY 等环境变量
    # 显式指定当前工作目录的 .env 文件路径，确保在 Windows 上正确加载
    cwd = Path.cwd()

    # 探测 cwd/.env：存在则按显式路径加载（确定性优先）。
    env_file = cwd / ".env"
    if env_file.exists():
        dotenv.load_dotenv(env_file)
    else:
        # 如果当前目录没有，尝试默认搜索（dotenv 自身向上查找）。
        dotenv.load_dotenv()

    # 将当前目录添加到Python路径，以便导入自定义模块
    sys.path.insert(0, str(cwd))

    try:
        # 导入并注册子命令：延迟到分发前（保持包导入轻量，
        # 各子命令的重依赖在其函数体内进一步延迟）。
        from app.ics_ai.cli.init import init_command
        from app.ics_ai.cli.run import run_command
        from app.ics_ai.cli.shell import shell_command
        from app.ics_ai.cli.export import export_command
        from app.ics_ai.cli.inspect import inspect_command

        # 逐个注册：顺序即 --help 呈现顺序。
        cli.add_command(init_command)
        cli.add_command(run_command)
        cli.add_command(shell_command)
        cli.add_command(export_command)
        cli.add_command(inspect_command)

        # 执行CLI：进入 click 分发（解析参数 → 匹配子命令 → 执行）。
        cli(args)
    except Exception as e:
        # 异常兜底：打印错误；命令行带 --debug 时透传原始堆栈
        # （args 为 None 时回退检查真实 argv），否则退出码 1。
        click.echo(f"错误: {e}", err=True)
        if "--debug" in (args or sys.argv):
            raise
        sys.exit(1)


# 导出清单：按 ASCII 字母序排列，供包外按名导入与 IDE 补全。
__all__ = [
    "ICSCLI",
    "cli",
    "main",
]
