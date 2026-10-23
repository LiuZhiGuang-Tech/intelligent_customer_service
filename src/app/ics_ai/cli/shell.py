"""
shell - 交互对话命令模块

核心功能：提供 `ics shell` 命令与 InteractiveShell 交互循环——在
          终端以 REPL 方式与加载好的 Agent 对话测试，支持 /quit、
          /reset、/slots、/history、/debug 等 Shell 内建命令。

执行流程：1. click 解析 --model（项目目录）与 --sender-id（会话
          标识）；2. 延迟导入并 Agent.load 装配项目；3. 构造
          InteractiveShell 并 asyncio.run 进入交互循环；4. 循环体：
          读 stdin → 空行跳过 → "/" 前缀走 Shell 命令分派 → 其余
          作为对话消息经 agent.handle_message 异步处理；5. EOF
          （Ctrl+D）或 /quit 退出循环；6. 异常兜底——导入错误/启动
          失败转退出码（--debug 透传），单条消息错误仅回显不中断会话。

副作用：读项目目录（Agent.load 装配）；读写 stdin/stdout（对话
        交互）；经 agent.reset_tracker 重置 TrackerStore 中的会话
        状态；/debug 切换根 logger 级别（进程级副作用）。

主要依赖：click（命令框架）、asyncio（异步事件循环）、
          app.ics_ai.agent.agent（Agent 装配与消息处理，延迟导入）。
"""

# 启用延迟注解求值：类型注解按字符串存储，TYPE_CHECKING 中的 Agent
# 无需真实导入即可用于签名标注。
from __future__ import annotations

# 导入 asyncio：事件循环驱动（asyncio.run 启动会话、逐命令执行协程）。
import asyncio

# 导入 logging：/debug 命令切换根 logger 级别，本模块 logger 留痕异常。
import logging

# 从标准库 pathlib 导入 Path：项目目录路径的对象化运算。
from pathlib import Path

# 导入 click：命令装饰器、参数选项与终端输出。
import click

# 仅类型检查导入 Agent：运行时由 shell_command 延迟导入，
# InteractiveShell.__init__ 的 agent 注解经 __future__ 保留为字符串。
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # 类型命名空间导入：仅为注解提供 Agent 符号。
    from app.ics_ai.agent.agent import Agent

# 创建模块级 logger：消息处理异常经 logger.exception 记录完整堆栈。
logger = logging.getLogger(__name__)


class InteractiveShell:
    """交互式对话Shell。

    职责：封装终端 REPL 交互循环——输入读取、Shell 内建命令分派
          （退出/重置/槽位/历史/调试）、对话消息异步处理与响应
          渲染（文本/按钮/自定义数据）。

    设计模式：命令循环（REPL）——单一 run() 主循环 + 私有 _handle_*
              分派方法；对话能力全部委托 agent 门面，本类不持有
              对话状态（状态在 TrackerStore）。

    生命周期：shell_command 装配 Agent 后构造；随 asyncio.run 结束
              （/quit 或 EOF）销毁；running 标志控制循环退出。
    """

    def __init__(self, agent: Agent, sender_id: str = "shell_user") -> None:
        """初始化Shell。

        业务流程：1. 保存 Agent 门面引用；2. 记录会话标识（决定
                  TrackerStore 中的会话键）；3. 置运行标志为 True
                  （run 循环的退出开关）。

        副作用：仅写实例属性。

        :param agent: 已装配的 Agent 实例（经 Agent.load 构造）。
        :param sender_id: 发送者ID，作为会话标识写入 TrackerStore，
                          默认 "shell_user"。
        """
        # 保存 Agent 门面：后续消息处理/重置/查询全部经它转发。
        self.agent = agent

        # 会话标识：同一 TrackerStore 下与其他渠道会话隔离。
        self.sender_id = sender_id

        # 运行标志：/quit 命令置 False 使 run 循环自然退出。
        self.running = True

    async def run(self) -> None:
        """运行交互式会话。

        业务流程：1. 打印欢迎横幅与命令速查；2. 循环读取用户输入——
                  EOF（Ctrl+D）打印告别并退出；空行/纯空白跳过；
                  "/" 前缀转 Shell 命令分派；其余作为对话消息异步
                  处理；3. Ctrl+C 不退出（提示改用 /quit），其余
                  异常仅回显错误继续循环（单条失败不影响会话）。

        副作用：读 stdin（阻塞等待输入）；经分派方法产生对话/查询
                副作用；KeyboardInterrupt 被吞掉（协程内捕获）。

        :return: 无返回；running 置 False 或 EOF 后循环结束。
        """
        # 欢迎横幅与可用命令速查（降低使用门槛）。
        self._print_welcome()

        # 主循环：running 标志由 /quit 置 False 时终止。
        while self.running:
            try:
                # 同步读 stdin：input() 阻塞直至回车；EOF 返回 None。
                user_input = self._get_input()

                if user_input is None:
                    # EOF (Ctrl+D)：标准输入关闭，视作正常退出。
                    click.echo("\n再见!")
                    break

                # 去除首尾空白：纯空白输入等价于空行。
                user_input = user_input.strip()

                if not user_input:
                    # 空行：不响应也不退出，继续等待有效输入。
                    continue

                # "/" 前缀：Shell 内建命令（不进入对话管线）。
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                    continue

                # 对话消息：异步经 Agent 图管线处理并渲染响应。
                await self._handle_message(user_input)

            except KeyboardInterrupt:
                # Ctrl+C：与 /quit 区分——提示保留会话继续对话。
                click.echo("\n使用 /quit 或 Ctrl+D 退出")
            except Exception as e:
                # 其余异常：仅回显不中断——交互循环的容错底线。
                click.echo(f"错误: {e}", err=True)

    def _print_welcome(self) -> None:
        """打印欢迎信息。

        业务流程：空行分隔后输出横幅、使用说明与四个高频命令速查。

        副作用：写 stdout。

        :return: 无返回。
        """
        # 顶部空行与横幅。
        click.echo()
        click.echo("=" * 50)
        click.echo("ICS AI - 交互式对话")
        click.echo("=" * 50)
        click.echo()

        # 使用说明与命令速查。
        click.echo("输入消息开始对话，或使用以下命令:")
        click.echo("  /help    - 显示帮助信息")
        click.echo("  /reset   - 重置对话")
        click.echo("  /slots   - 显示当前槽位值")
        click.echo("  /quit    - 退出")
        click.echo()

    def _get_input(self) -> str | None:
        """获取用户输入。

        业务流程：input() 同步阻塞读取一行；EOFError（Ctrl+D 或
                  管道结束）转 None 供主循环识别退出。

        副作用：读 stdin；打印 "You: " 提示符。

        :return: 用户输入的原始行；EOF 时返回 None。
        """
        try:
            # 同步读取一行："You: " 作为输入提示符。
            return input("You: ")
        except EOFError:
            # Ctrl+D / 输入流结束：以 None 信号主循环退出。
            return None

    def _handle_command(self, command: str) -> None:
        """处理Shell命令。

        核心功能：解析 "/" 内建命令并分派到对应处理逻辑；异步子
                  命令（reset/slots/history）经 asyncio.run 就地
                  执行对应协程。

        业务流程：1. 小写化并取首段（忽略多余参数）；2. 按命令别名
                  分派——quit/exit/q 置 running=False；help 打印
                  帮助；reset/slots/history 分别执行异步协程；debug
                  切换日志级别；3. 未知命令提示查帮助。

        副作用：随各子命令产生（退出标志/TrackerStore 重置/日志
                级别切换/终端输出）。

        :param command: 以 "/" 开头的原始输入行（含可能的尾部参数）。
        :return: 无返回；未知命令仅回显提示。
        """
        # 小写化后取首段：实现 /QUIT、/slots 任意大小写与多段输入容错。
        cmd = command.lower().split()[0]

        if cmd in ["/quit", "/exit", "/q"]:
            # 退出命令：置运行标志，主循环条件判断后自然终止。
            self.running = False
            click.echo("再见!")

        elif cmd in ["/help", "/h", "/?"]:
            # 帮助命令：打印全部内建命令说明。
            self._print_help()

        elif cmd == "/reset":
            # 重置会话：异步协程经 asyncio.run 就地执行（事件循环收尾）。
            asyncio.run(self._reset_conversation())

        elif cmd == "/slots":
            # 槽位查询：同上就地执行协程。
            asyncio.run(self._show_slots())

        elif cmd == "/history":
            # 历史查询：同上就地执行协程。
            asyncio.run(self._show_history())

        elif cmd == "/debug":
            # 调试开关：同步方法直接调用。
            self._toggle_debug()

        else:
            # 未知命令：回显并引导查帮助（不当作对话消息处理）。
            click.echo(f"未知命令: {cmd}")
            click.echo("使用 /help 查看可用命令")

    def _print_help(self) -> None:
        """打印帮助信息。

        业务流程：空行分隔后列出六组内建命令及其别名与用途。

        副作用：写 stdout。

        :return: 无返回。
        """
        # 命令清单：与 _handle_command 的分派分支一一对应。
        click.echo()
        click.echo("可用命令:")
        click.echo("  /help, /h, /?  - 显示帮助信息")
        click.echo("  /reset         - 重置对话，清除会话状态")
        click.echo("  /slots         - 显示当前所有槽位值")
        click.echo("  /history       - 显示对话历史")
        click.echo("  /debug         - 切换调试模式")
        click.echo("  /quit, /exit   - 退出程序")
        click.echo()

    async def _reset_conversation(self) -> None:
        """重置对话。

        业务流程：经 agent.reset_tracker 删除本会话的 Tracker（槽位、
                  对话栈、历史全部清空，下次消息重建新会话）。

        副作用：写 TrackerStore（删除会话记录）。

        :return: 无返回；失败时错误回显（不抛出，保持会话可用）。
        """
        try:
            # 委托 Agent 门面重置：TrackerStore 按本会话 sender_id 删除。
            await self.agent.reset_tracker(self.sender_id)
            click.echo("对话已重置")
        except Exception as e:
            # 失败回显：重置失败不应终止 Shell 进程。
            click.echo(f"重置失败: {e}", err=True)

    async def _show_slots(self) -> None:
        """显示槽位值。

        业务流程：1. 经 agent.get_tracker 取本会话 Tracker（不存在
                  说明尚未对话）；2. get_all_slots 取槽位名到值的
                  映射；3. 逐项打印——None 显示 "(未设置)" 占位。

        副作用：读 TrackerStore；写 stdout。

        :return: 无返回；查询失败时错误回显。
        """
        try:
            # 取本会话 Tracker：未发过消息时会话尚未创建，返回 None。
            tracker = await self.agent.get_tracker(self.sender_id)
            if tracker:
                # 读取全部槽位值：键为槽名，值为当前值（可能 None）。
                slots = tracker.get_all_slots()
                click.echo()
                click.echo("当前槽位值:")
                # 逐项打印：None 与已设值区分展示。
                for name, value in slots.items():
                    if value is not None:
                        click.echo(f"  {name}: {value}")
                    else:
                        click.echo(f"  {name}: (未设置)")
                click.echo()
            else:
                # 会话不存在：提示先发一条消息再查询。
                click.echo("会话不存在")
        except Exception as e:
            # 查询失败回显：不中断 Shell。
            click.echo(f"获取槽位失败: {e}", err=True)

    async def _show_history(self) -> None:
        """显示对话历史。

        业务流程：1. 经 agent.get_tracker 取本会话 Tracker；2.
                  get_messages_for_llm 取扁平化消息序列（user/
                  assistant 交替，含正文）；3. 逐条打印角色与正文
                  （正文截断 100 字符防刷屏）。

        副作用：读 TrackerStore；写 stdout。

        :return: 无返回；查询失败时错误回显。
        """
        try:
            # 取本会话 Tracker：未发过消息时为 None。
            tracker = await self.agent.get_tracker(self.sender_id)
            if tracker:
                # 扁平消息序列：与 LLM 提示词同源（role/content 键）。
                messages = tracker.get_messages_for_llm()
                click.echo()
                click.echo("对话历史:")
                # 逐条打印：序号从 1 起；角色映射 You/Bot；正文截断 100。
                for i, msg in enumerate(messages):
                    role = "You" if msg.get("role") == "user" else "Bot"
                    text = msg.get("content", "")
                    click.echo(f"  [{i + 1}] {role}: {text[:100]}...")
                click.echo()
            else:
                # 会话不存在：提示先发一条消息再查询。
                click.echo("会话不存在")
        except Exception as e:
            # 查询失败回显：不中断 Shell。
            click.echo(f"获取历史失败: {e}", err=True)

    def _toggle_debug(self) -> None:
        """切换调试模式。

        核心功能：在根 logger 的 DEBUG/INFO 两级间翻转——开启后
                  框架内部 DEBUG 日志（LLM 请求、命令解析等）实时
                  可见。

        业务流程：读根 logger 当前级别，DEBUG 则降回 INFO 并提示
                  关闭，否则升至 DEBUG 并提示开启。

        副作用：修改根 logger 级别（进程级，影响所有模块的日志
                输出），Shell 退出后不自动还原。

        :return: 无返回。
        """
        # 读取根 logger 当前级别：全局唯一，框架各模块共用。
        current_level = logging.getLogger().level

        if current_level == logging.DEBUG:
            # 当前已调试：降回 INFO（关闭）。
            logging.getLogger().setLevel(logging.INFO)
            click.echo("调试模式: 关闭")
        else:
            # 当前非调试：升至 DEBUG（开启）。
            logging.getLogger().setLevel(logging.DEBUG)
            click.echo("调试模式: 开启")

    async def _handle_message(self, text: str) -> None:
        """处理用户消息。

        业务流程：1. 经 agent.handle_message 走完整图管线（理解→
                  策略→动作→响应）拿 MessageResponse DTO；2. 遍历
                  DTO 的 messages 列表逐条渲染——每项 dict 拆解
                  text/buttons/custom 三类字段；3. 空消息列表打印
                  占位提示。

        副作用：推进对话状态（TrackerStore 读写、动作执行、可能的
                LLM 调用）；写 stdout。

        :param text: 用户输入的对话消息正文（已去首尾空白）。
        :return: 无返回；管线异常时错误回显并记录堆栈，不中断会话。
        """
        try:
            # 发送消息到 Agent：异步走图管线，返回统一响应 DTO
            # （messages 为回复列表，metadata 携带轨迹/错误）。
            response = await self.agent.handle_message(
                message=text,
                sender_id=self.sender_id,
            )

            # 逐条渲染 DTO 的消息列表：每项为 {"text": ...} 及
            # 可选 buttons/custom 键的字典。
            for msg in response.messages:
                # 正文：结构化响应的 text 字段（可能为空）。
                text = msg.get("text", "")
                if text:
                    click.echo(f"Bot: {text}")

                # 按钮：选项列表，仅打印标题（编号供用户参考）。
                buttons = msg.get("buttons", [])
                if buttons:
                    click.echo("  选项:")
                    for i, btn in enumerate(buttons, 1):
                        title = btn.get("title", "")
                        click.echo(f"    [{i}] {title}")

                # 自定义数据：非文本扩展载荷（如卡片）原样展示。
                custom = msg.get("custom")
                if custom:
                    click.echo(f"  (自定义: {custom})")

            if not response.messages:
                # 空响应：管线未产出任何回复（如仅内部动作）时占位。
                click.echo("Bot: (无响应)")

        except Exception as e:
            # 管线异常：终端回显简因 + 日志记录完整堆栈（排障依据）。
            click.echo(f"处理消息失败: {e}", err=True)
            logger.exception("Message handling error")


@click.command("shell", help="交互式对话测试")
@click.option(
    "--model",
    "-m",
    type=click.Path(exists=True),
    default=".",
    help="模型或项目目录路径",
)
@click.option(
    "--sender-id",
    type=str,
    default="shell_user",
    help="发送者ID",
)
@click.pass_context
def shell_command(
    ctx: click.Context,
    model: str,
    sender_id: str,
) -> None:
    """启动交互式对话Shell。

    核心功能：装配指定项目并进入终端对话 REPL，实时测试对话效果。

    业务流程：1. 从上下文取 verbose/debug 标志；2. 延迟导入 Agent
              并 Agent.load 装配；3. 以 sender_id 构造 InteractiveShell；
              4. asyncio.run 进入交互循环（阻塞至 /quit 或 EOF）；
              5. 异常兜底——导入错误/启动失败转退出码（--debug 透传）。

    副作用：装配读项目目录；交互读写 stdin/stdout；会话状态写入
            TrackerStore（由 InteractiveShell 转发）。

    :param ctx: click 上下文（承载主组的 verbose/debug 标志）。
    :param model: 项目目录路径（exists=True 已由 click 校验存在）。
    :param sender_id: 会话标识，默认 "shell_user"。
    :return: 无返回；失败路径抛 SystemExit(1)（debug 时透传原异常）。
    """
    # 读取主组注入的调试标志：verbose 留作细粒度输出扩展位。
    verbose = ctx.obj.get("verbose", False)

    # debug 标志控制异常处理策略：False 转为退出码，True 透传堆栈。
    debug = ctx.obj.get("debug", False)

    # 解析项目路径：字符串转 Path 对象（回显绝对路径用）。
    model_path = Path(model)

    try:
        # 延迟导入 Agent：仅执行本命令时才加载装配链重依赖。
        from app.ics_ai.agent.agent import Agent

        # 装配项目：domain/flows/actions/config/endpoints 全链加载。
        click.echo(f"加载模型: {model_path.absolute()}")
        agent = Agent.load(str(model_path))
        click.echo("模型加载完成")

        # 创建 Shell：绑定 Agent 门面与会话标识。
        shell = InteractiveShell(agent, sender_id)

        # 运行 Shell：asyncio.run 启动事件循环并阻塞至会话结束。
        asyncio.run(shell.run())

    except ImportError as e:
        # 导入错误单列：多为依赖缺失/装配断裂，提示语与运行错误区分。
        click.echo(f"导入错误: {e}", err=True)
        if debug:
            raise
        raise SystemExit(1)
    except Exception as e:
        # 兜底其余异常：打印错误；debug 透传堆栈，普通模式退出码 1。
        click.echo(f"启动Shell失败: {e}", err=True)
        if debug:
            raise
        raise SystemExit(1)


# 导出清单：cli 聚合层按名导入，InteractiveShell 同时暴露供单测复用。
__all__ = ["shell_command", "InteractiveShell"]
