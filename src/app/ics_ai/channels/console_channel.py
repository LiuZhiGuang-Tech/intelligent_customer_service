"""
console_channel - 控制台通道模块
核心功能：面向命令行交互的通道实现——ConsoleChannel 运行交互式
          读取循环（stdin 阻塞读取经线程池转异步），将用户输入封装
          为 UserMessage 交给处理回调；ConsoleOutputChannel 将机器人
          消息（文本/按钮/图片）打印到 stdout，构成最小可运行的
          本地对话体验。
执行流程：1. 应用启动时以 ConsoleChannel 为输入通道；2. run_interactive
          打印欢迎横幅后进入主循环；3. 每轮异步读取一行用户输入
          （线程池包装避免阻塞事件循环）；4. 空输入跳过、命中退出
          命令（exit/quit/bye//quit//exit，大小写不敏感）或 Ctrl+C
          时结束循环；5. 否则构造 UserMessage 调用 on_new_message
          回调，回复由 ConsoleOutputChannel 直接打印。
副作用：读写标准输入/输出流（input/print）；日志记录处理异常；
        不读写文件/数据库、不发起网络请求。
主要依赖：asyncio（线程池执行器包装阻塞读）、logging（错误日志）、
          app.ics_ai.channels.base_channel（输入/输出通道基类与
          UserMessage 消息载体）。
"""
# 延迟注解求值：所有类型注解按字符串处理，规避前向引用与版本差异问题
# （from __future__ 导入必须位于文件最前，仅可在 docstring 之后）。
from __future__ import annotations
# 导入 asyncio：将阻塞的 stdin 读取包装为线程池执行器异步任务。
import asyncio
# 从标准库 collections.abc 导入回调抽象类型：run_interactive 的回调
# 参数注解使用（PEP 585 推荐从 collections.abc 导入而非 typing）。
from collections.abc import Awaitable, Callable
# 导入 logging 模块：模块级 logger 记录交互循环中的处理异常。
import logging
# 从标准库 typing 导入 Any：消息字典与回调返回值的开放结构类型。
from typing import Any
# 从同包基类导入通道抽象接口与消息载体：ConsoleChannel 继承
# InputChannel、ConsoleOutputChannel 继承 OutputChannel，UserMessage
# 为交互循环构造的消息对象。
from app.ics_ai.channels.base_channel import (
    InputChannel,
    OutputChannel,
    UserMessage,
)
# 模块级日志器：以模块路径命名，继承全局日志配置
logger = logging.getLogger(__name__)
class ConsoleOutputChannel(OutputChannel):
    """控制台输出通道
    职责：将机器人消息渲染为终端可读格式并打印到 stdout——支持
          文本、编号按钮选项与图片占位提示三类内容。
    设计模式：策略模式——OutputChannel 抽象接口的控制台实现，
          与 REST/SocketIO 等投递型通道可互换。
    生命周期：由 ConsoleChannel.get_output_channel 每次调用新建，
          会话期间随调用方持有复用，进程结束即弃。
    """
    @property
    def name(self) -> str:
        """通道名称。
        @return: 固定返回 "console"，标识本通道为控制台实现。
        """
        # 固定名称：同时作为 UserMessage.input_channel 的来源标识
        return "console"
    async def send_response(
        self,
        recipient_id: str,
        message: dict[str, Any],
    ) -> None:
        """打印消息到控制台
        核心功能：将消息字典渲染为终端文本并打印（文本 + 编号按钮 +
                  图片占位）。
        业务流程：1. 提取 text 字段，非空打印 "Bot: <文本>"；2. 提取
                  buttons 列表，逐个打印编号选项（title 优先、回退
                  text 字段）；3. 提取 image 字段，非空打印占位提示。
        副作用：向 stdout 写入文本（print）；recipient_id 在单用户
                控制台场景不使用。
        @param recipient_id: 接收者 ID（本实现仅签名兼容，不使用）。
        @param message: 消息内容字典（可含 text/buttons/image 键）。
        @return: 无返回；本实现不抛异常。
        """
        # 第一步：提取消息文本，缺省空串
        text = message.get("text", "")
        # 条件打印：空文本不输出（避免打印孤立的 "Bot: " 前缀）
        if text:
            # 打印机器人回复正文
            print(f"Bot: {text}")
        # 第二步：提取按钮列表，缺省空列表
        buttons = message.get("buttons", [])
        # 条件渲染：有按钮时输出编号选项列表
        if buttons:
            # 选项列表标题
            print("选项:")
            # 逐个渲染按钮：enumerate 从 1 起编号（用户可读序号）
            for i, button in enumerate(buttons, 1):
                # 提取按钮文案：title 键优先，缺失回退 text 键，再缺省空串
                title = button.get("title", button.get("text", ""))
                # 打印编号选项行
                print(f"  {i}. {title}")
        # 第三步：提取图片地址，缺省 None
        image = message.get("image")
        # 条件渲染：有图片时打印占位提示（终端无法显示位图）
        if image:
            # 打印图片占位信息（含原始 URL 便于复制查看）
            print(f"[图片: {image}]")
class ConsoleChannel(InputChannel):
    """控制台输入通道
    职责：提供命令行交互入口——运行异步读取循环，将 stdin 输入封装
          为 UserMessage 交给处理回调，并支持退出命令与 Ctrl+C 优雅
          退出。
    设计模式：策略模式——InputChannel 抽象接口的控制台实现；
          交互循环内联于 run_interactive（无路由注册需求）。
    生命周期：随应用启动创建并常驻，run_interactive 阻塞至用户退出
          （退出命令/Ctrl+C），进程结束即弃。
    Attributes:
        sender_id: 交互会话的发送者标识
        exit_commands: 触发退出的命令词列表
    """
    def __init__(
        self,
        sender_id: str = "console_user",
        exit_commands: list[str] | None = None,
    ):
        """初始化控制台通道
        核心功能：记录会话标识与退出命令表。
        副作用：无（仅设置实例属性）。
        @param sender_id: 发送者 ID，默认 "console_user"（控制台单用户）。
        @param exit_commands: 退出命令列表；None 时使用默认集
                              （exit/quit/bye//quit//exit），传入空
                              列表则禁用命令退出（仅剩 Ctrl+C）。
        @return: 无返回。
        """
        # 保存发送者标识：写入每条 UserMessage.sender_id
        self.sender_id = sender_id
        # 保存退出命令表：未传入时兜底默认命令集（or 处理 None）
        self.exit_commands = exit_commands or ["exit", "quit", "bye", "/quit", "/exit"]
    @property
    def name(self) -> str:
        """通道名称。
        @return: 固定返回 "console"。
        """
        # 固定名称：写入 UserMessage.input_channel 标识消息来源
        return "console"
    def get_output_channel(self) -> OutputChannel:
        """获取关联的输出通道。
        核心功能：返回与控制台输入配对的输出通道实例。
        业务流程：每次调用返回全新 ConsoleOutputChannel（无状态打印
                  实现，新建成本可忽略）。
        副作用：无（仅构造内存对象）。
        @return: 新建的 ConsoleOutputChannel 实例。
        """
        # 新建输出通道：与基类契约一致，由调用方持有复用
        return ConsoleOutputChannel()
    async def run_interactive(
        self,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
    ) -> None:
        """运行交互式控制台
        核心功能：启动命令行主循环——读取输入、识别退出命令、构造
                  消息并移交回调处理，直至用户退出。
        业务流程：1. 打印欢迎横幅；2. 循环异步读取一行输入；3. 空输入
                  跳过本轮；4. 命中退出命令（大小写不敏感）则告别退出；
                  5. 否则构造 UserMessage 并 await 回调处理，处理后
                  空行分隔；6. KeyboardInterrupt（Ctrl+C）优雅退出；
                  其他异常记录日志并打印错误后继续循环（不中断会话）。
        副作用：读写 stdin/stdout；触发回调的下游处理副作用；异常时
                写错误日志。
        @param on_new_message: 消息处理回调——接收 UserMessage 并异步
                               完成处理（回复经输出通道打印），不能
                               为 None。
        @return: 无返回；用户退出后循环结束方法返回。
        """
        # 第一步：打印分隔线与欢迎横幅（提示交互方式与退出方式）
        print("=" * 50)
        # 横幅标题行：展示系统名称
        print("ICS AI 对话系统")
        # 退出提示行：引导用户使用退出命令
        print("输入 'exit' 或 'quit' 退出")
        # 横幅收尾分隔线
        print("=" * 50)
        # 空行分隔横幅与会话区
        print()
        # 第二步：进入主循环，直至用户主动退出
        while True:
            try:
                # 第三步：异步获取一行用户输入（去除首尾空白）
                user_input = await self._get_user_input()
                # 空输入守卫：直接跳过本轮，避免空消息进入处理链
                if not user_input:
                    # 继续下一轮读取
                    continue
                # 第四步：退出命令判定——转小写比较实现大小写不敏感
                if user_input.lower() in self.exit_commands:
                    # 打印告别语并退出循环
                    print("再见！")
                    # 跳出主循环，方法返回
                    break
                # 第五步：构造标准用户消息（ID/时间戳由 __post_init__ 补齐）
                user_message = UserMessage(
                    text=user_input,
                    sender_id=self.sender_id,
                    input_channel=self.name,
                )
                # 调用处理回调，将消息移交 Agent 处理链（回复直接打印）
                await on_new_message(user_message)
                # 处理后空行分隔，提升会话可读性
                print()
            # 精确捕获：Ctrl+C 中断视为用户主动退出（优雅结束）
            except KeyboardInterrupt:
                # 换行脱离当前输入行后打印告别语
                print("\n再见！")
                # 跳出主循环，方法返回
                break
            # 兜底捕获：回调处理异常——记录日志并提示后继续会话
            except Exception as e:
                # 记录错误详情（含原始异常信息）便于排查
                logger.error(f"Error: {e}")
                # 终端同步提示错误（控制台场景用户即运维）
                print(f"错误: {e}")
    async def _get_user_input(self) -> str:
        """获取用户输入（异步）
        核心功能：异步读取一行 stdin 输入并去除首尾空白。
        业务流程：input() 为阻塞调用，包装进默认线程池执行器执行，
                  避免卡死事件循环（异步框架下其余任务仍可调度）。
        副作用：阻塞读取标准输入（等待用户敲回车）。
        @return: 去除首尾空白后的用户输入字符串（可能为空串）。
        """
        # 获取当前事件循环引用：用于向线程池提交阻塞任务
        loop = asyncio.get_event_loop()
        # input("You: ") 提示符读取提交至默认执行器线程执行；
        # await 挂起协程直至用户回车，读取期间事件循环不被阻塞
        return await loop.run_in_executor(None, lambda: input("You: ").strip())
# 导出清单：按字母序排列，约束 from ... import * 的导出范围并为 IDE
# 提供准确的补全与跳转依据。
__all__ = [
    # 控制台输入通道（交互循环入口）
    "ConsoleChannel",
    # 控制台输出通道（终端打印）
    "ConsoleOutputChannel",
]