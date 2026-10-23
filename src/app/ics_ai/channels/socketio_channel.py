"""
socketio_channel - Socket.IO 实时通道模块
核心功能：基于 Socket.IO（WebSocket 双向通信）的通道实现——
          SocketIOChannel 管理服务器与事件处理器（连接/断开/会话请求/
          用户消息四类事件），SocketIOOutputChannel 经事件向客户端
          房间实时推送机器人消息，构成浏览器端长连接对话能力。
执行流程：1. 应用启动时经 create_app 构造 AsyncServer 与 ASGI 应用
          并注册事件处理器；2. 客户端 connect 后发 session_request
          加入会话房间并收到 session_confirm；3. 客户端经
          user_uttered 事件发送消息，handle_message 解析字段构造
          UserMessage 交给处理回调；4. 回复经 SocketIOOutputChannel
          以 bot_uttered 事件按房间推送；5. 处理异常时向该客户端
          房间回发错误提示消息。
副作用：Socket.IO 网络收发（emit/enter_room）；日志记录连接与异常；
        维护服务器引用等实例状态；不读写文件/数据库。
主要依赖：socketio（延迟导入的 AsyncServer/ASGIApp）、logging、
          app.ics_ai.channels.base_channel（输入/输出通道基类与
          UserMessage 消息载体）。
"""
# 延迟注解求值：所有类型注解按字符串处理，规避前向引用与版本差异问题
# （from __future__ 导入必须位于文件最前，仅可在 docstring 之后）。
from __future__ import annotations
# 从标准库 collections.abc 导入回调抽象类型：register_handlers/
# create_app 的回调参数注解使用（PEP 585 推荐从 collections.abc 导入）。
from collections.abc import Awaitable, Callable
# 导入 logging 模块：模块级 logger 记录连接、断开与处理异常。
import logging
# 从标准库 typing 导入 Any 与 TYPE_CHECKING：开放结构类型 + 仅类型
# 检查执行的导入分支标记。
from typing import Any, TYPE_CHECKING
# 从同包基类导入通道抽象接口与消息载体：SocketIOChannel 继承
# InputChannel、SocketIOOutputChannel 继承 OutputChannel，UserMessage
# 为消息事件构造的标准消息对象。
from app.ics_ai.channels.base_channel import (
    InputChannel,
    OutputChannel,
    UserMessage,
)
# 仅类型检查分支：AsyncServer 类型仅供注解使用，运行时由
# create_sio_server/create_app 延迟导入，保持模块级零 socketio 依赖。
if TYPE_CHECKING:
    # Socket.IO 异步服务器类型
    from socketio import AsyncServer
# 模块级日志器：以模块路径命名，继承全局日志配置
logger = logging.getLogger(__name__)
class SocketIOOutputChannel(OutputChannel):
    """Socket.IO 输出通道
    职责：将机器人消息以 Socket.IO 事件实时推送到目标客户端所在的
          房间（长连接推送，非请求-响应模型）。
    设计模式：策略模式——OutputChannel 抽象接口的 Socket.IO 实现，
          与 REST/控制台等通道可互换。
    生命周期：由 SocketIOChannel.get_output_channel 在服务器就绪后
          创建，随通道复用，进程结束即弃。
    Attributes:
        sio: Socket.IO 服务器
        bot_message_evt: 机器人消息事件名
    """
    def __init__(
        self,
        sio: AsyncServer,
        bot_message_evt: str = "bot_uttered",
    ) -> None:
        """初始化 Socket.IO 输出通道
        核心功能：绑定服务器实例与消息事件名。
        副作用：无（仅设置实例属性）。
        @param sio: Socket.IO 异步服务器实例，不能为 None。
        @param bot_message_evt: 推送机器人消息的事件名，默认
                                "bot_uttered"。
        @return: 无返回。
        """
        # 绑定 Socket.IO 服务器：emit 的事件发送载体
        self.sio = sio
        # 保存消息事件名：send_response 按此事件名推送
        self.bot_message_evt = bot_message_evt
    @property
    def name(self) -> str:
        """通道名称。
        @return: 固定返回 "socketio"，标识本通道为 Socket.IO 实现。
        """
        # 固定名称：同时作为 UserMessage.input_channel 的来源标识
        return "socketio"
    async def send_response(
        self,
        recipient_id: str,
        message: dict[str, Any],
    ) -> None:
        """发送消息
        核心功能：以 bot_uttered 事件将消息推送到接收者所在房间。
        业务流程：直接 emit——Socket.IO 房间模型下 room=recipient_id
                  精准投递到该会话的所有连接（多标签页同会话均收到）。
        副作用：向 Socket.IO 服务器发起事件广播（网络 IO）。
        @param recipient_id: 接收者 ID（映射为 Socket.IO 房间名，
                             经 session_request 加入）。
        @param message: 消息内容字典（可含 text/buttons 等键）。
        @return: 无返回；服务器未就绪或连接断开时由 socketio 层抛异常。
        """
        # 向目标房间推送消息事件：message 字典原样作为事件负载
        await self.sio.emit(
            self.bot_message_evt,
            message,
            room=recipient_id,
        )
class SocketIOChannel(InputChannel):
    """Socket.IO 通道
    职责：管理 Socket.IO 服务器生命周期与事件处理——创建 AsyncServer、
          注册连接/断开/会话请求/用户消息四类事件处理器，并将用户
          消息封装为 UserMessage 交给处理回调；输出经配对的
          SocketIOOutputChannel 按房间推送。
    设计模式：策略模式——InputChannel 抽象接口的 Socket.IO 实现；
          register_handlers 以闭包捕获通道配置构造事件处理器。
    生命周期：随应用启动经 create_app 创建 ASGI 应用常驻监听；
          服务器引用在 create_sio_server/register_handlers 时建立。
    事件：
    - connect: 客户端连接
    - disconnect: 客户端断开
    - user_uttered: 用户发送消息
    - bot_uttered: 机器人回复消息
    - session_request: 请求会话
    Attributes:
        user_message_evt: 用户消息事件名
        bot_message_evt: 机器人消息事件名
        session_persistence: 是否保持会话
        socketio_path: Socket.IO 路径
        sio: Socket.IO 异步服务器（创建后填充）
    """
    def __init__(
        self,
        user_message_evt: str = "user_uttered",
        bot_message_evt: str = "bot_uttered",
        session_persistence: bool = True,
        socketio_path: str = "/socket.io",
    ):
        """初始化 Socket.IO 通道
        核心功能：记录事件名与会话配置，初始化未创建的服务器引用。
        副作用：无（仅设置实例属性，不创建服务器）。
        @param user_message_evt: 接收用户消息的事件名，默认
                                 "user_uttered"。
        @param bot_message_evt: 推送机器人消息的事件名，默认
                                "bot_uttered"。
        @param session_persistence: 是否保持会话，默认 True（当前
                                    通道逻辑未消费该配置，仅透传保存）。
        @param socketio_path: Socket.IO 握手路径，默认 "/socket.io"。
        @return: 无返回。
        """
        # 保存用户消息事件名：register_handlers 按此注册接收处理器
        self.user_message_evt = user_message_evt
        # 保存机器人消息事件名：输出通道按此推送回复
        self.bot_message_evt = bot_message_evt
        # 保存会话保持配置：当前未参与连接逻辑，预留扩展
        self.session_persistence = session_persistence
        # 保存 Socket.IO 握手路径：ASGI 应用挂载时使用
        self.socketio_path = socketio_path
        # 服务器引用占位：create_sio_server/register_handlers 时填充，
        # 未就绪时 get_output_channel 返回 None
        self.sio: AsyncServer | None = None
    @property
    def name(self) -> str:
        """通道名称。
        @return: 固定返回 "socketio"。
        """
        # 固定名称：写入 UserMessage.input_channel 标识消息来源
        return "socketio"
    def get_output_channel(self) -> OutputChannel | None:
        """获取关联的输出通道
        核心功能：返回绑定当前服务器的输出通道实例。
        业务流程：服务器已创建则返回绑定它的 SocketIOOutputChannel；
                  未创建（尚未 create_sio_server/register_handlers）
                  返回 None——无服务器则无推送能力。
        副作用：无（仅构造内存对象）。
        @return: SocketIOOutputChannel 实例；服务器未就绪返回 None。
        """
        # 前置判断：服务器引用非空才具备推送能力
        if self.sio:
            # 返回绑定当前服务器与事件名的输出通道
            return SocketIOOutputChannel(self.sio, self.bot_message_evt)
        # 服务器未就绪：返回 None 由调用方降级处理
        return None
    def create_sio_server(self) -> AsyncServer:
        """创建 Socket.IO 服务器
        核心功能：构造 ASGI 异步模式的 AsyncServer 并保存引用。
        业务流程：延迟导入 socketio → 以 asgi 异步模式创建服务器 →
                  允许所有跨域来源（开发期宽松配置，生产环境应收紧
                  为具体域名列表）→ 保存引用供 get_output_channel
                  判定就绪状态。
        副作用：修改实例状态（self.sio 填充）；无网络 IO（服务器
                尚未监听）。
        @return: 新创建的 AsyncServer 实例。
        """
        # 延迟导入 socketio：未安装该库不影响本模块被导入
        import socketio
        # 创建 ASGI 异步模式服务器；cors_allowed_origins="*" 放开全部
        # 来源（开发调试便利，生产部署需替换为具体域名白名单）
        self.sio = socketio.AsyncServer(
            async_mode="asgi",
            cors_allowed_origins="*",
        )
        # 返回服务器实例：由调用方注册处理器或包装 ASGI 应用
        return self.sio
    def register_handlers(
        self,
        sio: AsyncServer,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
    ) -> None:
        """注册 Socket.IO 事件处理器
        核心功能：向服务器注册连接/断开/会话请求/用户消息四类事件的
                  处理闭包。
        业务流程：1. 保存服务器引用（get_output_channel 就绪判定依据）；
                  2. 注册 connect/disconnect 生命周期事件（日志记录）；
                  3. 注册 session_request——客户端加入会话房间并确认；
                  4. 注册 user_message_evt 消息事件——解析字段、构造
                  UserMessage、移交回调，异常时回发错误提示。
        副作用：修改实例状态（self.sio）；向服务器注册事件监听；事件
                触发时产生网络 IO 与回调下游副作用。
        @param sio: Socket.IO 异步服务器实例，不能为 None。
        @param on_new_message: 消息处理回调——接收 UserMessage 并异步
                               完成处理（回复经输出通道推送），不能
                               为 None。
        @return: 无返回。
        """
        # 第一步：保存服务器引用——get_output_channel 依赖它判定就绪
        self.sio = sio
        @sio.event
        async def connect(sid: str, environ: dict[str, Any]) -> bool:
            """客户端连接
            核心功能：记录客户端接入日志并接受连接。
            @param sid: Socket.IO 会话 ID（服务端为每个连接分配）。
            @param environ: 连接握手的环境信息（WSGI 风格字典）。
            @return: True 表示接受连接（False 将拒绝）。
            """
            # 记录接入日志：sid 便于追踪会话
            logger.info(f"Client connected: {sid}")
            # 接受该连接
            return True
        @sio.event
        async def disconnect(sid: str) -> None:
            """客户端断开
            核心功能：记录客户端断开日志（服务器自动清理其连接状态，
                      无需手动出房间）。
            @param sid: 断开连接的会话 ID。
            @return: 无返回。
            """
            # 记录断开日志
            logger.info(f"Client disconnected: {sid}")
        # 注册会话请求事件：on(event) 返回装饰器，是 socketio 文档的
        # 标准装饰器用法（运行时行为正确）；存根将 on 的返回类型推断
        # 为可选 handler（None），导致装饰器调用被误判"None 不可调用"，
        # 按第三方库存根滞后惯例压制
        @sio.on("session_request")  # pyright: ignore[reportCallIssue, reportOptionalCall]
        async def session_request(sid: str, data: dict[str, Any]) -> None:
            """会话请求
            核心功能：将会话加入指定房间并确认会话 ID。
            业务流程：1. 从请求取 session_id（缺省回退连接自身的
                      sid）；2. enter_room 将连接加入以会话 ID 命名的
                      房间（后续 bot_uttered 按房间推送）；3. 向该
                      连接回发 session_confirm 确认。
            副作用：网络 IO（入房间操作与确认事件）。
            @param sid: 发起请求的会话 ID。
            @param data: 请求数据（可含 session_id 键）。
            @return: 无返回。
            """
            # 第一步：解析目标会话 ID——未指定时以连接 sid 作为会话
            session_id = data.get("session_id", sid)
            # 记录会话请求日志
            logger.info(f"Session request from {sid}: {session_id}")
            # 第二步：将该连接加入会话房间——同一会话的多连接共享推送
            await sio.enter_room(sid, session_id)
            # 第三步：向发起连接单发确认事件（room=sid 精准回执）
            await sio.emit("session_confirm", {"session_id": session_id}, room=sid)
        # 注册用户消息事件：同上，装饰器调用形式的存根误报压制
        @sio.on(self.user_message_evt)  # pyright: ignore[reportCallIssue, reportOptionalCall]
        async def handle_message(sid: str, data: dict[str, Any]) -> None:
            """处理用户消息
            核心功能：解析用户消息事件，构造 UserMessage 交给回调
                      处理。
            业务流程：1. 提取文本（message/text 双字段名兼容回退）与
                      发送者（session_id/sender_id/连接 sid 三级回退）
                      及元数据；2. 空文本静默忽略（Socket.IO 无状态码
                      语义，直接返回）；3. 构造 UserMessage 并 await
                      回调；4. 任何异常记录日志并向该客户端房间回发
                      错误提示消息。
            副作用：触发回调的下游处理副作用；异常时向房间推送错误
                    消息（网络 IO）；写日志。
            @param sid: 发送消息的会话 ID。
            @param data: 消息负载数据（可含 message/text/session_id/
                         sender_id/metadata 键）。
            @return: 无返回；异常不向上抛（内部消化，保障连接存活）。
            """
            try:
                # 第一步：提取消息文本——先取 message 键，缺失回退
                # text 键，再缺失为空串（兼容两种客户端字段命名）
                text = data.get("message", data.get("text", ""))
                # 第一步：提取发送者——优先显式会话 ID，回退 sender_id，
                # 再回退连接自身的 sid（未建立会话的裸连接场景）
                sender_id = data.get("session_id", data.get("sender_id", sid))
                # 第一步：提取扩展元数据，缺省空字典
                metadata = data.get("metadata", {})
                # 第二步：空文本守卫——无有效内容直接忽略本轮
                if not text:
                    # 静默返回（不影响连接与后续消息）
                    return
                # 记录消息调试日志（低级别，生产默认不输出）
                logger.debug(f"Received message from {sender_id}: {text}")
                # 第三步：构造标准用户消息（ID/时间戳由 __post_init__ 补齐）
                user_message = UserMessage(
                    text=text,
                    sender_id=sender_id,
                    input_channel=self.name,
                    metadata=metadata,
                )
                # 调用处理回调，将消息移交 Agent 处理链
                # （回复经 SocketIOOutputChannel 以 bot_uttered 推送）
                await on_new_message(user_message)
            # 兜底捕获：处理异常内部消化——向上抛会破坏 Socket.IO 事件分发
            except Exception as e:
                # 记录错误详情（含原始异常信息）便于排查
                logger.error(f"Error handling message: {e}")
                # 向该客户端所在房间推送错误提示（用户可感知的降级反馈）
                await sio.emit(
                    self.bot_message_evt,
                    {"text": "抱歉，处理消息时出现错误。"},
                    room=sid,
                )
    def create_app(
        self,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
    ) -> Any:
        """创建 ASGI 应用
        核心功能：一步完成服务器创建、事件注册与 ASGI 应用包装。
        业务流程：1. create_sio_server 创建并保存 AsyncServer；2.
                  register_handlers 注册全部事件处理器；3. 以服务器
                  包装 ASGIApp（供 uvicorn 等服务器挂载监听）。
        副作用：创建服务器与注册处理器（见对应方法）；返回的 ASGI
                应用被挂载后开始网络监听。
        @param on_new_message: 消息处理回调（透传给事件注册），不能
                               为 None。
        @return: socketio.ASGIApp 实例（ASGI 协议应用）。
        """
        # 延迟导入 socketio：仅在真正构建应用时加载
        import socketio
        # 第一步：创建服务器（引用同步保存到 self.sio）
        sio = self.create_sio_server()
        # 第二步：注册全部事件处理器（连接/断开/会话/消息）
        self.register_handlers(sio, on_new_message)
        # 第三步：包装为 ASGI 应用——供 uvicorn/hypercorn 等挂载运行
        app = socketio.ASGIApp(sio)
        # 返回应用实例：由应用装配层挂载到 HTTP 服务
        return app
# 导出清单：按字母序排列，约束 from ... import * 的导出范围并为 IDE
# 提供准确的补全与跳转依据。
__all__ = [
    # Socket.IO 输入通道（服务器与事件管理）
    "SocketIOChannel",
    # Socket.IO 输出通道（房间推送）
    "SocketIOOutputChannel",
]