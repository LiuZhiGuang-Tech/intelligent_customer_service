"""
inspect_proxy - Inspect 调试代理模块
核心功能：为开发调试提供对话状态实时查看能力——InspectProxy 以装饰器
          模式包装任意输入通道（透传其路由），并附加 /inspect.html
          调试页面与 /tracker_stream WebSocket 端点；TrackerStream
          管理调试客户端连接集合并广播 Tracker 状态 JSON，页面端
          支持槽位/活动 Flow/原始状态三区展示与对话调试。
执行流程：1. 应用装配时以底层通道（如 RestChannel）构造 InspectProxy；
          2. create_routes 将底层通道路由与调试端点合并为单一
          APIRouter；3. 浏览器访问 /inspect.html 经 WebSocket 连入
          /tracker_stream 并请求 retrieve 拉取初始状态；4. Tracker
          变化时 on_tracker_updated 向全部客户端广播新状态 JSON；
          5. 页面消息发送走底层通道的 REST webhook（经包装回调透传）。
副作用：网络 IO（WebSocket 收发、HTTP 页面响应）；日志记录广播异常；
        维护客户端连接集合等实例状态；不读写文件/数据库。
主要依赖：asyncio（并发广播 gather）、json（状态序列化，default=str
          兜底非原生类型）、logging、fastapi（延迟导入的 WebSocket
          路由）、app.ics_ai.channels.base_channel（InputChannel 基类
          与 UserMessage）。
"""
# 延迟注解求值：所有类型注解按字符串处理，规避前向引用与版本差异问题
# （from __future__ 导入必须位于文件最前，仅可在 docstring 之后）。
from __future__ import annotations
# 导入 asyncio：broadcast 中并发发送多客户端消息（gather 聚合）。
import asyncio
# 从标准库 collections.abc 导入回调抽象类型：状态获取回调与消息处理
# 回调的参数注解使用（PEP 585 推荐从 collections.abc 导入而非 typing）。
from collections.abc import Awaitable, Callable
# 导入 json 模块：Tracker 状态字典与 JSON 字符串互转。
import json
# 导入 logging 模块：模块级 logger 记录广播失败等异常。
import logging
# 从标准库 typing 导入 Any 与 TYPE_CHECKING：开放结构类型 + 仅类型
# 检查执行的导入分支标记。
from typing import Any, TYPE_CHECKING
# 从同包基类导入输入通道抽象与消息载体：InspectProxy 继承 InputChannel，
# create_routes 的包装回调处理 UserMessage。
from app.ics_ai.channels.base_channel import InputChannel, UserMessage
# 仅类型检查分支：Tracker/TrackerStore 与 MessageProcessor 仅用于注解，
# 避免与 agent/core 模块的运行时循环导入。
if TYPE_CHECKING:
    # 对话状态跟踪器类型（on_tracker_updated 参数注解）
    from app.ics_ai.core.tracker import DialogueStateTracker
    # Tracker 存储抽象基类（tracker_store 属性注解）
    from app.ics_ai.core.stores.tracker_store import TrackerStore
    # 消息处理器类型（processor 属性注解）
    from app.ics_ai.agent.message_processor import MessageProcessor
# 模块级日志器：以模块路径命名，继承全局日志配置
logger = logging.getLogger(__name__)
class TrackerStream:
    """Tracker 状态流
    职责：管理经 WebSocket 连入的调试客户端集合，并向全部在线客户端
          广播 Tracker 状态 JSON 文本；单个客户端发送失败时自动剔除，
          不影响其余客户端。
    设计模式：观察者模式——客户端经 add/remove 订阅/退订，broadcast
          为事件分发点。
    生命周期：随 InspectProxy 创建常驻；客户端集合动态增减，随代理
          进程结束销毁。
    Attributes:
        get_tracker_state: 获取 Tracker 状态的回调
    """
    def __init__(
        self,
        get_tracker_state: Callable[[str], Awaitable[str]],
    ):
        """初始化 Tracker 状态流
        核心功能：保存状态获取回调，初始化空客户端集合。
        副作用：无（仅设置实例属性）。
        @param get_tracker_state: 按 sender_id 获取状态 JSON 的异步
                                  回调（InspectProxy 注入其私有实现）。
        @return: 无返回。
        """
        # 保存状态获取回调：供上层按需拉取指定会话状态
        self.get_tracker_state = get_tracker_state
        # 调试客户端集合：元素为 WebSocket 连接对象（Any 鸭子类型，
        # 兼容 fastapi WebSocket 及测试替身）；set 保证去重
        self._clients: set[Any] = set()
    def add_client(self, websocket: Any) -> None:
        """添加客户端。
        核心功能：将新连接加入广播列表（重复添加被 set 去重）。
        副作用：修改实例级客户端集合。
        @param websocket: WebSocket 连接对象（鸭子类型）。
        @return: 无返回。
        """
        # 加入集合：重复连接自动去重
        self._clients.add(websocket)
    def remove_client(self, websocket: Any) -> None:
        """移除客户端。
        核心功能：将连接移出广播列表。
        副作用：修改实例级客户端集合。
        @param websocket: WebSocket 连接对象；不存在时静默通过
                          （discard 与 remove 的差异：不抛 KeyError）。
        @return: 无返回。
        """
        # discard 语义：元素不存在时静默跳过，适配断连竞态
        self._clients.discard(websocket)
    async def broadcast(self, message: str) -> None:
        """广播消息给所有客户端
        核心功能：将状态文本并发发送给全部在线客户端，失败者自动剔除。
        业务流程：1. 无客户端直接返回；2. 遍历集合副本为每个客户端
                  创建发送协程（集合副本规避遍历中剔除引发的
                  RuntimeError）；3. gather 并发执行且吞单点异常
                  （return_exceptions=True），发送失败由 _send 内部剔除。
        副作用：向全部客户端发起 WebSocket 网络发送；可能修改客户端
                集合（剔除失败连接）。
        @param message: 待广播的 JSON 文本。
        @return: 无返回；单客户端失败不中断整体广播。
        """
        # 前置守卫：无订阅客户端时无需任何网络操作
        if not self._clients:
            # 直接返回，避免空 gather 开销
            return
        # 发送协程收集列表
        tasks = []
        # 遍历集合副本（copy）：_send 失败会修改原集合，副本保证遍历安全
        for client in self._clients.copy():
            try:
                # 创建发送协程并收集（协程创建不执行，gather 时才并发运行）
                tasks.append(self._send(client, message))
            # 兜底捕获：协程对象创建阶段即异常（如事件循环关闭）时剔除该客户端
            except Exception:
                # 从广播列表中移除坏连接
                self._clients.discard(client)
        # 条件并发：有可发送协程时统一执行
        if tasks:
            # return_exceptions=True：单客户端发送异常作为结果值返回
            # 而非中断 gather，其余客户端的发送不受影响
            await asyncio.gather(*tasks, return_exceptions=True)
    async def _send(self, websocket: Any, message: str) -> None:
        """发送消息给单个客户端
        核心功能：向单个 WebSocket 连接发送文本，失败时将其剔除出
                  广播列表。
        业务流程：await websocket.send_text 发送；任何异常（连接已断/
                  发送超时等）视为客户端失效，从集合移除。
        副作用：网络发送；失败时修改客户端集合。
        @param websocket: WebSocket 连接对象（鸭子类型）。
        @param message: 待发送的 JSON 文本。
        @return: 无返回；发送失败不抛异常（内部消化，保障广播健壮性）。
        """
        try:
            # 发送文本帧给该客户端
            await websocket.send_text(message)
        # 兜底捕获：发送失败统一视为连接失效
        except Exception:
            # 剔除失效连接，避免后续广播反复撞死连接
            self._clients.discard(websocket)
class InspectProxy(InputChannel):
    """Inspect 代理通道
    职责：包装底层输入通道并提供调试能力——透传底层路由与消息回调，
          附加 /inspect.html 调试页面（左对话区 + 右状态区，前端逻辑
          概述见 _get_inspect_html）与 /tracker_stream WebSocket 端点
          （retrieve 动作按 sender_id 拉取状态快照），并在 Tracker
          变化时向全部调试客户端广播。
    设计模式：装饰器模式——持有底层 InputChannel 并扩展调试路由，
          对使用者保持 InputChannel 接口不变；观察者模式（经
          TrackerStream 分发状态更新）。
    生命周期：随应用装配创建常驻；processor 可先空后经 set_processor
          注入（装配顺序解耦），随进程结束销毁。
    功能：
    - 包装任意输入通道
    - 提供/inspect.html页面查看对话状态
    - 通过WebSocket实时推送状态更新
    """
    def __init__(
        self,
        underlying_channel: InputChannel,
        tracker_store: TrackerStore | None = None,
        processor: MessageProcessor | None = None,
    ):
        """初始化 Inspect 代理
        核心功能：记录被包装通道、Tracker 存储与消息处理器，装配
                  状态流。
        副作用：构造 TrackerStream 实例（纯内存对象，无 IO）。
        @param underlying_channel: 被包装的底层通道（如 RestChannel），
                                   不能为 None。
        @param tracker_store: Tracker 存储实例——状态拉取的数据来源
                              （由 Agent 装配时传入其持有的存储）；允许
                              None（状态拉取降级为 "{}"，后续经
                              set_tracker_store 注入）。
        @param processor: 消息处理器；允许 None（预留引用，当前状态
                          拉取不依赖它，后续经 set_processor 注入）。
        @return: 无返回。
        """
        # 保存被包装通道：路由透传与 name 组合的来源
        self.underlying = underlying_channel
        # 保存 Tracker 存储：状态拉取的唯一数据通路
        # （注：Domain 不持有存储，存储由 Agent 装配并持有）
        self.tracker_store = tracker_store
        # 保存消息处理器：预留引用，允许后置注入
        self.processor = processor
        # 装配状态流：注入本类的私有状态获取回调作为拉取实现
        self.tracker_stream = TrackerStream(get_tracker_state=self._get_tracker_state)
    @property
    def name(self) -> str:
        """通道名称。
        @return: 组合名称 "inspect_<底层通道名>"（标识代理叠加关系）。
        """
        # f-string 组合底层通道名：如 inspect_rest
        return f"inspect_{self.underlying.name}"
    def set_processor(self, processor: MessageProcessor) -> None:
        """设置消息处理器
        核心功能：运行期后置注入处理器（应对先建代理后建处理器的
                  装配顺序）。
        副作用：修改实例属性 processor。
        @param processor: MessageProcessor 实例，不能为 None。
        @return: 无返回。
        """
        # 覆盖处理器引用：后续依赖处理器的功能走新实例
        self.processor = processor
    def set_tracker_store(self, tracker_store: TrackerStore) -> None:
        """设置 Tracker 存储
        核心功能：运行期后置注入存储（应对先建代理后建存储的装配
                  顺序，与 set_processor 对称）。
        副作用：修改实例属性 tracker_store。
        @param tracker_store: TrackerStore 实例，不能为 None。
        @return: 无返回。
        """
        # 覆盖存储引用：后续状态拉取走新存储
        self.tracker_store = tracker_store
    async def _get_tracker_state(self, sender_id: str) -> str:
        """获取 Tracker 状态的 JSON 字符串
        核心功能：按 sender_id 从 Tracker 存储拉取状态并序列化。
        业务流程：1. 无存储时返回 "{}"（降级——存储尚未注入）；2. 从
                  tracker_store 检索该会话的 Tracker；3. 不存在返回
                  "{}"；4. 存在则取当前状态快照并序列化（default=str
                  兜底 datetime 等非原生 JSON 类型，ensure_ascii=False
                  保留中文原文）。
        副作用：读取 Tracker 存储（可能触发存储后端 IO）；无写入。
        @param sender_id: 会话 ID。
        @return: 状态 JSON 字符串；无存储/无记录时为 "{}"。
        """
        # 第一步：无存储降级——返回空对象占位（页面显示暂无数据）
        if not self.tracker_store:
            # 返回空 JSON 对象
            return "{}"
        # 第二步：从 Tracker 存储检索该会话的状态记录
        # （存储由 Agent 装配持有，Domain 不承载存储）
        tracker = await self.tracker_store.retrieve(sender_id)
        # 第三步：无记录降级——该会话从未保存过状态
        if not tracker:
            # 返回空 JSON 对象
            return "{}"
        # 第四步：取当前状态快照（槽位/活动 Flow/事件等）
        state = tracker.current_state()
        # 序列化：default=str 将非原生类型降级为字符串表示，避免序列化报错
        return json.dumps(state, ensure_ascii=False, default=str)
    async def on_tracker_updated(self, tracker: DialogueStateTracker) -> None:
        """Tracker 更新时广播状态
        核心功能：将变更后的 Tracker 状态推送给全部调试客户端
                  （观察者的主题侧入口）。
        业务流程：取状态快照 → 序列化 → 经 TrackerStream 广播；任何
                  异常（序列化失败/网络故障）记录日志后吞掉——调试
                  功能故障不得影响正常对话流程。
        副作用：向全部客户端发起 WebSocket 发送；写错误日志。
        @param tracker: 刚发生更新的 Tracker 实例，不能为 None。
        @return: 无返回；广播失败静默（仅日志）。
        """
        try:
            # 取状态快照
            state = tracker.current_state()
            # 序列化为 JSON 文本（非原生类型降级字符串，中文保留原文）
            state_json = json.dumps(state, ensure_ascii=False, default=str)
            # 向全部在线调试客户端广播
            await self.tracker_stream.broadcast(state_json)
        # 兜底捕获：广播环节任何异常仅记日志（调试旁路不影响主流程）
        except Exception as e:
            # 记录失败详情便于排查调试通道问题
            logger.error(f"Failed to broadcast tracker state: {e}")
    def create_routes(
        self,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
    ) -> Any:
        """创建 FastAPI 路由
        核心功能：合并底层通道路由与调试端点，返回统一 APIRouter。
        业务流程：1. 延迟导入 fastapi；2. 创建调试路由器；3. 构造
                  包装回调（当前透传，预留状态广播切面）；4. 底层通道
                  支持 create_routes（鸭子类型探测）则将其路由并入；
                  5. 注册 GET /inspect.html 页面端点；6. 注册
                  WebSocket /tracker_stream 端点（accept → 注册客户端
                  → 循环处理 retrieve 请求 → 断连清理）。
        副作用：本方法自身无 IO（构造路由对象）；端点被调用时产生
                网络 IO 与回调下游副作用。
        @param on_new_message: 消息处理回调（底层通道端点最终调用它），
                               不能为 None。
        @return: FastAPI APIRouter 实例（含页面/WebSocket/底层路由）。
        """
        # 延迟导入 fastapi：仅在真正构造路由时加载
        from fastapi import APIRouter, WebSocket, WebSocketDisconnect
        # 延迟导入 HTML 响应类：inspect.html 页面返回
        from fastapi.responses import HTMLResponse
        # 创建调试路由器：无统一前缀（页面与流端点挂应用根路径）
        router = APIRouter(tags=["inspect"])
        # 包装消息处理回调，添加状态广播
        async def wrapped_handler(message: UserMessage) -> Any:
            # 透传调用原始回调，保留其返回值
            result = await on_new_message(message)
            # 广播状态更新由钩子处理（on_tracker_updated 由外部触发）
            return result
        # 添加底层通道的路由
        # 鸭子类型探测：仅支持具备 create_routes 能力的通道（如 RestChannel）；
        # create_routes 声明于具体子类而非 InputChannel 基类，pyright 不支持
        # hasattr 收窄到基类声明之外，按鸭子类型守卫惯例压制访问告警
        if hasattr(self.underlying, "create_routes"):
            # 将底层通道路由（webhook 等）并入调试路由器，
            # 传入包装回调维持调用链可插拔
            underlying_router = self.underlying.create_routes(wrapped_handler)  # pyright: ignore[reportAttributeAccessIssue]
            # 合并底层路由到统一路由器
            router.include_router(underlying_router)
        @router.get("/inspect.html", response_class=HTMLResponse)
        async def inspect_page():
            """Inspect 页面。
            核心功能：返回内嵌的调试页面 HTML（前端自动连接 WebSocket、
                      3 秒自动重连、retrieve 拉初始状态、消息经 REST
                      webhook 发送、状态三区渲染——槽位/活动 Flow/原始
                      JSON）。
            副作用：返回 HTTP 响应（网络 IO）。
            @return: HTMLResponse 页面响应。
            """
            # 返回页面 HTML 字符串（前端逻辑详见 _get_inspect_html）
            return self._get_inspect_html()
        @router.websocket("/tracker_stream")
        async def tracker_stream(websocket: WebSocket):
            """Tracker 状态 WebSocket 流
            核心功能：维护单个调试客户端的连接生命周期，处理其状态
                      拉取请求。
            业务流程：1. accept 握手并注册进广播集合；2. 循环读取
                      客户端文本消息并解析 JSON；3. action=retrieve
                      时按 sender_id 拉取状态并回发；4. 客户端断连
                      （WebSocketDisconnect）静默退出；5. finally 中
                      无论何种退出路径均从广播集合移除该连接。
            副作用：WebSocket 网络收发；修改广播客户端集合。
            @param websocket: FastAPI WebSocket 连接对象。
            @return: 无返回；客户端断连后协程结束。
            """
            # 第一步：完成 WebSocket 握手，连接进入可用状态
            await websocket.accept()
            # 注册进广播集合：后续 on_tracker_updated 会推送状态
            self.tracker_stream.add_client(websocket)
            try:
                # 第二步：常驻循环，逐条处理客户端请求
                while True:
                    # 阻塞等待客户端文本帧（断连时抛 WebSocketDisconnect）
                    data = await websocket.receive_text()
                    # 解析客户端 JSON 请求
                    message = json.loads(data)
                    # 第三步：处理获取状态请求
                    if message.get("action") == "retrieve":
                        # 提取目标会话 ID，缺省 default
                        sender_id = message.get("sender_id", "default")
                        # 按会话 ID 拉取状态 JSON
                        state = await self._get_tracker_state(sender_id)
                        # 将状态快照回发给请求方
                        await websocket.send_text(state)
            # 第四步：客户端正常断连——静默退出循环
            except WebSocketDisconnect:
                # 断连属预期行为，无需日志
                pass
            # 第五步：清理——无论正常断连还是异常退出都移除注册
            finally:
                # 从广播集合剔除，防止向死连接发送
                self.tracker_stream.remove_client(websocket)
        # 返回装配好的路由器：由应用统一 include_router 挂载
        return router
    def _get_inspect_html(self) -> str:
        """返回 Inspect 页面 HTML
        核心功能：返回内嵌单页调试器的完整 HTML（样式 + 布局 + 前端
                  脚本）。
        前端行为概述（页面内容为字符串字面量，不加行内注释以免污染
        输出）：连接 /tracker_stream（wss/ws 自适应协议）；onopen
        标记已连接并 retrieve 初始状态；onclose 3 秒后自动重连；
        onmessage 解析状态并渲染槽位表/活动 Flow/原始 JSON 三区；
        sendMessage 将输入经 POST /webhooks/rest/webhook 发送并渲染
        机器人回复；会话 ID 按时间戳生成保证独立调试会话。
        副作用：无（纯字符串返回；网络行为由浏览器端触发）。
        @return: 完整的 HTML 页面字符串。
        """
        # 返回内嵌页面：单引号三引号包裹，内部样式/脚本保持原样输出
        return """
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ICS AI - 对话调试器</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; }
        .container { display: flex; height: 100vh; }
        .panel { flex: 1; padding: 20px; overflow-y: auto; }
        .panel-left { background: white; border-right: 1px solid #ddd; }
        .panel-right { background: #fafafa; }
        h2 { margin-bottom: 15px; color: #333; }
        .chat-box { height: calc(100vh - 200px); overflow-y: auto; border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: #fff; margin-bottom: 15px; }
        .message { margin-bottom: 10px; padding: 10px 15px; border-radius: 18px; max-width: 80%; }
        .user-message { background: #007bff; color: white; margin-left: auto; }
        .bot-message { background: #e9ecef; color: #333; }
        .input-area { display: flex; gap: 10px; }
        .input-area input { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 25px; font-size: 14px; }
        .input-area button { padding: 12px 25px; background: #007bff; color: white; border: none; border-radius: 25px; cursor: pointer; }
        .input-area button:hover { background: #0056b3; }
        .state-section { margin-bottom: 20px; }
        .state-section h3 { margin-bottom: 10px; color: #666; font-size: 14px; text-transform: uppercase; }
        .state-content { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 15px; }
        .slot-item { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee; }
        .slot-name { font-weight: 500; }
        .slot-value { color: #007bff; }
        pre { font-size: 12px; overflow-x: auto; white-space: pre-wrap; }
        .status { padding: 5px 10px; border-radius: 4px; font-size: 12px; margin-bottom: 15px; }
        .status.connected { background: #d4edda; color: #155724; }
        .status.disconnected { background: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="container">
        <div class="panel panel-left">
            <h2>对话</h2>
            <div id="status" class="status disconnected">未连接</div>
            <div id="chat-box" class="chat-box"></div>
            <div class="input-area">
                <input type="text" id="message-input" placeholder="输入消息..." onkeypress="if(event.key==='Enter')sendMessage()">
                <button onclick="sendMessage()">发送</button>
            </div>
        </div>
        <div class="panel panel-right">
            <h2>对话状态</h2>
            <div class="state-section">
                <h3>槽位</h3>
                <div id="slots" class="state-content">暂无数据</div>
            </div>
            <div class="state-section">
                <h3>活动Flow</h3>
                <div id="flow" class="state-content">暂无</div>
            </div>
            <div class="state-section">
                <h3>原始状态</h3>
                <div class="state-content"><pre id="raw-state">{}</pre></div>
            </div>
        </div>
    </div>
    <script>
        const senderId = 'inspect_user_' + Date.now();
        let ws = null;
        function connect() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/tracker_stream`);
            ws.onopen = () => {
                document.getElementById('status').className = 'status connected';
                document.getElementById('status').textContent = '已连接';
                ws.send(JSON.stringify({action: 'retrieve', sender_id: senderId}));
            };
            ws.onclose = () => {
                document.getElementById('status').className = 'status disconnected';
                document.getElementById('status').textContent = '已断开 - 重连中...';
                setTimeout(connect, 3000);
            };
            ws.onmessage = (event) => {
                try {
                    const state = JSON.parse(event.data);
                    updateState(state);
                } catch (e) {
                    console.error('Parse error:', e);
                }
            };
        }
        function updateState(state) {
            // 更新槽位
            const slotsDiv = document.getElementById('slots');
            if (state.slots && Object.keys(state.slots).length > 0) {
                slotsDiv.innerHTML = Object.entries(state.slots)
                    .map(([name, value]) => `<div class="slot-item"><span class="slot-name">${name}</span><span class="slot-value">${JSON.stringify(value)}</span></div>`)
                    .join('');
            } else {
                slotsDiv.textContent = '暂无槽位';
            }
            // 更新Flow
            const flowDiv = document.getElementById('flow');
            flowDiv.textContent = state.active_flow || '暂无活动Flow';
            // 更新原始状态
            document.getElementById('raw-state').textContent = JSON.stringify(state, null, 2);
        }
        function addMessage(text, isUser) {
            const chatBox = document.getElementById('chat-box');
            const div = document.createElement('div');
            div.className = 'message ' + (isUser ? 'user-message' : 'bot-message');
            div.textContent = text;
            chatBox.appendChild(div);
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        async function sendMessage() {
            const input = document.getElementById('message-input');
            const text = input.value.trim();
            if (!text) return;
            addMessage(text, true);
            input.value = '';
            try {
                const response = await fetch('/webhooks/rest/webhook', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({sender: senderId, message: text})
                });
                const messages = await response.json();
                messages.forEach(msg => {
                    if (msg.text) addMessage(msg.text, false);
                });
            } catch (e) {
                addMessage('发送失败: ' + e.message, false);
            }
        }
        connect();
    </script>
</body>
</html>
"""
# 导出清单：按字母序排列，约束 from ... import * 的导出范围并为 IDE
# 提供准确的补全与跳转依据。
__all__ = [
    # Inspect 调试代理通道（装饰器式包装）
    "InspectProxy",
    # Tracker 状态广播流（观察者管理）
    "TrackerStream",
]