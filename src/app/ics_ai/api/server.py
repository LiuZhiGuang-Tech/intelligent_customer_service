"""
ICS AI FastAPI 应用服务器模块
核心功能：
    将对话 Agent 的能力以 Web 服务形式对外暴露，包含四类出口：
    - REST API：消息收发（/api/messages）、会话查询/重置、domain 与
      Flow 定义查看、完整 Tracker 状态导出；
    - WebSocket 实时通道（/api/stream）：按 session 分组管理连接，
      支持消息推送与心跳；
    - 健康检查（/ 与 /health）；
    - 调试页面（/inspect，读取本地模板 HTML）。
    另以 CORS 中间件放行跨域前端访问。
执行流程：
    1. 经工厂函数 create_app（或直接构造 ICSServer）创建服务器，
       __init__ 内调用 _create_app 生成 FastAPI 实例；
    2. _create_app 依次：挂载 CORS 中间件 → _register_routes 注册
       全部路由（含按 enable_inspect 决定是否挂调试页）；
    3. 消息类端点把请求封装后转交 Agent.handle_message，再把响应
       DTO 的 messages 列表适配为 JSON 返回，并同步广播到同会话的
       WebSocket 连接；
    4. run() 延迟导入 uvicorn 并阻塞式启动监听。
副作用：
    - 经 Agent 间接读写对话状态存储（tracker_store）；
    - 经 Agent 间接调用 LLM/检索等外部服务；
    - 维护进程内 WebSocket 连接表 _ws_connections（随连接增删而变）；
    - /inspect 端点读取本目录 templates/inspect.html 文件内容；
    - run() 占用网络端口监听（默认地址端口取自 shared.constants）。
主要依赖：
    fastapi / pydantic（Web 框架与请求-响应模型）、uvicorn（ASGI 服务器，
    run 内延迟导入）、app.ics_ai.shared.constants（默认监听地址端口）、
    app.ics_ai.channels.base_channel.UserMessage（用户消息 DTO）、
    app.ics_ai.agent.agent.Agent（对话核心，仅类型引用）。
名称备注：
    本模块的 Pydantic 响应模型 MessageResponse 与 agent.message_processor
    中的同名 DTO 是两个不同的类：前者描述 HTTP 响应结构（供 OpenAPI
    文档与校验），后者是 Agent.handle_message 的返回载体（含 messages/
    events/metadata）。注释中分别以"Pydantic 模型"与"响应 DTO"指代。
"""
# 启用延迟注解求值：类型注解按字符串存储，规避循环导入与旧版
# typing 写法兼容问题，FastAPI 经 get_type_hints 仍可正常解析。
from __future__ import annotations
# 导入 logging 模块：创建模块级 logger 记录处理异常。
import logging
# 从标准库 typing 导入 Any 与 TYPE_CHECKING：前者标注开放结构
# （元数据/槽位/按钮列表）的元素类型，后者包裹仅类型检查需要的导入。
from typing import Any, TYPE_CHECKING
# 从第三方 fastapi 导入核心构件：FastAPI 应用类、HTTPException（业务
# 错误转 HTTP 状态码）、WebSocket 与 WebSocketDisconnect（实时通道及
# 其断连异常）。
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
# 导入 CORS 中间件：为浏览器跨域请求（调试前端）放行响应头。
from fastapi.middleware.cors import CORSMiddleware
# 导入 HTMLResponse：/inspect 调试页面以此指定响应类型为 HTML。
from fastapi.responses import HTMLResponse
# 导入 pydantic BaseModel：四个请求/响应模型均继承自它获得校验能力。
from pydantic import BaseModel
# 导入用户消息 DTO：REST 端点用它封装请求字段后转交 Agent。
from app.ics_ai.channels.base_channel import UserMessage
# 导入默认监听配置：run() 的 host/port 形参缺省值来源。
from app.ics_ai.shared.constants import (
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
)
# 仅类型检查导入 Agent：运行时不执行（避免模块级重依赖），
# 函数签名中的 Agent 注解由 __future__ annotations 以字符串保留。
if TYPE_CHECKING:
    from app.ics_ai.agent.agent import Agent
# 创建模块级 logger：以 __name__（模块限定名）命名，供异常留痕。
logger = logging.getLogger(__name__)
# Pydantic模型定义
class MessageRequest(BaseModel):
    """消息请求模型。
    职责：承载 POST /api/messages 的请求体，自动完成入参校验
          （message 必填且为字符串）。
    设计模式：Pydantic 声明式 DTO。
    生命周期：随请求解析创建，端点返回后即可回收；无持久化。
    """
    # 发送者ID：作为 Agent 侧 tracker 的隔离键，同时回填响应的
    # recipient_id；缺省 "user"（单用户调试场景可省略）。
    sender: str = "user"
    # 消息正文：必填字段，无默认值；空串不在此拦截（交由 Agent 处理）。
    message: str
    # 消息元数据：渠道/时间戳等附加信息的开放字典；None 表示无元数据，
    # 转交 Agent 前经 `or {}` 兜底为空字典。
    metadata: dict[str, Any] | None = None
class MessageResponse(BaseModel):
    """消息响应模型（Pydantic）。
    职责：声明单条机器人回复的 JSON 结构，供 POST /api/messages 的
          response_model=List[MessageResponse] 做响应校验与 OpenAPI
          文档展示（与 agent.message_processor 的同名 DTO 无关）。
    设计模式：Pydantic 声明式 DTO。
    生命周期：随响应序列化创建，校验完成即可回收；无持久化。
    """
    # 接收者ID：回填请求的 sender，标识该条回复的目标用户。
    recipient_id: str
    # 回复文本：None 表示该条消息无纯文本（如纯按钮/纯图片消息）。
    text: str | None = None
    # 按钮列表：每项为按钮定义字典（如 title/payload）；None 表示无按钮。
    buttons: list[dict[str, Any]] | None = None
    # 图片URL：None 表示该条消息不含图片。
    image: str | None = None
    # 自定义数据：通道特定的扩展结构（如富媒体卡片）；None 表示无。
    custom: dict[str, Any] | None = None
class SessionInfo(BaseModel):
    """会话信息模型。
    职责：声明 GET /api/sessions/{session_id} 的响应结构，概括单个
          会话的槽位与对话进度。
    设计模式：Pydantic 声明式 DTO。
    生命周期：随响应序列化创建，校验完成即可回收；无持久化。
    """
    # 会话ID：tracker 的隔离键，与请求路径参数一致。
    session_id: str
    # 槽位快照：键为槽位名、值为当前填充值（Any 兼容各类型槽位）。
    slots: dict[str, Any]
    # 最近一条用户消息：字典由 UserMessage.to_dict() 生成；None 表示
    # 该会话尚无用户消息。
    latest_message: dict[str, Any] | None = None
    # 事件计数：当前为对话轮次数（dialogue_turns 长度），非负整数。
    events_count: int
class HealthResponse(BaseModel):
    """健康检查响应模型。
    职责：声明 / 与 /health 两个健康检查端点的响应结构。
    设计模式：Pydantic 声明式 DTO。
    生命周期：随响应序列化创建，校验完成即可回收；无持久化。
    """
    # 服务状态：固定 "ok"（能响应即视为存活）。
    status: str
    # 服务版本：读取 app.ics_ai 包级 __version__ 常量。
    version: str
    # Agent就绪标记：True 表示已装配 Agent 实例（可处理消息）。
    agent_ready: bool
class ICSServer:
    """FastAPI 应用服务器。
    职责：持有 Agent 实例与 FastAPI 应用，管理按 session 分组的
          WebSocket 连接表；把 HTTP/WebSocket 请求适配为 Agent 调用
          并将响应 DTO 适配为 JSON。
    设计模式：外观模式——把 Agent 的对话能力封装为一组 Web 端点；
          对外另配工厂函数 create_app 简化构造。
    生命周期：随应用装配创建（create_app 或直接构造），调用 run()
          后阻塞当前线程直至服务停止。
    """
    def __init__(
        self,
        agent: Agent | None = None,
        cors_origins: list[str] | None = None,
        enable_inspect: bool = True,
    ):
        """初始化服务器。
        业务流程：1. 保存 Agent 引用（允许暂缺，稍后经 set_agent 补齐）；
        2. 归一化 CORS 白名单；3. 保存调试页开关；4. 初始化空的
        WebSocket 连接表；5. 经 _create_app 构建 FastAPI 实例。
        副作用：创建进程内连接表字典与 FastAPI 应用实例（注册全部路由）。
        :param agent: Agent 实例，允许 None（装配后经 set_agent 注入，
                      未注入前消息端点返回 503）。
        :param cors_origins: 允许跨域的源列表（如 ["http://localhost:3000"]）；
                      None 表示放行所有源（"*"，仅调试场景）。
        :param enable_inspect: 是否注册 /inspect 调试页，默认 True。
        """
        # 保存 Agent 引用：None 表示尚未装配，端点请求时按 503 拒绝。
        self.agent = agent
        # 归一化 CORS 白名单：未提供时退化为放行所有源（调试友好）。
        self.cors_origins = cors_origins or ["*"]
        # 保存调试页开关：_register_routes 据此决定是否挂 /inspect。
        self.enable_inspect = enable_inspect
        # WebSocket连接管理：会话ID → 该会话的全部活跃连接列表
        # （同一会话允许多端同时在线，广播时逐一分发）。
        self._ws_connections: dict[str, list[WebSocket]] = {}
        # 创建FastAPI应用：构造期即完成中间件挂载与路由注册，
        # self.app 供 uvicorn 加载或直接导出。
        self.app = self._create_app()
    def _create_app(self) -> FastAPI:
        """创建FastAPI应用。
        业务流程：1. 实例化 FastAPI 并填写 OpenAPI 元信息；2. 挂载
        CORS 中间件放行跨域；3. 调用 _register_routes 注册全部端点。
        副作用：向新建的 app 实例添加中间件与路由（均限本实例内部）。
        :return: 配置完毕的 FastAPI 应用实例（未运行，待 run/uvicorn 启动）。
        """
        # 实例化应用：标题/描述/版本写入 OpenAPI 文档页眉，
        # docs_url/redoc_url 分别为 Swagger 与 ReDoc 文档路径。
        app = FastAPI(
            title="ICS AI",
            description="教学版对话系统API",
            version="0.1.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )
        # 添加CORS中间件：放行白名单源的全部方法与请求头；
        # allow_credentials=True 允许携带 Cookie 等凭证。
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        # 注册路由：全部端点以嵌套函数形式定义在 _register_routes 内，
        # 闭包直接捕获 self（访问 agent/连接表）。
        self._register_routes(app)
        # 返回配置完毕的应用实例。
        return app
    def _register_routes(self, app: FastAPI) -> None:
        """注册API路由。
        业务流程：在本方法内以装饰器依次注册——
            GET  /                            根路径健康检查；
            GET  /health                      健康检查端点；
            POST /api/messages                发送消息（核心入口）；
            GET  /api/sessions/{session_id}   查询会话状态；
            POST /api/sessions/{session_id}/reset  重置会话；
            GET  /api/domain                  查看 domain 配置；
            GET  /api/flows                   查看全部 Flow 定义；
            GET  /api/tracker/{session_id}/full  完整 Tracker 状态；
            WS   /api/stream                  WebSocket 实时消息流；
            GET  /inspect                     调试页面（按开关挂载）。
        副作用：向传入的 app 注册上述路由（修改 app 的路由表）。
        :param app: 待注册路由的 FastAPI 实例，不能为 None。
        """
        @app.get("/", response_model=HealthResponse)
        async def root():
            """根路径健康检查。
            业务流程：局部读取包版本号 → 组装 HealthResponse 字典
            （FastAPI 会按 response_model 校验并序列化）。
            :return: 含 status/version/agent_ready 三键的字典。
            """
            # 局部导入包版本号：仅响应时使用，避免模块级导入链路。
            from app.ics_ai import __version__
            # 组装响应字典：agent_ready 以"是否已装配 Agent 实例"判定。
            return {
                "status": "ok",
                "version": __version__,
                "agent_ready": self.agent is not None,
            }
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """健康检查端点。
            业务流程与 root() 完全一致，提供语义明确的探测路径
            （容器/负载均衡常用 /health 作为探针地址）。
            :return: 含 status/version/agent_ready 三键的字典。
            """
            # 局部导入包版本号：与 root() 同源，保证版本信息一致。
            from app.ics_ai import __version__
            # 组装响应字典：逻辑与 root() 相同。
            return {
                "status": "ok",
                "version": __version__,
                "agent_ready": self.agent is not None,
            }
        @app.post("/api/messages", response_model=list[MessageResponse])
        async def send_message(request: MessageRequest):
            """发送消息到对话系统。
            业务流程：1. 校验 Agent 已装配（否则 503）；2. 把请求体
            封装为 UserMessage DTO；3. 调 Agent.handle_message 驱动
            对话图得到响应 DTO；4. 把 DTO 的 messages 列表逐条适配为
            Pydantic 响应结构（回填 recipient_id）；5. 广播同会话的
            WebSocket 连接；6. 返回适配结果。任一步异常记日志并转 500。
            副作用：经 Agent 读写对话存储/调用 LLM；向同会话的活跃
            WebSocket 连接推送 bot_response 消息。
            :param request: 消息请求体（Pydantic 已完成字段校验）。
            :return: 机器人回复列表，每条含 recipient_id/text/buttons/
                     image/custom 五键；Agent 未装配抛 503，处理异常
                     抛 500（HTTPException）。
            """
            # 前置校验：Agent 缺位时返回 503 Service Unavailable。
            if not self.agent:
                raise HTTPException(status_code=503, detail="Agent not ready")
            try:
                # 创建用户消息：把请求体字段封装为通道层标准 DTO，
                # input_channel 标记来源为 REST 便于链路追踪。
                user_message = UserMessage(
                    text=request.message,
                    sender_id=request.sender,
                    input_channel="rest",
                    metadata=request.metadata or {},
                )
                # 处理消息：驱动 Agent 的 LangGraph 对话图（异步），
                # 返回 message_processor.MessageResponse 响应 DTO。
                response = await self.agent.handle_message(
                    message=user_message.text,
                    sender_id=user_message.sender_id,
                    metadata=user_message.metadata,
                )
                # 转换响应格式 - response 是 MessageResponse 响应 DTO，
                # 逐条把内部消息字典适配为 Pydantic 模型结构：
                # recipient_id 统一回填请求方 sender。
                result = []
                for msg in response.messages:
                    result.append(
                        {
                            "recipient_id": request.sender,
                            # msg.get 取不到时落 None，对应模型的可空字段。
                            "text": msg.get("text"),
                            "buttons": msg.get("buttons"),
                            "image": msg.get("image"),
                            "custom": msg.get("custom"),
                        }
                    )
                # 广播到WebSocket连接：让同会话的其他端（如调试页）
                # 实时收到这条回复，REST 响应本身照常返回。
                await self._broadcast_to_session(
                    request.sender,
                    {"type": "bot_response", "data": result},
                )
                # 返回适配结果：FastAPI 按 list[MessageResponse] 校验序列化。
                return result
            except Exception as e:
                # 异常留痕：记录完整错误信息便于排查。
                logger.error(f"Error processing message: {e}")
                # 转译为 500：避免把内部异常栈直接暴露给调用方。
                raise HTTPException(status_code=500, detail=str(e))
        @app.get("/api/sessions/{session_id}", response_model=SessionInfo)
        async def get_session(session_id: str):
            """获取会话状态。
            业务流程：1. 校验 Agent 已装配；2. 经 tracker_store 查找
            会话 tracker，不存在返回 404；3. 提取槽位/最近消息/轮次数
            组装 SessionInfo 字典。查询异常记日志并转 500。
            :param session_id: 会话ID（路径参数，tracker 隔离键）。
            :return: SessionInfo 结构字典；会话不存在抛 404，查询异常
                     抛 500（HTTPException）。
            """
            # 前置校验：Agent 缺位时返回 503。
            if not self.agent:
                raise HTTPException(status_code=503, detail="Agent not ready")
            try:
                # 查找 tracker：存储按 sender_id 隔离，未出现过的会话
                # 返回 None（不自动创建）。
                tracker = await self.agent.get_tracker(session_id)
                # 会话不存在：返回 404 交由调用方处理。
                if not tracker:
                    raise HTTPException(status_code=404, detail="Session not found")
                # 组装响应：latest_message 为 tracker.UserMessage 实例，
                # 须经 to_dict() 转字典才能通过 SessionInfo 的 dict 字段
                # 校验（直接传实例会被 Pydantic 拒绝）。
                return {
                    "session_id": session_id,
                    "slots": tracker.get_all_slots(),
                    "latest_message": (
                        tracker.latest_message.to_dict()
                        if tracker.latest_message
                        else None
                    ),
                    "events_count": len(tracker.dialogue_turns),
                }
            except HTTPException:
                # 404 等已定性的 HTTP 异常原样上抛，不转 500。
                raise
            except Exception as e:
                # 其余异常留痕并转译为 500。
                logger.error(f"Error getting session: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        @app.post("/api/sessions/{session_id}/reset")
        async def reset_session(session_id: str):
            """重置会话。
            业务流程：1. 校验 Agent 已装配；2. 调 Agent.reset_tracker
            清空该会话的槽位/对话栈/历史；3. 返回确认信息。异常记日志
            并转 500。
            副作用：清空 tracker_store 中该会话的持久化状态（不可恢复）。
            :param session_id: 会话ID（路径参数）。
            :return: {"status": "ok", "message": "..."} 确认字典；
                     重置异常抛 500（HTTPException）。
            """
            # 前置校验：Agent 缺位时返回 503。
            if not self.agent:
                raise HTTPException(status_code=503, detail="Agent not ready")
            try:
                # 执行重置：由 Agent 委派 tracker_store 清空会话状态。
                await self.agent.reset_tracker(session_id)
                # 返回确认信息：message 回显会话ID便于核对。
                return {"status": "ok", "message": f"Session {session_id} reset"}
            except Exception as e:
                # 异常留痕并转译为 500。
                logger.error(f"Error resetting session: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        @app.get("/api/domain")
        async def get_domain():
            """获取domain配置。
            业务流程：校验 Agent 已装配 → 调 Domain.to_dict() 把槽位/
            动作/响应模板/Flow 引用整体序列化。
            :return: domain 全量配置字典（结构见 Domain.to_dict）；
                     Agent 未装配抛 503（HTTPException）。
            """
            # 前置校验：Agent 缺位时返回 503。
            if not self.agent:
                raise HTTPException(status_code=503, detail="Agent not ready")
            # 序列化 domain：Domain 的序列化方法名为 to_dict
            # （原代码误写为 as_dict，运行时会抛 AttributeError）。
            return self.agent.domain.to_dict()
        @app.get("/api/flows")
        async def get_flows():
            """获取所有Flow定义。
            业务流程：1. 校验 Agent 已装配；2. 遍历 FlowsList 中的每个
            Flow；3. 对每个 Flow 再遍历其 steps，把步骤序列化为扁平
            字典（含步骤类型枚举取值）；4. 组装 [Flow, ...] 列表返回。
            :return: Flow 定义列表，每项含 id/description/steps 三键；
                     Agent 未装配抛 503（HTTPException）。
            """
            # 前置校验：Agent 缺位时返回 503。
            if not self.agent:
                raise HTTPException(status_code=503, detail="Agent not ready")
            # 初始化输出列表：逐个 Flow 追加序列化结果。
            flows_data = []
            # 防御性判断：agent.flows 构造期已兜底非 None（FlowsList
            # 数据类恒真值），此判断仅为防御异常装配情形。
            if self.agent.flows:
                # 外层遍历：flows 是 FlowsList 容器，.flows 才是列表。
                for flow in self.agent.flows.flows:
                    # 初始化本 Flow 的步骤序列化列表。
                    steps_data = []
                    # 内层遍历：逐个步骤抽字典。
                    for step in flow.steps:
                        # 组装单步骤字典：常规字段直接取值。
                        step_dict = {
                            "id": step.id,
                            "action": step.action,
                            "next": step.next,
                            "collect": step.collect,
                            # "set_slots": step.set_slots if hasattr(step, 'set_slots') else None,
                            # set_slots 存于步骤的 slots 属性（列表格式），
                            # 经 as_dict 还原；步骤未预设槽位时该键缺失，
                            # .get 落 None。
                            "set_slots": step.as_dict().get("set_slots"),
                            # step_type 为枚举时取其字符串值；防御式兼容
                            # 直接存字符串的情形（hasattr 分支）。
                            "step_type": step.step_type.value
                            if hasattr(step.step_type, "value")
                            else str(step.step_type),
                        }
                        # 追加进本 Flow 的步骤列表。
                        steps_data.append(step_dict)
                    # 组装单 Flow 字典：id/描述/步骤列表三键。
                    flows_data.append(
                        {
                            "id": flow.id,
                            "description": flow.description,
                            "steps": steps_data,
                        }
                    )
            # 返回全部 Flow 的序列化列表（无 Flow 时为空列表）。
            return flows_data
        @app.get("/api/tracker/{session_id}/full")
        async def get_full_tracker(session_id: str):
            """获取完整的Tracker状态。
            包括：slots, events, flow_stack, messages 等。
            业务流程：1. 校验 Agent 已装配；2. 查找 tracker（404 兜底）；
            3. 遍历对话轮次重建事件时间线（用户/Bot 消息各记一条）；
            4. 从对话栈过滤 Flow 帧生成当前栈快照；5. 摘录 Flow 执行
            历史；6. 组装完整状态字典返回。异常记日志并转 500。
            :param session_id: 会话ID（路径参数）。
            :return: 完整状态字典（sender_id/slots/events/flow_stack/
                     flow_history/active_flow/latest_action/
                     latest_message 七键）；会话不存在抛 404，异常抛 500。
            """
            # 前置校验：Agent 缺位时返回 503。
            if not self.agent:
                raise HTTPException(status_code=503, detail="Agent not ready")
            try:
                # 查找 tracker：未出现过的会话返回 None。
                tracker = await self.agent.get_tracker(session_id)
                # 会话不存在：返回 404。
                if not tracker:
                    raise HTTPException(status_code=404, detail="Session not found")
                # 构建完整的 tracker 状态：
                # 事件时间线：按轮次顺序重放用户与 Bot 消息。
                events = []
                # 遍历每个对话轮次。
                for turn in tracker.dialogue_turns:
                    # 本轮有用户消息才记录 user 事件。
                    if turn.user_message:
                        events.append(
                            {
                                "event": "user",
                                "text": turn.user_message.text,
                                "timestamp": turn.user_message.timestamp,
                            }
                        )
                    # Bot 消息一轮可多条，逐条记录 bot 事件。
                    for bot_msg in turn.bot_messages:
                        events.append(
                            {
                                "event": "bot",
                                "text": bot_msg.text,
                                # 防御式取时间戳：字段缺失时落 None。
                                "timestamp": getattr(bot_msg, "timestamp", None),
                            }
                        )
                # Flow 历史
                # 当前栈快照：只收集 Flow 类型的栈帧。
                flow_stack = []
                # 遍历对话栈全部帧（含非 Flow 帧）。
                for frame in tracker.dialogue_stack.frames:
                    # 局部导入 FlowStackFrame 类型：仅此 isinstance
                    # 判断使用（导入系统有缓存，循环内无重复开销）。
                    from app.ics_ai.dialogue_understanding.stack.stack_frame import (
                        FlowStackFrame,
                    )
                    # 类型过滤：仅 Flow 栈帧携带 flow_id/step_id。
                    if isinstance(frame, FlowStackFrame):
                        flow_stack.append(
                            {
                                "flow_id": frame.flow_id,
                                "step_id": frame.step_id,
                                "frame_id": frame.frame_id,
                            }
                        )
                # Flow 执行历史（包括已完成的）
                # 历史列表：逐条摘录起止时间与完成标记。
                flow_history = []
                # 遍历 tracker 累积的执行历史字典。
                for hist in tracker.flow_history:
                    flow_history.append(
                        {
                            # .get 兜底：历史条目缺键时落空值不报错。
                            "flow_id": hist.get("flow_name", ""),
                            "started_at": hist.get("started_at", ""),
                            "ended_at": hist.get("ended_at"),
                            "completed": hist.get("completed", False),
                        }
                    )
                # 组装完整响应：latest_message 按有无分别构造（有消息
                # 取其文本与时间戳，无则整体置 None）。
                return {
                    "sender_id": session_id,
                    "slots": tracker.get_all_slots(),
                    "events": events,
                    "flow_stack": flow_stack,
                    "flow_history": flow_history,
                    "active_flow": tracker.active_flow,
                    "latest_action": tracker.latest_action_name,
                    "latest_message": {
                        "text": tracker.latest_message.text
                        if tracker.latest_message
                        else None,
                        "timestamp": tracker.latest_message.timestamp
                        if tracker.latest_message
                        else None,
                    }
                    if tracker.latest_message
                    else None,
                }
            except HTTPException:
                # 404 等已定性的 HTTP 异常原样上抛。
                raise
            except Exception as e:
                # 其余异常留痕并转译为 500。
                logger.error(f"Error getting full tracker: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        @app.websocket("/api/stream")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket实时消息流。
            支持的消息类型：
            - connect: 连接并指定session_id
            - message: 发送消息
            - ping: 心跳
            业务流程：1. 接受握手；2. 循环收取 JSON 帧并按 type 分派：
               connect → 登记连接表并回执 connected；message → 调
               Agent 处理并回推 bot_response（异常回推 error）；ping →
               回 pong；3. 断连或异常时从连接表摘除本连接。
            副作用：读写 _ws_connections 连接表；经 Agent 读写对话
            存储/调用 LLM；向客户端推送响应帧。
            :param websocket: FastAPI 注入的 WebSocket 连接对象。
            """
            # 接受握手：完成 WebSocket 升级，此后可双向收发。
            await websocket.accept()
            # 会话ID初始为 None：connect 帧到达前连接尚未归属任何会话。
            session_id = None
            try:
                # 主循环：持续收取客户端 JSON 帧（阻塞等待下一条）。
                while True:
                    # 收取并解析一条 JSON 消息。
                    data = await websocket.receive_json()
                    # 取消息类型：缺省按 message 处理（宽松协议）。
                    msg_type = data.get("type", "message")
                    # 分支一：connect —— 登记连接并回执。
                    if msg_type == "connect":
                        # 取会话ID：未指定时落 "default"（共享兜底会话）。
                        session_id = data.get("session_id", "default")
                        # 登记进连接表：加入该会话的连接列表。
                        self._add_ws_connection(session_id, websocket)
                        # 回执连接成功：回显生效的会话ID。
                        await websocket.send_json(
                            {
                                "type": "connected",
                                "session_id": session_id,
                            }
                        )
                    # 分支二：message —— 处理对话消息。
                    elif msg_type == "message":
                        # Agent 缺位：回推 error 帧后继续循环（不断连）。
                        if not self.agent:
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "message": "Agent not ready",
                                }
                            )
                            continue
                        # 取消息文本：兼容 message/text 两种键名，
                        # 均缺失时落空串。
                        text = data.get("message", data.get("text", ""))
                        # 取发送者：未指定时沿用 connect 登记的会话ID，
                        # 连 connect 都没有则落 "default"。
                        sender_id = data.get("sender_id", session_id or "default")
                        try:
                            # 调 Agent 处理：驱动对话图（异步），
                            # 返回响应 DTO。
                            response = await self.agent.handle_message(
                                message=text,
                                sender_id=sender_id,
                            )
                            # response 是 MessageResponse 响应 DTO，
                            # 需要取出 messages 列表回推：客户端按
                            # bot_response 类型消费。
                            await websocket.send_json(
                                {
                                    "type": "bot_response",
                                    "data": response.messages,
                                }
                            )
                        except Exception as e:
                            # 单条消息失败不断连：回推 error 帧后
                            # 继续服务后续消息。
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "message": str(e),
                                }
                            )
                    # 分支三：ping —— 心跳保活，原样回 pong。
                    elif msg_type == "ping":
                        await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                # 正常断连：从连接表摘除本连接（已归属会话时）。
                if session_id:
                    self._remove_ws_connection(session_id, websocket)
            except Exception as e:
                # 异常断连：留痕后同样摘除，避免连接表泄漏死连接。
                logger.error(f"WebSocket error: {e}")
                if session_id:
                    self._remove_ws_connection(session_id, websocket)
        # 调试页面
        # 按开关挂载：关闭时不注册 /inspect，减少暴露面。
        if self.enable_inspect:
            @app.get("/inspect", response_class=HTMLResponse)
            async def inspect_page():
                """调试页面。
                业务流程：委托 _get_inspect_html 读取模板文件全文，
                以 HTML 响应返回给浏览器。
                :return: 调试页 HTML 字符串。
                """
                # 读取模板内容并返回。
                return self._get_inspect_html()
    def _add_ws_connection(self, session_id: str, ws: WebSocket) -> None:
        """添加WebSocket连接。
        业务流程：会话ID首次出现时先建空列表，再把连接追加进该会话
        的连接列表。
        副作用：修改 _ws_connections 连接表（可能新增键）。
        :param session_id: 会话ID（连接表键），不能为 None。
        :param ws: 待登记的 WebSocket 连接对象。
        """
        # 惰性建列表：该会话无连接时先初始化空列表再追加。
        if session_id not in self._ws_connections:
            self._ws_connections[session_id] = []
        # 追加连接：同一会话支持多端并存。
        self._ws_connections[session_id].append(ws)
    def _remove_ws_connection(self, session_id: str, ws: WebSocket) -> None:
        """移除WebSocket连接。
        业务流程：会话存在于连接表时，先移除指定连接，列表被清空后
        连同会话键一并删除（保持连接表不含空会话）。
        副作用：修改或收缩 _ws_connections 连接表。
        :param session_id: 会话ID（连接表键）。
        :param ws: 待摘除的 WebSocket 连接对象；不在表中时静默跳过。
        """
        # 会话存在才处理：避免摘除未登记会话时报 KeyError。
        if session_id in self._ws_connections:
            # 连接在列表中才移除：保证幂等（重复清理不报错）。
            if ws in self._ws_connections[session_id]:
                self._ws_connections[session_id].remove(ws)
            # 列表已空则删除会话键：防止连接表积累空条目。
            if not self._ws_connections[session_id]:
                del self._ws_connections[session_id]
    async def _broadcast_to_session(
        self,
        session_id: str,
        message: dict[str, Any],
    ) -> None:
        """广播消息到会话的所有连接。
        业务流程：1. 会话无连接直接返回；2. 逐个连接推送 JSON 帧，
        推送失败的连接记入待清理名单；3. 循环结束后统一摘除断开的
        连接（边遍历边删会跳元素，故先收集后清理）。
        副作用：向同会话全部活跃连接发送数据；可能收缩连接表。
        :param session_id: 目标会话ID（连接表键）。
        :param message: 待广播的消息字典（含 type 与业务数据键）。
        """
        # 会话无连接：无人可发，直接返回。
        if session_id not in self._ws_connections:
            return
        # 断连收集名单：推送失败的连接先记录，避免遍历中修改列表。
        disconnected = []
        # 遍历该会话全部连接逐一推送。
        for ws in self._ws_connections[session_id]:
            try:
                # 推送 JSON 帧。
                await ws.send_json(message)
            except Exception:
                # 推送失败视作连接已断，记入待清理名单。
                disconnected.append(ws)
        # 清理断开的连接：统一摘除，保持连接表干净。
        for ws in disconnected:
            self._remove_ws_connection(session_id, ws)
    def _get_inspect_html(self) -> str:
        """获取调试页面HTML。
        业务流程：1. 拼出与本文件同目录的 templates/inspect.html 绝对
        路径（经 os.path，保证任意工作目录启动都定位正确）；2. 全文
        读取并返回。
        副作用：读取磁盘上的模板文件。
        :return: 调试页 HTML 全文；模板缺失时抛 FileNotFoundError
                （未捕获，由 FastAPI 转 500）。
        """
        # 局部导入 os：仅本方法使用路径拼接。
        import os
        # 拼接模板绝对路径：以本文件所在目录为基准定位 templates 子目录。
        template_path = os.path.join(
            os.path.dirname(__file__), "templates", "inspect.html"
        )
        # 以 UTF-8 全文读取模板并返回字符串。
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    def set_agent(self, agent: Agent) -> None:
        """设置Agent实例。
        业务流程：以注入方式补齐/替换 Agent 引用，支持"先建服务器、
        后装配 Agent"的启动顺序。
        副作用：覆盖实例属性 agent（此后各端点按新 Agent 服务）。
        :param agent: Agent 实例，不能为 None。
        """
        # 保存 Agent 引用。
        self.agent = agent
    def run(
        self,
        host: str = DEFAULT_SERVER_HOST,
        port: int = DEFAULT_SERVER_PORT,
    ) -> None:
        """运行服务器。
        业务流程：1. 延迟导入 uvicorn（仅在真正启动时才需要）；2.
        以当前 self.app 为目标阻塞式启动 ASGI 服务。
        副作用：监听网络端口并阻塞当前线程直至服务停止；
        uvicorn 按其自身配置输出 info 级访问日志。
        :param host: 监听地址，缺省取 DEFAULT_SERVER_HOST（0.0.0.0，
                     即所有网卡）。
        :param port: 监听端口，缺省取 DEFAULT_SERVER_PORT（5005）。
        """
        # 延迟导入 uvicorn：模块级导入会拖累非服务场景的加载速度。
        import uvicorn
        # 启动留痕：输出监听地址与端口（排查端口占用的入口日志）。
        logger.info(f"Starting server on {host}:{port}")
        # 阻塞式启动：uvicorn 接管事件循环，直到进程被终止。
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info",
        )
def create_app(
    agent: Agent | None = None,
    cors_origins: list[str] | None = None,
    enable_inspect: bool = True,
) -> FastAPI:
    """创建FastAPI应用。
    工厂函数，便于在不同场景下创建应用。
    业务流程：以入参构造 ICSServer（构造期完成中间件挂载与路由
    注册）→ 直接返回其内部 FastAPI 实例（服务器对象仅作装配载体，
    路由闭包经 self 持有状态，实例被 app 的路由引用间接保活）。
    :param agent: Agent 实例，允许 None（稍后需自行经服务器对象注入）。
    :param cors_origins: 允许跨域的源列表；None 表示放行所有源。
    :param enable_inspect: 是否注册 /inspect 调试页，默认 True。
    :return: 配置完毕、可交给 uvicorn 运行的 FastAPI 应用实例。
    """
    # 构造服务器对象：全部装配工作在 __init__ 内完成。
    server = ICSServer(
        agent=agent,
        cors_origins=cors_origins,
        enable_inspect=enable_inspect,
    )
    # 返回内部 FastAPI 实例。
    return server.app
# 导出：按字母序排列，便于 IDE 自动补全与 diff 清晰。
__all__ = [
    # 服务器类（含连接管理与全部路由装配）。
    "ICSServer",
    # 健康检查响应模型。
    "HealthResponse",
    # 消息请求模型。
    "MessageRequest",
    # 消息响应模型（Pydantic）。
    "MessageResponse",
    # 会话信息模型。
    "SessionInfo",
    # FastAPI 应用工厂函数。
    "create_app",
]