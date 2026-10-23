"""
rest_channel - REST 通道模块
核心功能：基于 HTTP REST API 的通道实现——RestChannel 提供接收用户
          消息的 FastAPI 端点（POST /webhooks/rest/webhook 收消息、
          GET /webhooks/rest/ 健康检查）；RestOutputChannel 作为请求级
          输出缓冲，收集单次请求产生的全部机器人消息并随 HTTP 响应体
          一次性返回。
执行流程：1. 应用启动时经 create_routes 将路由挂载到 FastAPI；2. 客户端
          POST 消息 JSON 到 webhook 端点；3. 端点解析 sender/message/
          metadata 字段并构造 UserMessage；4. 调用 on_new_message 回调
          交给 Agent 处理链；5. 将本次请求的输出缓冲序列化为 JSON 响应
          返回；6. 文本为空返回 400，处理异常返回 500。
副作用：本模块构造路由无 IO；webhook 端点处理请求时读取 HTTP 请求体
        （网络 IO）并触发回调的下游副作用（Agent 处理、输出通道内存
        收集）；日志记录错误信息；不读写本地文件/数据库。
主要依赖：fastapi（延迟导入的路由框架，仅 create_routes 运行时需要）、
          logging（错误日志）、app.ics_ai.channels.base_channel（输入/
          输出通道基类与 UserMessage 消息载体）。
"""
# 延迟注解求值：所有类型注解按字符串处理，规避前向引用与版本差异问题
# （from __future__ 导入必须位于文件最前，仅可在 docstring 之后）。
from __future__ import annotations
# 从标准库 collections.abc 导入回调抽象类型：create_routes 的回调参数
# 注解使用（PEP 585 推荐从 collections.abc 导入而非 typing）。
from collections.abc import Awaitable, Callable
# 导入 logging 模块：模块级 logger 用于记录消息处理异常。
import logging
# 从标准库 typing 导入 Any 与 TYPE_CHECKING：开放结构元素类型 + 仅类型
# 检查执行的导入分支标记。
from typing import Any, TYPE_CHECKING
# 从同包基类导入通道抽象接口与消息载体：RestChannel 继承 InputChannel、
# RestOutputChannel 继承 OutputChannel，UserMessage 为端点构造的消息对象。
from app.ics_ai.channels.base_channel import (
    InputChannel,
    OutputChannel,
    CollectingOutputChannel,
    UserMessage,
)
# 仅类型检查分支：fastapi 类型仅供注解使用，运行时由 create_routes
# 延迟导入，保持模块级零 fastapi 依赖。
if TYPE_CHECKING:
    # FastAPI 应用与请求对象类型
    from fastapi import FastAPI, Request
    # JSON 响应类型
    from fastapi.responses import JSONResponse
# 模块级日志器：以模块路径命名，继承全局日志配置
logger = logging.getLogger(__name__)
class RestOutputChannel(OutputChannel):
    """REST 输出通道
    职责：作为请求级消息缓冲，收集单个 HTTP 请求处理期间产生的全部
          机器人消息，等待端点在响应体中一次性返回。
    设计模式：与 CollectingOutputChannel 同型的收集器模式——不做真实
          投递，仅按序记录 send_response 的入参。
    生命周期：每个 webhook 请求创建一个实例，响应序列化完成后即弃；
          实例级隔离，无跨请求共享状态。
    Attributes:
        messages: 收集的消息字典列表
    """
    def __init__(self) -> None:
        """初始化 REST 输出通道
        核心功能：创建空的消息收集列表。
        副作用：无（仅初始化实例属性）。
        @return: 无返回。
        """
        # 消息收集列表：按发送顺序追加端点回调产生的消息
        self.messages: list[dict[str, Any]] = []
    @property
    def name(self) -> str:
        """通道名称。
        @return: 固定返回 "rest"，标识本通道为 REST 实现。
        """
        # 固定名称：同时作为 UserMessage.input_channel 的来源标识
        return "rest"
    async def send_response(
        self,
        recipient_id: str,
        message: dict[str, Any],
    ) -> None:
        """收集消息
        核心功能：将消息追加到本请求的缓冲列表（不投递，等待响应返回）。
        业务流程：直接将 message 字典按序追加到 messages；recipient_id
                  在 REST 模式下不单独记录（单请求单用户，响应即回给
                  请求方）。
        副作用：修改实例内部状态（messages 列表增长）；无任何 IO。
        @param recipient_id: 接收者 ID（本实现仅签名兼容，不使用）。
        @param message: 消息内容字典。
        @return: 无返回；本实现不抛异常。
        """
        # 追加消息到请求级缓冲：响应体将按此顺序序列化
        self.messages.append(message)
    def get_messages(self) -> list[dict[str, Any]]:
        """获取所有消息。
        @return: 按发送顺序排列的消息字典列表（内部列表的直接引用）。
        """
        # 直接返回内部列表引用：端点将其作为 JSON 响应体
        return self.messages
class RestChannel(InputChannel):
    """REST API 通道
    职责：提供 HTTP 接入端点——接收用户消息 JSON、构造标准 UserMessage、
          经回调移交 Agent 处理，并将本次请求收集的机器人消息以 JSON
          响应返回。
    设计模式：策略模式——InputChannel 的 REST 接入实现，与其他通道
          （SocketIO/控制台）可互换；create_routes 内部以闭包捕获
          回调与通道状态构造端点。
    生命周期：随应用启动创建并注册路由，常驻监听；输出通道为请求级
          短生命周期对象（见 RestOutputChannel）。
    端点：
    - POST /webhooks/rest/webhook: 发送消息
    - GET /webhooks/rest/: 健康检查
    """
    @property
    def name(self) -> str:
        """通道名称。
        @return: 固定返回 "rest"。
        """
        # 固定名称：写入 UserMessage.input_channel 标识消息来源
        return "rest"
    def get_output_channel(self) -> OutputChannel:
        """获取关联的输出通道。
        核心功能：返回与 REST 接入配对的输出通道实例。
        业务流程：每次调用返回全新 RestOutputChannel——HTTP 为请求-
          响应模型，缓冲必须按请求隔离，不能跨请求复用。
        副作用：无（仅构造内存对象）。
        @return: 新建的 RestOutputChannel 实例。
        """
        # 每请求新建：保证消息缓冲按请求隔离，响应后即弃
        return RestOutputChannel()
    def create_routes(
        self,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
    ) -> Any:
        """创建 FastAPI 路由
        核心功能：构造挂载健康检查与消息接收两个端点的 APIRouter。
        业务流程：1. 延迟导入 fastapi（未安装时仅在本方法调用处报错，
                  不影响模块导入）；2. 创建带 /webhooks/rest 前缀的
                  路由器；3. 注册 GET / 健康检查端点；4. 注册 POST
                  /webhook 消息端点（解析→校验→构造 UserMessage→回调
                  处理→返回缓冲消息）；5. 返回路由器供应用挂载。
        副作用：本方法自身无 IO（仅构造路由对象）；webhook 端点被调用
                  时产生网络 IO 与回调下游副作用。
        @param on_new_message: 消息处理回调——接收 UserMessage 并异步
                               完成 Agent 处理，不能为 None。
        @return: FastAPI APIRouter 实例（含 GET / 与 POST /webhook
                 两个端点）。
        """
        # 延迟导入 fastapi：仅在真正构造路由时加载，未安装不影响模块级导入
        from fastapi import APIRouter, Request
        # 延迟导入 JSON 响应类：端点返回结构化 JSON
        from fastapi.responses import JSONResponse
        # 创建路由器：统一前缀 /webhooks/rest，OpenAPI 文档归入 rest 标签
        router = APIRouter(prefix="/webhooks/rest", tags=["rest"])
        @router.get("/")
        async def health_check():
            """健康检查端点
            核心功能：供运维探活，报告服务与通道状态。
            业务流程：直接返回固定状态字典。
            副作用：无。
            @return: {"status": "ok", "channel": 通道名} 字典。
            """
            # 返回健康状态与通道名，供探活方校验
            return {"status": "ok", "channel": self.name}
        @router.post("/webhook")
        async def receive_message(request: Request):
            """接收用户消息端点
            核心功能：解析消息 JSON，构造 UserMessage 交给回调处理，
                      返回本请求收集的机器人消息。
            业务流程：1. 读取请求体 JSON；2. 提取 sender（缺省 default）、
                      message/text（双字段名兼容）、metadata（缺省空）；3.
                      文本为空返回 400；4. 构造 UserMessage 与请求级输出
                      通道；5. await 回调完成 Agent 处理；6. 以缓冲消息
                      为响应体返回；7. 任意异常记录日志并返回 500。
            副作用：读取网络请求体；触发回调的下游处理副作用；异常时
                    写错误日志。
            @param request: FastAPI 请求对象（承载原始 JSON 请求体）。
            @return: JSONResponse——成功为消息列表（可为空），文本为空
                     返回 400 错误对象，处理异常返回 500 错误对象。
            """
            try:
                # 第一步：读取并解析请求体 JSON（非法 JSON 抛异常走 500）
                data = await request.json()
                # 第二步：提取发送者 ID，缺省 "default"（匿名单用户场景）
                sender_id = data.get("sender", "default")
                # 第二步：提取消息文本——先取 message 键，缺失再回退
                # text 键，再缺失为空串（兼容两种请求字段命名）
                text = data.get("message", data.get("text", ""))
                # 第二步：提取扩展元数据，缺省空字典
                metadata = data.get("metadata", {})
                # 前置校验：空文本无法构成有效对话轮次，直接拒绝
                if not text:
                    # 返回 400 与错误说明，提示调用方补全文本
                    return JSONResponse(
                        status_code=400,
                        content={"error": "No message text provided"},
                    )
                # 第三步：构造标准用户消息（ID/时间戳由 __post_init__ 补齐）
                user_message = UserMessage(
                    text=text,
                    sender_id=sender_id,
                    input_channel=self.name,
                    metadata=metadata,
                )
                # 第四步：创建请求级输出通道，缓冲本次处理产生的全部消息
                output_channel = RestOutputChannel()
                # 第五步：调用处理回调，将消息移交 Agent 处理链
                # （回调内部经 get_output_channel 等方式向通道投递回复）
                await on_new_message(user_message)
                # 第六步：以收集到的消息列表作为 JSON 响应体返回
                return JSONResponse(content=output_channel.get_messages())
            # 兜底捕获：解析失败、回调异常等任何处理错误统一降级为 500
            except Exception as e:
                # 记录错误详情（含原始异常信息）便于排查
                logger.error(f"Error processing message: {e}")
                # 返回 500 与错误摘要，避免向客户端泄漏堆栈细节
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e)},
                )
        # 返回装配好的路由器：由应用统一 include_router 挂载
        return router
# 导出清单：按字母序排列，约束 from ... import * 的导出范围并为 IDE
# 提供准确的补全与跳转依据。
__all__ = [
    # REST 输入通道（HTTP 端点接入）
    "RestChannel",
    # REST 输出通道（请求级消息缓冲）
    "RestOutputChannel",
]