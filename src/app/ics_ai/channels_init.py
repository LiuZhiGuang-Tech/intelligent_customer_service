"""
channels - 通道入口模块
核心功能：聚合导出对话系统与用户交互的全部通道能力——base_channel
          定义输入/输出通道抽象契约与 UserMessage 消息载体，
          rest_channel/console_channel/socketio_channel 分别提供
          HTTP 接入、命令行交互、WebSocket 长连接三种接入实现，
          CollectingOutputChannel 为测试收集替身，InspectProxy 为
          调试代理（包装底层通道附加状态查看页面）。
执行流程：1. 模块导入：聚合导出五个子模块的公开符号（无 IO，
          fastapi/socketio 均由子模块延迟导入）；2. 应用装配时按
          部署形态选择通道实现（REST 挂路由 / 控制台跑交互循环 /
          Socket.IO 包 ASGI 应用）；3. 消息统一以 UserMessage 流入
          处理回调，回复经 OutputChannel 抽象送出。
副作用：模块导入无 IO（仅类定义与聚合导出）；各通道的网络/终端
        副作用见对应子模块文档。
主要依赖：app.ics_ai.channels.base_channel（抽象契约来源）、
          app.ics_ai.channels.rest_channel（HTTP 通道）、
          app.ics_ai.channels.console_channel（控制台通道）、
          app.ics_ai.channels.socketio_channel（Socket.IO 通道）、
          app.ics_ai.channels.inspect_proxy（调试代理）。
"""
# 从基础模块导入抽象契约与消息载体：输入/输出通道接口、标准用户
# 消息对象与测试收集输出通道，是全部具体实现的公共依赖。
from app.ics_ai.channels.base_channel import (
    CollectingOutputChannel,
    InputChannel,
    OutputChannel,
    UserMessage,
)
# 从控制台通道模块导入命令行交互实现：stdin/stdout 收发消息
# （输入通道跑交互循环，配对的输出通道渲染终端文本）。
from app.ics_ai.channels.console_channel import ConsoleChannel, ConsoleOutputChannel
# 从调试代理模块导入状态查看代理：包装底层通道并附加调试端点。
from app.ics_ai.channels.inspect_proxy import InspectProxy
# 从 REST 通道模块导入 HTTP 接入实现：webhook 端点接收用户消息。
from app.ics_ai.channels.rest_channel import RestChannel
# 从 Socket.IO 通道模块导入长连接实现：WebSocket 双向实时通信。
from app.ics_ai.channels.socketio_channel import SocketIOChannel
# 导出清单：from channels import * 时仅暴露以下名字，按字母序排列
# 为 IDE 提供准确的补全与跳转依据。
__all__ = [
    # 测试用消息收集输出通道
    "CollectingOutputChannel",
    # 命令行交互通道
    "ConsoleChannel",
    # 命令行输出通道（终端文本渲染）
    "ConsoleOutputChannel",
    # 输入通道抽象基类
    "InputChannel",
    # 调试代理通道（包装底层通道）
    "InspectProxy",
    # 输出通道抽象基类
    "OutputChannel",
    # HTTP REST 接入通道
    "RestChannel",
    # Socket.IO 长连接通道
    "SocketIOChannel",
    # 标准用户消息载体
    "UserMessage",
]