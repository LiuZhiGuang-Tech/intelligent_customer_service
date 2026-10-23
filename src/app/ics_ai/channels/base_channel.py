"""
base_channel - 通道基类模块
核心功能：定义消息通道体系的抽象接口与标准数据载体——UserMessage/
          BotResponse 两个消息数据类、OutputChannel 输出通道抽象基类
          （含发送文本/按钮/图片/自定义 JSON 的便捷方法）、Collecting
          OutputChannel 收集型测试通道、InputChannel 输入通道抽象基类，
          以及 MessageHandler 异步消息处理回调类型。
执行流程：1. 输入通道子类（如 REST/WebSocket）接收原始请求并构造
          UserMessage；2. 消息经 MessageHandler 回调交给 Agent 处理链；
          3. Agent 产出 BotResponse 后，通过输出通道的 send_response
          及其便捷方法投递给用户；4. 测试场景以 CollectingOutputChannel
          替换真实输出通道，捕获全部外发消息做断言。
副作用：模块导入无 IO；UserMessage 构造时经 __post_init__ 读取系统
        时钟并生成 UUID（仅补齐缺失字段）；CollectingOutputChannel
        维护进程内消息列表（纯内存）；不读写文件/数据库、不发起网络请求。
主要依赖：abc（抽象基类机制）、dataclasses（消息载体自动构造）、
          collections.abc（Callable/Awaitable 回调类型）、time/uuid
          （时间戳与消息 ID 生成）。
"""
# 延迟注解求值：所有类型注解按字符串处理，规避前向引用与版本差异问题
# （from __future__ 导入必须位于文件最前，仅可在 docstring 之后）。
from __future__ import annotations
# 从标准库 abc 导入抽象基类机制：ABC 声明抽象类，abstractmethod 标记
# 子类必须实现的方法。
from abc import ABC, abstractmethod
# 从标准库 collections.abc 导入回调相关抽象类型：MessageHandler 别名
# 使用（PEP 585 推荐从 collections.abc 导入而非 typing）。
from collections.abc import Awaitable, Callable
# 从标准库 dataclasses 导入数据类工具：@dataclass 自动生成消息载体的
# 构造/比较方法，field 支持可变默认值（dict/list）。
from dataclasses import dataclass, field
# 导入 time 模块：__post_init__ 补齐缺失时间戳时读取系统时钟。
import time
# 从标准库 typing 导入 Any：metadata/buttons 等开放结构的元素类型。
from typing import Any
# 导入 uuid 模块：__post_init__ 补齐缺失消息 ID 时生成随机 UUID。
import uuid
@dataclass
class UserMessage:
    """用户消息数据类
    职责：作为用户输入的标准载体，贯通"通道接收 → Agent 处理 → 存储
          记录"全链路，统一承载文本、来源通道、会话标识与扩展元数据。
    设计模式：数据传输对象（DTO）——借助 @dataclass 自动生成 __init__/
          __repr__/__eq__，仅承载数据不含业务逻辑。
    生命周期：输入通道收到用户请求时构造，沿处理链传递至 Agent 与
          Tracker，单轮消息处理完成后由垃圾回收；无持久化。
    Attributes:
        text: 消息文本
        sender_id: 发送者ID
        input_channel: 输入通道名称
        message_id: 消息ID
        timestamp: 时间戳
        metadata: 元数据
    """
    # 消息正文文本：必填字段，无默认值
    text: str
    # 发送者标识：默认 "default"（单用户场景可缺省）
    sender_id: str = "default"
    # 来源输入通道名称：默认 "default"，用于区分 REST/WebSocket 等接入方
    input_channel: str = "default"
    # 消息唯一 ID：UUID 字符串；缺省时由 __post_init__ 自动生成
    message_id: str | None = None
    # 消息时间戳：Unix 秒级浮点数；缺省时由 __post_init__ 自动填充
    timestamp: float | None = None
    # 扩展元数据：通道相关的附加信息（如原始请求头）；默认空字典
    metadata: dict[str, Any] = field(default_factory=dict)
    def __post_init__(self):
        """构造后补齐自动生成的字段
        核心功能：为未提供的 message_id/timestamp 填充默认生成值。
        业务流程：1. message_id 为 None 则生成随机 UUID 字符串；2.
                  timestamp 为 None 则取当前 Unix 时间戳。
        副作用：读取系统时钟（time.time）；生成 UUID（随机数消耗）。
        @return: 无返回；仅就地修改实例字段。
        """
        # 前置判断：调用方未提供消息 ID 时自动生成，保证消息可追溯
        if self.message_id is None:
            # 生成随机 UUID v4 字符串作为消息唯一标识
            self.message_id = str(uuid.uuid4())
        # 前置判断：调用方未提供时间戳时自动填充当前时刻
        if self.timestamp is None:
            # 取 Unix 秒级浮点时间戳
            self.timestamp = time.time()
@dataclass
class BotResponse:
    """机器人响应数据类
    职责：作为机器人输出的标准载体，统一承载回复文本与可选的按钮、
          图片、附件、自定义数据等多媒体扩展字段。
    设计模式：数据传输对象（DTO）——仅承载数据不含业务逻辑。
    生命周期：Agent 处理链产出后构造，由输出通道消费发送，发送完成
          即可回收；无持久化。
    Attributes:
        text: 响应文本
        recipient_id: 接收者ID
        buttons: 按钮列表
        image: 图片URL
        attachment: 附件
        custom: 自定义数据
    """
    # 回复正文文本：默认空串（纯多媒体响应可无文本）
    text: str = ""
    # 接收者标识：对应用户的 sender_id；默认空串由发送方补齐
    recipient_id: str = ""
    # 按钮列表：每项为按钮定义字典（如 title/payload）；默认空列表
    buttons: list[dict[str, Any]] = field(default_factory=list)
    # 图片 URL：可选，指向机器人要展示的图片资源
    image: str | None = None
    # 附件标识/URL：可选，承载文件等附加资源
    attachment: str | None = None
    # 自定义数据字典：可选，透传通道特有的结构化内容
    custom: dict[str, Any] | None = None
class OutputChannel(ABC):
    """输出通道抽象基类
    职责：定义"发送消息给用户"的统一接口——子类只需实现 name 属性与
          send_response 核心方法，即可免费获得文本、按钮、图片、自定义
          JSON 四类便捷发送能力。
    设计模式：策略模式——不同投递介质（收集器/REST/控制台等）作为可
          互换策略；同时是模板方法模式——便捷方法（send_text_message
          等）为不变流程，send_response 为可变的介质适配点。
    生命周期：随 Agent 装配或输入通道关联而创建，整个运行期复用；
          本基类不可实例化。
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """通道名称。
        @return: 通道唯一名称字符串（如 "collector"），用于日志与路由。
        """
        # 抽象方法体：仅占位，强制子类提供实现
        raise NotImplementedError()
    @abstractmethod
    async def send_response(
        self,
        recipient_id: str,
        message: dict[str, Any],
    ) -> None:
        """发送响应消息（抽象方法）
        核心功能：将一条消息投递到具体介质（契约，由子类实现）。
        业务流程（子类实现要求）：1. 按 recipient_id 定位投递目标；
                  2. 将 message 字典按介质协议序列化并发送。
        副作用：由子类定义——收集器追加内存列表、REST 发起 HTTP 请求等。
        @param recipient_id: 接收者 ID，不能为空。
        @param message: 消息内容字典（text/buttons/image 等键的开放结构）。
        @return: 无返回；发送失败时子类自行决定异常策略。
        """
        # 抽象方法体：仅占位，强制子类提供实现
        raise NotImplementedError()
    async def send_text_message(
        self,
        recipient_id: str,
        text: str,
        **kwargs: Any,
    ) -> None:
        """发送文本消息（便捷方法）
        核心功能：以 {"text": ...} 消息结构复用 send_response 发送纯文本。
        业务流程：组装仅含 text（及额外键）的消息字典，委托抽象方法
                  send_response 完成实际投递。
        副作用：继承 send_response 的子类副作用。
        @param recipient_id: 接收者 ID。
        @param text: 消息文本，不能为 None。
        @param kwargs: 额外参数，原样并入消息字典顶层。
        @return: 无返回；异常沿 send_response 向上传播。
        """
        # 组装 {"text": 文本, **额外键} 结构并委托核心方法投递
        await self.send_response(recipient_id, {"text": text, **kwargs})
    async def send_text_with_buttons(
        self,
        recipient_id: str,
        text: str,
        buttons: list[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        """发送带按钮的文本消息（便捷方法）
        核心功能：以 {"text": ..., "buttons": [...]} 结构复用 send_response
                  发送带交互按钮的消息。
        业务流程：组装文本与按钮列表的消息字典，委托 send_response 投递。
        副作用：继承 send_response 的子类副作用。
        @param recipient_id: 接收者 ID。
        @param text: 消息文本。
        @param buttons: 按钮定义列表，每项为按钮属性字典。
        @param kwargs: 额外参数，原样并入消息字典顶层。
        @return: 无返回；异常沿 send_response 向上传播。
        """
        # 组装文本 + 按钮结构并委托核心方法投递
        await self.send_response(
            recipient_id,
            {"text": text, "buttons": buttons, **kwargs},
        )
    async def send_image_url(
        self,
        recipient_id: str,
        image: str,
        **kwargs: Any,
    ) -> None:
        """发送图片消息（便捷方法）
        核心功能：以 {"image": <URL>} 结构复用 send_response 发送图片。
        业务流程：组装图片 URL 消息字典，委托 send_response 投递。
        副作用：继承 send_response 的子类副作用。
        @param recipient_id: 接收者 ID。
        @param image: 图片资源 URL 字符串。
        @param kwargs: 额外参数，原样并入消息字典顶层。
        @return: 无返回；异常沿 send_response 向上传播。
        """
        # 组装图片 URL 结构并委托核心方法投递
        await self.send_response(recipient_id, {"image": image, **kwargs})
    async def send_custom_json(
        self,
        recipient_id: str,
        json_message: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """发送自定义JSON消息（便捷方法）
        核心功能：以 {"custom": ...} 结构复用 send_response 透传通道特有
                  的结构化内容。
        业务流程：将业务字典包进 custom 键，委托 send_response 投递。
        副作用：继承 send_response 的子类副作用。
        @param recipient_id: 接收者 ID。
        @param json_message: 自定义业务数据字典。
        @param kwargs: 额外参数，原样并入消息字典顶层。
        @return: 无返回；异常沿 send_response 向上传播。
        """
        # 组装 custom 包装结构并委托核心方法投递
        await self.send_response(recipient_id, {"custom": json_message, **kwargs})
class CollectingOutputChannel(OutputChannel):
    """收集输出通道
    职责：将所有外发消息按序收集到内存列表，供测试断言或同步处理读取。
    设计模式：测试替身（Spy 模式）——OutputChannel 的可观测实现，
          不做真实投递，仅记录调用参数。
    生命周期：测试用例中按需创建，用例结束即弃；实例级隔离，无全局状态。
    Attributes:
        messages: 收集的消息字典列表
    """
    def __init__(self) -> None:
        """初始化收集通道
        核心功能：创建空的消息收集列表。
        副作用：无（仅初始化实例属性）。
        @return: 无返回。
        """
        # 消息收集列表：按发送顺序追加 {"recipient_id": ..., **message}
        self.messages: list[dict[str, Any]] = []
    @property
    def name(self) -> str:
        """通道名称。
        @return: 固定返回 "collector"。
        """
        # 固定名称：标识本通道为收集器实现
        return "collector"
    async def send_response(
        self,
        recipient_id: str,
        message: dict[str, Any],
    ) -> None:
        """收集消息
        核心功能：将接收者 ID 与消息内容合并后追加到内存列表。
        业务流程：1. 以 recipient_id 为首键构造合并字典；2. 字典展开
                  (**message) 将消息各键并入；3. 键冲突时 message 侧
                  覆盖 recipient_id；4. 追加到 messages 列表。
        副作用：修改实例内部状态（messages 列表增长）；无任何 IO。
        @param recipient_id: 接收者 ID。
        @param message: 消息内容字典。
        @return: 无返回；本实现不抛异常。
        """
        # 合并接收者与消息内容后追加记录（模拟一次投递）
        self.messages.append(
            {
                "recipient_id": recipient_id,
                **message,
            }
        )
    def get_messages(self) -> list[dict[str, Any]]:
        """获取所有收集的消息。
        @return: 按发送顺序排列的消息字典列表（内部列表的直接引用，
                 修改会反映到本通道状态）。
        """
        # 直接返回内部列表引用：测试断言按序读取
        return self.messages
    def clear(self) -> None:
        """清空收集的消息。
        副作用：修改实例内部状态（messages 列表清空）。
        @return: 无返回。
        """
        # 原地清空列表：保持列表对象引用不变
        self.messages.clear()
class InputChannel(ABC):
    """输入通道抽象基类
    职责：定义"接收用户消息并交给 Agent 处理"的接入接口，并约定
          关联输出通道与请求元数据的提取能力。
    设计模式：策略模式——不同接入介质（REST/WebSocket/控制台）作为
          可互换实现；from_credentials 为类级工厂方法，按认证配置
          构造通道实例。
    生命周期：随应用启动创建（常驻监听），应用关闭时销毁；本基类
          不可实例化。
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """通道名称。
        @return: 通道唯一名称字符串，用于标识消息来源（UserMessage
                 .input_channel 即取自此处）。
        """
        # 抽象方法体：仅占位，强制子类提供实现
        raise NotImplementedError()
    def get_output_channel(self) -> OutputChannel | None:
        """获取关联的输出通道。
        核心功能：返回与本输入通道配对的输出通道（回复路径）。
        业务流程：默认实现返回 None——表示"无专用输出通道"，子类
                  （如需要同介质回传的实现）重写本方法返回配对实例。
        副作用：无。
        @return: OutputChannel 实例；无关联时返回 None。
        """
        # 默认无关联输出通道：回传路径由上层另行选择
        return None
    def get_metadata(self, request: Any) -> dict[str, Any] | None:
        """从请求中提取元数据。
        核心功能：解析原始请求对象中的附加信息（如认证上下文）。
        业务流程：默认实现返回 None——表示"无元数据"，子类按介质
                  协议重写提取逻辑。
        副作用：无。
        @param request: 原始请求对象（类型随介质而定，允许任意）。
        @return: 元数据字典；无可提取内容时返回 None。
        """
        # 默认不提取元数据：由子类按介质协议实现
        return None
    @classmethod
    def from_credentials(
        cls,
        credentials: dict[str, Any] | None,
    ) -> InputChannel:
        """从认证配置创建通道。
        核心功能：类级工厂方法——按凭据配置构造通道实例。
        业务流程：默认实现忽略凭据直接无参构造（适用于无需认证的
                  通道）；需认证的子类（如 Slack）重写本方法按凭据装配。
        副作用：由子类构造器决定；基类默认构造无副作用。
        @param credentials: 认证配置字典；允许 None（无凭据场景）。
        @return: 新建的 InputChannel 实例。
        """
        # 默认无参构造：忽略凭据，适用于开放通道
        return cls()
# 消息处理回调类型：接收 UserMessage 并返回可等待对象（异步处理结果）
# 的函数签名约定，输入通道以此回调把消息移交 Agent。
MessageHandler = Callable[[UserMessage], Awaitable[Any]]
# 导出清单：按字母序排列，约束 from ... import * 的导出范围并为 IDE
# 提供准确的补全与跳转依据。
__all__ = [
    # 机器人响应数据类
    "BotResponse",
    # 收集型输出通道（测试替身）
    "CollectingOutputChannel",
    # 输入通道抽象基类
    "InputChannel",
    # 消息处理回调类型别名
    "MessageHandler",
    # 输出通道抽象基类
    "OutputChannel",
    # 用户消息数据类
    "UserMessage",
]