"""
domain - 领域定义模块

功能：定义对话系统的领域(Domain)，包括槽位、动作、响应模板、Flow、表单等配置。
     Domain 是对话系统配置的核心入口，描述了系统能够处理的对话范围和能力边界。

执行流程：
    1. ResponseTemplate 定义单条响应的结构（文本/按钮/图片等）
    2. Domain 通过 load() 从 YAML 文件或目录加载配置字典
    3. from_dict() 将配置字典解析为 Domain 实例（槽位、动作、响应等）
    4. 运行时通过 get_slot()/get_response()/has_action() 等方法查询配置

副作用：
    - load() 会读取本地 YAML 文件（不涉及网络/数据库）
    - Domain 实例为纯内存配置对象，不修改全局状态

主要依赖：
    - dataclasses.dataclass, field：数据类装饰器和字段工厂
    - pathlib.Path：路径处理
    - typing.Any：类型通配符
    - app.ics_ai.shared.constants：系统默认动作常量、默认 domain 路径
    - app.ics_ai.shared.exceptions.DomainValidationError：domain 验证异常
    - app.ics_ai.shared.yaml_loader：YAML 文件读写工具
    - app.ics_ai.core.slots.Slot, create_slot：槽位定义和工厂
"""

from __future__ import annotations  # 延迟求值所有类型注解，一次解决类体内前向引用问题

from dataclasses import dataclass, field  # 数据类装饰器和字段工厂
from pathlib import Path  # 面向对象的路径处理
from typing import Any  # 类型通配符，保留自 typing（无内置替代）

from app.ics_ai.core.slots import Slot, create_slot  # 槽位数据类与工厂函数
from app.ics_ai.shared.constants import (  # 系统级预置动作常量
    ACTION_DEFAULT_FALLBACK,  # 默认兜底动作名
    ACTION_LISTEN,  # 监听用户输入动作名
    ACTION_RESTART,  # 重启对话动作名
    ACTION_SESSION_START,  # 会话启动动作名
    DEFAULT_DOMAIN_PATH,  # 默认 domain YAML 路径
)
from app.ics_ai.shared.exceptions import DomainValidationError  # domain 配置校验异常
from app.ics_ai.shared.yaml_loader import (
    merge_yaml_files,
    read_yaml_file,
)  # YAML 读取/合并工具


@dataclass  # 声明为数据类，自动生成 __init__/__repr__/__eq__ 等
class ResponseTemplate:
    """响应模板

    职责：定义 Bot 单条响应的结构化内容，支持多模态（文本/按钮/图片/自定义）
         和条件分支（根据状态或通道输出不同变体）。
    设计模式：数据载体（DTO），纯存储类无业务逻辑。
    生命周期：随 Domain.from_dict() 解析 YAML 配置时创建，运行时只读复用。

    属性：
        text: 响应纯文本，可为 None 表示无文本内容。
        buttons: 交互按钮列表，每个元素为字典（含 title/payload 等键），
                 空列表表示无按钮。
        image: 图片 URL 或资源标识，可为 None 表示无图片。
        custom: 任意自定义结构化数据（如卡片 JSON、富媒体载荷），
                留给上层业务解释，可为 None。
        condition: 条件表达式字符串，运行时求值决定该模板是否生效，
                   可为 None 表示无条件（始终可选）。
        channel: 限定适用的通讯通道名（如 "feishu"/"web"），
                 可为 None 表示通用于所有通道。
        metadata: 响应的元信息字典（如优先级、标签），默认为空字典。
    """

    text: str | None = None  # 响应纯文本；None 表示无文本内容
    buttons: list[dict[str, Any]] = field(
        default_factory=list
    )  # 按钮列表；默认空列表避免可变默认值陷阱
    image: str | None = None  # 图片 URL 或资源标识；None 表示无图片
    custom: dict[str, Any] | None = None  # 自定义结构化载荷；None 表示无自定义数据
    condition: str | None = None  # 条件表达式字符串；None 表示无条件
    channel: str | None = None  # 限定通道名；None 表示通用于所有通道
    metadata: dict[str, Any] = field(default_factory=dict)  # 元信息字典；默认空字典

    @classmethod  # 工厂方法，返回新实例而非 self
    def from_dict(cls, data: dict[str, Any] | str) -> ResponseTemplate:
        """从字典或字符串创建响应模板

        核心功能：将 YAML 配置中的响应条目（支持简写字符串或完整字典）
                 解析为 ResponseTemplate 实例。
        业务流程：1. 判断 data 类型；2. 字符串简写 → 仅填 text 字段；
                 3. 字典格式 → 逐 key 提取各属性。
        副作用：无，纯函数式创建对象。

        @param data: 响应配置，字符串形式表示纯文本响应，
                     字典形式表示完整的多模态响应配置；不能为 None。
        @return: ResponseTemplate 实例；若 data 为空字典则所有属性取默认值。
        """
        if isinstance(data, str):  # 简写格式：直接字符串视为纯文本响应
            return cls(text=data)  # 仅设置 text，其余字段取默认值

        # 完整字典格式：逐字段提取，缺失的 key 由 get() 返回 None 或默认值
        return cls(
            text=data.get("text"),  # 响应文本；无则 None
            buttons=data.get("buttons", []),  # 按钮列表；无则空列表
            image=data.get("image"),  # 图片 URL；无则 None
            custom=data.get("custom"),  # 自定义数据；无则 None
            condition=data.get("condition"),  # 条件表达式；无则 None
            channel=data.get("channel"),  # 限定通道；无则 None
            metadata=data.get("metadata", {}),  # 元信息；无则空字典
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式

        核心功能：将 ResponseTemplate 实例序列化为可写回 YAML 的字典。
        业务流程：逐字段判断是否有值，有值才写入结果字典（避免序列化冗余空字段）。
        副作用：无，只读 self 属性。

        @return: 仅包含非默认值字段的字典；若所有字段均为空则返回空字典。
        """
        result: dict[str, Any] = {}  # 初始化结果字典

        if self.text:  # 仅当 text 非 None 且非空字符串时才输出
            result["text"] = self.text  # 写入响应文本
        if self.buttons:  # 仅当按钮列表非空时才输出
            result["buttons"] = self.buttons  # 写入按钮列表
        if self.image:  # 仅当图片 URL 非 None 时才输出
            result["image"] = self.image  # 写入图片 URL
        if self.custom:  # 仅当自定义数据非 None 时才输出
            result["custom"] = self.custom  # 写入自定义数据
        if self.condition:  # 仅当条件表达式非 None 时才输出
            result["condition"] = self.condition  # 写入条件表达式
        if self.channel:  # 仅当通道限定非 None 时才输出
            result["channel"] = self.channel  # 写入通道限定
        if self.metadata:  # 仅当元信息字典非空时才输出
            result["metadata"] = self.metadata  # 写入元信息

        return result  # 返回构建好的序列化字典


@dataclass  # 声明为数据类，自动生成 __init__/__repr__/__eq__ 等
class Domain:
    """领域定义类

    职责：管理对话系统的全部领域配置（槽位、动作、响应、Flow、表单等），
         是对话引擎理解"系统能做什么、如何响应"的核心配置对象。
    设计模式：配置聚合根（Aggregate Root），将分散的 YAML 配置解析为
              强类型对象树；load()/from_dict() 为工厂方法。
    生命周期：应用启动时通过 load() 从 YAML 文件加载并单例复用；
             运行时只读，merge() 时生成新实例。

    属性：
        slots: 槽名字典，key 为槽位名，value 为 Slot 实例；默认空字典。
        actions: 动作名称集合，包含业务动作名 + 系统预置默认动作；
                 自动注入 ACTION_LISTEN/ACTION_RESTART/ACTION_SESSION_START/
                 ACTION_DEFAULT_FALLBACK。
        responses: 响应模板字典，key 为响应名（如 "utter_greet"），
                   value 为 ResponseTemplate 列表（同一响应名可有多条变体）。
        flows: Flow 流程名称列表，声明系统可用的 Flow 列表；默认空列表。
        forms: 表单定义字典，key 为表单名，value 为表单配置字典；默认空字典。
        session_config: 会话级配置（如超时、存储后端等），默认空字典。
        version: Domain 配置版本号，默认 "1.0"。
    """

    slots: dict[str, Slot] = field(default_factory=dict)  # 槽名字典；默认空字典
    actions: set[str] = field(default_factory=set)  # 动作名集合；默认空集合
    responses: dict[str, list[ResponseTemplate]] = field(
        default_factory=dict
    )  # 响应模板字典；默认空字典
    flows: list[str] = field(default_factory=list)  # Flow 名称列表；默认空列表
    forms: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )  # 表单定义字典；默认空字典
    session_config: dict[str, Any] = field(
        default_factory=dict
    )  # 会话配置字典；默认空字典
    version: str = "1.0"  # 配置版本号；默认 "1.0"

    def __post_init__(self) -> None:
        """初始化后处理

        核心功能：在 dataclass 自动生成的 __init__ 执行完毕后，
                 追加系统默认动作（不依赖外部配置，保证最小可用集）。
        业务流程：调用 _add_default_actions() 将 4 个系统动作注入 actions 集合。
        副作用：修改 self.actions 集合（追加元素）。

        @return: None
        """
        self._add_default_actions()  # 追加系统默认动作到集合

    def _add_default_actions(self) -> None:
        """添加系统默认动作（私有辅助方法）

        核心功能：确保 Domain 实例始终包含系统级必需的预置动作，
                 即使 YAML 配置中未显式声明。
        业务流程：1. 构造包含 4 个系统常量的集合；2. 用集合 update 一次性并入。
        副作用：修改 self.actions 集合。

        @return: None
        """
        default_actions = {  # 构造系统默认动作集合
            ACTION_LISTEN,  # 监听用户输入
            ACTION_RESTART,  # 重启对话
            ACTION_SESSION_START,  # 会话启动
            ACTION_DEFAULT_FALLBACK,  # 默认兜底
        }
        self.actions.update(default_actions)  # 集合并运算，自动去重

    @classmethod  # 工厂方法
    def load(
        cls,
        domain_path: str
        | Path = DEFAULT_DOMAIN_PATH,  # 文件或目录路径；默认为共享常量中定义的路径
    ) -> Domain:
        """从 YAML 文件或目录加载 Domain 配置

        核心功能：解析单个 YAML 文件或合并目录下所有 YAML 文件，
                 生成 Domain 实例。
        业务流程：
            1. pathlib.Path 包装路径参数，统一处理 str/Path 两种输入
            2. 目录模式：glob 匹配 *.yml 和 *.yaml → merge_yaml_files 合并
            3. 单文件模式：直接 read_yaml_file 读取
            4. 文件不存在或目录为空时抛 DomainValidationError
            5. 空字典兜底 → 交给 from_dict() 生成带默认动作的 Domain
        副作用：读取本地磁盘 YAML 文件；不修改全局状态。

        @param domain_path: Domain YAML 文件或包含多个 YAML 文件的目录路径；
                            可传 str 或 Path；默认为 DEFAULT_DOMAIN_PATH 常量。
        @return: Domain 实例；若文件完全为空则返回仅含系统默认动作的 Domain。
        @raises DomainValidationError: 路径不存在或目录内无 YAML 文件时抛出。
        """
        path = Path(domain_path)  # 统一转换为 Path 对象，便于后续路径操作

        if path.is_dir():  # 目录模式：合并目录下所有 YAML 文件
            # glob 同时匹配 .yml 和 .yaml 后缀，合并为单一列表
            yaml_files = list(path.glob("*.yml")) + list(path.glob("*.yaml"))
            if not yaml_files:  # 目录内无任何 YAML 文件 → 配置无效
                raise DomainValidationError(f"目录中没有找到YAML文件: {path}")
            domain_dict = merge_yaml_files(yaml_files)  # 合并多个 YAML 为单个字典
        elif path.exists():  # 单文件模式：路径存在且不是目录
            domain_dict = read_yaml_file(path)  # 读取单个 YAML 文件
        else:  # 路径既不存在也不是目录 → 配置无效
            raise DomainValidationError(f"Domain文件不存在: {path}")

        if not domain_dict:  # 空文件或解析结果为空 → 兜底为空字典
            domain_dict = {}  # 确保后续 from_dict 调用不会因 None 崩溃

        return cls.from_dict(domain_dict)  # 委托给 from_dict 完成实际解析

    @classmethod  # 工厂方法
    def from_dict(cls, data: dict[str, Any]) -> Domain:
        """从配置字典创建 Domain 实例

        核心功能：将解析好的 YAML 字典（含 slots/actions/responses/forms 等 key）
                 逐段解析为强类型对象，组装成 Domain 实例。
        业务流程：
            1. 解析 slots：遍历 data["slots"]，每个条目可为完整 dict 或简写字符串
               - 完整 dict：提取 mappings 首元素的 type 作为 mapping_type
               - 简写字符串：视为 slot_type 直接创建
            2. 解析 actions：data["actions"] 列表 → set 去重
            3. 解析 responses：data["responses"] 每个值可为列表或单条目 →
               统一转为 list[ResponseTemplate]
            4. 解析 flows/forms/session_config/version 等剩余字段
            5. 调用 cls(...) 构造，触发 __post_init__ 注入系统默认动作
        副作用：无，纯函数式创建对象。

        @param data: Domain 配置字典，通常由 YAML 解析而来；
                     可包含 slots/actions/responses/flows/forms/session_config/version 等 key，
                     所有 key 均可缺失（缺失时取默认空值）。
        @return: Domain 实例（含系统默认动作）。
        """
        # --- 解析槽位 ---
        slots: dict[str, Slot] = {}  # 初始化槽位字典
        # data.get("slots", {}) 取不到时返回空字典，.items() 不会报错
        for slot_name, slot_config in data.get("slots", {}).items():
            if isinstance(
                slot_config, dict
            ):  # 完整字典格式：含 mappings/type 等嵌套配置
                # mappings：槽位映射规则列表，用于决定槽位值的来源（LLM 提取/实体匹配/预设值等）
                mappings = slot_config.get("mappings", [])  # 无 mappings 则空列表
                mapping_type = "from_llm"  # 默认映射类型为 LLM 直接提取
                # 仅当 mappings 是非空列表且首元素为字典时，从中提取 type 字段
                if (
                    mappings and isinstance(mappings, list) and len(mappings) > 0
                ):  # 三重守卫：非空 + 是列表 + 有首元素
                    first_mapping = mappings[0]  # 取第一条映射规则
                    if isinstance(first_mapping, dict):  # 首元素必须是 dict 才有 .get()
                        mapping_type = first_mapping.get(
                            "type", "from_llm"
                        )  # 取 type 字段，缺省回退

                # 构建 create_slot 工厂函数的参数字典
                slot_kwargs: dict[str, Any] = {
                    "name": slot_name,  # 槽位名称
                    "slot_type": slot_config.get(
                        "type", "any"
                    ),  # 槽位值类型；默认 "any"
                    "mapping_type": mapping_type,  # 映射来源类型（从 mappings 首元素提取）
                    "description": slot_config.get(
                        "description"
                    ),  # 槽位描述；无则 None
                    "initial_value": slot_config.get(
                        "initial_value"
                    ),  # 初始值；无则 None
                    "influence_conversation": slot_config.get(
                        "influence_conversation",
                        True,  # 是否影响对话流程；默认 True
                    ),
                    "mappings": mappings,  # 完整映射规则列表
                }
                # 分类槽位的可选值：仅当 YAML 中显式声明了 values 才传入
                if (
                    "values" in slot_config
                ):  # 用 in 而非 .get()，区分"key 不存在"和"key 存在但值为 None"
                    slot_kwargs["values"] = slot_config.get("values")  # 传入可选值列表

                slots[slot_name] = create_slot(
                    **slot_kwargs
                )  # 展开 kwargs，调用工厂创建 Slot
            else:
                # 简化格式：slot_name: type_string → slot_config 直接就是类型字符串
                slots[slot_name] = create_slot(
                    name=slot_name,
                    slot_type=str(slot_config),  # 强制 str 转换兼容非字符串类型
                )

        # --- 解析动作 ---
        actions: set[str] = set(
            data.get("actions", [])
        )  # 列表 → set 去重；无 actions 则空集合

        # --- 解析响应模板 ---
        responses: dict[str, list[ResponseTemplate]] = {}  # 初始化响应字典
        # data.get("responses", {}).items() 遍历所有响应名及其配置
        for response_name, response_list in data.get("responses", {}).items():
            if isinstance(response_list, list):  # YAML 中为列表格式：多条变体
                # 列表推导式：逐元素转换为 ResponseTemplate
                responses[response_name] = [
                    ResponseTemplate.from_dict(r) for r in response_list
                ]
            else:  # YAML 中为单条格式（字符串或单一 dict）→ 包装成单元素列表
                responses[response_name] = [ResponseTemplate.from_dict(response_list)]

        # --- 解析其他配置 ---
        flows: list[str] = data.get("flows", [])  # Flow 名称列表；无则空列表
        forms: dict[str, Any] = data.get("forms", {})  # 表单定义字典；无则空字典
        session_config: dict[str, Any] = data.get(
            "session_config", {}
        )  # 会话配置字典；无则空字典
        version: str = data.get("version", "1.0")  # 配置版本号；默认 "1.0"

        # 构造 Domain 实例，触发 __post_init__ 注入系统默认动作
        return cls(
            slots=slots,  # 槽位字典
            actions=actions,  # 动作集合（之后会被 __post_init__ 追加系统默认动作）
            responses=responses,  # 响应模板字典
            flows=flows,  # Flow 列表
            forms=forms,  # 表单定义字典
            session_config=session_config,  # 会话配置字典
            version=version,  # 配置版本号
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式

        核心功能：将 Domain 实例完整序列化，包含所有槽位、动作、响应、表单等配置。
        业务流程：逐字段转换（嵌套对象递归调用各自的 to_dict），动作集合转为列表。
        副作用：无，只读 self 属性。

        @return: 完整配置字典；所有嵌套对象已递归序列化。
        """
        return {
            "version": self.version,  # 配置版本号直接返回
            # 槽位字典：逐 Slot 实例调用 to_dict() 序列化
            "slots": {name: slot.to_dict() for name, slot in self.slots.items()},
            "actions": list(self.actions),  # set → list（集合无法 JSON 序列化）
            # 响应字典：逐 ResponseTemplate 调用 to_dict() 序列化
            "responses": {
                name: [r.to_dict() for r in templates]
                for name, templates in self.responses.items()
            },
            "flows": self.flows,  # 列表直接返回
            "forms": self.forms,  # 字典直接返回
            "session_config": self.session_config,  # 字典直接返回
        }

    def get_slot(self, slot_name: str) -> Slot | None:
        """获取指定槽位

        核心功能：按名称查询槽位定义。
        业务流程：直接用 dict.get() 查询，避免 KeyError。
        副作用：无，只读 self.slots。

        @param slot_name: 槽位名称字符串；不能为空字符串。
        @return: Slot 实例；若槽位不存在则返回 None。
        """
        return self.slots.get(slot_name)  # dict.get 找不到返回 None 而非抛异常

    def get_response(self, response_name: str) -> list[ResponseTemplate]:
        """获取指定响应模板列表

        核心功能：按响应名查询响应模板，同一响应名可能有多条变体（条件分支）。
        业务流程：直接用 dict.get() 查询，缺省返回空列表。
        副作用：无，只读 self.responses。

        @param response_name: 响应名称（如 "utter_greet"）；不能为空字符串。
        @return: ResponseTemplate 列表；若响应名不存在则返回空列表 []。
        """
        return self.responses.get(response_name, [])  # dict.get 缺省返回空列表

    def has_action(self, action_name: str) -> bool:
        """检查动作是否存在

        核心功能：判断某个动作名是否在 Domain 的动作集合中。
        业务流程：直接用 set 的 in 运算，O(1) 查找。
        副作用：无，只读 self.actions。

        @param action_name: 动作名称字符串；不能为空。
        @return: True 表示动作已定义；False 表示不存在。
        """
        return action_name in self.actions  # set 成员判断，O(1)

    def has_flow(self, flow_name: str) -> bool:
        """检查 Flow 是否存在

        核心功能：判断某个 Flow 名称是否在 Domain 的 Flow 列表中。
        业务流程：直接用 list 的 in 运算，O(n) 查找（Flow 数量通常较小）。
        副作用：无，只读 self.flows。

        @param flow_name: Flow 名称字符串；不能为空。
        @return: True 表示 Flow 已声明；False 表示不存在。
        """
        return flow_name in self.flows  # list 成员判断

    def add_slot(self, slot: Slot) -> None:
        """添加槽位（运行时动态注入）

        核心功能：将外部创建的 Slot 实例注册到 Domain。
        业务流程：以 slot.name 为 key 直接写入 slots 字典，覆盖同名槽位。
        副作用：修改 self.slots 字典（写入/覆盖）。

        @param slot: Slot 实例；其 name 属性将作为字典 key。
        @return: None
        """
        self.slots[slot.name] = slot  # 以槽位名作为 key 写入字典，同名则覆盖

    def add_action(self, action_name: str) -> None:
        """添加动作（运行时动态注入）

        核心功能：向 Domain 的动作集合追加一个新动作名。
        业务流程：直接调用 set.add()，自动去重。
        副作用：修改 self.actions 集合（追加元素）。

        @param action_name: 动作名称字符串；不能为空。
        @return: None
        """
        self.actions.add(action_name)  # set.add 自动去重

    def add_response(
        self,
        response_name: str,
        templates: list[ResponseTemplate],
    ) -> None:
        """添加响应模板（运行时动态注入）

        核心功能：为某个响应名设置（或覆盖）其响应模板列表。
        业务流程：直接以 response_name 为 key 写入 responses 字典，
                  覆盖同名响应的所有既有模板。
        副作用：修改 self.responses 字典（写入/覆盖）。

        @param response_name: 响应名称（如 "utter_greet"）；不能为空。
        @param templates: ResponseTemplate 列表；不能为 None，可为空列表。
        @return: None
        """
        self.responses[response_name] = templates  # 直接赋值；同名响应会被完整覆盖

    def merge(self, other: Domain) -> Domain:
        """合并另一个 Domain

        核心功能：将当前 Domain 与另一个 Domain 的配置合并，返回新实例。
        业务流程：
            1. 槽位字典：{**self, **other} → other 同名槽位覆盖 self
            2. 动作集合：self | other → 并集，自动去重
            3. 响应字典：同槽位策略，other 同名响应覆盖 self
            4. Flow 列表：list(set(self.flows + other.flows)) → 拼接后去重再转列表
            5. 表单/会话配置：同槽位策略，other 同名 key 覆盖 self
            6. 构造新 Domain（触发 __post_init__ 再次注入系统默认动作，幂等）
        副作用：无（纯函数），不修改 self 也不修改 other。

        @param other: 待合并的 Domain 实例；不能为 None。
        @return: 新的合并后的 Domain 实例；self 和 other 均不被修改。
        """
        # 槽位字典：展开式合并，other 的同名 key 覆盖 self
        merged_slots: dict[str, Slot] = {**self.slots, **other.slots}
        # 动作集合：集合并运算（|），自动去重
        merged_actions: set[str] = self.actions | other.actions
        # 响应字典：同槽位策略，other 同名响应整体覆盖 self
        merged_responses: dict[str, list[ResponseTemplate]] = {
            **self.responses,
            **other.responses,
        }
        # Flow 列表：先拼接 → set 去重 → 转回 list（保持唯一，不保证顺序）
        merged_flows: list[str] = list(set(self.flows + other.flows))
        # 表单定义字典：同槽位策略
        merged_forms: dict[str, Any] = {**self.forms, **other.forms}

        # 构造新 Domain 实例（__post_init__ 会自动追加系统默认动作，幂等操作）
        return Domain(
            slots=merged_slots,  # 合并后的槽位字典
            actions=merged_actions,  # 合并后的动作集合
            responses=merged_responses,  # 合并后的响应字典
            flows=merged_flows,  # 合并后的 Flow 列表（已去重）
            forms=merged_forms,  # 合并后的表单字典
            # 会话配置字典：同槽位策略，other 同名 key 覆盖 self
            session_config={**self.session_config, **other.session_config},
        )

    def __repr__(self) -> str:
        """可读字符串表示

        核心功能：调试时打印 Domain 概要信息，包含核心配置数量。
        业务流程：f-string 拼接四个计数值。
        副作用：无。

        @return: 格式如 "Domain(slots=5, actions=12, responses=8, flows=3)" 的字符串。
        """
        return (  # f-string 拼接，展示核心配置数量
            f"Domain(slots={len(self.slots)}, actions={len(self.actions)}, "
            f"responses={len(self.responses)}, flows={len(self.flows)})"
        )
