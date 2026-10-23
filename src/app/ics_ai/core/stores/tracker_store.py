"""
tracker_store - Tracker 存储抽象基类模块

核心功能：定义 TrackerStore 抽象基类——统一对话状态持久化的接口契约
          （save/retrieve/delete/keys 四个抽象方法），并提供不依赖具体
          存储介质的通用能力：exists 存在性检查、create_tracker 按域
          槽位新建、get_or_create_tracker 获取或新建、retrieve_full_tracker
          完整状态获取、set_domain 运行期替换领域定义。

执行流程：1. 各存储后端（JsonTrackerStore/MySQLTrackerStore 等）继承本类
          并实现全部抽象方法；2. Agent 装配阶段经工厂实例化具体子类；
          3. 业务层调用 get_or_create_tracker 拿到 Tracker，处理消息后
          调 save 持久化；4. 存在性判断、新建等通用逻辑由本基类直接提供。

副作用：本模块自身无副作用（抽象接口不执行 IO；通用方法的副作用取决于
        子类 retrieve/save 的实现）；不读写文件/数据库、不发起网络请求。

主要依赖：abc（抽象基类机制）、collections.abc.Iterable（keys 返回类型）、
          app.ics_ai.core.tracker.DialogueStateTracker（被存储的对话状态）、
          app.ics_ai.core.domain.Domain（提供新建 Tracker 时的槽位定义）。
"""

# 从标准库 abc 导入抽象基类机制：ABC 作为基类声明抽象类，abstractmethod
# 装饰器强制子类实现指定方法。
from abc import ABC, abstractmethod

# 从标准库 collections.abc 导入 Iterable：keys() 返回值的容器抽象类型
# （PEP 585 推荐从 collections.abc 导入而非 typing）。
from collections.abc import Iterable

# 从项目核心模块导入领域模型：create_tracker 新建 Tracker 时复制其槽位定义。
from app.ics_ai.core.domain import Domain

# 从项目核心模块导入对话状态跟踪器：本接口持久化与恢复的目标对象。
from app.ics_ai.core.tracker import DialogueStateTracker


class TrackerStore(ABC):
    """Tracker 存储抽象基类

    职责：定义 Tracker 持久化存储的统一接口契约，所有存储后端必须实现
          四个抽象方法——save（保存）、retrieve（获取）、delete（删除）、
          keys（枚举全部 sender_id）；同时提供与介质无关的通用能力
          （存在性检查、按域新建、获取或新建等）。

    设计模式：模板方法模式——抽象方法 save/retrieve/delete/keys 由子类
          实现（变化部分），exists/create_tracker/get_or_create_tracker
          等具体方法基于抽象方法组合出通用流程（不变部分）；配合
          策略模式，具体存储后端作为可互换策略注入 Agent。

    生命周期：抽象类本身不实例化；由各子类在 Agent 装配阶段经工厂
          创建并持有，随应用整个运行期复用。

    所有存储后端必须实现以下方法：
    - save: 保存Tracker
    - retrieve: 获取Tracker
    - delete: 删除Tracker
    - keys: 获取所有sender_id
    """

    def __init__(self, domain: Domain | None = None) -> None:
        """初始化存储

        核心功能：保存 Domain 引用，供创建/反序列化 Tracker 时提供槽位定义。

        业务流程：仅将 domain 记录为实例属性，无其他初始化动作；
                  Domain 可在运行期经 set_domain 替换。

        副作用：无（仅设置实例属性）。

        @param domain: Domain 实例，用于创建 Tracker 时初始化槽位；
                       允许 None（此时按无槽位定义创建/反序列化）。
        """
        # 保存 Domain 引用：子类的 retrieve 反序列化与基类 create_tracker 均依赖它
        self.domain = domain

    @abstractmethod
    async def save(self, tracker: DialogueStateTracker) -> None:
        """保存 Tracker 状态（抽象方法）

        核心功能：将 Tracker 序列化后存储到后端（契约，由子类实现）。

        业务流程（子类实现要求）：1. 将 Tracker 状态序列化为可存储形式；
                  2. 以 tracker.sender_id 为键写入后端；3. 同键重复保存
                  应覆盖旧状态（UPSERT 语义）。

        副作用：由子类定义——文件后端写磁盘、数据库后端执行写事务。

        @param tracker: 要保存的 Tracker，不能为 None。
        @return: 无返回；保存失败时子类应抛 TrackerStoreException（不返回）。
        """
        # 抽象方法体：仅作占位，强制子类提供实现
        pass

    @abstractmethod
    async def retrieve(self, sender_id: str) -> DialogueStateTracker | None:
        """获取 Tracker 状态（抽象方法）

        核心功能：从存储后端加载指定 sender_id 的 Tracker（契约，由子类实现）。

        业务流程（子类实现要求）：1. 按 sender_id 定位存储的快照；2.
                  反序列化并重建 DialogueStateTracker；3. 不存在时返回
                  None 而非抛异常。

        副作用：由子类定义——通常为只读操作（读文件/执行 SELECT）。

        @param sender_id: 会话 ID；任意字符串。
        @return: DialogueStateTracker 实例；不存在返回 None；获取失败时
                 子类应抛 TrackerStoreException（不返回）。
        """
        # 抽象方法体：仅作占位，强制子类提供实现
        pass

    async def retrieve_full_tracker(
        self,
        sender_id: str,
    ) -> DialogueStateTracker | None:
        """获取完整的 Tracker 状态

        核心功能：获取包含所有历史会话的完整 Tracker。

        业务流程：默认实现直接转发给 retrieve（单存储场景两者等价）；
                  支持会话分割的后端（如按会话轮次分表存储）可重写本
                  方法以聚合跨会话的历史状态。

        副作用：取决于子类 retrieve 的实现（通常只读）。

        @param sender_id: 会话 ID。
        @return: DialogueStateTracker 实例；不存在返回 None（继承自
                 retrieve 的约定）。
        """
        # 默认委托给 retrieve：无会话分割需求的子类无需重写本方法
        return await self.retrieve(sender_id)

    @abstractmethod
    async def delete(self, sender_id: str) -> None:
        """删除 Tracker 状态（抽象方法）

        核心功能：从存储后端删除指定 sender_id 的 Tracker（契约，由子类实现）。

        业务流程（子类实现要求）：1. 按 sender_id 定位快照；2. 执行删除；
                  3. 目标不存在时应静默通过，不抛异常。

        副作用：由子类定义——文件后端删文件、数据库后端执行 DELETE 事务。

        @param sender_id: 会话 ID。
        @return: 无返回；删除失败时子类应抛 TrackerStoreException（不返回）。
        """
        # 抽象方法体：仅作占位，强制子类提供实现
        pass

    @abstractmethod
    async def keys(self) -> Iterable[str]:
        """获取所有 sender_id（抽象方法）

        核心功能：返回存储中所有 Tracker 的 sender_id（契约，由子类实现）。

        业务流程（子类实现要求）：枚举后端全部快照的键，以可迭代集合返回。

        副作用：由子类定义——通常为只读枚举（遍历目录/全表 SELECT）。

        @return: sender_id 的可迭代对象；空存储返回空集合。
        """
        # 抽象方法体：仅作占位，强制子类提供实现
        pass

    async def exists(self, sender_id: str) -> bool:
        """检查 Tracker 是否存在

        核心功能：判断指定 sender_id 的快照是否已存在于后端。

        业务流程：1. 调用 retrieve 尝试加载；2. 结果非 None 即视为存在。

        副作用：继承 retrieve 的只读副作用（子类实现决定）。

        @param sender_id: 会话 ID。
        @return: 存在返回 True；不存在（retrieve 得到 None）返回 False。
        """
        # 复用 retrieve：存在性等价于"能否加载到实例"，避免子类额外实现
        tracker = await self.retrieve(sender_id)

        # 非 None 判定：得到实例即存在
        return tracker is not None

    def create_tracker(
        self,
        sender_id: str,
    ) -> DialogueStateTracker:
        """创建新的 Tracker

        核心功能：使用 Domain 定义的槽位创建全新的空白 Tracker。

        业务流程：1. 先准备空槽位字典（无 Domain 时的兜底）；2. 若持有
                  Domain 则深拷贝其全部槽位定义（深拷贝保证每个 Tracker
                  独占槽位状态，互不串扰）；3. 以 sender_id 与槽位构造
                  DialogueStateTracker 返回。

        副作用：无（纯内存构造，不触发持久化——需保存请显式调用 save）。

        @param sender_id: 会话 ID；作为新 Tracker 的唯一标识。
        @return: 新的 DialogueStateTracker 实例（仅内存中，未写入后端）。
        """
        # 兜底初始化：无 Domain 时以空槽位字典创建
        slots = {}

        # 条件守卫：持有 Domain 才复制槽位定义
        if self.domain:
            # 延迟导入深拷贝工具：仅本分支需要，按需加载
            from copy import deepcopy

            # 字典推导逐项深拷贝槽位：键为槽位名，值为槽位定义的独立副本，
            # 深拷贝避免多个 Tracker 共享同一槽位对象导致状态互相污染
            slots = {name: deepcopy(slot) for name, slot in self.domain.slots.items()}

        # 以会话 ID 与槽位定义构造全新的空白 Tracker 并返回
        return DialogueStateTracker(sender_id=sender_id, slots=slots)

    async def get_or_create_tracker(
        self,
        sender_id: str,
    ) -> DialogueStateTracker:
        """获取或创建 Tracker

        核心功能：会话首次出现时新建 Tracker 并持久化，否则复用已有 Tracker，
                  保证调用方总能拿到可用的 Tracker 实例。

        业务流程：1. 调用 retrieve 尝试从后端加载；2. 加载到 None（不存在）
                  则经 create_tracker 新建，并立即 save 写入后端完成注册；
                  3. 返回已有或新建的 Tracker。

        副作用：新建分支会触发子类 save 的持久化副作用（写文件/写数据库）；
                  已存在分支仅有 retrieve 的只读副作用。

        @param sender_id: 会话 ID。
        @return: DialogueStateTracker 实例（后端已有则为其反序列化副本，
                 否则为本方法新建并已持久化的实例）。
        """
        # 第一步：尝试从后端加载既有 Tracker（不存在时得到 None）
        tracker = await self.retrieve(sender_id)

        # 条件分支：后端无此会话，走"创建并注册"流程
        if tracker is None:
            # 按域槽位新建空白 Tracker（仅内存）
            tracker = self.create_tracker(sender_id)

            # 立即持久化：让新 Tracker 在后端完成注册，下次 retrieve 可命中
            await self.save(tracker)

        # 返回获取或新建的 Tracker 实例
        return tracker

    def set_domain(self, domain: Domain) -> None:
        """设置 Domain

        核心功能：运行期替换存储持有的 Domain 引用。

        业务流程：直接覆盖 self.domain，此后 create_tracker/子类反序列化
                  均使用新的槽位定义（已存在的 Tracker 不受影响）。

        副作用：修改实例内部状态（影响后续新建与反序列化行为）。

        @param domain: Domain 实例，不能为 None（替换场景必然已有领域定义）。
        """
        # 覆盖 Domain 引用：后续槽位相关操作切换到新定义
        self.domain = domain
