"""
json_store - JSON 文件 Tracker 存储模块

核心功能：实现 JsonTrackerStore——将 DialogueStateTracker 的对话状态以
          JSON 文件形式持久化到本地文件系统，每个 sender_id 对应一个独立
          的 .json 文件；同时提供纯内存模式（in_memory=True，不落盘），
          供测试与临时场景使用。

执行流程：1. 构造时确定存储模式：文件模式按 path 创建存储目录，内存模式
          只准备进程内字典；2. save() 将 Tracker 序列化为字典，写入
          {path}/{sender_id}.json 或存入内存字典；3. retrieve() 读取对应
          快照，并借助 Domain 槽位定义反序列化重建 Tracker；4. delete()/
          keys()/clear_all() 分别提供单个删除、枚举 sender_id 与清空全部。

副作用：文件模式下读写本地文件系统（创建目录、写/读/删除 .json 文件）；
        内存模式下仅增删查进程内字典 _memory_store；不发起网络请求、
        不访问数据库。

主要依赖：json（JSON 序列化/反序列化）、pathlib.Path（路径与文件操作）、
          collections.abc.Iterable（keys 返回类型）、
          app.ics_ai.shared.constants.DEFAULT_ENCODING（统一文件编码）、
          app.ics_ai.shared.exceptions（存储异常体系）、
          app.ics_ai.core.tracker.DialogueStateTracker（被存储的对话状态）、
          app.ics_ai.core.domain.Domain（提供反序列化所需槽位定义）。
"""

# 从标准库 collections.abc 导入 Iterable：keys() 返回值的容器抽象类型
# （PEP 585 推荐 ITERABLE 类抽象从 collections.abc 导入而非 typing）。
from collections.abc import Iterable

# 导入 json 模块：负责 Tracker 状态字典与 JSON 文本之间的相互转换。
import json

# 从标准库 pathlib 导入 Path：以面向对象方式处理存储目录创建、路径拼接
# 与文件存在性判断。
from pathlib import Path

# 导入 Any：标注内存快照字典的值类型（to_dict 产出的任意结构）。
from typing import Any

# 从项目共享常量模块导入统一文件编码：JSON 文件读写均使用 utf-8，
# 避免跨平台默认编码不一致导致的乱码。
from app.ics_ai.shared.constants import DEFAULT_ENCODING

# 从项目核心模块导入领域模型：retrieve 反序列化时需借用其槽位定义。
from app.ics_ai.core.domain import Domain

# 从同包基类导入 TrackerStore：JsonTrackerStore 的抽象父类，
# 约定 save/retrieve/delete/keys 等统一接口。
from app.ics_ai.core.stores.tracker_store import TrackerStore

# 从项目核心模块导入对话状态跟踪器：本存储持久化与恢复的目标对象。
from app.ics_ai.core.tracker import DialogueStateTracker

# 从项目共享异常模块导入存储异常体系：序列化失败与通用存储失败分别抛出，
# 便于上层按需精确捕获。
from app.ics_ai.shared.exceptions import (
    TrackerSerializationError,
    TrackerStoreException,
)


class JsonTrackerStore(TrackerStore):
    """JSON 文件 Tracker 存储

    职责：将每个 Tracker 序列化为独立 JSON 文件落盘（或写入内存字典），
          并支持从文件/内存恢复 Tracker，实现对话状态的持久化与复用。

    设计模式：策略模式——作为 TrackerStore 抽象接口的一种具体存储策略，
          与 MySQLTrackerStore 可互换使用；由工厂函数 create_tracker_store
          根据 store_type 选择具体策略，调用方只面向基类接口编程。

    生命周期：由 Agent 装配阶段（create_tracker_store 工厂）创建并被
          应用持有，随应用整个运行期复用；进程退出时随之销毁，但文件
          模式下已落盘的数据不受影响，下次启动可继续读取。

    文件结构：
        {path}/
        ├── {sender_id_1}.json
        ├── {sender_id_2}.json
        └── ...

    属性：
        path: 存储目录路径（文件模式）；内存模式下为 None
        in_memory: 是否使用内存存储(不持久化)
    """

    def __init__(
        self,
        domain: Domain | None = None,
        path: str | None = "trackers",
        in_memory: bool = False,
    ) -> None:
        """初始化 JSON 存储

        核心功能：确定存储模式（文件/内存）并准备对应的存储介质。

        业务流程：1. 调用父类构造保存 Domain；2. 记录内存模式开关并初始化
                  空的内存字典；3. 非内存模式且给定了 path 时，将路径包装
                  为 Path 并递归创建存储目录；否则 self.path 置 None。

        副作用：文件模式下会在磁盘上递归创建存储目录（目录已存在则跳过）。

        @param domain: Domain 实例，反序列化 Tracker 时提供槽位定义；
                       允许 None（此时按无槽位定义反序列化）。
        @param path: 存储目录路径，默认 "trackers"（相对当前工作目录）；
                     允许 None（表示无落盘位置，仅内存模式有意义）。
        @param in_memory: 是否使用内存存储；True 时所有读写仅发生在
                          进程内字典，不产生任何磁盘 IO。
        """
        # 调用抽象基类 TrackerStore 的构造函数：保存 domain 供后续反序列化使用
        super().__init__(domain)

        # 记录内存模式开关：True 表示所有读写仅发生在进程内字典
        self.in_memory = in_memory

        # 初始化内存存储字典：键为 sender_id（str），值为 Tracker 序列化后的快照字典
        self._memory_store: dict[str, dict[str, Any]] = {}

        # 条件守卫：只有"非内存模式"且"给定了路径"才启用文件存储——
        # 内存模式必须完全不落盘；path 为 None 表示调用方未指定落盘位置
        if not in_memory and path:
            # 将字符串路径包装为 Path 对象，便于后续目录创建与文件路径拼接
            self.path = Path(path)

            # 递归创建存储目录：parents=True 逐级补齐父目录，
            # exist_ok=True 保证目录已存在时不抛异常
            self.path.mkdir(parents=True, exist_ok=True)
        else:
            # 内存模式或未给路径：置 None 表示"无文件存储位置"
            self.path = None

    def _get_file_path(self, sender_id: str) -> Path:
        """获取 Tracker 文件路径（私有辅助方法）

        核心功能：将 sender_id 清洗为文件名安全的字符串，拼出目标 .json
                  文件的完整路径。

        业务流程：1. 逐字符过滤 sender_id——仅保留字母/数字/下划线/连字符，
                  其余字符（如 "/"、"." 等可能构造路径穿越或非法文件名的
                  字符）替换为 "_"；2. 与存储目录拼接为 "{safe_id}.json"。

        副作用：无（纯路径计算，不触碰文件系统）。

        @param sender_id: 会话 ID；任意字符串，特殊字符会被替换为下划线。
        @return: Path 对象，指向 "{path}/{safe_id}.json"。
        """
        # 生成器逐字符遍历 sender_id：字母/数字/下划线/连字符原样保留，
        # 其余替换为下划线——防止 sender_id 携带路径分隔符或非法文件名
        # 字符，导致目录穿越或文件创建失败
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in sender_id)

        # 用 Path 的 "/" 运算符拼接存储目录与文件名，返回完整文件路径
        return self.path / f"{safe_id}.json"  # pyright: ignore[reportOptionalOperand]  # 文件模式调用链中 path 必非 None（__init__ 已守卫）

    async def save(self, tracker: DialogueStateTracker) -> None:
        """保存 Tracker 到 JSON 文件

        核心功能：将 Tracker 序列化后按存储模式写入文件或内存字典。

        业务流程：1. 调用 tracker.to_dict() 得到状态快照字典；2. 内存模式
                  直接以 sender_id 为键覆盖写入 _memory_store；3. 文件模式
                  计算目标文件路径后，以 utf-8 编码写入 JSON（中文保留
                  原文、两级缩进美化）。

        副作用：文件模式下写/覆盖磁盘上的 {sender_id}.json；内存模式下
                覆盖 _memory_store 中的同名键。

        @param tracker: 要保存的 DialogueStateTracker 实例，不能为 None。
        @return: 无返回；序列化或写文件失败时抛 TrackerSerializationError
                 （不返回）。
        """
        try:
            # 将 Tracker 的完整状态（槽位、轮次、对话栈等）序列化为普通字典
            tracker_data = tracker.to_dict()

            # 分支：内存模式——不触碰磁盘，直接更新内存字典
            if self.in_memory:
                # 以 sender_id 为键存入快照，同键重复保存即覆盖旧快照
                self._memory_store[tracker.sender_id] = tracker_data
            else:
                # 文件模式——根据 sender_id 计算目标 .json 文件路径
                file_path = self._get_file_path(tracker.sender_id)

                # 以 utf-8 编码打开目标文件（"w" 模式：无则创建、有则截断覆盖）
                with open(file_path, "w", encoding=DEFAULT_ENCODING) as f:
                    # ensure_ascii=False 让中文以原文写入（而非 \uXXXX 转义），
                    # indent=2 输出两级缩进，便于人工直接查看排查
                    json.dump(tracker_data, f, ensure_ascii=False, indent=2)

        # 兜底捕获：无论是序列化异常还是磁盘 IO 异常，统一包装为存储异常
        except Exception as e:
            # 包装原始异常信息后抛出，保证异常类型符合基类接口约定
            raise TrackerSerializationError(f"保存Tracker失败: {e}")

    async def retrieve(self, sender_id: str) -> DialogueStateTracker | None:
        """从 JSON 文件加载 Tracker

        核心功能：按 sender_id 读取持久化快照并反序列化为 Tracker 实例。

        业务流程：1. 内存模式从 _memory_store 安全取快照（可能为 None）；
                  文件模式计算路径，文件不存在直接返回 None；2. 读取并
                  json.load 解析文件内容；3. 取 Domain 的槽位定义（无
                  Domain 时为 None）；4. 调用 DialogueStateTracker.from_dict
                  重建 Tracker 实例并返回。

        副作用：文件模式下读取磁盘上的 {sender_id}.json；不修改任何状态。

        @param sender_id: 会话 ID；任意字符串（文件模式下会经同样规则清洗
                          定位到文件名）。
        @return: 反序列化得到的 DialogueStateTracker；快照不存在返回 None；
                 JSON 格式损坏抛 TrackerSerializationError、其他失败抛
                 TrackerStoreException（均不返回）。
        """
        try:
            # 分支：内存模式——从内存字典取快照
            if self.in_memory:
                # dict.get 安全取值：键不存在时返回 None 而非抛 KeyError
                tracker_data = self._memory_store.get(sender_id)
            else:
                # 文件模式——计算该 sender_id 对应的 .json 文件路径
                file_path = self._get_file_path(sender_id)

                # 前置判断：文件不存在说明该会话从未保存过，直接返回 None，
                # 交由上层（如 get_or_create_tracker）决定是否新建
                if not file_path.exists():
                    # 返回 None 表示"无此 Tracker"
                    return None

                # 以统一编码 utf-8 打开并读取 JSON 文本
                with open(file_path, "r", encoding=DEFAULT_ENCODING) as f:
                    # 将 JSON 文本解析为字典快照；格式非法时抛 JSONDecodeError
                    tracker_data = json.load(f)

            # 双重保险：内存模式 get 到 None（键不存在）时同样返回 None
            if tracker_data is None:
                # 无快照可恢复，返回 None
                return None

            # 取 Domain 的全部槽位定义；无 Domain 时为 None，from_dict 按无槽位处理
            domain_slots = self.domain.slots if self.domain else None

            # 将快照字典与槽位定义交给 Tracker 的类方法，重建完整 Tracker 实例
            return DialogueStateTracker.from_dict(tracker_data, domain_slots)

        # 精确捕获：JSON 文本格式损坏（非法 JSON）属于序列化层面的问题
        except json.JSONDecodeError as e:
            # 包装后抛出序列化错误，便于上层区分"数据损坏"与"IO 故障"
            raise TrackerSerializationError(f"解析Tracker JSON失败: {e}")
        # 兜底捕获：其余异常（磁盘 IO 故障、内存数据结构异常等）统一包装
        except Exception as e:
            # 抛出通用存储异常，保留原始错误信息便于排查
            raise TrackerStoreException(f"加载Tracker失败: {e}")

    async def delete(self, sender_id: str) -> None:
        """删除 Tracker 文件

        核心功能：删除指定 sender_id 的持久化快照。

        业务流程：1. 内存模式从 _memory_store 弹出对应键（不存在则忽略）；
                  2. 文件模式计算路径，文件存在则执行 unlink 删除。

        副作用：文件模式下删除磁盘上的 .json 文件；内存模式下移除字典
                条目。目标不存在时静默忽略，不报错。

        @param sender_id: 会话 ID；对应要删除的文件名（同样经过字符清洗）。
        @return: 无返回；删除失败时抛 TrackerStoreException（不返回）。
        """
        try:
            # 分支：内存模式——从内存字典移除该键
            if self.in_memory:
                # dict.pop 带默认值 None：键不存在时不抛 KeyError，静默通过
                self._memory_store.pop(sender_id, None)
            else:
                # 文件模式——计算该 sender_id 对应的 .json 文件路径
                file_path = self._get_file_path(sender_id)

                # 先判存在再删除：避免对不存在的文件调用 unlink 抛 FileNotFoundError
                if file_path.exists():
                    # unlink 删除磁盘上的单个文件
                    file_path.unlink()

        # 兜底捕获：删除过程中的磁盘异常统一包装
        except Exception as e:
            # 抛出通用存储异常，保留原始错误信息
            raise TrackerStoreException(f"删除Tracker失败: {e}")

    async def keys(self) -> Iterable[str]:
        """获取所有 sender_id

        核心功能：枚举当前存储中全部会话 ID。

        业务流程：1. 内存模式返回内存字典全部键的列表副本；2. 文件模式
                  先确认存储目录存在，再遍历目录下所有 .json 文件，以
                  文件名（去扩展名）作为 sender_id 收集返回。

        副作用：文件模式下只读遍历目录，不修改任何内容。

        @return: sender_id 的可迭代集合（实际为 list[str]）；文件模式下
                 存储目录不存在（path 为 None）时返回空列表。
        """
        # 分支：内存模式——直接返回内存字典的全部键
        if self.in_memory:
            # list() 浅拷贝键集合，避免调用方操作影响内部字典
            return list(self._memory_store.keys())

        # 防御判断：构造时未指定路径（path=None）则没有可扫描的目录
        if self.path is None:
            # 返回空列表表示"无任何已存储会话"
            return []

        # 结果收集列表：存放从文件名还原出的全部 sender_id
        sender_ids = []

        # glob("*.json") 扫描存储目录下所有 JSON 文件（不递归子目录）
        for file_path in self.path.glob("*.json"):
            # 从文件名还原 sender_id：stem 即去掉 ".json" 扩展名的主文件名
            sender_id = file_path.stem

            # 追加到结果列表
            sender_ids.append(sender_id)

        # 返回收集到的全部 sender_id
        return sender_ids

    async def clear_all(self) -> None:
        """清除所有 Tracker

        核心功能：清空存储中的全部会话快照。

        业务流程：1. 内存模式一次性清空内存字典；2. 文件模式先通过 keys()
                  枚举全部 sender_id，再逐个调用 delete 删除对应文件。

        副作用：文件模式下批量删除磁盘上的全部 .json 文件；内存模式下
                清空 _memory_store。此操作不可逆！

        @return: 无返回；底层 delete 失败时异常向上传播（不返回）。
        """
        # 分支：内存模式——一次性清空整个内存字典
        if self.in_memory:
            # dict.clear 移除全部键值对
            self._memory_store.clear()
        else:
            # 文件模式——先异步枚举全部 sender_id，再逐个删除
            for sender_id in await self.keys():
                # 复用 delete 的删除逻辑，保持异常处理行为与单删一致
                await self.delete(sender_id)
