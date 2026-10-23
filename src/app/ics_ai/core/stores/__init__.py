"""
stores - Tracker 存储入口模块

核心功能：作为 core.stores 包的统一出口，汇聚 Tracker 持久化存储的三个
          后端实现与工厂函数——TrackerStore 抽象基类、JsonTrackerStore
          （本地文件/内存）、MySQLTrackerStore（关系数据库），对外暴露
          create_tracker_store 按 store_type 一站式创建存储实例。

执行流程：1. 导入本包时依次加载三个子模块（均无重量级顶层依赖）；2.
          Agent 装配阶段调用 create_tracker_store(store_type, **kwargs)
          ——工厂按类型归一化后分发到对应后端构造器，多余配置经 kwargs
          透传；3. 业务层面向 TrackerStore 基类接口编程，不感知具体后端。

副作用：仅执行子模块导入（三个子模块顶层均无文件/网络 IO）；不读写
        文件/数据库、不发起网络请求、不修改全局状态。

主要依赖：app.ics_ai.core.stores.tracker_store（抽象基类）、
          app.ics_ai.core.stores.json_store（文件/内存后端）、
          app.ics_ai.core.stores.mysql_store（数据库后端）。
"""

# 导入抽象基类：三个后端的统一接口契约，同时作为工厂函数的返回类型。
from app.ics_ai.core.stores.tracker_store import (
    TrackerStore,
)

# 导入 JSON 文件存储后端：落盘文件或纯内存两种模式，适合开发与测试。
from app.ics_ai.core.stores.json_store import (
    JsonTrackerStore,
)

# 导入 MySQL 数据库存储后端：关系数据库持久化，适合生产与多实例部署。
from app.ics_ai.core.stores.mysql_store import (
    MySQLTrackerStore,
)

# 公开导出清单：按字母序排列，约束 from core.stores import * 的导出范围
# 并为 IDE 提供准确的补全与跳转依据。
__all__ = [
    # JSON 文件/内存存储后端类
    "JsonTrackerStore",
    # MySQL 数据库存储后端类
    "MySQLTrackerStore",
    # Tracker 存储抽象基类
    "TrackerStore",
    # 存储实例工厂函数
    "create_tracker_store",
]


def create_tracker_store(
    store_type: str = "json",
    **kwargs,
) -> TrackerStore:
    """创建 Tracker 存储实例

    核心功能：工厂函数——按 store_type 选择并构造对应的存储后端实例。

    业务流程：1. 将 store_type 统一转小写（容忍 "JSON"/"MySQL" 等大小写
              变体）；2. 按归一化结果三分支分发——json 构造 JsonTrackerStore、
              mysql 构造 MySQLTrackerStore（两者均透传 kwargs 配置）、
              memory 复用 JsonTrackerStore 的内存模式（path=None 且
              in_memory=True，不产生任何磁盘 IO）；3. 未知类型抛 ValueError。

    副作用：构造 JsonTrackerStore（文件模式）时会在磁盘上创建存储目录；
            构造 MySQLTrackerStore 本身不建连（惰性初始化，首次数据操作
            才真正连接数据库）。

    @param store_type: 存储类型，取值 json/mysql/memory（大小写不敏感），
                       默认 "json"；其他值抛 ValueError。
    @param kwargs: 透传给对应后端构造器的配置参数（如 path、host、url 等）。
    @return: TrackerStore 实例（具体子类由 store_type 决定）。

    示例：
        >>> store = create_tracker_store("json", path="./trackers")
        >>> await store.save(tracker)
    """
    # 类型归一化：统一转小写，消除调用方大小写书写差异
    store_type = store_type.lower()

    # 分支：JSON 文件存储——kwargs 透传 path 等配置
    if store_type == "json":
        # 构造文件存储后端并返回
        return JsonTrackerStore(**kwargs)
    # 分支：MySQL 数据库存储——kwargs 透传 host/url 等连接配置
    elif store_type == "mysql":
        # 构造数据库存储后端并返回（惰性建连，构造阶段不触库）
        return MySQLTrackerStore(**kwargs)
    # 分支：纯内存存储——复用 JSON 存储实现，关闭落盘能力
    elif store_type == "memory":
        # path=None 且 in_memory=True：全部读写走进程内字典，零磁盘 IO
        return JsonTrackerStore(path=None, in_memory=True)
    # 兜底分支：未知的存储类型直接拒绝并给出合法取值提示
    else:
        # 抛出值错误：附上实际传入值与支持的三种类型，便于调用方修正
        raise ValueError(
            f"不支持的存储类型: {store_type}。支持的类型: json, mysql, memory"
        )
