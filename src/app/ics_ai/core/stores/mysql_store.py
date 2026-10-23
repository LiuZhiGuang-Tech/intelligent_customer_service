"""
mysql_store - MySQL 数据库 Tracker 存储模块

核心功能：实现 MySQLTrackerStore——将 DialogueStateTracker 的对话状态以
          JSON 字符串形式存入 MySQL 表（sender_id 主键 + state 状态列），
          采用 UPSERT 语义实现"存在即更新、不存在即插入"。

执行流程：1. 构造时仅记录连接参数，不发起任何数据库连接（惰性初始化）；
          2. 首次执行 save/retrieve/delete/keys 时由 _ensure_initialized
          延迟导入 sqlalchemy、创建引擎、建表与会话工厂；3. save() 将
          Tracker 序列化为 JSON 写入（ON DUPLICATE KEY UPDATE）；4.
          retrieve() 查询 state 列并反序列化重建 Tracker；5. delete()/
          keys() 提供删除与枚举，close() 释放连接池。

副作用：读写 MySQL 数据库（建表 DDL、INSERT/UPDATE/DELETE/SELECT）；
        维护进程内连接池与初始化标记等实例状态；不读写本地文件、
        不发起数据库之外的网络请求。

主要依赖：json（状态序列化）、sqlalchemy + pymysql（延迟导入的数据库
          ORM 与驱动）、app.ics_ai.shared.exceptions（存储异常体系）、
          app.ics_ai.core.tracker.DialogueStateTracker（被存储的对话状态）、
          app.ics_ai.core.domain.Domain（提供反序列化所需槽位定义）。
"""

# 从标准库 collections.abc 导入 Iterable：keys() 返回值的容器抽象类型
# （PEP 585 推荐从 collections.abc 导入而非 typing）。
from collections.abc import Iterable

# 导入 json 模块：负责 Tracker 状态字典与 JSON 字符串之间的相互转换。
import json

# 导入 TYPE_CHECKING：标记"仅类型检查时执行"的导入分支，运行时自动跳过，
# 用于在不破坏延迟导入设计的前提下为惰性属性提供 sqlalchemy 类型注解。
from typing import TYPE_CHECKING

# 从项目核心模块导入领域模型：retrieve 反序列化时需借用其槽位定义。
from app.ics_ai.core.domain import Domain

# 从同包基类导入 TrackerStore：MySQLTrackerStore 的抽象父类，
# 约定 save/retrieve/delete/keys 等统一接口。
from app.ics_ai.core.stores.tracker_store import TrackerStore

# 从项目核心模块导入对话状态跟踪器：本存储持久化与恢复的目标对象。
from app.ics_ai.core.tracker import DialogueStateTracker

# 从项目共享异常模块导入存储异常体系：序列化失败、连接失败与通用存储
# 失败分别抛出，便于上层按需精确捕获。
from app.ics_ai.shared.exceptions import (
    TrackerSerializationError,
    TrackerStoreConnectionError,
    TrackerStoreException,
)

# 仅类型检查分支：sqlalchemy 类型仅供静态注解使用，运行时仍由
# _ensure_initialized 延迟导入，保持模块级零 sqlalchemy 依赖。
if TYPE_CHECKING:
    # 数据库引擎与表对象类型：用于 _engine/_table 属性注解
    from sqlalchemy import Engine, Table

    # 会话工厂类型：用于 _session_maker 属性注解
    from sqlalchemy.orm import sessionmaker


class MySQLTrackerStore(TrackerStore):
    """MySQL Tracker 存储

    职责：将每个 Tracker 序列化为 JSON 字符串存入 MySQL 表（sender_id
          作主键、state 列承载状态），并提供按会话 ID 的查询恢复、删除
          与全量枚举能力。

    设计模式：策略模式——作为 TrackerStore 抽象接口的一种具体存储策略，
          与 JsonTrackerStore 可互换使用，由工厂函数 create_tracker_store
          按 store_type 选择；同时采用惰性初始化（首次数据操作才建连），
          将数据库依赖与连接开销推迟到真正需要时。

    生命周期：由 Agent 装配阶段（create_tracker_store 工厂）创建并被
          应用持有；首次数据操作时建立连接池并建表，随应用运行期复用；
          调用 close() 或进程退出时释放连接池。

    表结构：
        CREATE TABLE trackers (
            sender_id VARCHAR(255) PRIMARY KEY,
            state JSON NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        );

    属性：
        host: 数据库主机
        port: 数据库端口
        db: 数据库名称
        username: 用户名
        password: 密码
        table_name: 表名
    """

    def __init__(
        self,
        domain: Domain | None = None,
        host: str = "localhost",
        port: int = 3306,
        db: str = "ics",
        username: str = "root",
        password: str = "",
        url: str | None = None,
        table_name: str = "trackers",
        **kwargs,
    ) -> None:
        """初始化 MySQL 存储

        核心功能：记录数据库连接参数；不在此阶段建连（惰性初始化）。

        业务流程：1. 调用父类构造保存 Domain；2. 逐项保存主机/端口/库名/
                  用户名/密码/连接 URL/表名；3. 将引擎、会话工厂、表对象
                  与初始化标记全部置空/置 False，等待首次数据操作时由
                  _ensure_initialized 真正建立。

        副作用：无（仅设置实例属性，不访问网络与文件）。

        @param domain: Domain 实例，反序列化 Tracker 时提供槽位定义；
                       允许 None（此时按无槽位定义反序列化）。
        @param host: 数据库主机地址，默认 "localhost"。
        @param port: 数据库端口，默认 3306（MySQL 默认端口）。
        @param db: 数据库名称，默认 "ics"。
        @param username: 数据库用户名，默认 "root"。
        @param password: 数据库密码，默认空串。
        @param url: 完整的数据库连接 URL；允许 None。传入时优先于
                    host/port 等零散参数。
        @param table_name: Tracker 状态表表名，默认 "trackers"。
        @param kwargs: 接收多余关键字参数（工厂透传额外配置时保持兼容），
                       当前不参与连接逻辑。
        """
        # 调用抽象基类 TrackerStore 的构造函数：保存 domain 供反序列化使用
        super().__init__(domain)

        # 保存数据库主机地址
        self.host = host

        # 保存数据库端口号（int）
        self.port = port

        # 保存数据库名称
        self.db = db

        # 保存数据库用户名
        self.username = username

        # 保存数据库密码
        self.password = password

        # 保存完整连接 URL（优先于零散参数）；未传入时为 None
        self.url = url

        # 保存 Tracker 状态表表名
        self.table_name = table_name

        # 数据库引擎占位：惰性创建，初始化完成后持有连接池；
        # 注解 Engine | None 供静态检查识别（实例属性注解运行时不求值）
        self._engine: Engine | None = None

        # 会话工厂占位：惰性创建，用于开启数据库会话；
        # 注解 sessionmaker | None（同上，运行时不求值）
        self._session_maker: sessionmaker | None = None

        # 表对象占位：惰性创建，承载 trackers 表的 ORM 表定义；
        # 注解 Table | None（同上，运行时不求值）
        self._table: Table | None = None

        # 初始化标记：False 表示尚未建连，首次数据操作时触发建连流程
        self._initialized = False

    def _get_connection_url(self) -> str:
        """获取数据库连接 URL（私有辅助方法）

        核心功能：优先返回显式配置的完整连接 URL，否则按各连接参数
                  拼接 SQLAlchemy 格式的 MySQL URL。

        业务流程：1. 构造时传入了 url 则直接返回（显式配置优先级最高）；
                  2. 否则拼出 "mysql+pymysql://user:pass@host:port/db"
                  并追加 utf8mb4 字符集参数。

        副作用：无（纯字符串拼接）。

        @return: SQLAlchemy 可识别的连接 URL 字符串。
        """
        # 前置判断：显式传入的 url 优先，直接覆盖 host/port 等零散参数
        if self.url:
            # 返回调用方给定的完整连接 URL
            return self.url

        # 按参数拼接连接 URL：驱动选 pymysql（纯 Python 实现，免编译），
        # charset=utf8mb4 保证 Emoji 等 4 字节 Unicode 字符可正常存取
        return (
            f"mysql+pymysql://{self.username}:{self.password}"
            f"@{self.host}:{self.port}/{self.db}?charset=utf8mb4"
        )

    async def _ensure_initialized(self) -> None:
        """确保数据库连接已初始化（私有辅助方法）

        核心功能：惰性初始化——首次数据操作前建立引擎、建表与会话工厂。

        业务流程：1. 已初始化则直接返回（幂等短路）；2. 延迟导入
                  sqlalchemy（依赖未安装时抛带安装指引的 ImportError）；
                  3. 生成连接 URL 并创建带探活/回收策略的引擎；4. 定义
                  trackers 表结构并执行建表（不存在才建）；5. 创建会话
                  工厂并标记初始化完成。

        副作用：连接 MySQL 并可能执行建表 DDL；修改实例内部状态
                （_engine/_session_maker/_table/_initialized）。

        @return: 无返回；sqlalchemy 或 pymysql 未安装抛 ImportError、
                 连接/建表失败抛 TrackerStoreConnectionError（均不返回）。
        """
        # 幂等短路：已初始化过则跳过全部建连流程，避免重复开销
        if self._initialized:
            # 直接返回，后续操作复用既有连接池
            return

        try:
            # 延迟导入 sqlalchemy：将重量级依赖的导入推迟到真正需要数据库时，
            # 未安装该库也不影响本模块被导入与其他功能的静态分析
            from sqlalchemy import (
                # 列定义类：表中每个字段的声明载体
                Column,
                # 日期时间列类型：用于 created_at/updated_at 时间戳字段
                DateTime,
                # 元数据容器：汇集表定义，供 create_all 执行建表
                MetaData,
                # 变长字符串列类型：用于 sender_id 主键列
                String,
                # 表构造类：按列定义组装出表对象（用于声明 trackers 表结构）
                Table,
                # 长文本列类型：别名 SQLText 与 Python 字符串类型概念区分，
                # 用于承载 Tracker 状态的 JSON 文本
                Text as SQLText,
                # 引擎工厂：根据连接 URL 创建数据库引擎（连接池）
                create_engine,
                # SQL 函数 now()：为时间戳列提供数据库端默认值
                func,
            )

            # 导入会话工厂：sessionmaker 生成绑定引擎的 Session 类
            from sqlalchemy.orm import sessionmaker

            # 生成连接 URL：显式 url 优先，否则按 host/port 等参数拼接
            connection_url = self._get_connection_url()

            # 创建数据库引擎（此刻惰性，不真正建连）：
            # pool_pre_ping=True 取连接前先 ping，自动剔除被服务端断开的死连接；
            # pool_recycle=3600 连接最多复用 1 小时，规避 MySQL wait_timeout 断连
            self._engine = create_engine(
                connection_url,
                pool_pre_ping=True,
                pool_recycle=3600,
            )

            # 创建元数据容器：管理本存储的表定义集合
            metadata = MetaData()

            # 定义 trackers 表结构：sender_id 主键 + state 状态文本 +
            # 数据库端自动维护的创建/更新时间戳
            self._table = Table(
                # 表名：默认 "trackers"，可经构造参数自定义
                self.table_name,
                # 表定义归属的元数据容器
                metadata,
                # 会话 ID 列：变长字符串主键，最长 255 字符
                Column("sender_id", String(255), primary_key=True),
                # 状态列：Tracker 序列化后的 JSON 文本，不允许为空
                Column("state", SQLText, nullable=False),
                # 创建时间列：插入时由数据库 now() 填充
                Column("created_at", DateTime, default=func.now()),
                # 更新时间列：插入时填充，行更新时由 onupdate 自动刷新
                Column("updated_at", DateTime, default=func.now(), onupdate=func.now()),
            )

            # 执行建表：create_all 仅在表不存在时创建，已存在则跳过（幂等）
            metadata.create_all(self._engine)

            # 创建会话工厂：绑定引擎，后续每次数据操作用它开启独立会话
            self._session_maker = sessionmaker(bind=self._engine)

            # 标记初始化完成：后续调用走幂等短路
            self._initialized = True

        # 精确捕获：sqlalchemy/pymysql 未安装属于依赖缺失问题
        # （ModuleNotFoundError 是 ImportError 的子类，同样在此被捕获）
        except ImportError:
            # 抛出带安装指引的 ImportError，提示用户补装驱动依赖
            raise ImportError(
                "使用MySQL存储需要安装sqlalchemy和pymysql包。"
                "请运行: pip install sqlalchemy pymysql -i https://pypi.tuna.tsinghua.edu.cn/simple"
            )
        # 兜底捕获：连接失败、建表失败等数据库层异常统一包装
        except Exception as e:
            # 抛出连接异常，保留原始错误信息便于排查
            raise TrackerStoreConnectionError(f"连接MySQL失败: {e}")

    async def save(self, tracker: DialogueStateTracker) -> None:
        """保存 Tracker 到 MySQL

        核心功能：将 Tracker 序列化为 JSON 后以 UPSERT 语义写入状态表
                  （存在则更新，不存在则插入）。

        业务流程：1. 确保连接已初始化；2. 延迟导入 MySQL 方言的 insert
                  构造器；3. 调用 tracker.to_dict() 得到状态快照并序列化
                  为 JSON 字符串（中文保留原文）；4. 构造插入语句并在
                  会话中执行 ON DUPLICATE KEY UPDATE，提交事务。

        副作用：向 MySQL 写入/更新一行 trackers 记录（含 updated_at
                自动刷新）；保持会话在 with 块结束时自动关闭。

        @param tracker: 要保存的 DialogueStateTracker 实例，不能为 None。
        @return: 无返回；初始化失败沿 _ensure_initialized 异常向上抛，
                 写入失败抛 TrackerSerializationError（均不返回）。
        """
        # 惰性建连：保证引擎/会话工厂/表对象就绪（已初始化则直接返回）
        await self._ensure_initialized()

        try:
            # 延迟导入通用插入构造器（本方法实际使用 MySQL 方言版本）
            from sqlalchemy import insert

            # 延迟导入 MySQL 方言插入构造器：支持 ON DUPLICATE KEY UPDATE 子句
            from sqlalchemy.dialects.mysql import insert as mysql_insert

            # 将 Tracker 的完整状态（槽位、轮次、对话栈等）序列化为普通字典
            tracker_data = tracker.to_dict()

            # 序列化为 JSON 字符串；ensure_ascii=False 让中文以原文入库
            state_json = json.dumps(tracker_data, ensure_ascii=False)

            # 开启独立数据库会话：with 块结束时自动关闭会话释放资源
            # （压制 reportOptionalCall：上方 _ensure_initialized 已保证
            # _session_maker 必非 None，静态检查器无法跨方法追踪该不变式）
            with self._session_maker() as session:  # pyright: ignore[reportOptionalCall]
                # 构造 MySQL 插入语句：写入 sender_id 主键与状态 JSON 文本
                # （压制 reportArgumentType：同上，_table 惰性初始化后必非 None）
                stmt = mysql_insert(self._table).values(  # pyright: ignore[reportArgumentType]
                    sender_id=tracker.sender_id,
                    state=state_json,
                )

                # 追加 MySQL 特有的 UPSERT 子句：主键冲突时把 state 列
                # 更新为本语句待插入的值（实现"存在即更新"）
                stmt = stmt.on_duplicate_key_update(
                    state=stmt.inserted.state,
                )

                # 在当前会话中执行 UPSERT 语句
                session.execute(stmt)

                # 提交事务：将变更持久化到数据库
                session.commit()

        # 兜底捕获：序列化异常与数据库写入异常统一包装为序列化错误
        except Exception as e:
            # 包装原始异常信息后抛出，保留错误上下文便于排查
            raise TrackerSerializationError(f"保存Tracker到MySQL失败: {e}")

    async def retrieve(self, sender_id: str) -> DialogueStateTracker | None:
        """从 MySQL 加载 Tracker

        核心功能：按 sender_id 查询状态 JSON 并反序列化为 Tracker 实例。

        业务流程：1. 确保连接已初始化；2. 延迟导入 select 构造器；3.
                  开启会话，查询该 sender_id 对应的 state 列（取第一行）；
                  4. 无结果返回 None；5. 有结果则解析 JSON 并借助 Domain
                  槽位定义重建 Tracker。

        副作用：对 MySQL 执行一次只读 SELECT；不修改任何状态。

        @param sender_id: 会话 ID；对应表中 sender_id 主键列的值。
        @return: 反序列化得到的 DialogueStateTracker；记录不存在返回
                 None；JSON 解析失败抛 TrackerSerializationError、其他
                 失败抛 TrackerStoreException（均不返回）。
        """
        # 惰性建连：保证引擎/会话工厂/表对象就绪（已初始化则直接返回）
        await self._ensure_initialized()

        try:
            # 延迟导入查询构造器：按需加载，维持模块级零 sqlalchemy 依赖
            from sqlalchemy import select

            # 开启独立数据库会话：with 块结束时自动关闭
            # （压制 reportOptionalCall：上方 _ensure_initialized 已保证
            # _session_maker 必非 None，静态检查器无法跨方法追踪该不变式）
            with self._session_maker() as session:  # pyright: ignore[reportOptionalCall]
                # 构造 SELECT state FROM trackers WHERE sender_id = ? 查询
                # （压制 reportOptionalMemberAccess：同上，_table 必非 None）
                stmt = select(self._table.c.state).where(  # pyright: ignore[reportOptionalMemberAccess]
                    self._table.c.sender_id == sender_id  # pyright: ignore[reportOptionalMemberAccess]
                )

                # 执行查询并取第一行；主键查询至多一行，无匹配时为 None
                result = session.execute(stmt).fetchone()

                # 前置判断：无记录说明该会话从未保存过，直接返回 None
                if result is None:
                    # 返回 None 表示"无此 Tracker"，由上层决定是否新建
                    return None

                # 取出第一列（state）的 JSON 字符串
                state_json = result[0]

                # 将 JSON 字符串解析为字典快照；格式非法时抛 JSONDecodeError
                tracker_data = json.loads(state_json)

                # 取 Domain 的全部槽位定义；无 Domain 时为 None，按无槽位处理
                domain_slots = self.domain.slots if self.domain else None

                # 将快照字典与槽位定义交给 Tracker 的类方法，重建完整实例
                return DialogueStateTracker.from_dict(tracker_data, domain_slots)

        # 精确捕获：状态列内容格式损坏（非法 JSON）属于序列化层面问题
        except json.JSONDecodeError as e:
            # 包装后抛出序列化错误，便于上层区分"数据损坏"与"IO 故障"
            raise TrackerSerializationError(f"解析Tracker JSON失败: {e}")
        # 兜底捕获：查询执行失败等数据库异常统一包装
        except Exception as e:
            # 抛出通用存储异常，保留原始错误信息便于排查
            raise TrackerStoreException(f"从MySQL加载Tracker失败: {e}")

    async def delete(self, sender_id: str) -> None:
        """从 MySQL 删除 Tracker

        核心功能：删除指定 sender_id 的状态记录。

        业务流程：1. 确保连接已初始化；2. 延迟导入 delete 构造器；3.
                  开启会话，构造并执行按 sender_id 过滤的 DELETE 语句，
                  提交事务。

        副作用：从 MySQL 删除匹配的记录行；记录不存在时 DELETE 影响
                0 行，静默通过不报错。

        @param sender_id: 会话 ID；对应表中 sender_id 主键列的值。
        @return: 无返回；删除失败抛 TrackerStoreException（不返回）。
        """
        # 惰性建连：保证引擎/会话工厂/表对象就绪（已初始化则直接返回）
        await self._ensure_initialized()

        try:
            # 延迟导入删除构造器：按需加载
            from sqlalchemy import delete

            # 开启独立数据库会话：with 块结束时自动关闭
            # （压制 reportOptionalCall：上方 _ensure_initialized 已保证
            # _session_maker 必非 None，静态检查器无法跨方法追踪该不变式）
            with self._session_maker() as session:  # pyright: ignore[reportOptionalCall]
                # 构造 DELETE FROM trackers WHERE sender_id = ? 语句
                # （压制 reportArgumentType/reportOptionalMemberAccess：_table 必非 None）
                stmt = delete(self._table).where(self._table.c.sender_id == sender_id)  # pyright: ignore[reportArgumentType, reportOptionalMemberAccess]

                # 在当前会话中执行删除语句（不存在的 ID 影响 0 行，不报错）
                session.execute(stmt)

                # 提交事务：使删除生效
                session.commit()

        # 兜底捕获：删除执行失败等数据库异常统一包装
        except Exception as e:
            # 抛出通用存储异常，保留原始错误信息
            raise TrackerStoreException(f"从MySQL删除Tracker失败: {e}")

    async def keys(self) -> Iterable[str]:
        """获取所有 sender_id

        核心功能：查询状态表中全部会话 ID。

        业务流程：1. 确保连接已初始化；2. 延迟导入 select 构造器；3.
                  开启会话，查询全部行的 sender_id 列；4. 将结果行
                  解包为字符串列表返回。

        副作用：对 MySQL 执行一次只读 SELECT；不修改任何状态。

        @return: sender_id 字符串列表；表为空时返回空列表。
        """
        # 惰性建连：保证引擎/会话工厂/表对象就绪（已初始化则直接返回）
        await self._ensure_initialized()

        try:
            # 延迟导入查询构造器：按需加载
            from sqlalchemy import select

            # 开启独立数据库会话：with 块结束时自动关闭
            # （压制 reportOptionalCall：上方 _ensure_initialized 已保证
            # _session_maker 必非 None，静态检查器无法跨方法追踪该不变式）
            with self._session_maker() as session:  # pyright: ignore[reportOptionalCall]
                # 构造 SELECT sender_id FROM trackers 全表查询（仅取主键列）
                # （压制 reportOptionalMemberAccess：同上，_table 必非 None）
                stmt = select(self._table.c.sender_id)  # pyright: ignore[reportOptionalMemberAccess]

                # 执行查询并取回全部结果行
                results = session.execute(stmt).fetchall()

                # 列表推导逐行解包：每行取第 0 列（sender_id）组成字符串列表
                return [row[0] for row in results]

        # 兜底捕获：查询执行失败等数据库异常统一包装
        except Exception as e:
            # 抛出通用存储异常，保留原始错误信息
            raise TrackerStoreException(f"获取sender_id列表失败: {e}")

    async def close(self) -> None:
        """关闭数据库连接

        核心功能：释放连接池资源并重置初始化标记。

        业务流程：1. 引擎存在时调用 dispose() 关闭连接池中的全部连接；
                  2. 将 _initialized 置 False，后续数据操作会重新走
                  _ensure_initialized 建连流程。

        副作用：关闭与 MySQL 的全部池化连接（释放服务端会话资源）；
                修改实例内部状态。

        @return: 无返回；引擎未初始化时静默跳过，不报错。
        """
        # 前置判断：引擎已创建才需要释放（未初始化时静默跳过）
        if self._engine:
            # dispose 关闭连接池中的全部连接，归还/释放数据库资源
            self._engine.dispose()

            # 重置初始化标记：允许下次数据操作重新建立连接池
            self._initialized = False
