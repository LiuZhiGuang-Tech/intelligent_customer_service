"""
db_table_class - 数据库表 ORM 映射类模块

核心功能：定义电商客服示例库（ecs）全部 13 张业务表与 2 张多对多
          关联表的 SQLAlchemy ORM 映射类——物流、物流公司、物流投诉
          对照、订单状态、售后状态、商品类别、区域、用户、物流投诉
          记录、售后原因、收货信息、商品、订单、订单明细、售后，供
          各 Action 查询与更新对话业务数据。

执行流程：1. 使用方经 db.SessionLocal 开启会话；2. 以本模块类执行
          查询/更新（关系经 relationship 惰性加载，查询侧多用
          joinedload 预加载规避 N+1）；3. 表结构变更时经 db.py 的
          __main__ 反向工程重新生成本文件。

副作用：模块导入仅注册类与元数据（无 IO，不建连）；实例操作由
        使用方的会话提交驱动。

主要依赖：sqlalchemy（声明式基类/列类型/关系）、
          app.ics_ecs.actions.db（引擎与会话工厂，本模块不依赖它）。

生成说明：本文件由 sqlacodegen 反向工程自动生成（db.py __main__ /
          export_db_table_class），列注释取自 MySQL 表注释；手工
          补充的类 docstring 在再生成后会被覆盖，需重新补充。
"""

# 导入 Optional：可空列的 Mapped 注解类型（生成器标准输出风格）。
from typing import Optional

# 导入 datetime/decimal：时间戳列与金额列的 Python 映射类型。
import datetime
import decimal

# 导入 sqlalchemy 列类型与表构造工具：各表列定义与关联表声明使用。
from sqlalchemy import (
    BigInteger,
    Column,
    DECIMAL,
    Enum,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    TIMESTAMP,
    Table,
    text,
)

# 导入 MySQL 方言专属类型：TINYINT 布尔标记列（MySQL 特有）。
from sqlalchemy.dialects.mysql import TINYINT

# 导入 SQLAlchemy 2.0 声明式 ORM 构件：DeclarativeBase 基类、Mapped
# 类型注解、mapped_column 列声明与 relationship 关系声明。
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """声明式基类：全部 ORM 类的公共父类，持有统一元数据注册表。"""

    pass


class Logistics(Base):
    """物流表

    职责：记录每条物流单的轨迹明细与类别（退货/换货退货/换货发货）；
          经 order_logistics/postsale_logistics 两张关联表与订单、
          售后形成多对多关系（一单多段物流、一次售后多段物流）。
    """

    __tablename__ = "logistics"
    __table_args__ = {"comment": "物流表"}

    # 物流单唯一标识（主键）
    logistics_id: Mapped[str] = mapped_column(
        String(50), primary_key=True, comment="物流ID"
    )

    # 物流创建时间（不可空）
    create_time: Mapped[datetime.datetime] = mapped_column(
        TIMESTAMP, nullable=False, comment="创建时间"
    )

    # 签收时间（可空：未签收时为 NULL）
    delivered_time: Mapped[Optional[datetime.datetime]] = mapped_column(
        TIMESTAMP, comment="签收时间"
    )

    # 物流轨迹明细文本（多行，Action 取最后一行作为"最近物流"）
    logistics_tracking: Mapped[Optional[str]] = mapped_column(
        String(500), comment="物流明细"
    )

    # 物流类别枚举：退货 / 换货退货 / 换货发货（可空）
    logistics_category: Mapped[Optional[str]] = mapped_column(
        Enum("退货", "换货退货", "换货发货"), comment="物流类别"
    )

    # 多对多：本物流关联的订单集合（经 order_logistics 关联表）
    order: Mapped[list["OrderInfo"]] = relationship(
        "OrderInfo", secondary="order_logistics", back_populates="logistics"
    )

    # 多对多：本物流关联的售后集合（经 postsale_logistics 关联表）
    postsale: Mapped[list["Postsale"]] = relationship(
        "Postsale", secondary="postsale_logistics", back_populates="logistics"
    )

    # 一对多：本物流的投诉记录集合
    logistics_complaints_record: Mapped[list["LogisticsComplaintsRecord"]] = (
        relationship("LogisticsComplaintsRecord", back_populates="logistics")
    )


class LogisticsCompany(Base):
    """物流公司表：维护物流公司名称字典（单列主键表）。"""

    __tablename__ = "logistics_company"
    __table_args__ = {"comment": "物流公司表"}

    # 物流公司名称即主键（业务键直作主键）
    company_name: Mapped[str] = mapped_column(
        String(20), primary_key=True, comment="物流公司名称"
    )


class LogisticsComplaint(Base):
    """物流投诉内容对照表：物流状态到标准投诉话术的映射字典
    （联合主键：物流状态 + 投诉内容）。"""

    __tablename__ = "logistics_complaint"
    __table_args__ = {"comment": "物流投诉内容对照表"}

    # 物流状态（联合主键之一）
    logistics_status: Mapped[str] = mapped_column(
        String(20), primary_key=True, comment="物流状态"
    )

    # 该状态下可用的投诉内容（联合主键之二）
    logistics_complaint: Mapped[str] = mapped_column(
        String(100), primary_key=True, comment="物流投诉内容"
    )


class OrderStatus(Base):
    """订单状态表：订单状态字典并携带状态码——Action 依 status_code
    数值段筛选订单（如 ≤310 可取消、≤320 可改地址、≥330 可售后、
    ≥400 已进入售后）。"""

    __tablename__ = "order_status"
    __table_args__ = {"comment": "订单状态表"}

    # 订单状态名即主键（与 OrderInfo.order_status 外键对应）
    order_status: Mapped[str] = mapped_column(
        String(20), primary_key=True, comment="订单状态"
    )

    # 状态码：控制可执行操作的数值门槛（可空）
    status_code: Mapped[Optional[int]] = mapped_column(Integer, comment="状态码")

    # 一对多：处于该状态的订单集合（反向引用 order_status_）
    order_info: Mapped[list["OrderInfo"]] = relationship(
        "OrderInfo", back_populates="order_status_"
    )


class PostsaleStatus(Base):
    """售后状态表：售后状态字典并携带三类业务标记（是否退款/退货/
    换货）与状态码，驱动售后流程分支。"""

    __tablename__ = "postsale_status"
    __table_args__ = {"comment": "售后状态表"}

    # 售后状态名即主键（与 Postsale.postsale_status 外键对应）
    postsale_status: Mapped[str] = mapped_column(
        String(20), primary_key=True, comment="售后状态"
    )

    # 是否退款标记（TINYINT 0/1，不可空）
    is_refund: Mapped[int] = mapped_column(
        TINYINT(1), nullable=False, comment="是否退款"
    )

    # 是否退货标记（TINYINT 0/1，不可空）
    is_return: Mapped[int] = mapped_column(
        TINYINT(1), nullable=False, comment="是否退货"
    )

    # 是否换货标记（TINYINT 0/1，不可空）
    is_exchange: Mapped[int] = mapped_column(
        TINYINT(1), nullable=False, comment="是否换货"
    )

    # 状态码：售后流程推进门槛（可空）
    status_code: Mapped[Optional[int]] = mapped_column(Integer, comment="状态码")

    # 一对多：处于该状态的售后单集合
    postsale: Mapped[list["Postsale"]] = relationship(
        "Postsale", back_populates="postsale_status_"
    )


class ProductCategory(Base):
    """商品类别表：类别字典，向下关联售后原因与商品信息。"""

    __tablename__ = "product_category"
    __table_args__ = {"comment": "商品类别表"}

    # 商品类别名即主键
    product_category: Mapped[str] = mapped_column(
        String(20), primary_key=True, comment="商品类别"
    )

    # 一对多：该类别下的售后原因集合
    postsale_reason: Mapped[list["PostsaleReason"]] = relationship(
        "PostsaleReason", back_populates="product_category_"
    )

    # 一对多：该类别下的商品集合
    sku_info: Mapped[list["SkuInfo"]] = relationship(
        "SkuInfo", back_populates="product_category"
    )


class Region(Base):
    """区域表：省/市/区三级行政区划字典（三列联合主键），收货信息
    表的三列外键指向此处；Action 逐级去重查询生成省→市→区按钮。"""

    __tablename__ = "region"
    __table_args__ = {"comment": "区域表"}

    # 省（联合主键之一）
    province: Mapped[str] = mapped_column(String(20), primary_key=True, comment="省")

    # 市（联合主键之二）
    city: Mapped[str] = mapped_column(String(20), primary_key=True, comment="市")

    # 区（联合主键之三）
    district: Mapped[str] = mapped_column(String(20), primary_key=True, comment="区")

    # 一对多：引用该行政区划的收货地址集合
    receive_info: Mapped[list["ReceiveInfo"]] = relationship(
        "ReceiveInfo", back_populates="region"
    )


class UserInfo(Base):
    """用户信息表：用户主档（单列 user_id），订单/收货地址/投诉记录
    均以其为外键锚点。"""

    __tablename__ = "user_info"
    __table_args__ = {"comment": "用户信息表"}

    # 用户唯一标识（主键；对话中 user_id 槽缺省回退 "1001"）
    user_id: Mapped[str] = mapped_column(String(50), primary_key=True, comment="用户ID")

    # 一对多：该用户的物流投诉记录
    logistics_complaints_record: Mapped[list["LogisticsComplaintsRecord"]] = (
        relationship("LogisticsComplaintsRecord", back_populates="user")
    )

    # 一对多：该用户的收货地址
    receive_info: Mapped[list["ReceiveInfo"]] = relationship(
        "ReceiveInfo", back_populates="user"
    )

    # 一对多：该用户的订单
    order_info: Mapped[list["OrderInfo"]] = relationship(
        "OrderInfo", back_populates="user"
    )


class LogisticsComplaintsRecord(Base):
    """物流投诉记录表：用户对某条物流的投诉事实记录（物流 + 用户
    双外键）。"""

    __tablename__ = "logistics_complaints_record"
    __table_args__ = (
        # 外键：投诉记录 → 物流单
        ForeignKeyConstraint(
            ["logistics_id"],
            ["logistics.logistics_id"],
            name="logistics_complaints_record_ibfk_2",
        ),
        # 外键：投诉记录 → 用户
        ForeignKeyConstraint(
            ["user_id"],
            ["user_info.user_id"],
            name="logistics_complaints_record_ibfk_1",
        ),
        # 普通索引：按物流查询投诉
        Index("logistics_id", "logistics_id"),
        # 普通索引：按用户查询投诉
        Index("user_id", "user_id"),
        {"comment": "物流投诉记录表"},
    )

    # 投诉记录自增主键
    record_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, comment="投诉记录ID"
    )

    # 被投诉的物流单 ID（不可空）
    logistics_id: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="物流ID"
    )

    # 投诉内容文本（不可空）
    logistics_complaint: Mapped[str] = mapped_column(
        String(500), nullable=False, comment="物流投诉内容"
    )

    # 投诉时间（不可空）
    complaint_time: Mapped[datetime.datetime] = mapped_column(
        TIMESTAMP, nullable=False, comment="投诉时间"
    )

    # 投诉用户 ID（不可空）
    user_id: Mapped[str] = mapped_column(String(50), nullable=False, comment="用户ID")

    # 多对一：投诉指向的物流单
    logistics: Mapped["Logistics"] = relationship(
        "Logistics", back_populates="logistics_complaints_record"
    )

    # 多对一：投诉发起用户
    user: Mapped["UserInfo"] = relationship(
        "UserInfo", back_populates="logistics_complaints_record"
    )


class PostsaleReason(Base):
    """售后原因表：按商品类别归组的售后原因话术字典。"""

    __tablename__ = "postsale_reason"
    __table_args__ = (
        # 外键：售后原因 → 商品类别
        ForeignKeyConstraint(
            ["product_category"],
            ["product_category.product_category"],
            name="postsale_reason_ibfk_1",
        ),
        # 普通索引：按类别筛选原因
        Index("product_category", "product_category"),
        {"comment": "售后原因表"},
    )

    # 售后原因话术即主键
    postsale_reason: Mapped[str] = mapped_column(
        String(100), primary_key=True, comment="售后原因"
    )

    # 所属商品类别（可空：通用原因不挂类别）
    product_category: Mapped[Optional[str]] = mapped_column(
        String(20), comment="商品类别"
    )

    # 多对一：原因所属的商品类别（尾下划线规避与列名同名冲突）
    product_category_: Mapped[Optional["ProductCategory"]] = relationship(
        "ProductCategory", back_populates="postsale_reason"
    )


class ReceiveInfo(Base):
    """收货信息表：用户的收货地址簿——用户外键 + 三级行政区划外键；
    四项地址要素设唯一索引（同用户同地址不重复入库，Action 修改
    收货信息时以此判重）。"""

    __tablename__ = "receive_info"
    __table_args__ = (
        # 外键：省市区三列联合 → region 表对应三列
        ForeignKeyConstraint(
            ["receive_province", "receive_city", "receive_district"],
            ["region.province", "region.city", "region.district"],
            name="receive_info_ibfk_2",
        ),
        # 外键：收货地址 → 用户
        ForeignKeyConstraint(
            ["user_id"], ["user_info.user_id"], name="receive_info_ibfk_1"
        ),
        # 普通索引：按行政区划筛选
        Index(
            "receive_province", "receive_province", "receive_city", "receive_district"
        ),
        # 唯一索引：同一用户的完全相同地址仅允许一条
        Index(
            "user_id",
            "user_id",
            "receiver_name",
            "receiver_phone",
            "receive_province",
            "receive_city",
            "receive_district",
            "receive_street_address",
            unique=True,
        ),
        {"comment": "收货信息表"},
    )

    # 收货地址唯一标识（主键；新建时按 "rec"+16 位 hex 生成）
    receive_id: Mapped[str] = mapped_column(
        String(50), primary_key=True, comment="收货信息ID"
    )

    # 所属用户 ID（不可空）
    user_id: Mapped[str] = mapped_column(String(50), nullable=False, comment="用户ID")

    # 收货人姓名（不可空）
    receiver_name: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="收货人姓名"
    )

    # 收货人电话（不可空）
    receiver_phone: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="收货人电话"
    )

    # 收货省（不可空，外键组件）
    receive_province: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="收货省"
    )

    # 收货市（不可空，外键组件）
    receive_city: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="收货市"
    )

    # 收货区（不可空，外键组件）
    receive_district: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="收货区"
    )

    # 收货详细地址（不可空）
    receive_street_address: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="收货详细地址"
    )

    # 多对一：地址指向的行政区划
    region: Mapped["Region"] = relationship("Region", back_populates="receive_info")

    # 多对一：地址所属用户
    user: Mapped["UserInfo"] = relationship("UserInfo", back_populates="receive_info")

    # 一对多：使用该地址的订单集合（反向引用 receive）
    order_info: Mapped[list["OrderInfo"]] = relationship(
        "OrderInfo", back_populates="receive"
    )

    # 一对多：使用该地址的售后单集合
    postsale: Mapped[list["Postsale"]] = relationship(
        "Postsale", back_populates="receive"
    )


class SkuInfo(Base):
    """商品信息表：SKU 主档——名称/单价/类别/库存，订单明细经
    sku_id 外键引用。"""

    __tablename__ = "sku_info"
    __table_args__ = (
        # 外键：商品 → 商品类别
        ForeignKeyConstraint(
            ["sku_category"],
            ["product_category.product_category"],
            name="sku_info_ibfk_1",
        ),
        # 普通索引：按类别筛选商品
        Index("sku_category", "sku_category"),
        {"comment": "商品信息表"},
    )

    # 商品唯一标识（主键）
    sku_id: Mapped[str] = mapped_column(String(50), primary_key=True, comment="商品ID")

    # 商品名称（不可空；订单明细冗余同名快照列）
    sku_name: Mapped[str] = mapped_column(
        String(100), nullable=False, comment="商品名称"
    )

    # 商品单价，DECIMAL(10,2) 精确小数（不可空）
    sku_price: Mapped[decimal.Decimal] = mapped_column(
        DECIMAL(10, 2), nullable=False, comment="商品单价"
    )

    # 商品类别（不可空，外键）
    sku_category: Mapped[str] = mapped_column(
        String(20), nullable=False, comment="商品类别"
    )

    # 商品库存数（不可空）
    sku_count: Mapped[int] = mapped_column(Integer, nullable=False, comment="商品库存")

    # 多对一：商品所属类别
    product_category: Mapped["ProductCategory"] = relationship(
        "ProductCategory", back_populates="sku_info"
    )

    # 一对多：包含该商品的订单明细集合
    order_detail: Mapped[list["OrderDetail"]] = relationship(
        "OrderDetail", back_populates="sku"
    )


class OrderInfo(Base):
    """订单表：电商客服核心主档——用户/收货地址/订单状态三外键，
    四个关键时间点（创建/支付/签收/完成）驱动状态流转；经
    order_logistics 关联表与物流多对多。"""

    __tablename__ = "order_info"
    __table_args__ = (
        # 外键：订单 → 订单状态字典
        ForeignKeyConstraint(
            ["order_status"], ["order_status.order_status"], name="order_info_ibfk_3"
        ),
        # 外键：订单 → 收货地址
        ForeignKeyConstraint(
            ["receive_id"], ["receive_info.receive_id"], name="order_info_ibfk_2"
        ),
        # 外键：订单 → 用户
        ForeignKeyConstraint(
            ["user_id"], ["user_info.user_id"], name="order_info_ibfk_1"
        ),
        # 普通索引：按状态筛选订单
        Index("order_status", "order_status"),
        # 普通索引：按地址反查订单
        Index("receive_id", "receive_id"),
        # 普通索引：按用户查订单列表
        Index("user_id", "user_id"),
        {"comment": "订单表"},
    )

    # 订单唯一标识（主键；对话中 order_id 槽经按钮选择写入）
    order_id: Mapped[str] = mapped_column(
        String(50), primary_key=True, comment="订单ID"
    )

    # 订单创建时间（不可空；3 日内完成判定基准之一）
    create_time: Mapped[datetime.datetime] = mapped_column(
        TIMESTAMP, nullable=False, comment="创建时间"
    )

    # 下单用户 ID（不可空）
    user_id: Mapped[str] = mapped_column(String(50), nullable=False, comment="用户ID")

    # 收货地址 ID（不可空；修改收货信息即更新此列）
    receive_id: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="收货信息ID"
    )

    # 订单状态名（不可空；取消订单即改写为"已取消"）
    order_status: Mapped[str] = mapped_column(
        String(20), nullable=False, comment="订单状态"
    )

    # 支付时间（可空：未支付为 NULL）
    payment_time: Mapped[Optional[datetime.datetime]] = mapped_column(
        TIMESTAMP, comment="支付时间"
    )

    # 签收时间（可空：未签收为 NULL）
    delivered_time: Mapped[Optional[datetime.datetime]] = mapped_column(
        TIMESTAMP, comment="签收时间"
    )

    # 完成时间（可空：未完成为 NULL；取消订单时也会写入）
    complete_time: Mapped[Optional[datetime.datetime]] = mapped_column(
        TIMESTAMP, comment="完成时间"
    )

    # 多对多：订单关联的物流段集合（经 order_logistics）
    logistics: Mapped[list["Logistics"]] = relationship(
        "Logistics", secondary="order_logistics", back_populates="order"
    )

    # 多对一：订单状态字典行（尾下划线规避与列名同名冲突）
    order_status_: Mapped["OrderStatus"] = relationship(
        "OrderStatus", back_populates="order_info"
    )

    # 多对一：订单的收货地址
    receive: Mapped["ReceiveInfo"] = relationship(
        "ReceiveInfo", back_populates="order_info"
    )

    # 多对一：下单用户
    user: Mapped["UserInfo"] = relationship("UserInfo", back_populates="order_info")

    # 一对多：订单明细行集合
    order_detail: Mapped[list["OrderDetail"]] = relationship(
        "OrderDetail", back_populates="order"
    )


class OrderDetail(Base):
    """订单明细表：订单行项目——商品快照（名称/数量）与三级金额
    （总计 - 优惠 = 实付），售后单以明细行外键挂在行级。"""

    __tablename__ = "order_detail"
    __table_args__ = (
        # 外键：明细 → 订单
        ForeignKeyConstraint(
            ["order_id"], ["order_info.order_id"], name="order_detail_ibfk_2"
        ),
        # 外键：明细 → 商品
        ForeignKeyConstraint(
            ["sku_id"], ["sku_info.sku_id"], name="order_detail_ibfk_1"
        ),
        # 普通索引：按订单查明细
        Index("order_id", "order_id"),
        # 普通索引：按商品反查明细
        Index("sku_id", "sku_id"),
        {"comment": "订单明细表"},
    )

    # 明细行唯一标识（主键；售后按此定位行级售后）
    order_detail_id: Mapped[str] = mapped_column(
        String(50), primary_key=True, comment="订单明细ID"
    )

    # 所属订单 ID（不可空）
    order_id: Mapped[str] = mapped_column(String(50), nullable=False, comment="订单ID")

    # 商品 ID（不可空）
    sku_id: Mapped[str] = mapped_column(String(50), nullable=False, comment="商品ID")

    # 商品名称快照（不可空；下单时冗余存储防商品改名）
    sku_name: Mapped[str] = mapped_column(
        String(100), nullable=False, comment="商品名称"
    )

    # 购买数量（不可空）
    sku_count: Mapped[int] = mapped_column(Integer, nullable=False, comment="商品数量")

    # 总计金额，DECIMAL(10,2)（不可空）
    total_amount: Mapped[decimal.Decimal] = mapped_column(
        DECIMAL(10, 2), nullable=False, comment="总计金额"
    )

    # 实付金额，DECIMAL(10,2)（不可空）
    final_amount: Mapped[decimal.Decimal] = mapped_column(
        DECIMAL(10, 2), nullable=False, comment="实付金额"
    )

    # 优惠金额（可空；库端默认 '0.00'）
    discount_amount: Mapped[Optional[decimal.Decimal]] = mapped_column(
        DECIMAL(10, 2), server_default=text("'0.00'"), comment="优惠金额"
    )

    # 多对一：明细所属订单
    order: Mapped["OrderInfo"] = relationship(
        "OrderInfo", back_populates="order_detail"
    )

    # 多对一：明细指向的商品
    sku: Mapped["SkuInfo"] = relationship("SkuInfo", back_populates="order_detail")

    # 一对多：该明细行的售后单集合
    postsale: Mapped[list["Postsale"]] = relationship(
        "Postsale", back_populates="order_detail"
    )


# 订单与物流关联表：多对多中间表（订单 ID + 物流 ID 联合主键），
# 以显式 Table 声明（无 ORM 类，供 relationship secondary 引用）
t_order_logistics = Table(
    "order_logistics",
    Base.metadata,
    # 联合主键之一：订单 ID
    Column("order_id", String(50), primary_key=True, comment="订单ID"),
    # 联合主键之二：物流 ID
    Column("logistics_id", String(50), primary_key=True, comment="物流ID"),
    # 外键：中间表 → 物流表
    ForeignKeyConstraint(
        ["logistics_id"], ["logistics.logistics_id"], name="order_logistics_ibfk_2"
    ),
    # 外键：中间表 → 订单表
    ForeignKeyConstraint(
        ["order_id"], ["order_info.order_id"], name="order_logistics_ibfk_1"
    ),
    # 普通索引：按物流反查订单
    Index("logistics_id", "logistics_id"),
    comment="订单与物流关联表",
)


class Postsale(Base):
    """售后表：行级售后单——挂在订单明细行上，携带售后类型（退款/
    退货/换货）、退款金额与状态；经 postsale_logistics 关联表与物流
    多对多（换货发货段亦记于此）。"""

    __tablename__ = "postsale"
    __table_args__ = (
        # 外键：售后 → 订单明细行
        ForeignKeyConstraint(
            ["order_detail_id"],
            ["order_detail.order_detail_id"],
            name="postsale_ibfk_2",
        ),
        # 外键：售后 → 售后状态字典
        ForeignKeyConstraint(
            ["postsale_status"],
            ["postsale_status.postsale_status"],
            name="postsale_ibfk_3",
        ),
        # 外键：售后 → 收货地址（退货寄回/换货收货地址）
        ForeignKeyConstraint(
            ["receive_id"], ["receive_info.receive_id"], name="postsale_ibfk_1"
        ),
        # 普通索引：按明细行查售后
        Index("order_detail_id", "order_detail_id"),
        # 普通索引：按状态筛选售后
        Index("postsale_status", "postsale_status"),
        # 普通索引：按地址反查售后
        Index("receive_id", "receive_id"),
        {"comment": "售后表"},
    )

    # 售后单唯一标识（主键）
    postsale_id: Mapped[str] = mapped_column(
        String(50), primary_key=True, comment="售后ID"
    )

    # 售后创建时间（不可空；"最新一条售后"判定基准）
    create_time: Mapped[datetime.datetime] = mapped_column(
        TIMESTAMP, nullable=False, comment="创建时间"
    )

    # 关联的订单明细行 ID（不可空）
    order_detail_id: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="订单明细ID"
    )

    # 售后原因文本（不可空）
    postsale_reason: Mapped[str] = mapped_column(
        String(500), nullable=False, comment="售后原因"
    )

    # 售后状态名（不可空）
    postsale_status: Mapped[str] = mapped_column(
        String(20), nullable=False, comment="售后状态"
    )

    # 收货地址 ID（不可空）
    receive_id: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="收货信息ID"
    )

    # 售后完成时间（可空：进行中为 NULL）
    complete_time: Mapped[Optional[datetime.datetime]] = mapped_column(
        TIMESTAMP, comment="完成时间"
    )

    # 退款金额（可空；库端默认 '0.00'）
    refund_amount: Mapped[Optional[decimal.Decimal]] = mapped_column(
        DECIMAL(10, 2), server_default=text("'0.00'"), comment="退款金额"
    )

    # 售后类型枚举：退款 / 退货 / 换货（可空）
    postsale_type: Mapped[Optional[str]] = mapped_column(
        Enum("退款", "退货", "换货"), comment="售后类型"
    )

    # 多对多：售后关联的物流段集合（经 postsale_logistics）
    logistics: Mapped[list["Logistics"]] = relationship(
        "Logistics", secondary="postsale_logistics", back_populates="postsale"
    )

    # 多对一：售后挂靠的订单明细行
    order_detail: Mapped["OrderDetail"] = relationship(
        "OrderDetail", back_populates="postsale"
    )

    # 多对一：售后状态字典行（尾下划线规避与列名同名冲突）
    postsale_status_: Mapped["PostsaleStatus"] = relationship(
        "PostsaleStatus", back_populates="postsale"
    )

    # 多对一：售后使用的收货地址
    receive: Mapped["ReceiveInfo"] = relationship(
        "ReceiveInfo", back_populates="postsale"
    )


# 售后与物流关联表：多对多中间表（售后 ID + 物流 ID 联合主键），
# 以显式 Table 声明（无 ORM 类，供 relationship secondary 引用）
t_postsale_logistics = Table(
    "postsale_logistics",
    Base.metadata,
    # 联合主键之一：售后 ID
    Column("postsale_id", String(50), primary_key=True, comment="售后ID"),
    # 联合主键之二：物流 ID
    Column("logistics_id", String(50), primary_key=True, comment="物流ID"),
    # 外键：中间表 → 物流表
    ForeignKeyConstraint(
        ["logistics_id"], ["logistics.logistics_id"], name="postsale_logistics_ibfk_2"
    ),
    # 外键：中间表 → 售后表
    ForeignKeyConstraint(
        ["postsale_id"], ["postsale.postsale_id"], name="postsale_logistics_ibfk_1"
    ),
    # 普通索引：按物流反查售后
    Index("logistics_id", "logistics_id"),
    comment="售后与物流关联表",
)
