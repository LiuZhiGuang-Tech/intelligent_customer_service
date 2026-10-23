"""
gen_data.py - 电商库（ecs）测试数据生成脚本（离线数据准备工具）

核心功能：
    基于 db_table_class.py 的 ORM 表类，向 MySQL ecs 库批量生成仿真业务数据：
    收货信息、订单（含状态机推演的支付/物流/售后时间线）、物流轨迹、售后单。
    产物供订单/物流/售后三个业务动作（actions/action_*.py）联调使用。

执行流程（__main__ 入口）：
    1. clear_tables(True)：按外键依赖顺序（子表在前）清空先前生成的数据；
    2. import_receive_info(30)：导入 30 条收货信息（随机地区 + faker 仿真）；
    3. import_order_info(200)：为拥有收货信息的用户生成 200 笔订单，
       按订单状态逐级推演支付时间/物流/售后，级联生成明细与售后单。

副作用：
    - import 期：模块级创建 MySQL 连接引擎、faker 实例，并从 ecs 库加载
      全量地区表 regions 常驻内存（后续所有地址/轨迹生成都从其中随机取样）；
    - 运行期：破坏性写库——先清空 8 张业务表再批量插入，仅触碰 ecs 库；
    - 本脚本不在服务运行链路上（不随 Agent 启动被导入），独立入口在 __main__；
      模块导入期先加载仓库根 .env，连接密码经环境变量 MYSQL_PASSWORD 注入，
      与 actions/db.py 的凭据方案一致（禁止硬编码）。

主要依赖：
    - sqlalchemy（引擎与 ORM 查询）、pymysql（MySQL 驱动）
    - faker（zh_CN 中文仿真数据：姓名/电话/街道地址）
    - app.ics_ecs.actions.db_table_class（ORM 表类与关联表，星号导入）
"""

# db_table_class：全量 ORM 表类与关联表（UserInfo/OrderInfo/Logistics 等），
# 星号导入使后续生成函数可直接按类名引用
from app.ics_ecs.actions.db_table_class import *

# random：随机取样/加权选择，驱动各实体属性的仿真分布
import random

# os：读取环境变量 MYSQL_PASSWORD（敏感凭据经 .env 注入，与 actions/db.py 方案一致）
import os

# Path：定位仓库根 .env 的路径（密码注入源）
from pathlib import Path

# uuid4：为各实体主键生成 "前缀 + 16位hex" 形式的唯一ID
from uuid import uuid4

# dotenv：加载仓库根 .env（模块导入期注入 MYSQL_PASSWORD 等敏感配置）
import dotenv

# Faker：中文仿真数据生成器（姓名/电话/街道地址）
from faker import Faker

# datetime/timedelta：订单与物流时间线的推演基准
from datetime import datetime, timedelta

# overload：为 gen_new_time 声明重载签名（None 入出 None，datetime 入出 datetime），
# 使调用链无需逐处判空即可通过静态检查
from typing import overload

# create_engine：创建 MySQL 连接引擎；or_：组装售后原因的 OR 过滤条件
from sqlalchemy import create_engine, or_

# Session：ORM 会话（短生命周期，随 with 块开关）；joinedload：预加载关联收货信息
from sqlalchemy.orm import Session, joinedload

# 创建数据库引擎
# 定位仓库根 .env：本文件位于 src/app/ics_ecs 下，包目录（parent）向上三级
# （ics_ecs → app → src）即仓库根，.env 固定位于此处
repo_env = Path(__file__).resolve().parent.parents[2] / ".env"
# 密码读取发生在模块导入期，.env 加载必须先于下方 os.environ.get 执行：
# 优先仓库根的确定位置（cwd 无关），缺失时交由 dotenv 默认搜索兜底
if repo_env.exists():
    # 显式按路径加载（确定性优先，键值写入 os.environ）
    dotenv.load_dotenv(repo_env)
else:
    # 兜底：dotenv 从当前目录向上递归搜索 .env
    dotenv.load_dotenv()
# 本地开发库连接四要素（密码经 MYSQL_PASSWORD 环境变量注入，见文件头副作用说明）
db_host = "localhost"
# MySQL 默认端口
db_port = 3306
# 目标业务库名（电商库）
db_name = "ecs"
# 连接用户名
db_user_name = "root"
# 连接密码：经环境变量 MYSQL_PASSWORD 注入（仓库根 .env 提供，禁止硬编码）
db_password = os.environ.get("MYSQL_PASSWORD", "")
# 拼接 SQLAlchemy 连接 URL：mysql+pymysql 驱动 + utf8 字符集
url = f"mysql+pymysql://{db_user_name}:{db_password}@{db_host}:{db_port}/{db_name}?charset=utf8"
# 模块级创建引擎：连接池随模块常驻，供所有生成函数共用
engine = create_engine(url)

# 中文 Locale 的 faker 实例：生成姓名/电话/街道地址等仿真字段
fake = Faker("zh_CN")

# 模块级预载地区表：一次性加载全量 Region 记录常驻内存，
# 后续收货地址与物流轨迹的省市区取样全部从该列表随机选取
with Session(engine) as session:
    # 查询全量地区记录（省/市/区三元组）
    regions = session.query(Region).all()


@overload
def gen_new_time(this_time: None, delta_seconds: int) -> None: ...


@overload
def gen_new_time(this_time: datetime, delta_seconds: int) -> datetime: ...


def gen_new_time(this_time: datetime | None, delta_seconds: int) -> datetime | None:
    """生成处于[this_time, this_time+delta_seconds)，且不超过当前时间的新时间

    核心功能：在基准时间与上限窗口间随机取一时刻，保证事件时序递增且不越过现在。
    重载签名：None 入出 None，datetime 入出 datetime——时间链推演中调用方无需判空。

    业务流程：1. 基准为 None 则原样返回 None（调用方无前置时间时跳过推演）；
              2. 窗口上限收紧为 min(delta_seconds, 距当前剩余秒数)，防止生成未来时间；
              3. 在 [0, 收紧后的窗口] 内随机取秒数偏移并返回新时刻。

    副作用：无（纯函数，不触碰全局 state 与数据库）。

    参数：
        this_time: 基准时间，datetime；允许 None（视为无前置事件，直接透传）。
        delta_seconds: 随机窗口上限，单位秒，正整数。

    返回：新时刻 datetime；基准为 None 时返回 None。
    """
    # 空值兜底：无前置时间时无法推演，原样透传 None
    if this_time is None:
        return None
    # 判定逻辑：窗口上限与"距当前剩余秒数"取较小值——事件链推演到当前时刻为止，
    # 避免生成晚于 now 的未来时间（如刚创建的订单不会有未来的支付时间）
    delta_seconds = int(
        min(delta_seconds, (datetime.now() - this_time).total_seconds())
    )
    # 在 [0, delta_seconds] 内随机取偏移秒数，叠加到基准时间得到新时刻
    return this_time + timedelta(seconds=random.randint(0, delta_seconds))


def import_receive_info(nums: int = 50):
    """导入收货信息

    核心功能：批量生成收货信息记录（随机地区 + faker 仿真收件人）。

    业务流程：1. nums 为 0 时直接返回（支持按需跳过）；
              2. 从地区表随机取 nums 条省市区；
              3. 开会话查询全量用户，逐条构造 ReceiveInfo（ID 前缀 rec）；
              4. add_all + commit 批量入库。

    副作用：向 ecs 库 receive_info 表插入 nums 行。

    参数：
        nums: 生成条数，整数，缺省 50；传 0 表示不生成。

    返回：无返回值（None）。
    """
    # 数量为 0 时跳过：无收货信息可供后续订单引用
    if nums == 0:
        return
    # 从全量地区中随机放回取样 nums 条（省/市/区与收货信息一一对应）
    region = random.choices(regions, k=nums)
    # 开短会话：查询用户列表（仅读，with 块结束自动关闭）
    with Session(engine) as session:
        # 查询全量用户信息，供逐条随机挂接收货信息
        user_infos = session.query(UserInfo).all()
        # 列表推导批量构造：每条收货信息 = 唯一ID + 随机用户 + faker 仿真收件人 + 对应地区
        receive_infos = [
            ReceiveInfo(
                # 收货信息ID："rec" 前缀 + 16 位 uuid hex，保证全局唯一
                receive_id="rec" + uuid4().hex[:16],
                # 归属用户：从全量用户中随机选取（一用户可有多条收货信息）
                user_id=random.choice(user_infos).user_id,
                # 收件人姓名：faker 中文仿真
                receiver_name=fake.name(),
                # 收件人电话：faker 中文手机号
                receiver_phone=fake.phone_number(),
                # 省市区：取自本条对应的随机地区记录
                receive_province=region[i].province,
                # 市
                receive_city=region[i].city,
                # 区
                receive_district=region[i].district,
                # 街道门牌：faker 仿真，与省市区拼成完整地址
                receive_street_address=fake.street_address(),
            )
            # 按取样下标逐一生成 nums 条
            for i in range(nums)
        ]
        # 批量加入会话
        session.add_all(receive_infos)
        # 提交事务落库
        session.commit()


def import_order_info(nums: int = 50):
    """导入订单信息

    核心功能：为拥有收货信息的用户批量生成订单（含级联的明细/物流/售后）。

    业务流程：1. nums 为 0 时直接返回；
              2. 查询"至少有一条收货信息"的用户（joinedload 预加载关联避免懒加载开销）；
              3. 无合格用户则抛异常终止（订单必须挂真实收货信息）；
              4. 逐单调 gen_order_info 推演完整状态机并生成订单对象；
              5. add_all + commit 批量入库（级联带出明细/物流/售后）。

    副作用：向 ecs 库写入订单及其级联数据（order_info/order_detail/
            logistics/t_order_logistics/postsale 等多表）。

    参数：
        nums: 生成订单数，整数，缺省 50；传 0 表示不生成。

    返回：无返回值（None）；缺少合格用户时抛 Exception 不返回。
    """
    # 数量为 0 时跳过：不生成任何订单
    if nums == 0:
        return
    # 开短会话查询合格用户：仅取"有收货信息"的用户（any() 存在性过滤），
    # joinedload 预加载收货信息集合，供 gen_order_info 内直接随机选用
    with Session(engine) as session:
        user_infos = (
            session.query(UserInfo)
            .options(joinedload(UserInfo.receive_info))
            .filter(UserInfo.receive_info.any())
            .all()
        )
    # 空值兜底：没有任何带收货信息的用户则订单无法挂地址，直接终止造数
    if not user_infos:
        raise Exception("缺少收货信息")
    # 列表推导：在会话外逐单调 gen_order_info（其内部自行开关会话查询状态字典）
    order_infos = [gen_order_info(random.choice(user_infos)) for _ in range(nums)]
    # 重新开会话：批量写入订单及其级联对象（明细/物流/售后）
    with Session(engine) as session:
        # 批量加入会话（级联关系由 ORM relationship 自动带出子对象）
        session.add_all(order_infos)
        # 提交事务落库
        session.commit()


def gen_logistics_tracking(
    create_time: datetime,
    shipping_info: ReceiveInfo,
    receive_info: ReceiveInfo,
    delivered: bool,
):
    """生成物流轨迹

    核心功能：按"揽收→若干中转→到达派送→签收"的标准链路生成轨迹文本与签收时间。

    业务流程：1. 揽收节点=发货地（shipping_info 的省市区），时间在基准后 6 小时窗口内；
              2. 随机 1-3 轮中转：每轮"离开旧地发往新地"+"到达新地"两条轨迹；
              3. 末段离开中转地发往收货地（省+市），再补区县到达与派送（附派件员信息）；
              4. 签收时间在派送后 3 小时窗口内；
              5. delivered=True 返回完整轨迹+签收时间；否则随机截断（未签收）返回 None。

    副作用：无写库（纯生成；写库由调用方 gen_logistics 的返回值携带）。

    参数：
        create_time: 物流创建（揽收推演）基准时间，datetime，不为 None。
        shipping_info: 发货方收货信息对象（提供发货地省市区），不为 None。
        receive_info: 收货方收货信息对象（提供收货地省市区与街道），不为 None。
        delivered: 是否已签收，布尔；True 生成全量轨迹，False 截断。

    返回：二元组 (轨迹文本 str（多条以换行连接）, 签收时间 datetime | None)；
          未签收时签收时间为 None。
    """
    # 轨迹行列表：每条为"时间 + 地点 + 动作"拼接的文本（显式注解元素类型）
    tracking: list[str] = []
    # 揽收时间：在基准时间后 6 小时窗口内随机
    this_time = gen_new_time(create_time, 6 * 60 * 60)
    # 当前所在地初始化为发货地省市区（揽收起点）
    this_address = (
        shipping_info.receive_province
        + shipping_info.receive_city
        + shipping_info.receive_district
    )
    # 追加揽收轨迹行："时间 发货地 已揽收"
    tracking.append(
        this_time.strftime("%Y-%m-%d %H:%M:%S") + " " + this_address + " " + "已揽收"
    )
    # 中转段：随机 1-3 轮，模拟包裹在枢纽间的流转
    for _ in range(random.randint(1, 3)):
        # 中转出发时间：上一节点后 1 小时窗口内
        this_time = gen_new_time(this_time, 1 * 60 * 60)
        # 记录离开前的所在地，供"离开…发往…"文案引用
        pre_address = this_address
        # 新中转地：随机选一个地区，取其省+市（中转枢纽粒度到市）
        region = random.choice(regions)
        this_address = region.province + region.city
        # 追加"离开旧地，发往新地"轨迹行
        tracking.append(
            this_time.strftime("%Y-%m-%d %H:%M:%S")
            + " 离开 "
            + pre_address
            + "，发往 "
            + this_address
        )
        # 到达时间：出发后 6 小时窗口内
        this_time = gen_new_time(this_time, 6 * 60 * 60)
        # 追加"到达新地"轨迹行
        tracking.append(
            this_time.strftime("%Y-%m-%d %H:%M:%S") + " 到达 " + this_address
        )
    # 末段出发：离开最后一个中转地，时间在其后 1 小时窗口内
    this_time = gen_new_time(this_time, 1 * 60 * 60)
    # 记录离开地供文案引用
    pre_address = this_address
    # 收货地：省+市（先到市一级）
    this_address = receive_info.receive_province + receive_info.receive_city
    # 追加"离开中转地，发往收货市"轨迹行
    tracking.append(
        this_time.strftime("%Y-%m-%d %H:%M:%S")
        + " 离开 "
        + pre_address
        + "，发往 "
        + this_address
    )
    # 到达收货市的时间：出发后 6 小时窗口内
    this_time = gen_new_time(this_time, 6 * 60 * 60)
    # 地址细化到区县
    this_address += receive_info.receive_district
    # 追加"到达收货区"轨迹行
    tracking.append(this_time.strftime("%Y-%m-%d %H:%M:%S") + " 到达 " + this_address)
    # 派送时间：到达后 3 小时窗口内
    this_time = gen_new_time(this_time, 3 * 60 * 60)
    # 地址进一步细化到街道门牌（派送末端粒度）
    this_address += receive_info.receive_street_address
    # 追加派送轨迹行："时间 详细地址 派件员姓名电话 派送中"
    tracking.append(
        this_time.strftime("%Y-%m-%d %H:%M:%S")
        + " "
        + this_address
        + " "
        + fake.name()
        + fake.phone_number()
        + " 派送中"
    )
    # 签收时间：派送后 3 小时窗口内
    delivered_time = gen_new_time(this_time, 3 * 60 * 60)
    # 追加签收轨迹行
    tracking.append(
        delivered_time.strftime("%Y-%m-%d %H:%M:%S") + " " + this_address + " 已签收"
    )
    # 分支判定：已签收返回全量轨迹+签收时间
    if delivered:
        # 多条轨迹以换行连接为单段文本，随签收时间一并返回
        return "\n".join(tracking), delivered_time
    else:
        # 未签收：随机截断到揽收后 1 ~ 末行前一行之间（不含签收行），签收时间为 None
        return "\n".join(tracking[0 : random.randint(1, len(tracking) - 1)]), None


def gen_logistics(
    create_time: datetime,
    receive_info: ReceiveInfo,
    logistics_category: str | None,
    delivered: bool,
) -> Logistics:
    """生成物流信息

    核心功能：构造 Logistics 记录——先合成虚拟发货地，再委托轨迹生成并包装。

    业务流程：1. 校验物流类别必须属于 [None, 退货, 换货退货, 换货发货]，非法则抛 ValueError；
              2. 随机生成一个"虚拟发货地"收货信息对象（无主键/用户/收件人，仅地址有效）；
              3. 揽收时间在基准后 6 小时窗口内随机；
              4. 退货/换货退货类别下调换发货与收货方向（用户寄回商家）；
              5. 委托 gen_logistics_tracking 生成轨迹文本与签收时间；
              6. 包装为 Logistics 对象返回（ID 前缀 lgt）。

    副作用：无写库（返回对象由调用方挂到订单/售后上统一落库）。

    参数：
        create_time: 物流推演基准时间，datetime，不为 None。
        receive_info: 收货方向的真实收货信息对象，不为 None。
        logistics_category: 物流类别，仅允许 [None, "退货", "换货退货", "换货发货"]。
        delivered: 是否已签收，布尔，透传给轨迹生成。

    返回：Logistics 对象（含轨迹文本与 delivered_time）；
          类别非法时抛 ValueError 不返回。
    """
    # 防御校验：类别必须属于白名单，防止误传产生语义混乱的物流单
    if logistics_category not in [None, "退货", "换货退货", "换货发货"]:
        # 非法类别直接抛错，尽早暴露调用方逻辑错误
        raise ValueError(
            "logistics_category must in [None, '退货', '换货退货','换货发货']"
        )
    # 随机取一个地区作为虚拟发货地（商家仓库位置仿真）
    region = random.choice(regions)
    # 发货信息：构造仅含地址的 ReceiveInfo（receive_id/user/收件人均置 None，
    # 表示这是商家侧虚拟发货地而非用户真实收货信息）
    shipping_info = ReceiveInfo(
        # 无主键：不入库的临时对象
        receive_id=None,
        # 无归属用户
        user_id=None,
        # 无收件人
        receiver_name=None,
        # 无电话
        receiver_phone=None,
        # 发货地省
        receive_province=region.province,
        # 发货地市
        receive_city=region.city,
        # 发货地区
        receive_district=region.district,
        # 发货地街道：faker 仿真
        receive_street_address=fake.street_address(),
    )
    # 揽收时间：订单支付后 6 小时窗口内随机（发货有一定延迟）
    create_time = gen_new_time(create_time, 6 * 60 * 60)
    # 分支判定：退货/换货退货时包裹方向反转——用户是发货方、商家仓是收货方
    if logistics_category in ["退货", "换货退货"]:
        # 调换发货信息和收货信息
        shipping_info, receive_info = receive_info, shipping_info
    # 委托轨迹生成：返回轨迹文本与签收时间（未签收时为 None）
    logistics_tracking, delivered_time = gen_logistics_tracking(
        create_time, shipping_info, receive_info, delivered
    )
    # 包装为 Logistics 对象（主键 "lgt" 前缀 + 16 位 hex）
    return Logistics(
        # 物流单ID
        logistics_id="lgt" + uuid4().hex[:16],
        # 创建（揽收）时间
        create_time=create_time,
        # 签收时间：未签收为 None
        delivered_time=delivered_time,
        # 完整轨迹文本（换行分隔）
        logistics_tracking=logistics_tracking,
        # 物流类别（正向单为 None）
        logistics_category=logistics_category,
    )


def gen_postsale(
    create_time: datetime,
    order_detail: OrderDetail,
    receive_info: ReceiveInfo,
    postsale_type: str,
    postsale_status: PostsaleStatus,
) -> Postsale:
    """订单明细关联售后信息

    核心功能：为指定订单明细生成售后单，按售后状态码分支推演完成时间与退/换货物流。

    业务流程：1. 申请时间在基准后 6 小时窗口内随机；
              2. 查询明细对应 SKU 的分类，再取该分类（或通用）的售后原因池随机选用；
              3. 构造 Postsale（ID 前缀 pts，退款金额默认取明细实付额）；
              4. 按状态码分支：≤420 处理中前置态不推演；900-920 完成态仅补完成时间；
                 其余按类型生成退货/换货物流并推演完成时间；
              5. 把售后单挂到订单明细的级联集合上并返回。

    副作用：读库（SKU/售后原因查询）；生成的对象随调用方的订单级联统一落库。

    参数：
        create_time: 推演基准时间（通常为订单签收时间），datetime。
        order_detail: 目标订单明细对象（提供明细ID/SKU/实付额），不为 None。
        receive_info: 用户收货信息对象（退换货地址），不为 None。
        postsale_type: 售后类型字符串，取值 "退款"/"退货"/"换货"。
        postsale_status: 售后状态字典记录对象（提供状态码与状态名），不为 None。

    返回：Postsale 对象（已挂到 order_detail.postsale 级联）。
    """
    # 状态码守卫：PostsaleStatus.status_code 为可空列（Mapped[Optional[int]]），
    # 数值比较前先收窄为局部变量，避免 None 参与比较抛 TypeError
    status_code = postsale_status.status_code
    if status_code is None:
        raise ValueError("售后状态字典行缺少状态码，无法推演售后分支")
    # 售后申请时间：基准（签收）后 6 小时窗口内随机
    create_time = gen_new_time(create_time, 6 * 60 * 60)
    # 开短会话查询售后原因池：订单明细->商品信息->商品分类->售后原因
    with Session(engine) as session:
        # 查询明细对应 SKU 的商品信息（取其分类）
        sku_info = (
            session.query(SkuInfo).filter(SkuInfo.sku_id == order_detail.sku_id).first()
        )
        # 空值守卫：first() 在 sku_id 于库中不存在时返回 None；正常流程 sku_id
        # 来自刚随机选中的 SKU 必命中，查不到说明库被外部改动，显式报错终止
        if sku_info is None:
            raise ValueError(f"SKU {order_detail.sku_id} 不存在，无法生成售后")
        # SKU 所属商品分类（售后原因按分类过滤；守卫后 sku_info 已收窄为非 None）
        product_category = sku_info.sku_category
        # 售后原因池：匹配本分类的原因 + 分类为 None 的通用原因
        postsale_reasons = (
            session.query(PostsaleReason)
            .filter(
                or_(
                    # 分类精确匹配的原因
                    PostsaleReason.product_category == product_category,
                    # 通用原因（不限分类）
                    PostsaleReason.product_category.is_(None),
                )
            )
            .all()
        )
    # 生成售后信息：构造 Postsale 主体（状态码细分推演见下方分支）
    postsale = Postsale(
        # 售后单ID："pts" 前缀 + 16 位 hex
        postsale_id="pts" + uuid4().hex[:16],
        # 申请时间
        create_time=create_time,
        # 关联的订单明细ID
        order_detail_id=order_detail.order_detail_id,
        # 售后原因：从原因池随机选用
        postsale_reason=random.choice(postsale_reasons).postsale_reason,
        # 售后状态：取状态字典记录的状态名
        postsale_status=postsale_status.postsale_status,
        # 完成时间：初始置 None，按状态分支回填
        complete_time=None,
        # 退款金额：默认取明细实付额（换货分支会置 None）
        refund_amount=order_detail.final_amount,
        # 售后类型：退款/退货/换货
        postsale_type=postsale_type,
        # 关联收货信息ID（退换货地址）
        receive_id=receive_info.receive_id,
    )
    # 分支判定：状态码 ≤420 属处理中的前置态（审核/沟通阶段），无物流无完成时间
    if status_code <= 420:
        # 直接保持初始态返回
        pass
    # 分支判定：900-920 为纯退款/无需物流的完结态，仅补完成时间
    elif status_code >= 900 and status_code <= 920:
        # 生成完成时间：申请后 24 小时窗口内随机
        postsale.complete_time = gen_new_time(create_time, 24 * 60 * 60)
    # 分支判定：退货类型——生成退货物流（用户寄回商家），完成时间在签收后退回
    elif postsale_type == "退货":
        # 生成完成时间和退货物流信息：方向反转为退货；是否签收取决于状态码是否≥900
        logistics = gen_logistics(create_time, receive_info, "退货", status_code >= 900)
        # 退货物流挂到售后单级联集合（经 t_postsale_logistics 关联表落库）
        postsale.logistics.append(logistics)
        # 完成时间：退货签收后 12 小时窗口内随机；未签收时 delivered_time 为 None → 完成时间 None
        postsale.complete_time = gen_new_time(logistics.delivered_time, 12 * 60 * 60)
    # 分支判定：换货类型——先退回旧货，可能再发新货；无退款金额
    elif postsale_type == "换货":
        # 换货不退款：退款金额置 None
        postsale.refund_amount = None
        # 生成完成时间和换货物流信息：换货退货方向；签收与否随机，
        # 或状态码≥900（完结态强制签收），或状态名为"换发货"（已进入补发阶段必然签收）
        logistics = gen_logistics(
            create_time,
            receive_info,
            "换货退货",
            random.choice([True, False])
            or status_code >= 900
            or postsale_status.postsale_status == "换发货",
        )
        # 换货退货物流挂到售后单级联
        postsale.logistics.append(logistics)
        # 分支判定：退货段已签收 且 状态未停留在"换退货"时，继续推演新货发出段
        if (
            logistics.delivered_time is not None
            and postsale_status.postsale_status != "换退货"
        ):
            # 生成换货发货物流：以退货签收时间为基准，方向恢复为商家发用户
            logistics = gen_logistics(
                logistics.delivered_time,
                receive_info,
                "换货发货",
                status_code >= 900,
            )
            # 新货物流追加到售后单级联
            postsale.logistics.append(logistics)
            # 完成时间：新货签收后 6 小时窗口内随机
            postsale.complete_time = gen_new_time(logistics.delivered_time, 6 * 60 * 60)
    # 售后单挂到订单明细的级联集合（随订单统一落库）
    order_detail.postsale.append(postsale)
    # 返回售后单对象
    return postsale


def gen_order_detail(order_id: str) -> OrderDetail:
    """生成不包含售后信息的订单明细信息

    核心功能：随机选品并生成单条订单明细（数量加权 + 三段金额推演）。

    业务流程：1. 随机选取一条 SKU 商品信息；
              2. 加权随机购买数量（1 件占 70%，2-6 件递减）；
              3. 金额推演：总额=单价×数量，优惠额在 [0, 总额] 均匀随机，
                 实付额=总额-优惠额；
              4. 构造 OrderDetail（ID 前缀 ordd）返回。

    副作用：读库（随机选品查询）。

    参数：
        order_id: 所属订单ID，字符串，不为 None。

    返回：OrderDetail 对象（不含售后级联，售后由 gen_postsale 追加）。
    """
    # 开短会话随机选品：全量 SKU 中随机一条（仅读）
    with Session(engine) as session:
        # 随机选取商品信息
        sku_info = random.choice(session.query(SkuInfo).all())
    # 选择商品数量：加权随机——1 件权重 70%，2/3/4/5/6 件各 10/5/5/5/5，
    # choices 返回列表故取 [0]
    sku_count = random.choices([1, 2, 3, 4, 5, 6], weights=[70, 10, 5, 5, 5, 5])[0]
    # 订单总额：单价 × 购买数量
    total_amount = sku_info.sku_price * sku_count
    # 优惠金额：在 [0, 总额] 区间均匀随机（转 float 保证与 Decimal 运算兼容）
    discount_amount = random.uniform(0, float(total_amount))
    # 实付金额：总额减去优惠
    final_amount = float(total_amount) - discount_amount
    # 构造订单明细对象
    order_detail = OrderDetail(
        # 明细ID："ordd" 前缀 + 16 位 hex
        order_detail_id="ordd" + uuid4().hex[:16],
        # 所属订单ID
        order_id=order_id,
        # 商品ID
        sku_id=sku_info.sku_id,
        # 商品名称快照（下单时刻冗余）
        sku_name=sku_info.sku_name,
        # 购买数量
        sku_count=sku_count,
        # 订单总额
        total_amount=total_amount,
        # 实付金额
        final_amount=final_amount,
        # 优惠金额
        discount_amount=discount_amount,
    )
    # 返回明细对象（售后信息由调用方按需追加）
    return order_detail


def gen_order_info(user_info: UserInfo) -> OrderInfo:
    """生成订单信息

    核心功能：为指定用户推演一笔完整订单——状态机驱动的支付/物流/售后时间线。

    业务流程：1. 随机选用用户的一条收货信息，随机取订单状态字典；
              2. 创建时间定在 2-14 天前，构造 OrderInfo（ID 前缀 ord）；
              3. 加权生成 1-3 条明细；按状态分支：待支付直接返回；
                 已取消补完成时间（可能支付前取消，交换支付/完成时间语义）；
                 待发货直接返回；
              4. 其余状态生成物流（状态码≥330 判定已签收），回填签收时间；
              5. 状态码≥400（已完成/售后中）按 20/80 加权决定是否产生售后：
                 已完成取状态码≥900 的完结态，售后中取≤499 的处理态；
              6. 售后单完成时间取各售后最大值作为订单完成时间。

    副作用：读库（状态字典查询）；返回对象由 import_order_info 统一落库。

    参数：
        user_info: 目标用户对象（需已预加载 receive_info 集合），不为 None。

    返回：OrderInfo 对象（级联明细/物流/售后）。
    """
    # 选择收货信息：从用户已预加载的收货信息集合随机取一条
    receive_info = random.choice(user_info.receive_info)
    # 选择订单状态：从状态字典表随机一条（含状态码与状态名）
    with Session(engine) as session:
        order_status = random.choice(session.query(OrderStatus).all())
    # 状态码守卫：OrderStatus.status_code 为可空列（Mapped[Optional[int]]），
    # 数值比较前先收窄为局部变量，避免 None 参与比较抛 TypeError
    status_code = order_status.status_code
    if status_code is None:
        raise ValueError("订单状态字典行缺少状态码，无法推演订单时间线")
    # 创建时间：当前时间2-14天之前（保证后续时间推演有窗口）
    create_time = datetime.now() - timedelta(
        seconds=random.randint(2 * 24 * 60 * 60, 14 * 24 * 60 * 60)
    )
    # 生成订单信息：主体字段先置空，时间线按状态分支回填
    order_info = OrderInfo(
        # 订单ID："ord" 前缀 + 16 位 hex
        order_id="ord" + uuid4().hex[:16],
        # 创建时间
        create_time=create_time,
        # 归属用户ID
        user_id=user_info.user_id,
        # 收货信息ID
        receive_id=receive_info.receive_id,
        # 订单状态名
        order_status=order_status.order_status,
        # 支付时间：初始 None，按状态回填
        payment_time=None,
        # 发货（签收）时间：初始 None，有物流时回填
        delivered_time=None,
        # 完成时间：初始 None，按状态回填
        complete_time=None,
    )
    # 生成订单明细信息：加权随机 1-3 条（1 条占 80%，2 条 18%，3 条 2%）
    nums = random.choices([1, 2, 3], weights=[80, 18, 2])[0]
    # 逐条生成明细（不含售后）
    order_details = [gen_order_detail(order_info.order_id) for _ in range(nums)]
    # 明细挂到订单级联集合
    order_info.order_detail = order_details
    # 根据状态生成并调整其他信息
    # 分支判定：待支付——订单刚创建，无任何后续时间线
    if order_status.order_status == "待支付":
        # 直接返回初始态订单
        return order_info
    # 生成支付时间：创建后 15 分钟窗口内随机
    order_info.payment_time = gen_new_time(create_time, 15 * 60)
    # 分支判定：已取消——补完成时间，并可能表现为"支付前取消"
    if order_status.order_status == "已取消":
        # 生成完成时间：创建后 6 小时窗口内随机
        order_info.complete_time = gen_new_time(create_time, 6 * 60 * 60)
        # 可能在支付前取消：随机决定
        if random.choice([True, False]):
            # 删除支付时间：把"完成时间"语义移到支付位——支付时间为 None 表示未支付，
            # 完成时间保留（支付前取消的订单以取消动作收尾）
            order_info.complete_time = order_info.payment_time
            order_info.payment_time = None
        # 返回已取消订单
        return order_info
    # 分支判定：待发货——已支付未发货，无物流时间线
    if order_status.order_status == "待发货":
        # 返回待发货订单
        return order_info
    # 生成物流信息：状态码≥330（运输中及之后）判定已签收，否则在途截断
    delivered = True if status_code >= 330 else False
    # 支付时间收窄：模型字段声明为可空（Optional[datetime]），但本分支上文刚生成
    # 支付时间必非 None，取局部变量并守卫以满足静态检查（理论不可达）
    payment_time = order_info.payment_time
    if payment_time is None:
        raise ValueError("订单缺少支付时间，无法生成物流时间线")
    # 生成正向物流（类别 None），基准为支付时间
    logistics = gen_logistics(payment_time, receive_info, None, delivered)
    # 物流挂到订单级联集合（经 t_order_logistics 关联表落库）
    order_info.logistics = [logistics]
    # 回填发货/签收时间（未签收为 None）
    order_info.delivered_time = logistics.delivered_time
    # 分支判定：状态码<400（运输中/待收货）——尚未进入售后阶段
    if status_code < 400:
        # 返回在途订单
        return order_info
    # 是否产生售后：加权随机（20% 有售后 / 80% 无）
    have_postsale = random.choices([True, False], weights=[20, 80])[0]
    # 如果已完成且没有售后记录，不生成售后信息
    if not have_postsale and order_status.order_status != "售后中":
        # 生成完成时间：签收后 6 小时窗口内随机（"售后中"强制有售后，不适用此分支）
        order_info.complete_time = gen_new_time(order_info.delivered_time, 6 * 60 * 60)
        # 返回无售后的完结订单
        return order_info
    # 确定售后类型和售后状态：类型三选一（退款/退货/换货）
    postsale_type = random.choice(["退款", "退货", "换货"])
    # 类型→状态字典过滤条件的映射（SQLAlchemy 列比较表达式）
    type_condition = {
        # 退款类型
        "退款": PostsaleStatus.is_refund == True,
        # 退货类型
        "退货": PostsaleStatus.is_return == True,
        # 换货类型
        "换货": PostsaleStatus.is_exchange == True,
    }
    # 状态码过滤条件：已完成订单取完结态（≥900），售后中取处理态（≤499）
    status_condition = (
        PostsaleStatus.status_code >= 900
        if order_status.order_status == "已完成"
        else PostsaleStatus.status_code <= 499
    )
    # 开短会话按类型+状态码条件筛选售后状态字典并随机选用（仅读）
    with Session(engine) as session:
        postsale_status = random.choice(
            session.query(PostsaleStatus)
            .filter(
                # 类型匹配条件（is_refund/is_return/is_exchange）
                type_condition[postsale_type],
                # 状态码区间条件
                (status_condition),
            )
            .all()
        )
    # 选择若干个订单明细关联售后信息：从本单明细中随机取 1 到全部条
    order_details = random.choices(
        order_info.order_detail,
        k=random.randint(1, len(order_info.order_detail)),
    )
    # 签收时间收窄：本分支状态码≥400 ⇒ ≥330 ⇒ 必已签收 ⇒ delivered_time 必非 None
    # （模型字段声明可空故需守卫，理论不可达）
    delivered_time = order_info.delivered_time
    if delivered_time is None:
        raise ValueError("已签收订单缺少签收时间，无法推演售后时间线")
    # 逐条明细生成售后单（含退/换货物流推演）
    postsales = [
        gen_postsale(
            delivered_time,
            order_detail,
            receive_info,
            postsale_type,
            postsale_status,
        )
        for order_detail in order_details
    ]
    # 分支判定：售后中——订单未完结，完成时间保持 None
    if order_status.order_status == "售后中":
        # 返回售后进行中的订单
        return order_info
    # 生成完成时间：取全部售后单完成时间的最大值（订单以最后一笔售后完结为准）；
    # 过滤 None 仅为满足静态检查——完结态分支必已回填完成时间，运行时结果不变
    complete_times = [
        postsale.complete_time
        for postsale in postsales
        if postsale.complete_time is not None
    ]
    order_info.complete_time = max(complete_times)
    # 返回带售后的完结订单
    return order_info


def clear_tables(run: bool = False):  # pyright: ignore[reportUnusedParameter]
    """清空生成的数据

    核心功能：按外键依赖顺序（子表在前）删除 8 张业务表的全部行。

    业务流程：1. 组装待清空表列表（投诉记录/售后物流关联/订单物流关联/物流/
                 售后/明细/订单/收货信息，引用方在先避免外键冲突）；
              2. 单会话内逐表 delete + commit；
              3. 任一失败则整体回滚并打印原因。

    副作用：破坏性清空 ecs 库 8 张业务表数据。

    参数：
        run: 占位开关，布尔，缺省 False——原实现未使用该参数（无论传什么都
             执行清空），调用方以 clear_tables(True) 显式表达清库意图。

    返回：无返回值（None）。
    """
    # 待清空表列表：按外键依赖排序，先删引用方（子表）再删被引用方（父表）
    tables = [
        # 投诉记录（引用售后）
        LogisticsComplaintsRecord,
        # 售后-物流关联表
        t_postsale_logistics,
        # 订单-物流关联表
        t_order_logistics,
        # 物流单（被关联表引用）
        Logistics,
        # 售后单（被关联表/投诉引用）
        Postsale,
        # 订单明细（被售后引用）
        OrderDetail,
        # 订单（被明细/关联表引用）
        OrderInfo,
        # 收货信息（被订单引用）
        ReceiveInfo,
    ]

    # 单会话完成全部删除（保证原子性）
    with Session(engine) as session:
        try:
            # 逐表删除全部行
            for table in tables:
                # delete() 不带过滤条件即清空全表（返回受影响行数，此处无需使用，
                # 赋给 _ 标记有意忽略）
                _ = session.query(table).delete()
            # 提交事务
            session.commit()
            # 控制台确认清空成功
            print("已清空先前生成的数据")
        except Exception as e:
            # 任一表删除失败则整体回滚，避免留下半清空状态
            session.rollback()
            # 打印失败原因便于排查
            print(f"清空失败: {e}")


if __name__ == "__main__":
    # 清空先前生成的数据（显式传 True 解锁破坏性清空）
    clear_tables(True)  # 清空先前生成的数据
    # 导入收货信息（30 条，供订单随机挂载）
    import_receive_info(30)  # 导入收货信息
    # 导入订单信息（200 笔，级联生成明细/物流/售后）
    import_order_info(200)  # 导入订单信息
