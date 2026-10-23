"""
action_order - 订单业务 Action 模块

核心功能：实现电商客服订单域的动作集（共 7 个 Action 类）——统一
          订单查询（ActionAskOrderId，按 goto 槽区分六种查询场景）、
          订单详情（ActionGetOrderDetail，含金额合计/收货信息/物流
          轨迹/售后记录）、收货地址选择与省市区三级联动（ActionAsk
          ReceiveId / Province / City / District）、收货信息确认与
          修改入库（ActionAskSetReceiveInfo）、订单取消（Action
          CancelOrder）。

执行流程：1. 对话流程经 action_ask_order_id 询问订单号——run 从
          tracker 读取 user_id 与 goto 槽，_get_query_condition 把
          goto 动作名映射为 SQLAlchemy 查询条件，联表查询后按命中
          数量（0/1/多）生成回复与按钮（payload 为 /SetSlots 命令
          回填 order_id）；2. 用户点选后 order_id 落槽，action_get
          _order_detail 拉取明细/收货/物流并聚合金额，status_code
          >= 400（已进售后）时再按订单明细查各自最新售后记录；3.
          修改收货信息经 action_ask_receive_id 列出地址按钮 → 省/
          市/区三级行政区划联动选择 → action_ask_set_receive_info
          （首调展示待确认信息，确认后查重、入库并回写订单的
          receive_id）；4. action_cancel_order 将订单状态置"已取消"
          并写完成时间，待发货订单附退款提示。

副作用：每个 Action 均在 run 内延迟导入 SessionLocal 访问 MySQL
        （订单/明细/收货/区域/售后表）；写路径包括收货信息新增与
        订单收货地址回写、订单取消（session.commit）；向 tracker
        写槽位（order_id/receive_* /set_receive_info 等）；响应
        统一经 ActionResult.add_response 收集；所有数据库异常在
        方法内捕获、记日志并返回错误话术，不向上抛出。

主要依赖：app.ics_ai.agent.actions（Action 抽象基类与 ActionResult
          DTO）；app.ics_ecs.actions.db 与 db_table_class（本包数据库会
          话工厂与 ORM 表类，run 内延迟导入，避免模块导入期即连库）。
"""

# 导入 logging：各 Action 的数据库异常经本模块 logger 记录错误日志。
import logging

# 从 datetime 导入日期时间类型与时间差：查询"3日内完成"的时间边界、
# 取消订单时写入完成时间戳。
from datetime import datetime, timedelta

# 导入 Any：tracker/domain 等鸭子类型参数的通配注解。
from typing import Any

# 从动作框架导入基类与结果 DTO：Action 定义 name/run 契约，
# ActionResult 承载响应列表；本模块所有 Action 均继承前者。
from app.ics_ai.agent.actions import Action, ActionResult

# 本模块专属日志器：模块内所有异常日志统一挂在此命名空间下。
logger = logging.getLogger(__name__)


class ActionAskOrderId(Action):
    """
    统一的订单查询 Action。

    职责：按 goto 槽的值决定查询场景并检索用户订单，将结果以文本+
          按钮形式返回，供用户点选回填 order_id 槽。
    根据 goto 槽的值决定查询条件：
    - action_ask_order_id_before_completed_3_days: 进行中或3日内已完成的订单
    - action_ask_order_id_before_delivered: 已签收之前的订单（可修改收货信息）
    - action_ask_order_id_before_shipped: 待发货之前的订单（可取消）
    - action_ask_order_id_shipped: 已发货的订单（查物流）
    - action_ask_order_id_shipped_delivered: 已发货和已签收的订单（投诉物流）
    - action_ask_order_id_after_delivered: 已签收后的订单（售后）

    设计模式：模板方法——run 固定"取槽→建条件→查库→按数量分支"骨架，
              场景差异全部收敛到 _get_query_condition 的映射逻辑。

    生命周期：由动作注册表随 domain 装配创建、全流程复用；run 无
              实例状态，仅依赖入参 tracker。
    """

    @property
    def name(self) -> str:
        """动作名：domain/flow 中引用的唯一标识。"""
        # 返回动作注册名，与流程定义中的 action 引用一致。
        return "action_ask_order_id"

    async def run(
        self,
        tracker: Any,
        domain: Any | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """执行订单查询。

        核心功能：按 goto 场景检索当前用户订单，返回订单列表供点选。

        业务流程：1. 延迟导入数据库会话与 ORM 表类（含 SQLAlchemy
                  联表/预加载工具）；2. 读取 user_id（缺省演示账号
                  1001）与 goto 场景槽；3. 开启会话，将 goto 映射为
                  查询条件，联订单状态表、预加载订单明细，按创建时间
                  倒序取全部订单；4. 零单——回复"暂无订单"、置
                  order_id=false 并打断监听；5. 单单——拼详情文本+
                  确认/返回按钮；6. 多单——每单拼一个按钮，末尾追加
                  返回按钮；7. 任意异常兜底记日志并返回错误话术。

        副作用：访问 MySQL（只读查询）；向 tracker 写 order_id 槽
                （零单分支）；修改 ActionResult.responses。

        参数：
            tracker: 对话状态跟踪器，鸭子类型（需 get_slot/set_slot），
                     不允许为 None。
            domain: 当前域对象，本动作未使用，允许为 None。
            **kwargs: 框架预留扩展参数，本动作不消费。

        返回：
            ActionResult——responses 含订单文案与按钮；数据库异常时
            返回含错误话术的结果（不抛异常）。
        """
        # 延迟导入会话工厂：把数据库依赖压缩到调用时刻，模块导入期不连库。
        from app.ics_ecs.actions.db import SessionLocal

        # 延迟导入订单与订单状态 ORM 表类：查询与条件构建使用。
        from app.ics_ecs.actions.db_table_class import OrderInfo, OrderStatus

        # 延迟导入 joinedload：预加载一对多的订单明细，规避 N+1 查询。
        from sqlalchemy.orm import joinedload

        # 延迟导入 and_/or_：组合多表查询条件。
        from sqlalchemy import and_, or_

        # 创建结果 DTO：承载本次动作的全部响应。
        result = ActionResult()

        # 读取用户槽：缺失时兜底演示账号"1001"（演示环境无真实登录态）。
        user_id = tracker.get_slot("user_id") or "1001"

        # 读取 goto 场景槽：值为发起本次查询的流程动作名，决定查询场景。
        goto = tracker.get_slot("goto")

        try:
            # 进入会话上下文：退出时自动归还连接，事务随上下文管理。
            with SessionLocal() as session:
                # 根据 goto 值构建查询条件（场景差异集中在此映射）。
                query_condition = self._get_query_condition(user_id, goto)

                # 链式构建查询：
                # 1) 以订单表为主表；
                order_infos = (
                    session.query(OrderInfo)
                    # 2) 内联订单状态表（供条件中的状态码比较使用）；
                    .join(OrderInfo.order_status_)
                    # 3) 预加载订单明细关系（避免逐单再查明细）；
                    .options(joinedload(OrderInfo.order_detail))
                    # 4) 应用 goto 场景过滤条件；
                    .filter(query_condition)
                    # 5) 按创建时间倒序（最新订单在前）；
                    .order_by(OrderInfo.create_time.desc())
                    # 6) 执行查询取全部命中行。
                    .all()
                )

                # 统计命中订单数，决定后续"无/单/多"分支。
                order_nums = len(order_infos)

                # 没有订单
                if order_nums == 0:
                    # 回复无订单话术。
                    result.add_response("暂无订单")

                    # 将 order_id 槽置哨兵值 false：向流程表达"未选中任何订单"。
                    tracker.set_slot("order_id", "false")

                    # 设置打断监听标志：请求上层跳过本轮 action_listen；
                    # 字段已在 ActionResult 正式声明（上层暂未消费，属预留约定）。
                    result.reject_action_listen = True

                    # 无单即结束，不再走点选分支。
                    return result

                # 只有一个订单
                if order_nums == 1:
                    # 直接取唯一命中行。
                    order_info = order_infos[0]

                    # 拼接消息头：状态徽标 + 加粗订单 ID。
                    message = [
                        "查找到一个订单",
                        f"[{order_info.order_status}]**订单ID**：{order_info.order_id}",
                    ]

                    # 逐条追加订单明细行（商品名 × 数量）。
                    for order_detail in order_info.order_detail:
                        message.append(
                            f"- {order_detail.sku_name} × {order_detail.sku_count}"
                        )

                    # 构建点选按钮：确认回填订单 ID，返回置 false。
                    buttons = [
                        {
                            # 确认按钮：payload 回填该订单 ID 到 order_id 槽。
                            "title": "确认",
                            "payload": f"/SetSlots(order_id={order_info.order_id})",
                        },
                        # 返回按钮：order_id 置 false，交回流程重选。
                        {"title": "返回", "payload": "/SetSlots(order_id=false)"},
                    ]

                    # 输出单订单消息（多行文本合并 + 按钮组）。
                    result.add_response("\n".join(message), buttons=buttons)

                # 多个订单
                else:
                    # 初始化按钮列表：每单对应一个多行标题按钮。
                    buttons = []

                    # 遍历命中订单，逐单生成按钮。
                    for order_info in order_infos:
                        # 按钮标题首行：状态徽标 + 订单 ID。
                        title_parts = [
                            f"[{order_info.order_status}]订单ID：{order_info.order_id}",
                        ]

                        # 标题续行：该单的每条明细（商品名 × 数量）。
                        for order_detail in order_info.order_detail:
                            title_parts.append(
                                f"- {order_detail.sku_name} × {order_detail.sku_count}"
                            )

                        # 追加按钮：标题为多行文本，payload 回填订单 ID。
                        buttons.append(
                            {
                                "title": "\n".join(title_parts),
                                "payload": f"/SetSlots(order_id={order_info.order_id})",
                            }
                        )

                    # 末尾追加统一的返回按钮：置 false 交回流程。
                    buttons.append(
                        {"title": "返回", "payload": "/SetSlots(order_id=false)"}
                    )

                    # 输出选择提示与按钮组。
                    result.add_response("请选择订单", buttons=buttons)

        # 任意异常兜底：数据库连接/查询失败不让异常上抛，转为用户话术。
        except Exception as e:
            # 记录错误日志（含异常信息），便于排查。
            logger.error(f"查询订单失败: {e}")

            # 返回通用错误话术。
            result.add_response("查询订单时出错，请稍后重试。")

        # 返回结果 DTO（成功或错误话术二选一）。
        return result

    def _get_query_condition(self, user_id: str, goto: str | None):
        """根据 goto 值构建查询条件。

        核心功能：把 goto 场景动作名翻译为 SQLAlchemy 过滤条件。

        业务流程：match 匹配 goto——六种具名场景各返回一组 and_/or_
                  组合条件；未匹配（含 goto 为 None）走默认分支，仅
                  过滤"非已取消"订单。

        副作用：无（纯条件构建，不访问数据库）。

        参数：
            user_id: 用户 ID，字符串，不允许为空（调用方已兜底）。
            goto: 场景动作名或 None；None/未知值走默认分支。

        返回：
            SQLAlchemy 条件表达式（供 filter 使用）；无异常路径。
        """
        # 延迟导入订单与订单状态表类：条件构建需要列引用。
        from app.ics_ecs.actions.db_table_class import OrderInfo, OrderStatus

        # 延迟导入 and_/or_：组合条件。
        from sqlalchemy import and_, or_

        # match 按场景动作名分发（goto 为 None 时走 case _ 兜底）。
        match goto:
            case "action_ask_order_id_shipped":
                # 查询已发货的订单
                # 条件：属于该用户 且 状态恰为"已发货"。
                return and_(
                    OrderInfo.user_id == user_id,
                    OrderInfo.order_status == "已发货",
                )
            case "action_ask_order_id_shipped_delivered":
                # 查询已发货和已签收的订单
                # 条件：属于该用户 且 状态在"已发货/已签收"集合内。
                return and_(
                    OrderInfo.user_id == user_id,
                    OrderInfo.order_status.in_(["已发货", "已签收"]),
                )
            case "action_ask_order_id_before_completed_3_days":
                # 查询进行中，或3日内已完成的订单
                # 条件：属于该用户 且 未取消 且（未完成 或 3日内完成）。
                return and_(
                    OrderInfo.user_id == user_id,
                    OrderInfo.order_status != "已取消",
                    or_(
                        OrderInfo.order_status != "已完成",
                        OrderInfo.complete_time > datetime.now() - timedelta(days=3),
                    ),
                )
            case "action_ask_order_id_before_delivered":
                # 查询已签收之前状态的订单（可修改收货信息）
                # 条件：属于该用户 且 未取消 且 状态码 <= 320（签收前，
                # 借助外层 join 的状态表比较状态码）。
                return and_(
                    OrderInfo.user_id == user_id,
                    OrderInfo.order_status != "已取消",
                    OrderStatus.status_code <= 320,
                )
            case "action_ask_order_id_before_shipped":
                # 查询已发货之前状态的订单（可取消）
                # 条件：属于该用户 且 未取消 且 状态码 <= 310（发货前）。
                return and_(
                    OrderInfo.user_id == user_id,
                    OrderInfo.order_status != "已取消",
                    OrderStatus.status_code <= 310,
                )
            case "action_ask_order_id_after_delivered":
                # 查询已签收、售后中、已完成的订单（售后）
                # 条件：属于该用户 且 未取消 且 状态码 >= 330（签收后）。
                return and_(
                    OrderInfo.user_id == user_id,
                    OrderInfo.order_status != "已取消",
                    OrderStatus.status_code >= 330,
                )
            case _:
                # 默认：查询所有非取消订单
                # 兜底条件：仅按用户与未取消过滤，不限定状态段。
                return and_(
                    OrderInfo.user_id == user_id,
                    OrderInfo.order_status != "已取消",
                )


class ActionGetOrderDetail(Action):
    """获取订单详情

    职责：按 order_id 槽拉取订单全量信息——状态/时间线/明细金额合计/
          收货信息/最近物流；已进售后（status_code>=400）时追加各
          明细的最新售后记录。

    设计模式：模板方法——固定"校验槽→查库→拼装文本→售后追加"骨架。

    生命周期：随 domain 装配创建、全流程复用；run 无实例状态。
    """

    @property
    def name(self) -> str:
        """动作名：domain/flow 中引用的唯一标识。"""
        # 返回动作注册名。
        return "action_get_order_detail"

    async def run(
        self,
        tracker: Any,
        domain: Any | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """执行订单详情查询。

        核心功能：聚合订单状态、时间、明细金额、收货与物流信息并返回。

        业务流程：1. 延迟导入会话、订单/售后表类与查询工具；2. 读取
                  order_id 槽——缺失或为 false 直接回复"未找到"；
                  3. 查询订单（预加载明细/物流/收货/状态四关系），
                  查无此单同样提前返回；4. 拼状态头行与存在的时间项；
                  5. 遍历明细累加三种金额并输出合计；6. 拼收货人/
                  电话/四级地址；7. 有物流时取首条记录轨迹最后一行；
                  8. 状态码>=400 时构建子查询取各明细最新售后并逐条
                  追加响应；9. 异常兜底记日志返回错误话术。

        副作用：访问 MySQL（只读）；修改 ActionResult.responses；
                排序售后物流时在会话内改写关系属性（不提交事务）。

        参数：
            tracker: 对话状态跟踪器（需 get_slot），不允许为 None。
            domain: 当前域对象，本动作未使用，允许为 None。
            **kwargs: 框架预留扩展参数，本动作不消费。

        返回：
            ActionResult——responses 含详情文案（及售后记录）；槽
            缺失/订单不存在/异常时返回对应提示，不抛异常。
        """
        # 延迟导入会话工厂：调用时刻才连库。
        from app.ics_ecs.actions.db import SessionLocal

        # 延迟导入订单与售后表类：查询主体与售后分支使用。
        from app.ics_ecs.actions.db_table_class import OrderInfo, Postsale

        # 延迟导入 joinedload：预加载关联关系避免 N+1。
        from sqlalchemy.orm import joinedload

        # 延迟导入 and_/func：售后子查询的组合条件与聚合函数。
        from sqlalchemy import and_, func

        # 创建结果 DTO。
        result = ActionResult()

        # 读取订单 ID 槽：由上游 action_ask_order_id 点选回填。
        order_id = tracker.get_slot("order_id")

        # 槽缺失或为哨兵值 false：说明上游未选中订单，直接提示返回。
        if not order_id or order_id == "false":
            # 回复未找到话术。
            result.add_response("未找到该订单，请检查订单号是否正确。")

            # 提前返回，不进入查库分支。
            return result

        try:
            # 进入会话上下文。
            with SessionLocal() as session:
                # 链式构建查询：以订单表为主表，依次预加载
                # 明细/物流/收货/状态四组关系，按订单 ID 过滤取首条。
                order_info = (
                    session.query(OrderInfo)
                    .options(joinedload(OrderInfo.order_detail))
                    .options(joinedload(OrderInfo.logistics))
                    .options(joinedload(OrderInfo.receive))
                    .options(joinedload(OrderInfo.order_status_))
                    .filter_by(order_id=order_id)
                    .first()
                )

                # 查无此单：订单号非法或已删除，提示后返回。
                if not order_info:
                    # 回复未找到话术。
                    result.add_response("未找到该订单，请检查订单号是否正确。")

                    # 提前返回。
                    return result

                # 拼接订单信息
                # 首行：状态徽标 + 加粗订单 ID。
                message = [
                    f"- [{order_info.order_status}]**订单ID**：{order_info.order_id}"
                ]

                # 时间信息
                # 遍历"标签→时间值"字典，仅输出已存在（非空）的时间项。
                for k, v in {
                    "创建时间": order_info.create_time,
                    "支付时间": order_info.payment_time,
                    "签收时间": order_info.delivered_time,
                    "完成时间": order_info.complete_time,
                }.items():
                    # 空值（未支付/未签收等）不输出该行。
                    if v:
                        message.append(f"  - {k}：{v}")

                # 订单明细
                # 输出明细段落标题。
                message.append("- **订单明细**：")

                # 初始化三项金额合计（总价/优惠/实付），单位与明细一致。
                total_total_amount = 0.0
                total_discount_amount = 0.0
                total_final_amount = 0.0

                # 遍历明细：输出单行并累加金额。
                for order_detail in order_info.order_detail:
                    # 单行格式：商品名 × 数量 | 总价-优惠=实付。
                    message.append(
                        f"  - {order_detail.sku_name} × {order_detail.sku_count} | "
                        f"{order_detail.total_amount}-{order_detail.discount_amount}={order_detail.final_amount}"
                    )

                    # 累加总价（DB 字段经 float 转换防止 Decimal/str 混算）。
                    total_total_amount += float(order_detail.total_amount)

                    # 累加优惠金额：该列可空（None 表示无优惠），兜底按 0 计，
                    # 避免 float(None) 抛 TypeError 导致整个详情查询失败。
                    total_discount_amount += float(order_detail.discount_amount or 0)

                    # 累加实付金额。
                    total_final_amount += float(order_detail.final_amount)

                # 输出三项合计行。
                message.append(
                    f"  - **合计**：{total_total_amount}-{total_discount_amount}={total_final_amount}"
                )

                # 收货信息
                # 段落标题 + 收货人/电话/四级地址（省市区+街道逐段拼接）。
                message.extend(
                    [
                        "- **收货信息**：",
                        f"  - 收货人：{order_info.receive.receiver_name}",
                        f"  - 联系电话：{order_info.receive.receiver_phone}",
                        f"  - 收货地址：{order_info.receive.receive_province}"
                        f"{order_info.receive.receive_city}"
                        f"{order_info.receive.receive_district}"
                        f"{order_info.receive.receive_street_address}",
                    ]
                )

                # 最近物流信息
                # 取订单的物流关系列表。
                logistics = order_info.logistics

                # 有物流记录才输出该段。
                if logistics:
                    # 取列表首条记录的跟踪文本——该列可空，None 时跳过整段展示。
                    tracking = logistics[0].logistics_tracking

                    # 跟踪文本非空才输出：标题与最新轨迹（按行拆分取最后一行）。
                    if tracking:
                        # 段落标题。
                        message.append("- **最近物流信息**：")

                        # 取最后一行（最新轨迹）。
                        message.append(f"  - {tracking.splitlines()[-1]}")

                # 输出整段订单详情。
                result.add_response("\n".join(message))

                # 检查是否有售后信息（status_code >= 400）
                # 状态码列为可空，先取出便于判空。
                status_code = order_info.order_status_.status_code

                # 状态码非空且 >= 400 表示订单已进入售后阶段，追加售后记录；
                # None 视为状态未知，不进入售后分支。
                if status_code is not None and status_code >= 400:
                    # 查询售后信息
                    # 收集该订单全部明细 ID，作为售后表的过滤范围。
                    order_detail_ids = [
                        order_detail.order_detail_id
                        for order_detail in order_info.order_detail
                    ]

                    # 子查询：每个订单详情的最新售后
                    # 按"明细 ID + 创建时间最大值"分组，锁定每条明细
                    # 最近一次售后的（明细 ID, 时间）对。
                    subquery = (
                        session.query(
                            Postsale.order_detail_id,
                            func.max(Postsale.create_time).label("max_time"),
                        )
                        .filter(Postsale.order_detail_id.in_(order_detail_ids))
                        .group_by(Postsale.order_detail_id)
                        .subquery()
                    )

                    # 主查询：售后表与子查询按（明细 ID、最大时间）双键
                    # 关联，得到每条明细的最新售后记录，并预加载其明细
                    # 与物流关系。
                    postsales = (
                        session.query(Postsale)
                        .join(
                            subquery,
                            and_(
                                Postsale.order_detail_id == subquery.c.order_detail_id,
                                Postsale.create_time == subquery.c.max_time,
                            ),
                        )
                        .options(joinedload(Postsale.order_detail))
                        .options(joinedload(Postsale.logistics))
                        .all()
                    )

                    # 有售后记录才逐条输出。
                    if postsales:
                        # 遍历各明细的最新售后。
                        for postsale in postsales:
                            # 首行：售后状态徽标 + 加粗售后 ID。
                            ps_message = [
                                f"- [{postsale.postsale_status}]**售后ID**：{postsale.postsale_id}"
                            ]

                            # 输出订单明细段落标题。
                            ps_message.append("- **订单明细**：")

                            # 输出该售后对应的明细行（商品名 × 数量）。
                            ps_message.append(
                                f"  -{postsale.order_detail.sku_name} × {postsale.order_detail.sku_count}"
                            )

                            # 输出退款金额行。
                            ps_message.append(
                                f"- **退款金额**：{postsale.refund_amount}"
                            )

                            # 该售后存在物流记录时追加最近轨迹。
                            if postsale.logistics:
                                # 按创建时间倒序排列物流记录（最新在前），
                                # 就地改写会话内关系属性（仅内存生效，不提交）。
                                postsale.logistics = sorted(
                                    postsale.logistics,
                                    key=lambda x: x.create_time,
                                    reverse=True,
                                )

                                # 取首条（最新）记录的跟踪文本——该列可空，None 时跳过展示。
                                ps_tracking = postsale.logistics[0].logistics_tracking

                                # 跟踪文本非空才输出：标题与最新轨迹（按行拆分取最后一行）。
                                if ps_tracking:
                                    # 段落标题。
                                    ps_message.append("- **最近物流信息**：")

                                    # 取最后一行（最新轨迹）。
                                    ps_message.append(
                                        f"  - {ps_tracking.splitlines()[-1]}"
                                    )

                            # 每条售后作为独立响应输出。
                            result.add_response("\n".join(ps_message))

        # 任意异常兜底：不让异常上抛。
        except Exception as e:
            # 记录错误日志。
            logger.error(f"获取订单详情失败: {e}")

            # 返回通用错误话术。
            result.add_response("获取订单详情时出错，请稍后重试。")

        # 返回结果 DTO。
        return result


class ActionAskReceiveId(Action):
    """
    展示收货地址列表，供选择操作。
    选项包括：现有收货地址、修改并新建收货信息、取消

    职责：列出用户全部收货地址为按钮（payload 回填 receive_id），
          并预填当前订单的收货信息到槽位，供后续修改流程直接使用。

    设计模式：模板方法——固定"取槽→查库→造按钮→预填槽位"骨架。

    生命周期：随 domain 装配创建、全流程复用；run 无实例状态。
    """

    @property
    def name(self) -> str:
        """动作名：domain/flow 中引用的唯一标识。"""
        # 返回动作注册名。
        return "action_ask_receive_id"

    async def run(
        self,
        tracker: Any,
        domain: Any | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """执行收货地址列表展示。

        核心功能：输出地址选择按钮并预填当前订单收货信息。

        业务流程：1. 延迟导入会话与收货/订单表类；2. 读取 user_id
                  （兜底 1001）与 order_id 槽；3. 查用户全部收货
                  记录；4. 有订单时查其当前收货信息；5. 每条地址生成
                  一个按钮，末尾追加"修改并新建/取消"两个固定按钮；
                  6. 输出选择提示；7. 把当前订单收货信息逐字段写入
                  receiver_* / receive_* 槽；8. 异常兜底记日志返回
                  错误话术。

        副作用：访问 MySQL（只读）；向 tracker 写多个槽位；修改
                ActionResult.responses。

        参数：
            tracker: 对话状态跟踪器（需 get_slot/set_slot），不允许
                     为 None。
            domain: 当前域对象，本动作未使用，允许为 None。
            **kwargs: 框架预留扩展参数，本动作不消费。

        返回：
            ActionResult——responses 含地址按钮组；异常时返回错误
            话术，不抛异常。
        """
        # 延迟导入会话工厂。
        from app.ics_ecs.actions.db import SessionLocal

        # 延迟导入收货信息与订单表类。
        from app.ics_ecs.actions.db_table_class import ReceiveInfo, OrderInfo

        # 创建结果 DTO。
        result = ActionResult()

        # 读取用户槽：缺失时兜底演示账号"1001"。
        user_id = tracker.get_slot("user_id") or "1001"

        # 读取订单 ID 槽：存在时用于预填该单当前收货信息。
        order_id = tracker.get_slot("order_id")

        # 初始化空列表（本方法后续未再使用，属预留变量）。
        slots_to_set = []

        try:
            # 进入会话上下文。
            with SessionLocal() as session:
                # 获取用户所有收货信息
                # 按用户 ID 过滤取全部收货记录。
                receive_infos = (
                    session.query(ReceiveInfo).filter_by(user_id=user_id).all()
                )

                # 获取当前订单的收货信息
                # 先置 None：无订单/查无此单时保持空。
                current_receive_info = None

                # 仅在持有订单 ID 时才查询当前订单。
                if order_id:
                    # 按订单 ID 查订单。
                    order_info = (
                        session.query(OrderInfo).filter_by(order_id=order_id).first()
                    )

                    # 订单存在时取其收货信息对象。
                    if order_info:
                        current_receive_info = order_info.receive

                # 初始化按钮列表。
                buttons = []

                # 遍历收货信息，生成按钮
                for receive_info in receive_infos:
                    # 每条地址一个按钮：标题为收货人/电话/四级地址拼接，
                    # payload 把该地址 ID 回填 receive_id 槽。
                    buttons.append(
                        {
                            "title": f"收货人姓名：{receive_info.receiver_name} - "
                            f"联系电话：{receive_info.receiver_phone} - "
                            f"收货地址：{receive_info.receive_province}"
                            f"{receive_info.receive_city}"
                            f"{receive_info.receive_district}"
                            f"{receive_info.receive_street_address}",
                            "payload": f"/SetSlots(receive_id={receive_info.receive_id})",
                        }
                    )

                # 添加"修改并新建收货信息"和"取消"按钮
                # 固定两按钮：modify 哨兵进入修改流程，false 哨兵取消。
                buttons.extend(
                    [
                        {
                            # 修改并新建：receive_id 置 modify 哨兵值。
                            "title": "修改并新建收货信息",
                            "payload": "/SetSlots(receive_id=modify)",
                        },
                        # 取消：receive_id 置 false 哨兵值。
                        {"title": "取消", "payload": "/SetSlots(receive_id=false)"},
                    ]
                )

                # 输出选择提示与按钮组。
                result.add_response(
                    "请选择现有的收货信息，或修改并新建收货信息", buttons=buttons
                )

                # 将当前订单的收货信息更新到对应的槽中
                # 仅当前订单确有收货信息时执行预填。
                if current_receive_info:
                    # 预填收货人姓名。
                    tracker.set_slot(
                        "receiver_name", current_receive_info.receiver_name
                    )

                    # 预填联系电话。
                    tracker.set_slot(
                        "receiver_phone", current_receive_info.receiver_phone
                    )

                    # 预填收货省份。
                    tracker.set_slot(
                        "receive_province", current_receive_info.receive_province
                    )

                    # 预填收货城市。
                    tracker.set_slot("receive_city", current_receive_info.receive_city)

                    # 预填收货区县。
                    tracker.set_slot(
                        "receive_district", current_receive_info.receive_district
                    )

                    # 预填街道详细地址。
                    tracker.set_slot(
                        "receive_street_address",
                        current_receive_info.receive_street_address,
                    )

        # 任意异常兜底。
        except Exception as e:
            # 记录错误日志。
            logger.error(f"查询地址失败: {e}")

            # 返回通用错误话术。
            result.add_response("查询地址时出错，请稍后重试。")

        # 返回结果 DTO。
        return result


class ActionAskReceiveProvince(Action):
    """询问收货省份

    职责：从行政区划表去重查询全部省份，生成省份选择按钮。

    设计模式：模板方法；与市/区两级构成三级联动链的前端。

    生命周期：随 domain 装配创建、全流程复用；run 无实例状态。
    """

    @property
    def name(self) -> str:
        """动作名：domain/flow 中引用的唯一标识。"""
        # 返回动作注册名。
        return "action_ask_receive_province"

    async def run(
        self,
        tracker: Any,
        domain: Any | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """执行省份选择展示。

        核心功能：查询全部省份并输出按钮组。

        业务流程：1. 延迟导入会话与行政区划表类；2. 查询 province
                  列去重结果（注意：查询在会话上下文内执行）；3.
                  每省生成按钮（payload 回填 receive_province 槽）；
                  4. 输出提示；5. 异常兜底记日志返回错误话术。

        副作用：访问 MySQL（只读）；修改 ActionResult.responses。

        参数：
            tracker: 对话状态跟踪器，本动作未读槽，不允许为 None。
            domain: 当前域对象，未使用，允许为 None。
            **kwargs: 框架预留扩展参数，不消费。

        返回：
            ActionResult——responses 含省份按钮组；异常时返回错误
            话术，不抛异常。
        """
        # 延迟导入会话工厂。
        from app.ics_ecs.actions.db import SessionLocal

        # 延迟导入行政区划表类。
        from app.ics_ecs.actions.db_table_class import Region

        # 创建结果 DTO。
        result = ActionResult()

        try:
            # 进入会话上下文：省份查询在此内完成。
            with SessionLocal() as session:
                # 查询 province 列并去重，返回单列元组列表。
                provinces = session.query(Region.province).distinct().all()

            # 会话外构建按钮：每省一项，payload 回填省份名到槽。
            buttons = [
                {
                    "title": province[0],
                    "payload": f"/SetSlots(receive_province={province[0]})",
                }
                for province in provinces
            ]

            # 输出选择提示与按钮组。
            result.add_response("请选择省份", buttons=buttons)

        # 任意异常兜底。
        except Exception as e:
            # 记录错误日志。
            logger.error(f"查询省份失败: {e}")

            # 返回通用错误话术。
            result.add_response("查询省份时出错，请稍后重试。")

        # 返回结果 DTO。
        return result


class ActionAskReceiveCity(Action):
    """询问收货城市

    职责：按已选省份过滤行政区划表，去重查询城市并生成选择按钮；
          省/市/区三级联动的中段。

    设计模式：模板方法。

    生命周期：随 domain 装配创建、全流程复用；run 无实例状态。
    """

    @property
    def name(self) -> str:
        """动作名：domain/flow 中引用的唯一标识。"""
        # 返回动作注册名。
        return "action_ask_receive_city"

    async def run(
        self,
        tracker: Any,
        domain: Any | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """执行城市选择展示。

        核心功能：按省份查询城市列表并输出按钮组。

        业务流程：1. 延迟导入会话与行政区划表类；2. 读取
                  receive_province 槽作为过滤条件；3. 查询该省的
                  city 列去重结果；4. 每城生成按钮（payload 回填
                  receive_city 槽）；5. 输出提示；6. 异常兜底记日志
                  返回错误话术。

        副作用：访问 MySQL（只读）；修改 ActionResult.responses。

        参数：
            tracker: 对话状态跟踪器（需 get_slot），不允许为 None。
            domain: 当前域对象，未使用，允许为 None。
            **kwargs: 框架预留扩展参数，不消费。

        返回：
            ActionResult——responses 含城市按钮组；异常时返回错误
            话术，不抛异常。
        """
        # 延迟导入会话工厂。
        from app.ics_ecs.actions.db import SessionLocal

        # 延迟导入行政区划表类。
        from app.ics_ecs.actions.db_table_class import Region

        # 创建结果 DTO。
        result = ActionResult()

        # 读取已选省份槽：作为城市查询的过滤条件。
        receive_province = tracker.get_slot("receive_province")

        try:
            # 进入会话上下文：城市查询在此内完成。
            with SessionLocal() as session:
                # 查询该省份下的 city 列并去重。
                cities = (
                    session.query(Region.city)
                    .filter(Region.province == receive_province)
                    .distinct()
                    .all()
                )

            # 会话外构建按钮：每城一项，payload 回填城市名到槽。
            buttons = [
                {"title": city[0], "payload": f"/SetSlots(receive_city={city[0]})"}
                for city in cities
            ]

            # 输出选择提示与按钮组。
            result.add_response("请选择城市", buttons=buttons)

        # 任意异常兜底。
        except Exception as e:
            # 记录错误日志。
            logger.error(f"查询城市失败: {e}")

            # 返回通用错误话术。
            result.add_response("查询城市时出错，请稍后重试。")

        # 返回结果 DTO。
        return result


class ActionAskReceiveDistrict(Action):
    """询问收货区县

    职责：按已选城市过滤行政区划表，去重查询区县并生成选择按钮；
          省/市/区三级联动的末段。

    设计模式：模板方法。

    生命周期：随 domain 装配创建、全流程复用；run 无实例状态。
    """

    @property
    def name(self) -> str:
        """动作名：domain/flow 中引用的唯一标识。"""
        # 返回动作注册名。
        return "action_ask_receive_district"

    async def run(
        self,
        tracker: Any,
        domain: Any | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """执行区县选择展示。

        核心功能：按城市查询区县列表并输出按钮组。

        业务流程：1. 延迟导入会话与行政区划表类；2. 读取
                  receive_city 槽作为过滤条件；3. 查询该市的
                  district 列去重结果；4. 每区生成按钮（payload
                  回填 receive_district 槽）；5. 输出提示；6. 异常
                  兜底记日志返回错误话术。

        副作用：访问 MySQL（只读）；修改 ActionResult.responses。

        参数：
            tracker: 对话状态跟踪器（需 get_slot），不允许为 None。
            domain: 当前域对象，未使用，允许为 None。
            **kwargs: 框架预留扩展参数，不消费。

        返回：
            ActionResult——responses 含区县按钮组；异常时返回错误
            话术，不抛异常。
        """
        # 延迟导入会话工厂。
        from app.ics_ecs.actions.db import SessionLocal

        # 延迟导入行政区划表类。
        from app.ics_ecs.actions.db_table_class import Region

        # 创建结果 DTO。
        result = ActionResult()

        # 读取已选城市槽：作为区县查询的过滤条件。
        receive_city = tracker.get_slot("receive_city")

        try:
            # 进入会话上下文：区县查询在此内完成。
            with SessionLocal() as session:
                # 查询该城市下的 district 列并去重。
                districts = (
                    session.query(Region.district)
                    .filter(Region.city == receive_city)
                    .distinct()
                    .all()
                )

            # 会话外构建按钮：每区一项，payload 回填区县名到槽。
            buttons = [
                {
                    "title": district[0],
                    "payload": f"/SetSlots(receive_district={district[0]})",
                }
                for district in districts
            ]

            # 输出选择提示与按钮组。
            result.add_response("请选择城区", buttons=buttons)

        # 任意异常兜底。
        except Exception as e:
            # 记录错误日志。
            logger.error(f"查询区县失败: {e}")

            # 返回通用错误话术。
            result.add_response("查询区县时出错，请稍后重试。")

        # 返回结果 DTO。
        return result


class ActionAskSetReceiveInfo(Action):
    """
    设置收货信息。
    - 首次调用时展示收货信息并询问确认
    - 确认后执行修改入库

    职责：两阶段动作——按 receive_id 槽取得目标收货信息（新建路径
          从槽位构造 / 已有路径查库），首调展示并询问确认，确认后
          （set_receive_info 槽为真）查重、入库并回写订单收货地址。

    设计模式：状态机式模板方法——同一 run 依槽位组合走"展示确认"
              或"执行修改"两个阶段。

    生命周期：随 domain 装配创建、全流程复用；run 无实例状态。
    """

    @property
    def name(self) -> str:
        """动作名：domain/flow 中引用的唯一标识。"""
        # 返回动作注册名。
        return "action_ask_set_receive_info"

    async def run(
        self,
        tracker: Any,
        domain: Any | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """执行收货信息设置。

        核心功能：确定目标收货信息→首调展示确认→确认后入库回写订单。

        业务流程：1. 延迟导入会话、订单/收货表类与 uuid4；2. 读取
                  receive_id 与 set_receive_info 槽；3. receive_id
                  为 modify/modified 哨兵时从槽位构造新 ReceiveInfo
                  （ID 取 uuid 前缀），否则按 ID 查库（查无/异常均
                  提前返回）；4. set_receive_info 为真——查订单，
                  新建路径先查重（存在复用旧行、否则入库 flush），
                  回写订单 receive_id 并 commit，回复成功并记日志；
                  5. 为假（首调）——拼收货信息文本并输出确认/取消
                  按钮；6. 异常兜底记日志返回错误话术。

        副作用：访问 MySQL；写路径含收货信息新增与订单收货地址更新
                （session.commit）；向 tracker 无写槽（槽由按钮
                payload 回填）；修改 ActionResult.responses。

        参数：
            tracker: 对话状态跟踪器（需 get_slot），不允许为 None。
            domain: 当前域对象，本动作未使用，允许为 None。
            **kwargs: 框架预留扩展参数，本动作不消费。

        返回：
            ActionResult——首调返回待确认信息+确认按钮，确认后返回
            修改结果话术；槽缺失/订单不存在/异常时返回对应提示，
            不抛异常。
        """
        # 延迟导入会话工厂。
        from app.ics_ecs.actions.db import SessionLocal

        # 延迟导入订单与收货信息表类。
        from app.ics_ecs.actions.db_table_class import OrderInfo, ReceiveInfo

        # 延迟导入 uuid4：新建收货信息时生成唯一 ID。
        from uuid import uuid4

        # 创建结果 DTO。
        result = ActionResult()

        # 读取收货 ID 槽：已有地址 ID 或 modify/modified 哨兵。
        receive_id = tracker.get_slot("receive_id")

        # 读取确认标志槽：true 表示用户已确认修改（二次进入）。
        set_receive_info = tracker.get_slot("set_receive_info")

        # 根据收货id，获取收货信息
        # 哨兵值 modify（选择"修改并新建"）或 modified（修改态回入）
        # 均表示走槽位构造路径，而非选择已有地址。
        if receive_id in ("modify", "modified"):
            # 从槽中获取收货信息
            # 用槽中预填的六项字段构造未入库的临时收货对象。
            receive_info = ReceiveInfo(
                # 新 ID：rec 前缀 + 16 位十六进制随机串。
                receive_id="rec" + uuid4().hex[:16],
                # 归属用户：缺失兜底演示账号"1001"。
                user_id=tracker.get_slot("user_id") or "1001",
                # 收货人姓名：取自槽。
                receiver_name=tracker.get_slot("receiver_name"),
                # 联系电话：取自槽。
                receiver_phone=tracker.get_slot("receiver_phone"),
                # 省份：取自槽。
                receive_province=tracker.get_slot("receive_province"),
                # 城市：取自槽。
                receive_city=tracker.get_slot("receive_city"),
                # 区县：取自槽。
                receive_district=tracker.get_slot("receive_district"),
                # 街道详细地址：取自槽。
                receive_street_address=tracker.get_slot("receive_street_address"),
            )
        else:
            # 从数据库中查询对应收货信息
            # 用户选择了某个已有地址 ID：查库取记录。
            try:
                # 进入会话上下文。
                with SessionLocal() as session:
                    # 按收货 ID 查询，取首条。
                    receive_info = (
                        session.query(ReceiveInfo)
                        .filter_by(receive_id=receive_id)
                        .first()
                    )

                    # 查无此地址：提示后提前返回（receive_info 未定义，
                    # 后续分支依赖此前置返回保证安全）。
                    if not receive_info:
                        # 回复未找到话术。
                        result.add_response("未找到收货信息，请重新选择。")

                        # 提前返回。
                        return result

            # 查询异常兜底：提示后提前返回。
            except Exception as e:
                # 记录错误日志。
                logger.error(f"查询收货信息失败: {e}")

                # 返回通用错误话术。
                result.add_response("查询收货信息时出错，请稍后重试。")

                # 提前返回。
                return result

        # 如果确认修改，执行修改
        # set_receive_info 槽为真表示用户已点确认，进入写库阶段。
        if set_receive_info:
            # 读取订单 ID 槽：确定要回写收货地址的订单。
            order_id = tracker.get_slot("order_id")

            try:
                # 进入会话上下文：本阶段包含写操作。
                with SessionLocal() as session:
                    # 按订单 ID 查询订单。
                    order_info = (
                        session.query(OrderInfo).filter_by(order_id=order_id).first()
                    )

                    # 查无此单：提示后提前返回。
                    if not order_info:
                        # 回复未找到订单话术。
                        result.add_response("未找到订单，请重新操作。")

                        # 提前返回。
                        return result

                    # 如果是新建收货信息
                    # 哨兵路径：需先查重/入库，再回写订单。
                    if receive_id in ("modify", "modified"):
                        # 检查收货信息是否已存在
                        # 按用户 + 七项地址字段全等匹配查旧记录。
                        old_receive_info = (
                            session.query(ReceiveInfo)
                            .filter(
                                ReceiveInfo.user_id == receive_info.user_id,
                                ReceiveInfo.receiver_name == receive_info.receiver_name,
                                ReceiveInfo.receiver_phone
                                == receive_info.receiver_phone,
                                ReceiveInfo.receive_province
                                == receive_info.receive_province,
                                ReceiveInfo.receive_city == receive_info.receive_city,
                                ReceiveInfo.receive_district
                                == receive_info.receive_district,
                                ReceiveInfo.receive_street_address
                                == receive_info.receive_street_address,
                            )
                            .first()
                        )

                        # 已存在相同地址：复用旧记录，不重复建行。
                        if old_receive_info:
                            # 指向复用的旧记录。
                            receive_info = old_receive_info

                            # 回复提示已存在。
                            result.add_response("此收货信息已存在，将不再重复添加")

                        # 不存在：入库新记录。
                        else:
                            # 加入会话待写。
                            session.add(receive_info)

                            # flush 使 INSERT 落库并生成主键，但不提交事务
                            # （保证后续回写与新增在同一事务内原子生效）。
                            session.flush()

                    # 更新订单的收货信息
                    # 把目标地址 ID 回写到订单行。
                    order_info.receive_id = receive_info.receive_id

                    # 提交事务：新增收货信息（若有）与订单回写一并生效。
                    session.commit()

                # 回复修改成功话术。
                result.add_response("订单收货信息已修改")

                # 记录信息日志：订单与新地址 ID 的对应关系。
                logger.info(
                    f"订单 {order_id} 收货信息更新为: {receive_info.receive_id}"
                )

            # 写库异常兜底。
            except Exception as e:
                # 记录错误日志。
                logger.error(f"修改收货信息失败: {e}")

                # 返回通用错误话术。
                result.add_response("修改失败，请稍后重试。")

        # 首次执行（set_receive_info 未确认）：仅展示待确认信息。
        else:
            # 初次执行，展示收货信息，询问是否确认修改
            # 拼接六项收货字段为多行文本。
            message = [
                f"- 收货人姓名：{receive_info.receiver_name}",
                f"- 联系电话：{receive_info.receiver_phone}",
                f"- 收货省份：{receive_info.receive_province}",
                f"- 收货城市：{receive_info.receive_city}",
                f"- 收货城区：{receive_info.receive_district}",
                f"- 收货地址：{receive_info.receive_street_address}",
            ]

            # 输出收货信息文本。
            result.add_response("\n".join(message))

            # 发送确认按钮
            # 确认/取消按钮经 payload 回填 set_receive_info 槽，
            # 触发本动作二次进入走写库分支。
            result.add_response(
                "是否确认修改？",
                buttons=[
                    {"title": "确认", "payload": "/SetSlots(set_receive_info=true)"},
                    {"title": "取消", "payload": "/SetSlots(set_receive_info=false)"},
                ],
            )

        # 返回结果 DTO。
        return result


class ActionCancelOrder(Action):
    """取消订单

    职责：按 order_id 槽将订单状态置"已取消"并写完成时间；原状态为
          待发货时附退款提示。

    设计模式：模板方法——固定"校验槽→查库→更新状态→拼话术"骨架。

    生命周期：随 domain 装配创建、全流程复用；run 无实例状态。
    """

    @property
    def name(self) -> str:
        """动作名：domain/flow 中引用的唯一标识。"""
        # 返回动作注册名。
        return "action_cancel_order"

    async def run(
        self,
        tracker: Any,
        domain: Any | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """执行订单取消。

        核心功能：更新订单状态为已取消并按原状态附加退款提示。

        业务流程：1. 延迟导入会话与订单表类；2. 读取 order_id 槽，
                  缺失直接提示返回；3. 查订单，查无提示返回；4. 暂存
                  原状态→状态置"已取消"→完成时间写当前时刻→commit；
                  5. 拼话术（原状态为待发货时追加 24 小时退款提示）；
                  6. 记录信息日志；7. 异常兜底记日志返回错误话术。

        副作用：访问 MySQL 并写库（订单状态与完成时间，
                session.commit）；修改 ActionResult.responses。

        参数：
            tracker: 对话状态跟踪器（需 get_slot），不允许为 None。
            domain: 当前域对象，本动作未使用，允许为 None。
            **kwargs: 框架预留扩展参数，本动作不消费。

        返回：
            ActionResult——responses 含取消结果话术；槽缺失/订单不
            存在/异常时返回对应提示，不抛异常。
        """
        # 延迟导入会话工厂。
        from app.ics_ecs.actions.db import SessionLocal

        # 延迟导入订单表类。
        from app.ics_ecs.actions.db_table_class import OrderInfo

        # 创建结果 DTO。
        result = ActionResult()

        # 读取订单 ID 槽。
        order_id = tracker.get_slot("order_id")

        # 槽缺失：上游未选中订单，无法取消。
        if not order_id:
            # 回复订单信息丢失话术。
            result.add_response("订单信息丢失，请重新操作。")

            # 提前返回。
            return result

        try:
            # 进入会话上下文：本方法包含写操作。
            with SessionLocal() as session:
                # 按订单 ID 查询订单。
                order_info = (
                    session.query(OrderInfo).filter_by(order_id=order_id).first()
                )

                # 查无此单：提示后提前返回。
                if not order_info:
                    # 回复未找到话术。
                    result.add_response("未找到该订单，请检查订单号是否正确。")

                    # 提前返回。
                    return result

                # 获取当前订单状态
                # 暂存原状态：commit 后内存值即被改写，话术分支需要旧值。
                old_order_status = order_info.order_status

                # 更新订单状态为已取消
                order_info.order_status = "已取消"

                # 同步写完成时间：以取消时刻作为订单的终止时间戳。
                order_info.complete_time = datetime.now()

                # 提交事务：状态与时间变更落库。
                session.commit()

            # 生成回复消息
            # 基础话术。
            message = "订单已取消"

            # 如果订单状态为待发货，则添加退款金额提示
            if old_order_status == "待发货":
                # 待发货未出库：附 24 小时退款承诺。
                message += "，退款金额将在24小时内返还您的账户"

            # 输出取消结果话术。
            result.add_response(message)

            # 记录信息日志：订单 ID 与取消前状态。
            logger.info(f"订单 {order_id} 已取消，原状态: {old_order_status}")

        # 写库异常兜底。
        except Exception as e:
            # 记录错误日志。
            logger.error(f"取消订单失败: {e}")

            # 返回通用错误话术。
            result.add_response("取消失败，请稍后重试。")

        # 返回结果 DTO。
        return result
