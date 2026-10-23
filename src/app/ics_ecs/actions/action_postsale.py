"""
action_postsale - 售后业务 Action 模块

核心功能：实现售后域的四个动作——可售后订单查询（ActionAsk
          OrderIdAfterDelivered）、售后资格校验（ActionCheckPostsale
          Eligible，状态段 + 7 天期限双重门槛）、售后原因候选查询
          （ActionAskPostsaleReason，按商品类别过滤原因库）、售后
          申请提交（ActionApplyPostsale，逐明细建售后单并更新订单
          状态）。

执行流程：1. action_ask_order_id_after_delivered 查 status_code
          >= 330（已签收及之后）且未取消的订单，按订单生成点选
          按钮（无单置 order_id=NO_ORDER 哨兵）；2. action_check_
          postsale_eligible 校验订单状态段（空值或<330 拒绝）与售后期限
          （签收时间优先、兜底创建时间，超 7 天拒绝），通过后输出
          售后类型菜单；3. action_ask_postsale_reason 取订单明细
          的商品类别，查适配的售后原因（类别匹配或通用原因），
          无候选时降级为自由输入；4. action_apply_postsale 校验
          槽位完整性→逐明细创建售后单（换货不填退款金额）→订单
          状态置"售后中"→commit→输出申请回执。

副作用：各 run 延迟导入 SessionLocal 访问 MySQL；写路径为售后
        单批量新增与订单状态更新（session.commit）；向 tracker
        写槽位（order_id 哨兵、读取 postsale_type/reason）；响应
        经 ActionResult.add_response 收集；异常在方法内捕获、记
        日志并返回错误话术，不向上抛出。

主要依赖：app.ics_ai.agent.actions（Action 基类与 ActionResult
          DTO）；app.ics_ecs.actions.db 与 db_table_class（本包
          数据库会话工厂与 ORM 表类，run 内延迟导入，避免模块
          导入期连库）。
"""

# 导入 logging：各 Action 的数据库异常经本模块 logger 记录错误日志。
import logging

# 从 datetime 导入日期时间：售后期限计算与售后单创建时间戳。
from datetime import datetime

# 导入 Any：tracker/domain 等鸭子类型参数的通配注解。
from typing import Any

# 从 uuid 导入 uuid4：生成售后单唯一 ID。
from uuid import uuid4

# 从动作框架导入基类与结果 DTO：Action 定义 name/run 契约，
# ActionResult 承载响应列表；本模块四个 Action 均继承前者。
from app.ics_ai.agent.actions import Action, ActionResult

# 本模块专属日志器：模块内所有异常日志统一挂在此命名空间下。
logger = logging.getLogger(__name__)


class ActionAskOrderIdAfterDelivered(Action):
    """查询已签收的订单，用于售后申请

    职责：检索用户"已签收及之后"（status_code >= 330）且未取消的
          订单，生成订单点选按钮（订单 ID 与商品摘要截断展示），
          供售后申请流程选择目标订单。

    设计模式：模板方法——固定"取槽→查库→造按钮→输出"骨架。

    生命周期：随 domain 装配创建、全流程复用；run 无实例状态。
    """

    @property
    def name(self) -> str:
        """动作名：domain/flow 中引用的唯一标识。"""
        # 返回动作注册名。
        return "action_ask_order_id_after_delivered"

    async def run(
        self,
        tracker: Any,
        domain: Any | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """执行可售后订单查询。

        核心功能：输出可申请售后的订单按钮列表。

        业务流程：1. 延迟导入会话、订单/状态表类与 joinedload；
                  2. 读取 user_id（兜底演示账号 1001）；3. 联状态表
                  查询该用户未取消且状态码 >= 330 的订单，预加载
                  明细、按创建时间倒序；4. 无单——回复话术、置
                  order_id=NO_ORDER 哨兵并提前返回；5. 逐单生成
                  按钮（订单 ID 取前 12 字符、商品摘要取前 2 项并
                  截断 15 字符）；6. 异常兜底记日志返回错误话术。

        副作用：访问 MySQL（只读）；向 tracker 写 order_id 槽
                （无单分支）；修改 ActionResult.responses。

        参数：
            tracker: 对话状态跟踪器（需 get_slot/set_slot），不允许
                     为 None。
            domain: 当前域对象，本动作未使用，允许为 None。
            **kwargs: 框架预留扩展参数，本动作不消费。

        返回：
            ActionResult——responses 含订单按钮组；无单/异常时返回
            对应话术，不抛异常。
        """
        # 延迟导入会话工厂：调用时刻才连库。
        from app.ics_ecs.actions.db import SessionLocal

        # 延迟导入订单与状态表类：查询主体与状态码过滤使用。
        from app.ics_ecs.actions.db_table_class import OrderInfo, OrderStatus

        # 延迟导入 joinedload：预加载订单明细关系。
        from sqlalchemy.orm import joinedload

        # 创建结果 DTO。
        result = ActionResult()

        # 读取用户槽：缺失时兜底演示账号"1001"。
        user_id = tracker.get_slot("user_id") or "1001"

        try:
            # 进入会话上下文。
            with SessionLocal() as session:
                # 查询已签收、售后中、已完成的订单
                # 链式构建：联状态表（供状态码比较）、预加载明细、
                # 过滤（本用户 + 未取消 + 已签收及之后）、按创建时间
                # 倒序取全部。
                order_infos = (
                    session.query(OrderInfo)
                    .join(OrderInfo.order_status_)
                    .options(joinedload(OrderInfo.order_detail))
                    .filter(
                        OrderInfo.user_id == user_id,
                        OrderInfo.order_status != "已取消",
                        OrderStatus.status_code >= 330,  # 已签收及之后
                    )
                    .order_by(OrderInfo.create_time.desc())
                    .all()
                )

                # 无可售后订单：置 NO_ORDER 哨兵后提前返回。
                if not order_infos:
                    # 回复无单话术。
                    result.add_response("您没有可申请售后的订单。")

                    # 将 order_id 置 NO_ORDER 哨兵：向流程表达"无可选订单"，
                    # 与普通查询流程的 false 哨兵区分。
                    tracker.set_slot("order_id", "NO_ORDER")
                    return result

                # 拼接选择提示语。
                msg = "请选择要申请售后的订单："

                # 初始化按钮列表：每单对应一个摘要按钮。
                buttons = []

                # 遍历订单生成按钮（enumerate 提供序号 i，当前未用于文案）。
                for i, order in enumerate(order_infos, 1):
                    # 收集该单全部明细的"商品名×数量"摘要。
                    sku_info = []
                    for detail in order.order_detail:
                        sku_info.append(f"{detail.sku_name}×{detail.sku_count}")

                    # 仅取前 2 项商品，控制按钮标题长度。
                    sku_str = ", ".join(sku_info[:2])

                    # 添加按钮
                    # 标题截断：订单 ID 前 12 字符 + 商品摘要前 15 字符，
                    # payload 回填完整订单 ID。
                    buttons.append(
                        {
                            "title": f"{order.order_id[:12]}... ({sku_str[:15]})",
                            "payload": f"/SetSlots(order_id={order.order_id})",
                        }
                    )

                # 输出选择提示与按钮组。
                result.add_response(msg, buttons=buttons)

        # 任意异常兜底：不让异常上抛。
        except Exception as e:
            # 记录错误日志。
            logger.error(f"查询订单失败: {e}")

            # 返回通用错误话术。
            result.add_response("查询订单时出错，请稍后重试。")

        # 返回结果 DTO。
        return result


class ActionCheckPostsaleEligible(Action):
    """检查订单是否支持售后

    职责：对已选订单做售后资格双重校验——状态段（已签收及之后，
          status_code >= 330）与售后期限（签收/创建后 7 天内），
          通过后输出售后类型菜单。

    设计模式：模板方法——固定"校验槽→查库→双重门槛→出菜单"骨架，
              任一门槛不满足即提前返回。

    生命周期：随 domain 装配创建、全流程复用；run 无实例状态。
    """

    @property
    def name(self) -> str:
        """动作名：domain/flow 中引用的唯一标识。"""
        # 返回动作注册名。
        return "action_check_postsale_eligible"

    async def run(
        self,
        tracker: Any,
        domain: Any | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """执行售后资格校验。

        核心功能：校验订单状态段与售后期限，通过则给出类型菜单。

        业务流程：1. 延迟导入会话与表类；2. 读取 order_id 槽——
                  缺失或 false 提示先选订单；3. 联状态表查订单，
                  查无置 NO_ORDER 哨兵提前返回；4. 状态码为空或
                  < 330 拒绝并提示；5. 取签收时间（兜底创建时间）计算
                  距今天数，超 7 天拒绝；6. 拼售后类型菜单（退款/
                  退货/换货）输出；7. 异常兜底记日志返回错误话术。

        副作用：访问 MySQL（只读）；向 tracker 写 order_id 槽
                （查无此单分支）；修改 ActionResult.responses。

        参数：
            tracker: 对话状态跟踪器（需 get_slot/set_slot），不允许
                     为 None。
            domain: 当前域对象，本动作未使用，允许为 None。
            **kwargs: 框架预留扩展参数，本动作不消费。

        返回：
            ActionResult——通过校验时 responses 含类型菜单，否则含
            对应拒绝/提示话术；不抛异常。
        """
        # 延迟导入会话工厂。
        from app.ics_ecs.actions.db import SessionLocal

        # 延迟导入订单与状态表类。
        from app.ics_ecs.actions.db_table_class import OrderInfo, OrderStatus

        # 延迟导入 joinedload：预加载订单明细。
        from sqlalchemy.orm import joinedload

        # 创建结果 DTO。
        result = ActionResult()

        # 读取订单 ID 槽：由上游订单查询动作点选回填。
        order_id = tracker.get_slot("order_id")

        # 槽缺失或为哨兵值 false：上游未选中订单。
        if not order_id or order_id == "false":
            # 提示先选择订单。
            result.add_response("请先选择要申请售后的订单。")
            return result

        try:
            # 进入会话上下文。
            with SessionLocal() as session:
                # 联状态表查订单（供状态码比较），预加载明细，按 ID 取首条。
                # filter_by 在 join 后会默认作用到最后一个实体（OrderStatus）上，
                # 该表无 order_id 列会报 InvalidRequestError，故改用显式
                # filter(OrderInfo.order_id == ...) 限定实体避免歧义。
                order_info = (
                    session.query(OrderInfo)
                    .join(OrderInfo.order_status_)
                    .options(joinedload(OrderInfo.order_detail))
                    .filter(OrderInfo.order_id == order_id)
                    .first()
                )

                # 查无此单：置 NO_ORDER 哨兵后提前返回。
                if not order_info:
                    # 回复未找到话术。
                    result.add_response("未找到该订单。")

                    # NO_ORDER 哨兵：向流程表达"订单不存在"。
                    tracker.set_slot("order_id", "NO_ORDER")
                    return result

                # 检查订单状态是否支持售后 (已签收及之后)
                # 读取状态码：该列可空，取出后先做 None 守卫再做数值比较。
                status_code = order_info.order_status_.status_code

                # 状态码为空（状态字典缺失、无法确认已签收）或 < 330（未签收）不支持售后。
                if status_code is None or status_code < 330:
                    # 回复不支持售后话术后提前返回。
                    result.add_response(
                        "抱歉，该订单当前状态不支持售后申请。需要订单已签收后才能申请售后。"
                    )
                    return result

                # 检查是否在售后期限内（7天内）
                # 计时基准：签收时间优先（售后从签收起算），未签收兜底创建时间。
                check_time = order_info.delivered_time or order_info.create_time

                # 基准时间存在才校验期限（两者皆空的极端情况跳过）。
                if check_time:
                    # 计算距今天数（ timedelta 整数日）。
                    days_since = (datetime.now() - check_time).days

                    # 超 7 天拒绝售后。
                    if days_since > 7:
                        # 回复超期话术后提前返回。
                        result.add_response("抱歉，该订单已超过7天售后期限。")
                        return result

                # 双重校验通过：拼售后类型菜单。
                msg = f"订单 {order_id} 符合售后条件。\n"
                msg += "请选择售后类型：\n"
                msg += "1. 退款 - 仅退款不退货\n"
                msg += "2. 退货 - 退货并退款\n"
                msg += "3. 换货 - 退回商品换新\n"

                # 输出菜单。
                result.add_response(msg)

        # 任意异常兜底：不让异常上抛。
        except Exception as e:
            # 记录错误日志。
            logger.error(f"检查售后资格失败: {e}")

            # 返回通用错误话术。
            result.add_response("检查售后资格时出错，请稍后重试。")

        # 返回结果 DTO。
        return result


class ActionAskPostsaleReason(Action):
    """询问售后原因

    职责：按订单商品的类别检索适配的售后原因库（类别匹配或通用
          原因），生成编号列表 + "其他原因"选项；无候选或订单无
          明细时降级为让用户自由输入。

    设计模式：模板方法——固定"查明细→查类别→过滤原因库→出候选"
              骨架；降级路径（自由输入）贯穿各分支。

    生命周期：随 domain 装配创建、全流程复用；run 无实例状态。
    """

    @property
    def name(self) -> str:
        """动作名：domain/flow 中引用的唯一标识。"""
        # 返回动作注册名。
        return "action_ask_postsale_reason"

    async def run(
        self,
        tracker: Any,
        domain: Any | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """执行售后原因候选查询。

        核心功能：按商品类别输出可选售后原因列表。

        业务流程：1. 延迟导入会话、原因/订单/明细/SKU 表类与 or_；
                  2. 查订单（预加载明细）——订单或明细为空时提示
                  自由输入提前返回；3. 收集明细的 sku_id 集合，查
                  SKU 表得到类别并集合去重；4. 查原因库：类别命中
                  或 product_category 为 NULL（通用原因）；5. 有
                  候选——编号列表 + "其他原因"输出；无候选——提示
                  自由输入；6. 异常兜底同样降级为自由输入。

        副作用：访问 MySQL（只读）；修改 ActionResult.responses。

        参数：
            tracker: 对话状态跟踪器（需 get_slot），不允许为 None。
            domain: 当前域对象，本动作未使用，允许为 None。
            **kwargs: 框架预留扩展参数，本动作不消费。

        返回：
            ActionResult——responses 含原因候选列表或自由输入提示；
            不抛异常。
        """
        # 延迟导入会话工厂。
        from app.ics_ecs.actions.db import SessionLocal

        # 延迟导入原因/订单/明细/SKU 表类。
        from app.ics_ecs.actions.db_table_class import (
            PostsaleReason,
            OrderInfo,
            OrderDetail,
            SkuInfo,
        )

        # 延迟导入 or_：组合"类别匹配或通用原因"条件。
        from sqlalchemy import or_

        # 延迟导入 joinedload：预加载订单明细。
        from sqlalchemy.orm import joinedload

        # 创建结果 DTO。
        result = ActionResult()

        # 读取订单 ID 槽。
        order_id = tracker.get_slot("order_id")

        try:
            # 进入会话上下文。
            with SessionLocal() as session:
                # 获取订单中商品的类别
                # 查订单并预加载明细（原因候选依赖商品类别）。
                order_info = (
                    session.query(OrderInfo)
                    .options(joinedload(OrderInfo.order_detail))
                    .filter_by(order_id=order_id)
                    .first()
                )

                # 订单不存在或无明细：无法推断类别，降级自由输入。
                if not order_info or not order_info.order_detail:
                    # 提示自由输入后提前返回。
                    result.add_response("请输入售后原因：")
                    return result

                # 获取商品类别
                # 收集全部明细的 sku_id。
                sku_ids = [detail.sku_id for detail in order_info.order_detail]

                # 查 SKU 表取商品信息。
                sku_infos = (
                    session.query(SkuInfo).filter(SkuInfo.sku_id.in_(sku_ids)).all()
                )

                # 类别集合去重（set 无序，仅用于 in_ 匹配，不影响展示顺序）。
                categories = list(set(sku.sku_category for sku in sku_infos))

                # 查询适用的售后原因
                # 条件：原因的适用类别命中订单商品类别，或类别为 NULL
                # （NULL 表示不限类别的通用原因）。
                reasons = (
                    session.query(PostsaleReason)
                    .filter(
                        or_(
                            PostsaleReason.product_category.in_(categories),
                            PostsaleReason.product_category.is_(None),
                        )
                    )
                    .all()
                )

                # 有候选原因：输出编号列表。
                if reasons:
                    # 菜单标题。
                    msg = "请选择售后原因：\n"

                    # 逐条编号拼接原因文本。
                    for i, reason in enumerate(reasons, 1):
                        msg += f"{i}. {reason.postsale_reason}\n"

                    # 末尾追加"其他原因"选项，覆盖候选不全的情况。
                    msg += f"{len(reasons) + 1}. 其他原因\n"

                    # 输出候选列表。
                    result.add_response(msg)

                # 无候选原因：同样降级自由输入。
                else:
                    result.add_response("请输入售后原因：")

        # 任意异常兜底：降级为自由输入而非错误话术（保持流程可继续）。
        except Exception as e:
            # 记录错误日志。
            logger.error(f"获取售后原因失败: {e}")

            # 返回自由输入提示。
            result.add_response("请输入售后原因：")

        # 返回结果 DTO。
        return result


class ActionApplyPostsale(Action):
    """提交售后申请

    职责：汇总订单/类型/原因槽位，为订单的每个明细创建一条售后
          单（退款/退货填退款金额，换货不填），订单状态置"售后中"
          并输出申请回执。

    设计模式：模板方法——固定"校验槽→查单→建单→更新状态→回执"
              骨架；类型差异收敛到状态映射与金额判空两处。

    生命周期：随 domain 装配创建、全流程复用；run 无实例状态。
    """

    @property
    def name(self) -> str:
        """动作名：domain/flow 中引用的唯一标识。"""
        # 返回动作注册名。
        return "action_apply_postsale"

    async def run(
        self,
        tracker: Any,
        domain: Any | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """执行售后申请提交。

        核心功能：逐明细创建售后单并把订单置为售后中。

        业务流程：1. 延迟导入会话、售后/订单/售后状态表类；2. 读取
                  order_id/postsale_type/postsale_reason/user_id 槽，
                  任一缺失提示重新申请；3. 查订单，查无提示返回；
                  4. 按类型映射初始售后状态（未知类型兜底退款待审
                  核）；5. 逐明细建 Postsale（pts+uuid16 唯一 ID，
                  换货 refund_amount 为 None）；6. 订单状态置"售后中"
                  并 commit；7. 记录信息日志；8. 会话外拼申请回执
                  （类型文案映射：退货→"退货退款"）输出；9. 异常
                  兜底记日志返回错误话术。

        副作用：访问 MySQL；写库（售后单批量新增 + 订单状态更新，
                session.commit）；修改 ActionResult.responses；日志
                记录申请摘要。

        参数：
            tracker: 对话状态跟踪器（需 get_slot），不允许为 None。
            domain: 当前域对象，本动作未使用，允许为 None。
            **kwargs: 框架预留扩展参数，本动作不消费。

        返回：
            ActionResult——成功时 responses 含申请回执文本；槽缺失/
            订单不存在/异常时返回对应提示，不抛异常。
        """
        # 延迟导入会话工厂。
        from app.ics_ecs.actions.db import SessionLocal

        # 延迟导入售后单/订单/售后状态表类。
        from app.ics_ecs.actions.db_table_class import (
            Postsale,
            OrderInfo,
            PostsaleStatus,
        )
        from sqlalchemy.orm import joinedload

        # 创建结果 DTO。
        result = ActionResult()

        # 读取订单 ID 槽。
        order_id = tracker.get_slot("order_id")

        # 读取售后类型槽：退款/退货/换货
        postsale_type = tracker.get_slot("postsale_type")

        # 类型归一化：对外话术允许"退货退款"等表述（LLM 按模板原样提取），
        # 而售后表 postsale_type 列是 ENUM("退款","退货","换货")，先映射
        # 回值域内再写库，避免 Data truncated；未知值原样透传由兜底状态承接。
        postsale_type = {"退货退款": "退货", "仅退款": "退款"}.get(
            postsale_type, postsale_type
        )

        # 读取售后原因槽：候选选项或自由输入文本。
        postsale_reason = tracker.get_slot("postsale_reason")

        # 读取用户槽：缺失兜底演示账号"1001"。
        user_id = tracker.get_slot("user_id") or "1001"

        # 三项关键信息任一缺失即无法建单。
        # 判定逻辑：逐项显式判空而非 all([...])——两者运行时语义等价
        # （任一为假即整体为假），但静态检查器无法经 all() 收窄元素类型，
        # 显式 or 链可在守卫后将 order_id/postsale_type/postsale_reason
        # 收窄为 str，消除后续 dict.get 键参的 reportCallIssue。
        if not order_id or not postsale_type or not postsale_reason:
            # 提示重新申请后提前返回。
            result.add_response("售后信息不完整，请重新申请。")
            return result

        try:
            # 进入会话上下文：本方法包含写操作。
            with SessionLocal() as session:
                # 获取订单信息
                # 查订单并预加载明细（售后单按明细逐条创建）。
                order_info = (
                    session.query(OrderInfo)
                    .options(joinedload(OrderInfo.order_detail))
                    .filter_by(order_id=order_id)
                    .first()
                )

                # 查无此单：提示后提前返回。
                if not order_info:
                    result.add_response("未找到该订单。")
                    return result

                # 确定初始售后状态
                # 售后状态字典表（postsale_status）并无"xx待审核"值，售后申请
                # 的初始态固定为"审核中"（status_code=410），后续按审核结论
                # 流转到退款中/退货中/换发货等中间态与 9xx 终态；硬编码
                # "退款待审核/退货待审核/换货待审核"会触发外键约束失败。
                initial_status = "审核中"

                # 为每个订单明细创建售后记录
                # 收集已建售后单，供日志统计数量。
                created_postsales = []
                for detail in order_info.order_detail:
                    # 构造售后单：唯一 ID（pts+16 位随机 hex）、创建时间、
                    # 原因/状态/类型透传槽值；收货信息沿用订单当前地址；
                    # 退款金额取明细实付（换货无退款，置 None）。
                    postsale = Postsale(
                        postsale_id="pts" + uuid4().hex[:16],
                        create_time=datetime.now(),
                        order_detail_id=detail.order_detail_id,
                        postsale_reason=postsale_reason,
                        postsale_status=initial_status,
                        receive_id=order_info.receive_id,
                        refund_amount=detail.final_amount
                        if postsale_type != "换货"
                        else None,
                        postsale_type=postsale_type,
                    )

                    # 加入会话待写。
                    session.add(postsale)

                    # 登记到已建列表。
                    created_postsales.append(postsale)

                # 更新订单状态为售后中
                order_info.order_status = "售后中"

                # 提交事务：全部售后单与订单状态变更一并落库。
                session.commit()

                # 记录信息日志：订单/类型/建单数量摘要。
                logger.info(
                    f"售后申请已创建: 订单{order_id}, 类型{postsale_type}, 数量{len(created_postsales)}"
                )

            # 会话已结束，拼申请回执（类型文案映射：退货对外表述为"退货退款"）。
            type_text = {"退款": "退款", "退货": "退货退款", "换货": "换货"}.get(
                postsale_type, "售后"
            )

            # 输出回执：类型文案 + 订单号/类型/原因明细 + 处理时限承诺。
            result.add_response(
                f"您的{type_text}申请已提交！\n\n"
                f"- 订单号: {order_id}\n"
                f"- 申请类型: {postsale_type}\n"
                f"- 申请原因: {postsale_reason}\n\n"
                f"我们会在1-3个工作日内处理，请耐心等待。"
            )

        # 写库异常兜底：不让异常上抛。
        except Exception as e:
            # 记录错误日志。
            logger.error(f"提交售后申请失败: {e}")

            # 返回通用错误话术。
            result.add_response("提交申请时出错，请稍后重试。")

        # 返回结果 DTO。
        return result
