"""
action_logistics - 物流业务 Action 模块

核心功能：实现物流域的两个查询动作——快递公司列表（ActionGet
          LogisticsCompanys，供用户选择查询渠道）与订单物流轨迹
          （ActionGetLogisticsInfo，按 order_id 槽查单并展示明细/
          轨迹，同时回填 logistics_id 槽）。

执行流程：1. action_get_logistics_companys 全表查 LogisticsCompany，
          拼接公司名清单（空结果兜底"- 无"）；2. action_get_
          logistics_info 先校验 order_id 槽（缺失或 false 提示先选
          订单），查订单（预加载物流与明细关系）——查无此单/无物流
          分别提前返回；3. 取首条物流记录，拼订单 ID、明细清单、
          物流 ID 与轨迹行（tracking 文本按换行拆分为列表项），
          输出响应并把物流 ID 写入 tracker 槽位。

副作用：两个 Action 均在 run 内延迟导入 SessionLocal 访问 MySQL
        （只读查询，不写库）；ActionGetLogisticsInfo 向 tracker 写
        logistics_id 槽；响应经 ActionResult.add_response 收集；
        异常在方法内捕获、记日志并返回错误话术，不向上抛出。

主要依赖：app.ics_ai.agent.actions（Action 基类与 ActionResult
          DTO）；app.ics_ecs.actions.db 与 db_table_class（本包数据
          库会话工厂与 ORM 表类，run 内延迟导入，避免模块导入期连库）。
"""

# 导入 logging：数据库异常经本模块 logger 记录错误日志。
import logging

# 导入 Any：tracker/domain 等鸭子类型参数的通配注解。
from typing import Any

# 从动作框架导入基类与结果 DTO：Action 定义 name/run 契约，
# ActionResult 承载响应列表；本模块两个 Action 均继承前者。
from app.ics_ai.agent.actions import Action, ActionResult

# 本模块专属日志器：模块内所有异常日志统一挂在此命名空间下。
logger = logging.getLogger(__name__)


class ActionGetLogisticsCompanys(Action):
    """查询支持的快递公司

    职责：全表查询 LogisticsCompany，把公司名拼接为列表文本返回，
          供用户选择物流查询渠道。

    设计模式：模板方法——固定"查库→拼接→兜底→输出"骨架。

    生命周期：随 domain 装配创建、全流程复用；run 无实例状态。
    """

    @property
    def name(self) -> str:
        """动作名：domain/flow 中引用的唯一标识。"""
        # 返回动作注册名。
        return "action_get_logistics_companys"

    async def run(
        self,
        tracker: Any,
        domain: Any | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """执行快递公司列表查询。

        核心功能：输出全部支持快递公司的清单文本。

        业务流程：1. 延迟导入会话与快递公司表类；2. 全表查询并
                  关闭会话（查询在上下文内完成）；3. 逐行拼接
                  "- 公司名" 清单；4. 空结果兜底为"- 无"；5. 输出
                  响应；6. 异常兜底记日志返回错误话术。

        副作用：访问 MySQL（只读）；修改 ActionResult.responses。

        参数：
            tracker: 对话状态跟踪器，本动作未读槽，不允许为 None。
            domain: 当前域对象，本动作未使用，允许为 None。
            **kwargs: 框架预留扩展参数，本动作不消费。

        返回：
            ActionResult——responses 含快递公司清单；异常时返回
            错误话术，不抛异常。
        """
        # 延迟导入会话工厂：调用时刻才连库。
        from app.ics_ecs.actions.db import SessionLocal

        # 延迟导入快递公司表类。
        from app.ics_ecs.actions.db_table_class import LogisticsCompany

        # 创建结果 DTO。
        result = ActionResult()

        try:
            # 进入会话上下文：全表查询在此内完成。
            with SessionLocal() as session:
                # 查询全部快递公司记录。
                logistics_companys = session.query(LogisticsCompany).all()

            # 按照company_name字段，拼接快递公司名称
            # join 空列表得到空串，作为下方兜底判断的哨兵值。
            logistics_list = "".join(
                [f"- {i.company_name}\n" for i in logistics_companys]
            )

            # 如果没有查询到快递公司名称，返回"- 无"
            if logistics_list == "":
                logistics_list = "- 无"

            # 输出清单文本。
            result.add_response(f"支持的快递有:\n{logistics_list}")

        # 任意异常兜底：不让异常上抛。
        except Exception as e:
            # 记录错误日志。
            logger.error(f"查询快递公司失败: {e}")

            # 返回通用错误话术。
            result.add_response("查询快递公司时出错，请稍后重试。")

        # 返回结果 DTO。
        return result


class ActionGetLogisticsInfo(Action):
    """查询指定订单的物流信息

    职责：按 order_id 槽查询订单物流——展示订单 ID/明细/物流 ID/
          完整轨迹行，并把物流 ID 回填 logistics_id 槽供后续动作
          （如催单）消费。

    设计模式：模板方法——固定"校验槽→查库→三级前置校验→拼装→
              回填槽位"骨架。

    生命周期：随 domain 装配创建、全流程复用；run 无实例状态。
    """

    @property
    def name(self) -> str:
        """动作名：domain/flow 中引用的唯一标识。"""
        # 返回动作注册名。
        return "action_get_logistics_info"

    async def run(
        self,
        tracker: Any,
        domain: Any | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """执行订单物流查询。

        核心功能：输出订单物流轨迹并回填物流 ID 槽位。

        业务流程：1. 延迟导入会话、订单表类与 joinedload；2. 读取
                  order_id 槽——缺失或为 false 提示先选订单后提前
                  返回；3. 查订单（预加载物流/明细），查无此单、
                  无物流记录分别提示后提前返回；4. 取首条物流记录，
                  拼订单 ID/明细清单/物流 ID/轨迹行（tracking 按
                  换行拆分，可空列守卫已有）；5. 输出响应并把物流
                  ID 写入槽位；6. 异常兜底记日志返回错误话术。

        副作用：访问 MySQL（只读）；向 tracker 写 logistics_id 槽；
                修改 ActionResult.responses。

        参数：
            tracker: 对话状态跟踪器（需 get_slot/set_slot），不允许
                     为 None。
            domain: 当前域对象，本动作未使用，允许为 None。
            **kwargs: 框架预留扩展参数，本动作不消费。

        返回：
            ActionResult——responses 含物流轨迹文本；槽缺失/订单
            不存在/无物流/异常时返回对应提示，不抛异常。
        """
        # 延迟导入会话工厂。
        from app.ics_ecs.actions.db import SessionLocal

        # 延迟导入订单表类。
        from app.ics_ecs.actions.db_table_class import OrderInfo

        # 延迟导入 joinedload：预加载一对多关系，规避 N+1 查询。
        from sqlalchemy.orm import joinedload

        # 创建结果 DTO。
        result = ActionResult()

        # 读取订单 ID 槽：由上游 action_ask_order_id 点选回填。
        order_id = tracker.get_slot("order_id")

        # 槽缺失或为哨兵值 false：说明上游未选中订单。
        if not order_id or order_id == "false":
            # 提示先选择订单。
            result.add_response("请先选择要查询物流的订单。")

            # 提前返回，不进入查库分支。
            return result

        try:
            # 进入会话上下文。
            with SessionLocal() as session:
                # 链式构建查询：预加载物流与明细关系，按订单 ID 过滤取首条。
                order_info = (
                    session.query(OrderInfo)
                    .options(joinedload(OrderInfo.logistics))
                    .options(joinedload(OrderInfo.order_detail))
                    .filter_by(order_id=order_id)
                    .first()
                )

                # 查无此单：订单号非法或已删除。
                if not order_info:
                    # 回复未找到话术后提前返回。
                    result.add_response("未找到该订单，请检查订单号是否正确。")
                    return result

                # 无物流记录：订单尚未发货。
                if not order_info.logistics:
                    # 回复暂无物流话术后提前返回。
                    result.add_response("该订单暂无物流信息，可能还未发货。")
                    return result

                # 获取订单物流信息
                # 取首条物流记录（当前数据模型一单一条物流）。
                logistics = order_info.logistics[0]

                # 拼接订单id
                message = [f"- **订单ID**：{order_id}"]

                # 拼接sku名称和对应的数量
                for order_detail in order_info.order_detail:
                    message.append(
                        f"  - {order_detail.sku_name} × {order_detail.sku_count}"
                    )

                # 拼接物流id
                message.append(f"- **物流ID**：{logistics.logistics_id}")

                # 拼接"物流信息"标题
                message.append("- **物流信息**：")

                # 拼接物流详情，对logistics表的logistics_tracking字段按照换行符进行分割
                # 该列可空：None/空串时跳过轨迹行展示。
                if logistics.logistics_tracking:
                    message.append(
                        "  - " + "\n  - ".join(logistics.logistics_tracking.split("\n"))
                    )

                # 输出整段物流信息。
                result.add_response("\n".join(message))

                # 将物流id存入槽中
                # 供后续动作（如按物流催单）直接消费。
                tracker.set_slot("logistics_id", logistics.logistics_id)

        # 任意异常兜底：不让异常上抛。
        except Exception as e:
            # 记录错误日志。
            logger.error(f"查询物流失败: {e}")

            # 返回通用错误话术。
            result.add_response("查询物流时出错，请稍后重试。")

        # 返回结果 DTO。
        return result
