"""
actions - 电商客服动作入口模块

核心功能：ics_ecs 电商域全部自定义 Action 的聚合导出入口——覆盖
          订单（查询/详情/改址/取消）、物流（快递公司/轨迹查询）、
          售后（可售单查询/资格校验/原因候选/申请提交）三大业务域
          共 14 个动作，供 ics_ai 框架按动作名装配调度。

执行流程：1. 模块导入：聚合导出三个动作子模块的公开 Action 类
          （无 IO）；2. 装配期由 domain 收集各动作实例（以 name
          属性为唯一标识）；3. 执行期按流程预测的动作名分发到对应
          run()，完成查库、写槽与响应输出。

副作用：模块导入无 IO（仅类定义与聚合导出）；各 Action run 的
        数据库读写、tracker 槽位写入、响应收集与异常内吞等真实
        副作用见各动作类文档。

主要依赖：app.ics_ecs.actions.action_order / action_logistics /
          action_postsale（三大业务域的动作实现，动作基类 Action
          与结果 DTO ActionResult 由各子模块自行导入）。
"""

# 从物流动作子模块导入物流域动作：快递公司字典与订单轨迹查询。
from .action_logistics import (
    # 快递公司查询动作：全表列出支持的快递公司，供用户选择查询渠道
    ActionGetLogisticsCompanys,
    # 物流轨迹查询动作：按订单展示完整轨迹并回填 logistics_id 槽
    ActionGetLogisticsInfo,
)

# 从订单动作子模块导入订单域动作：分场景查询、改址三级联动与取消。
from .action_order import (
    # 统一订单查询动作：按 goto 槽分场景检索订单，按钮回填 order_id
    ActionAskOrderId,
    # 订单详情查询动作：按 order_id 拉取状态/时间线/明细/物流全量信息
    ActionGetOrderDetail,
    # 收货地址列表动作：列出全部收货地址为按钮并预填当前订单收货信息
    ActionAskReceiveId,
    # 询问收货省份动作：行政区划去重查询全部省份，三级联动前端
    ActionAskReceiveProvince,
    # 询问收货城市动作：按已选省份过滤行政区划，三级联动中段
    ActionAskReceiveCity,
    # 询问收货区县动作：按已选城市过滤行政区划，三级联动末段
    ActionAskReceiveDistrict,
    # 设置收货信息动作：两阶段——展示确认后查重入库并回写订单地址
    ActionAskSetReceiveInfo,
    # 取消订单动作：订单状态置"已取消"并写完成时间，待发货附退款提示
    ActionCancelOrder,
)

# 从售后动作子模块导入售后域动作：可售单查询、校验、原因与提交。
from .action_postsale import (
    # 售后申请提交动作：逐明细创建售后单并把订单状态置"售后中"
    ActionApplyPostsale,
    # 可售后订单查询动作：检索已签收（status_code >= 330）且未取消的订单
    ActionAskOrderIdAfterDelivered,
    # 售后资格校验动作：状态段（空值或 < 330 拒绝）+ 7 天期限双重门槛
    ActionCheckPostsaleEligible,
    # 售后原因候选查询动作：按商品类别过滤原因库，无候选降级自由输入
    ActionAskPostsaleReason,
)

# 导出清单：from actions import * 时仅暴露以下名字，
# 按字母序排列为 IDE 提供准确的补全与跳转依据。
__all__ = [
    # 售后申请提交
    "ActionApplyPostsale",
    # 统一订单查询
    "ActionAskOrderId",
    # 可售后订单查询
    "ActionAskOrderIdAfterDelivered",
    # 售后原因候选查询
    "ActionAskPostsaleReason",
    # 询问收货城市
    "ActionAskReceiveCity",
    # 询问收货区县
    "ActionAskReceiveDistrict",
    # 收货地址列表
    "ActionAskReceiveId",
    # 询问收货省份
    "ActionAskReceiveProvince",
    # 设置收货信息
    "ActionAskSetReceiveInfo",
    # 取消订单
    "ActionCancelOrder",
    # 售后资格校验
    "ActionCheckPostsaleEligible",
    # 快递公司查询
    "ActionGetLogisticsCompanys",
    # 物流轨迹查询
    "ActionGetLogisticsInfo",
    # 订单详情查询
    "ActionGetOrderDetail",
]
