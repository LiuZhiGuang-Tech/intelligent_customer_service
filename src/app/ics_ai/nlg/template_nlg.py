"""
template_nlg - 模板 NLG 生成器模块

核心功能：NLGGenerator 的模板实现——从 Domain 的 responses 中按
          响应动作名取模板（同一动作可配多条变体，随机选取实现
          回复多样化），对模板文本做 {slot_name} 槽位变量替换后
          产出 NLGResponse（文本/按钮/图片/自定义/元数据五字段
          透传自模板）。

执行流程：
    1. 装配期：agent.py 按 NLG 配置创建 TemplateNLG（可选注入
       NLGConfig，本实现为纯模板路径，不依赖 LLM）；
    2. 生成期：动作层以 utter_action + tracker 调用 generate——
       domain 缺失或模板缺失均返回空文本响应（降级不抛错）；
    3. 变量替换期：_fill_slots 以正则扫描 {slot_name} 占位符，
       命中槽位则替换为槽位值字符串，未命中保持占位符原样。

副作用：无——本模块只读 tracker 槽位与 Domain 模板，不修改任何
        状态、不读写文件、不发起网络请求。

主要依赖：
    - random：多模板变体的随机选取；
    - re：槽位占位符的扫描与替换；
    - app.ics_ai.nlg.nlg_generator：NLGGenerator 基类与
      NLGResponse/NLGConfig DTO；
    - app.ics_ai.core.tracker / core.domain：DialogueStateTracker
      与 Domain（仅注解用，TYPE_CHECKING 隔离）。
"""

# 启用 PEP 563 延迟注解求值：注解以字符串形式保存，TYPE_CHECKING
# 分支中的类型前向引用无需引号，PEP 604 联合类型注解不受运行时
# 版本限制。
from __future__ import annotations

# 导入随机模块：generate 中多模板变体的随机选取（回复多样化）。
import random

# 导入正则引擎：_fill_slots 中 {slot_name} 占位符的扫描与替换。
import re

# 导入类型注解工具：Any 类型通配符（PEP 585/604 新式注解下
# Dict/Optional 已不需要）。
from typing import Any, TYPE_CHECKING

# 从 NLG 基类模块导入：抽象基类（本类是其模板路径实现）与
# 响应/配置 DTO。
from app.ics_ai.nlg.nlg_generator import NLGGenerator, NLGConfig, NLGResponse

# 类型检查期专属导入：这些类型只出现在方法注解中，运行时不导入
# （避免 NLG 包与 core 包在导入期相互依赖）。
if TYPE_CHECKING:
    # 对话状态追踪器：槽位变量替换的数据源（仅注解用）。
    from app.ics_ai.core.tracker import DialogueStateTracker

    # 业务域定义：响应模板的静态来源（仅注解用）。
    from app.ics_ai.core.domain import Domain


class TemplateNLG(NLGGenerator):
    """模板 NLG 生成器。

    职责：从 Domain 的 responses 中按 utter_action 取响应模板
          （ResponseTemplate 列表），随机选取变体并做槽位变量
          替换，生成结构化回复——纯模板路径，无 LLM 依赖。
    设计模式：策略模式——NLGGenerator 的具体策略之一（与
              ResponseRephraser 的 LLM 改写路径并列，可按配置
              切换）。
    生命周期：随 agent 装配创建，跨多轮对话复用（无状态，随机
              选取不依赖历史）。

    使用方（构造/调用位置）：
        - agent.py load()：NLG 配置未启用 LLM 时的默认生成器；
        - 动作层：utter_* 动作经 generate 获取回复文本。
    """

    async def generate(
        self,
        utter_action: str,
        tracker: DialogueStateTracker,
        domain: Domain | None = None,
        **kwargs: Any,
    ) -> NLGResponse:
        """按响应动作名生成模板回复。

        业务流程：1. domain 缺失 → 返回空文本响应（无模板来源）；
                  2. 按 utter_action 查模板列表，空列表 → 空文本
                  响应（动作名未配置，降级不抛错）；3. 随机选取
                  一条变体（同一响应名多条模板时回复多样化）；
                  4. 取模板文本（None 兜底空串）并做槽位变量替换；
                  5. 组装 NLGResponse——模板的按钮/图片/自定义/
                  元数据四字段原样透传（各自 None 兜底为空值）。

        副作用：无（只读 domain 模板与 tracker 槽位）。

        参数：
            utter_action: 响应动作名称（如 utter_greet，对应
                          Domain responses 的键）。
            tracker: 对话状态追踪器——槽位变量替换的数据源。
            domain: 业务域定义，允许 None（缺失时返回空响应）。
            **kwargs: 额外参数（本实现不消费，仅满足基类契约）。

        返回：
            NLGResponse——模板未命中时 text 为空串的降级响应；
            正常路径携带模板全部内容。
        """
        # 降级一：无 Domain 则无模板来源，返回空文本响应。
        if domain is None:
            return NLGResponse(text="")

        # 按响应名查模板列表（Domain.get_response 缺省返回空列表）。
        templates = domain.get_response(utter_action)
        # 降级二：动作名未配置任何模板，返回空文本响应。
        if not templates:
            return NLGResponse(text="")

        # 随机选取一条变体：同一响应名的多条模板随机命中，避免
        # 机器人每次回复同一句（config.use_llm 在本实现不消费，
        # 纯模板路径始终走此分支）。
        template = random.choice(templates)

        # 取模板文本：text 允许 None（ResponseTemplate 定义），
        # 兜底为空串供后续替换。
        text = template.text or ""

        # 槽位变量替换：{slot_name} 占位符 → tracker 槽位值。
        text = self._fill_slots(text, tracker)

        # 组装统一响应 DTO：模板的四个附加字段原样透传
        # （buttons/image/custom/metadata 的 None 均兜底为空值）。
        response = NLGResponse(
            text=text,
            buttons=template.buttons or [],
            image=template.image,
            custom=template.custom,
            metadata=template.metadata or {},
        )

        return response

    def _fill_slots(
        self,
        text: str,
        tracker: DialogueStateTracker,
    ) -> str:
        """替换文本中的槽位变量占位符。

        业务流程：1. 空文本直接返回；2. 一次性取全量槽位快照
                  （tracker.get_all_slots，替换过程中值不再变动）；
                  3. 正则扫描 {slot_name} 占位符，回调函数逐个
                  判定——槽位命中则替换为值的字符串形式，未命中
                  保持占位符原样（未配置槽位的问题以可见形式暴露，
                  不静默吞掉）。

        副作用：无（纯字符串变换）。

        参数：
            text: 含 {slot_name} 占位符的原始模板文本。
            tracker: 对话状态追踪器——槽位值来源。

        返回：
            替换后的文本（未命中占位符保留原样）。
        """
        # 空文本短路：无占位符可替换，避免无谓的正则开销。
        if not text:
            return text

        # 一次性取全量槽位快照：后续替换只查字典，不再触碰 tracker。
        slots = tracker.get_all_slots()

        # 占位符正则（逐段拆解）：
        #   \{        ——字面量左花括号（占位符起始定界符）
        #   (\w+)     ——捕获组 1：槽位名，\w 匹配字母/数字/下划线
        #               （与槽位命名约定一致，覆盖 user_name 等）
        #   \}        ——字面量右花括号（占位符结束定界符）
        pattern = r"\{(\w+)\}"

        def replace_slot(match: re.Match[str]) -> str:
            """单次占位符命中的替换回调（re.sub 逐命中调用）。

            参数：
                match: 正则命中对象——group(1) 为槽位名，
                       group(0) 为完整占位符原文。

            返回：
                命中槽位 → 槽位值的字符串形式；未命中 → 占位符
                原文（保持可见，便于发现未配置槽位）。
            """
            # 取捕获组 1：占位符内的槽位名（不含花括号）。
            slot_name = match.group(1)
            # 查槽位值（缺省 None 表示该槽位未设置）。
            value = slots.get(slot_name)
            # 命中判定：值非 None 才替换——显式以 None 存储的槽位
            # 与未设置槽位同等处理，均保持占位符原样。
            if value is not None:
                # 槽位值统一转字符串（数字/布尔等类型安全展开）。
                return str(value)
            # 未命中：返回 group(0) 即占位符原文 {slot_name}。
            return match.group(0)

        # 执行替换：re.sub 扫描全部命中并逐个调用回调返回替换文本。
        return re.sub(pattern, replace_slot, text)


# 包级公开名单：本模块对外仅暴露 TemplateNLG 生成器类。
__all__ = ["TemplateNLG"]
