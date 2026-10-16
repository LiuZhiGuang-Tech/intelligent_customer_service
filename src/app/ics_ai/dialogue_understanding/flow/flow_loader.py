"""
flow_loader - Flow 加载器

核心功能：把 flows 配置（YAML 文件 / 目录 / YAML 字符串）加载并解析为内存中的
          FlowsList 对象——单个流程的对象化委托 flow.py 的 Flow.from_dict 完成，
          对外提供 load / load_from_string 两个方法及两个便捷函数。

执行流程：load() 先按路径类型分流——文件走 _load_from_file（读 YAML → 解析）；
          目录走 _load_from_directory（glob 匹配全部 .yml/.yaml → 逐个读取并
          合并）；两条路径最终都汇入 _parse_flows_data 统一解析：兼容
          "flows: {...}" 与"顶层直接是流程字典"两种格式，跳过 version/metadata/
          imports 等非流程键，逐流程构建 Flow 对象后组装为 FlowsList。

副作用：1. 读取磁盘 YAML 文件（只读，不写文件/数据库、不发起网络请求）；
        2. 经 logger 输出 debug/warning/error 日志；
        3. 目录加载时单个文件失败仅记 error 日志后继续，不中断、不向调用方
           抛出，因此可能返回"部分加载"的结果。

主要依赖：app.ics_ai.dialogue_understanding.flow.flow（Flow/FlowsList）、
          app.ics_ai.shared.yaml_loader（read_yaml_file/read_yaml_string）、
          app.ics_ai.shared.exceptions（ConfigurationException）。
"""

# 从 __future__ 导入 annotations：启用 PEP 563 延迟注解求值，
# 注解以字符串形式保存，可用 PEP 604 的 X | Y 写法而不受运行时版本限制。
from __future__ import annotations

# 导入 logging：模块级日志器，用于记录加载过程与异常。
import logging

# 导入 Path：面向对象的路径操作（判断文件/目录、glob 匹配）。
from pathlib import Path

# 导入类型注解工具：仅保留 Any（未弃用且无内置替代）；
# Union/Dict/List/Optional 已按 PEP 604/585 迁移为 str | Path、
# dict[str, Any] 等写法，List/Optional 因全文件未使用一并删除。
from typing import Any

# 导入 Flow 与 FlowsList：解析目标——Flow 承载单个流程，FlowsList 汇总全部流程。
from app.ics_ai.dialogue_understanding.flow.flow import Flow, FlowsList

# 导入配置异常：路径不存在时抛出，提示调用方配置有误。
from app.ics_ai.shared.exceptions import ConfigurationException

# 导入 YAML 读取函数：read_yaml_file 读单个文件（read_yaml_string 仅在
# load_from_string 内局部使用，故不在此声明）。
from app.ics_ai.shared.yaml_loader import read_yaml_file

# 模块级日志器：按当前模块名命名，日志来源可精确定位到本文件。
logger = logging.getLogger(__name__)


class FlowLoader:
    """Flow加载器。

    从YAML文件加载Flow定义。

    支持的文件格式：
    - 单个flows.yml文件
    - flows目录下的多个YAML文件

    职责：按路径读取 flows 配置并解析为 FlowsList；统一处理文件/目录分流、
          格式兼容与错误降级（单文件失败不影响整体加载）。
    设计模式：无状态服务类——所有方法只依赖入参，不持有实例状态，
              因此同一实例可安全复用；另配 load_flows / load_flows_from_string
              两个便捷函数封装"创建实例 + 调用"。
    生命周期：由调用方按需创建（便捷函数内即用即建），加载完成后即可丢弃；
              无状态故创建与销毁成本极低。
    """

    def __init__(self):
        """初始化加载器。

        业务流程：空实现——加载逻辑不依赖任何实例状态，全部输入来自方法入参。

        副作用：无。
        """
        # 无需初始化字段：本类无实例状态，所有方法仅使用入参。
        pass

    def load(self, path: str | Path) -> FlowsList:
        """加载Flow。

        业务流程：1. 将入参统一转为 Path（兼容字符串与 Path 两种入参）；
        2. 是文件→委托 _load_from_file；3. 是目录→委托 _load_from_directory；
        4. 两者都不是（路径不存在）→抛 ConfigurationException。

        副作用：读取磁盘 YAML 文件；记录日志；路径不存在时抛异常。

        参数：
            path: 文件或目录路径，str 或 Path，必须真实存在；
                  不存在的路径会抛异常而非返回空结果。

        返回：
            FlowsList 实例；配置为空/格式非法时返回空 FlowsList
            （不抛异常）；路径不存在时抛 ConfigurationException，不返回。
        """
        # 统一转为 Path 对象：后续 is_file/is_dir/glob 均为 Path 提供的方法，
        # 一次转换规避 str/Path 双分支处理。
        path = Path(path)

        # 分支1：是文件——按单文件流程加载。
        if path.is_file():
            return self._load_from_file(path)
        # 分支2：是目录——按目录批量加载（内部合并为单个 FlowsList）。
        elif path.is_dir():
            return self._load_from_directory(path)
        # 分支3：既非文件也非目录——路径不存在或无权限，属配置错误，
        # 直接抛出异常提示调用方（返回空集合会掩盖配置问题）。
        else:
            raise ConfigurationException(f"Flow path not found: {path}")

    def _load_from_file(self, file_path: Path) -> FlowsList:
        """从单个文件加载。

        业务流程：1. 记录 debug 日志；2. 读取 YAML（读取失败/空文件按空处理）；
        3. 数据为空（None 或空字典）时记 warning 并返回空 FlowsList；
        4. 否则委托 _parse_flows_data 解析。

        副作用：读取磁盘 YAML 文件；输出 debug/warning 日志。

        参数：
            file_path: 文件路径，Path 类型，调用方（load）已确保其存在。

        返回：
            FlowsList 实例；文件为空或内容非法（read_yaml_file 返回假值）时
            返回空 FlowsList，不抛异常。
        """
        # 记录加载来源，便于排查"为什么某个流程没被加载"。
        logger.debug(f"Loading flows from file: {file_path}")

        # 读取 YAML：转成 str 后交给共享工具，返回解析结果（失败时通常为 None）。
        data = read_yaml_file(str(file_path))
        # 空值兜底：None（读取/解析失败）或空字典（文件无有效内容）都视为无流程，
        # 记 warning 提示配置可疑但不影响启动。
        if not data:
            logger.warning(f"Empty or invalid flow file: {file_path}")
            return FlowsList()

        # 内容有效：进入统一解析流程。
        return self._parse_flows_data(data)

    def _load_from_directory(self, dir_path: Path) -> FlowsList:
        """从目录加载。

        业务流程：1. 记录 debug 日志；2. glob 匹配目录下所有 .yml 与 .yaml 文件；
        3. 无匹配文件时记 warning 并返回空 FlowsList；4. 逐个文件调用
        _load_from_file，把其中所有 Flow 追加到结果容器；
        5. 单个文件异常仅记 error 日志，继续处理后续文件。

        副作用：读取磁盘 YAML 文件；输出 debug/warning/error 日志；
                单文件失败被吞掉（不抛出），可能返回部分加载的结果。

        参数：
            dir_path: 目录路径，Path 类型，调用方（load）已确保其存在。

        返回：
            FlowsList 实例；目录无 YAML 文件时返回空 FlowsList；
            部分文件加载失败时返回成功加载的那部分（不抛异常）。
        """
        # 记录加载来源目录。
        logger.debug(f"Loading flows from directory: {dir_path}")

        # 查找所有YAML文件：分别匹配 .yml 与 .yaml 两种后缀再拼接，
        # 避免 glob 单个模式无法同时覆盖（glob 不支持花括号扩展）。
        yaml_files = list(dir_path.glob("*.yml")) + list(dir_path.glob("*.yaml"))

        # 空目录兜底：没有任何 YAML 文件时记 warning 返回空集合，
        # 避免后续空循环产生"静默无流程"的困惑。
        if not yaml_files:
            logger.warning(f"No YAML files found in: {dir_path}")
            return FlowsList()

        # 加载所有文件：结果容器，跨文件汇总（add_flow 会同步维护 ID 索引）。
        all_flows = FlowsList()

        # 逐文件加载：一个文件可含多个流程，全部并入结果容器。
        for yaml_file in yaml_files:
            try:
                # 复用单文件加载逻辑（含空值兜底与解析）。
                flows_list = self._load_from_file(yaml_file)
                # 遍历该文件解析出的流程，逐个登记到结果容器。
                for flow in flows_list:
                    all_flows.add_flow(flow)
            # 单文件失败（读取异常/解析异常）不中断整体加载：
            # 记 error 日志后继续，保证其余流程文件仍可用。
            except Exception as e:
                logger.error(f"Failed to load flow file {yaml_file}: {e}")

        # 返回汇总结果（可能因个别文件失败而少于目录实际流程数）。
        return all_flows

    def _parse_flows_data(self, data: dict[str, Any]) -> FlowsList:
        """解析Flow数据。

        业务流程：1. 取 flows 键作为流程字典，缺失时回退为整个 data（兼容
        "顶层直接是流程字典"的写法）；2. 非字典则记 warning 返回空 FlowsList；
        3. 逐键遍历：跳过 version/metadata/imports 等非流程键，跳过值不是
        字典的键（记 warning）；4. 对每个合法流程调用 Flow.from_dict 构建对象，
        单个失败仅记 error 日志并继续；5. 组装 FlowsList 返回。

        副作用：输出 warning/error/debug 日志；不读写文件。

        参数：
            data: YAML 数据，dict 类型；允许为嵌套/平铺两种流程格式。

        返回：
            FlowsList 实例；flows 值非字典时返回空 FlowsList；
            个别流程解析失败时跳过该流程、其余正常返回（不抛异常）。
        """
        # 流程对象暂存列表：解析完成后一次性交给 FlowsList 建索引。
        flows = []

        # 支持两种格式：
        # 1. data = flows: { flow_id: {...}, ... }
        # 2. 直接是 data = { flow_id: {...}, ... }
        # get 的第二个参数为默认值：无 flows 键时回退为整个 data，
        # 从而同时兼容 Rasa 风格与简化写法。
        flows_data = data.get("flows", data)

        # 类型兜底：flows 值不是字典（如写成列表/字符串）时无法遍历，
        # 记 warning 并返回空集合。
        if not isinstance(flows_data, dict):
            logger.warning(f"Invalid flows data format: {type(flows_data)}")
            return FlowsList()

        # 逐流程解析：键为流程 ID，值为该流程的配置字典。
        for flow_id, flow_config in flows_data.items():
            # 跳过非Flow字段：顶层杂项键（版本/元数据/导入）不是流程定义，
            # 若不跳过会因缺少 steps 被误判为非法流程。
            if flow_id in ("version", "metadata", "imports"):
                continue

            # 值类型兜底：配置不是字典时无法解析，记 warning 后跳过该键，
            # 不影响其他流程。
            if not isinstance(flow_config, dict):
                logger.warning(
                    f"Invalid flow config for {flow_id}: {type(flow_config)}"
                )
                continue

            try:
                # 委托 Flow.from_dict 完成对象化（步骤解析、next 自动串联、
                # 步骤索引构建均由该方法内部完成）。
                flow = Flow.from_dict(flow_id, flow_config)
                # 收集解析成功的流程。
                flows.append(flow)
                # debug 日志记录流程 ID 与步骤数，便于核对加载结果。
                logger.debug(f"Loaded flow: {flow_id} with {len(flow.steps)} steps")
            # 单个流程失败不拖垮整体：记 error 日志后继续解析后续流程。
            except Exception as e:
                logger.error(f"Failed to parse flow {flow_id}: {e}")

        # 组装并返回：FlowsList 构造时会建立"流程ID → Flow"索引。
        return FlowsList(flows=flows)

    def load_from_string(self, yaml_string: str) -> FlowsList:
        """从YAML字符串加载。

        业务流程：1. 局部导入 read_yaml_string；2. 解析字符串为数据；
        3. 空数据直接返回空 FlowsList；4. 否则委托 _parse_flows_data 解析。

        副作用：不读文件；解析失败时可能输出 warning/error 日志。

        参数：
            yaml_string: YAML 字符串，内容格式与文件一致；
                         空串/非法 YAML 时按空数据处理。

        返回：
            FlowsList 实例；字符串为空或解析失败时返回空 FlowsList，
            不抛异常。
        """
        # 局部导入：read_yaml_string 仅本方法使用，放在函数内可缩小模块级依赖。
        from app.ics_ai.shared.yaml_loader import read_yaml_string

        # 解析 YAML 字符串（失败时通常返回 None）。
        data = read_yaml_string(yaml_string)
        # 空值兜底：无有效内容时直接返回空集合，省去后续判空分支。
        if not data:
            return FlowsList()

        # 内容有效：与文件加载走同一套解析逻辑，保证两种入口行为一致。
        return self._parse_flows_data(data)


# 便捷函数


def load_flows(path: str | Path) -> FlowsList:
    """便捷函数：加载Flow。

    业务流程：创建 FlowLoader 实例并调用其 load 方法。

    副作用：同 FlowLoader.load——读取磁盘 YAML；路径不存在时抛
            ConfigurationException。

    参数：
        path: 文件或目录路径，str 或 Path，必须真实存在。

    返回：
        FlowsList 实例；解析为空时返回空集合；路径不存在时抛异常，不返回。
    """
    # 无状态类：即用即建，无需缓存实例。
    loader = FlowLoader()
    # 委托实例方法完成实际加载。
    return loader.load(path)


def load_flows_from_string(yaml_string: str) -> FlowsList:
    """便捷函数：从字符串加载Flow。

    业务流程：创建 FlowLoader 实例并调用其 load_from_string 方法。

    副作用：同 FlowLoader.load_from_string——不读文件，仅可能记录日志。

    参数：
        yaml_string: YAML 字符串，空串/非法内容按空数据处理。

    返回：
        FlowsList 实例；无有效内容时返回空集合，不抛异常。
    """
    # 无状态类：即用即建。
    loader = FlowLoader()
    # 委托实例方法完成实际解析。
    return loader.load_from_string(yaml_string)


# 导出清单：from flow_loader import * 时仅暴露以下三个名字。
__all__ = [
    # 列表项1：加载器类本体。
    "FlowLoader",
    # 列表项2：按路径加载的便捷函数。
    "load_flows",
    # 列表项3：按字符串加载的便捷函数。
    "load_flows_from_string",
]
