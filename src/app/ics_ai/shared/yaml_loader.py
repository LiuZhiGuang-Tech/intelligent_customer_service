"""
yaml_loader - YAML 文件安全加载器模块

功能：提供 YAML 文件的安全加载/解析/写入/合并能力，防止 YAML 注入攻击。
     支持单文档、多文档（--- 分隔）、目录合并三种加载模式。
     自定义 SafeLineLoader 保留行号信息，便于错误定位。

执行流程：
    1. SafeLineLoader 继承 yaml.SafeLoader，注册自定义 _construct_mapping
       以保留节点行号（flatten_mapping + construct_pairs）
    2. read_yaml_file 包装 open + yaml.safe_load，返回 dict | None
    3. read_yaml_files 批量读取多文件并过滤 None 结果
    4. merge_yaml_files 逐文件 read_yaml_file + _deep_merge 深度合并
       （后面文件覆盖前面文件同名键）
    5. write_yaml_file / dump_yaml_string 负责序列化输出

副作用：
    - read_yaml_file / read_yaml_multi_document：读取本地磁盘 YAML 文件
    - write_yaml_file：写入本地磁盘 YAML 文件（创建父目录）
    - write_yaml_file / dump_yaml_string：都依赖 PyYAML 第三方库
    - 无全局状态修改、无网络访问、无数据库操作

主要依赖：
    - pathlib.Path：路径处理
    - collections.abc.Sequence：协变序列类型（函数入参用，接受 list/tuple）
    - typing.Any：类型通配符
    - yaml：PyYAML 库，YAML 解析引擎
    - app.ics_ai.shared.constants.DEFAULT_ENCODING：统一文件编码常量
"""

from __future__ import annotations  # 延迟求值所有类型注解，支持前向引用

from collections.abc import Sequence  # 协变序列（比 list 更通用的函数入参类型）
from pathlib import Path  # 面向对象的路径处理
from typing import Any  # 类型通配符，保留自 typing（无内置替代）

import yaml  # PyYAML 第三方库，YAML 解析引擎

from app.ics_ai.shared.constants import (
    DEFAULT_ENCODING,
)  # 统一文件编码常量（如 "utf-8"）


class SafeLineLoader(yaml.SafeLoader):
    """安全的 YAML 加载器，保留行号信息

    职责：继承 yaml.SafeLoader（防 YAML 注入，不解析 Python 对象标签），
         通过自定义 construct_mapping 保留每个节点的行号，便于错误定位。
    设计模式：装饰器/注册器模式（add_constructor 将 _construct_mapping
             注入到 SafeLoader 的 mapping 构建流程中）。
    生命周期：模块导入时一次性注册构造函数，之后全局复用，不实例化。
    """

    pass  # 空类：仅作为 SafeLoader 的子类身份标记，所有行为由父类 + 注册的构造函数提供


def _construct_mapping(
    loader: SafeLineLoader, node: yaml.MappingNode
) -> dict[str, Any]:
    """构造映射时保留行号信息（私有辅助函数）

    核心功能：flatten_mapping 处理合并键（<<），construct_pairs 按节点顺序
             构造 (key, value) 对，最后组装成 dict 返回。
    业务流程：三步 → flatten → construct_pairs → dict()。
    副作用：修改 loader 内部状态（flatten_mapping 会修改 node）。

    @param loader: SafeLineLoader 实例；由 yaml 解析引擎自动传入。
    @param node: yaml.MappingNode（YAML 映射节点，对应 {key: value} 块或行内对象）；
                 PyYAML 运行时保证传入 MappingNode（因为本函数注册给 DEFAULT_MAPPING_TAG）。
    @return: 解析后的 dict[str, Any]。
    """
    loader.flatten_mapping(
        node
    )  # 处理 YAML 的合并键语法 <<:，将合并节点的内容展开到当前 node
    pairs = loader.construct_pairs(node)  # 按节点有序构造 (key, value) 列表，保留行号
    return dict(pairs)  # 组装成普通字典返回


# 模块导入时立即注册：将 _construct_mapping 替换 SafeLoader 默认的 mapping 构造函数
SafeLineLoader.add_constructor(  # 类方法调用，修改类级别的构造函数表
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,  # 默认 YAML 映射标签 "tag:yaml.org,2002:map"
    _construct_mapping,  # 我们的自定义构造函数
)


def read_yaml_file(path: str | Path) -> dict[str, Any] | None:
    """读取单个 YAML 文件

    核心功能：安全地读取 YAML 文件，返回解析后的字典；空文件返回 None。
    业务流程：
        1. Path(path) 统一 str/Path 入参
        2. 检查文件存在性，不存在抛 FileNotFoundError
        3. open(..., encoding=DEFAULT_ENCODING) 打开
        4. yaml.safe_load 解析（防注入，不执行 !!python/object 等标签）
    副作用：读取本地磁盘文件。

    @param path: YAML 文件路径；str 或 Path；不能为空。
    @return: 解析后的 dict[str, Any]；文件为空或仅含注释时返回 None。
    @raises FileNotFoundError: 路径不存在时抛出。
    @raises yaml.YAMLError: YAML 语法错误时抛出。
    """
    path = Path(path)  # 统一转换为 Path 对象，兼容 str/Path 入参

    if not path.exists():  # 文件不存在 → 立即抛异常
        raise FileNotFoundError(f"YAML文件不存在: {path}")

    # 以只读 + UTF-8 编码打开文件，保证跨平台一致性
    with open(path, "r", encoding=DEFAULT_ENCODING) as f:  # with 保证文件句柄关闭
        content = yaml.safe_load(f)  # 安全解析：拒绝 !!python/object 等危险标签

    return content  # safe_load 对空文件返回 None，调用方需判空


def read_yaml_string(yaml_string: str) -> dict[str, Any] | None:
    """解析 YAML 字符串

    核心功能：将内存中的 YAML 字符串解析为字典；空字符串返回 None。
    业务流程：直接调用 yaml.safe_load 解析，无磁盘 I/O。
    副作用：无（纯计算）。

    @param yaml_string: YAML 格式的字符串；不能为空。
    @return: 解析后的 dict[str, Any]；空字符串或仅含注释时返回 None。
    @raises yaml.YAMLError: YAML 语法错误时抛出。
    """
    return yaml.safe_load(yaml_string)  # 直接解析字符串，空串返回 None


def read_yaml_files(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """批量读取多个 YAML 文件

    核心功能：遍历路径列表逐个调用 read_yaml_file，过滤掉空文件（返回 None），
             只收集成功解析的字典。
    业务流程：for 循环 → read_yaml_file → if content is not None 则 append。
    副作用：读取多个本地磁盘文件。

    @param paths: YAML 文件路径序列（list/tuple 均可，Sequence 是协变接口）；
                  不能为空（空序列返回空列表）。
    @return: 成功解析的 dict[str, Any] 列表；空文件或失败文件会被过滤；
             若所有文件都为空则返回空列表。
    """
    results: list[dict[str, Any]] = []  # 初始化结果列表
    for path in paths:  # 遍历每个路径
        content = read_yaml_file(
            path
        )  # 调用单文件读取（内部处理 str/Path 转换和存在性检查）
        if content is not None:  # 仅当文件非空时才收集（空文件 safe_load 返回 None）
            results.append(content)  # 追加到结果列表
    return results  # 返回过滤后的结果列表


def read_yaml_multi_document(path: str | Path) -> list[dict[str, Any]]:
    """读取多文档 YAML 文件

    核心功能：解析包含 `---` 分隔符的多文档 YAML 文件，每个分隔段解析为独立字典。
    业务流程：
        1. Path(path) 标准化 + 存在性检查
        2. yaml.safe_load_all 返回生成器 → list() 全部消费
        3. 列表推导式过滤 None 文档（空段）
    副作用：读取本地磁盘文件。

    @param path: 多文档 YAML 文件路径；str 或 Path；不能为空。
    @return: 解析后的 dict[str, Any] 列表；空段（仅 ---）被过滤；
             文件为空时返回空列表。
    @raises FileNotFoundError: 路径不存在时抛出。
    @raises yaml.YAMLError: YAML 语法错误时抛出。
    """
    path = Path(path)  # 统一转换为 Path 对象

    if not path.exists():  # 文件不存在 → 抛异常
        raise FileNotFoundError(f"YAML文件不存在: {path}")

    # 以只读 + UTF-8 打开文件，用 safe_load_all 逐文档解析（返回生成器）
    with open(path, "r", encoding=DEFAULT_ENCODING) as f:  # with 保证文件句柄关闭
        documents = list(
            yaml.safe_load_all(f)
        )  # 消费生成器，得到所有文档（含 None 空段）

    # 列表推导式过滤掉 None（safe_load_all 对空文档段返回 None）
    return [doc for doc in documents if doc is not None]


def write_yaml_file(
    data: dict[str, Any],  # 待序列化的字典
    path: str | Path,  # 目标文件路径
    allow_unicode: bool = True,  # 是否允许 Unicode 字符；默认 True（输出中文原样）
    default_flow_style: bool = False,  # 是否使用流式风格（单行内联）；默认 False（块风格多行）
) -> None:
    """写入 YAML 文件

    核心功能：将 dict 序列化为 YAML 格式写入磁盘，自动创建父目录。
    业务流程：
        1. Path(path) 标准化
        2. path.parent.mkdir 创建父目录（exist_ok 允许已存在）
        3. open(..., "w") 打开 → yaml.dump 序列化
    副作用：写入本地磁盘文件（覆盖已存在文件），可能创建目录。

    @param data: 要序列化的字典；不能为空（空字典生成空 YAML 文档）。
    @param path: 目标文件路径；str 或 Path；父目录若不存在会自动创建。
    @param allow_unicode: True 输出 Unicode 原样（中文可读），False 用 \\u 转义；
                          默认 True。
    @param default_flow_style: True 输出单行内联（压缩大文件），False 输出块风格多行（人类可读）；
                               默认 False。
    @return: None
    """
    path = Path(path)  # 统一转换为 Path 对象

    # 确保父目录存在：parents=True 允许递归创建多层；exist_ok=True 已存在不报错
    path.parent.mkdir(parents=True, exist_ok=True)

    # 以写入 + UTF-8 编码打开文件，yaml.dump 写入
    with open(path, "w", encoding=DEFAULT_ENCODING) as f:  # with 保证文件句柄关闭
        yaml.dump(  # PyYAML 序列化
            data,  # 待序列化字典
            f,  # 写入目标文件流
            allow_unicode=allow_unicode,  # Unicode 模式开关
            default_flow_style=default_flow_style,  # 输出风格开关
            sort_keys=False,  # 保留原始键顺序（Python 3.7+ dict 有序）
        )


def dump_yaml_string(
    data: dict[str, Any],  # 待序列化的字典
    allow_unicode: bool = True,  # 是否允许 Unicode 字符；默认 True
    default_flow_style: bool = False,  # 是否使用流式风格；默认 False
) -> str:
    """将字典转换为 YAML 字符串（不落盘）

    核心功能：与 write_yaml_file 同参数，但返回字符串而非写文件。
    业务流程：直接调用 yaml.dump，目标写 StringIO 内部缓冲，返回 str。
    副作用：无（纯计算）。

    @param data: 要序列化的字典；不能为空。
    @param allow_unicode: True 输出 Unicode 原样；默认 True。
    @param default_flow_style: True 单行内联，False 多行块风格；默认 False。
    @return: YAML 格式字符串（末尾可能带 \\n，PyYAML 默认行为）。
    """
    return yaml.dump(  # 直接返回字符串，无需 open 文件
        data,  # 待序列化字典
        allow_unicode=allow_unicode,  # Unicode 模式开关
        default_flow_style=default_flow_style,  # 输出风格开关
        sort_keys=False,  # 保留原始键顺序
    )


def merge_yaml_files(paths: Sequence[str | Path]) -> dict[str, Any]:
    """深度合并多个 YAML 文件

    核心功能：按给定顺序逐个读取 YAML 文件，深度合并到 accumulator 字典。
             **后面文件覆盖前面文件的同名键**，嵌套字典递归合并。
    业务流程：
        1. 初始化空结果 dict
        2. for 循环逐个 read_yaml_file → if content（非空）→ _deep_merge
        3. 返回最终合并结果
    副作用：读取多个本地磁盘文件（调用 read_yaml_file）。

    @param paths: YAML 文件路径序列（list/tuple 均可，Sequence 协变接口，接受 str | Path 的任意子类集合）；
                  不能为空（空序列返回空字典）。
    @return: 合并后的 dict[str, Any]；若所有文件都为空则返回空字典。
    """
    result: dict[str, Any] = {}  # 初始化空字典作为合并累积器

    for path in paths:  # 逐个处理路径列表
        content = read_yaml_file(
            path
        )  # 读取单个文件（内部处理 str/Path 转换 + 存在性检查）
        if content:  # 空文件 safe_load 返回 None → 跳过；空字典 {} 也会进入（因为 bool({}) 为 False）
            result = _deep_merge(
                result, content
            )  # 深度合并：result 是 base，content 覆盖它

    return result  # 返回最终合并后的完整字典


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """深度合并两个字典（私有辅助函数）

    核心功能：override 中的值覆盖 base 中的值，**但当两者都是 dict 时递归合并**。
    业务流程：
        1. result = base.copy()（浅拷贝，避免修改传入的 base）
        2. 遍历 override 的所有 (key, value)
        3. 若 result 中已存在该 key 且两边值都是 dict → 递归调用 _deep_merge
        4. 否则直接赋值覆盖
    副作用：无（纯函数），不修改 base 也不修改 override。

    @param base: 基础字典（被合并的基底）；不能为空（空 dict 时所有键来自 override）。
    @param override: 覆盖字典（值优先级更高）；不能为空。
    @return: 合并后的新 dict[str, Any]；base 和 override 均不被修改。
    """
    result = base.copy()  # 浅拷贝：避免污染调用方传入的 base 字典

    for key, value in override.items():  # 遍历 override 的所有键值对
        # 三重判定：result 中已有 key + result 值是 dict + override 值也是 dict → 递归合并
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)  # 递归：嵌套字典深度合并
        else:  # 任一条件不满足 → 直接覆盖（包括两边都是非 dict 的情况，或 result 中无此 key）
            result[key] = value  # 直接赋值覆盖

    return result  # 返回合并后的新字典
