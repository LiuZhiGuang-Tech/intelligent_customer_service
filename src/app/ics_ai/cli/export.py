"""
export - 导出命令模块

核心功能：提供 `ics export` 命令，将对话机器人项目打包导出为三种
          形态——JSON（仅 domain 结构化导出）、目录（domain/config/
          endpoints/data/actions 按约定布局复制）、压缩包（tar.gz/
          zip，先落临时目录再归档）。

执行流程：1. click 解析 --model（源项目目录）、--output（目标路径）、
          --format（四种输出格式）、--include-data/--include-config
          开关；2. 按格式分派 _export_json / _export_directory /
          _export_archive 三个实现；3. 压缩包模式复用目录导出（先写
          TemporaryDirectory 再 make_archive）；4. 全程异常兜底——
          普通模式打印错误并退出码 1，--debug 透传原始堆栈。

副作用：读源目录的 domain/config/endpoints 文件与 data/actions 目录；
        向 --output 目标写文件/目录/压缩包；归档时创建并清理系统临时
        目录；JSON 模式经 Domain.load 读 yml 文件。

主要依赖：click（命令框架）、shutil/json/tempfile（文件打包）、
          pathlib（路径运算）、app.ics_ai.core.domain（Domain 域
          定义加载，延迟导入）。
"""

# 启用延迟注解求值：类型注解按字符串存储，函数体内延迟导入的
# Domain 等符号无需出现在模块级命名空间。
from __future__ import annotations

# 导入 json：元数据与 domain 结构化导出的 JSON 序列化。
import json

# 导入 logging：创建模块级 logger 记录导出过程异常。
import logging

# 导入 shutil：目录复制（copy2/copytree）与压缩包创建（make_archive）。
import shutil

# 从标准库 datetime 导入 datetime：元数据记录导出时间戳（ISO 格式）。
from datetime import datetime

# 从标准库 pathlib 导入 Path：源/目标/临时路径的对象化运算。
from pathlib import Path

# 导入 tempfile：压缩包模式先写临时目录再归档（自动清理）。
import tempfile

# 导入 click：命令装饰器、参数选项与终端输出。
import click

# 创建模块级 logger：异常兜底分支留痕（debug 透传场景之外）。
logger = logging.getLogger(__name__)


@click.command("export", help="导出模型或配置")
@click.option(
    "--model",
    "-m",
    type=click.Path(exists=True),
    default=".",
    help="模型或项目目录路径",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="输出路径",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["tar.gz", "zip", "dir", "json"]),
    default="tar.gz",
    help="输出格式",
)
@click.option(
    "--include-data/--no-data",
    default=True,
    help="包含训练数据",
)
@click.option(
    "--include-config/--no-config",
    default=True,
    help="包含配置文件",
)
@click.pass_context
def export_command(
    ctx: click.Context,
    model: str,
    output: str,
    format: str,
    include_data: bool,
    include_config: bool,
) -> None:
    """导出模型或配置。

    核心功能：将源项目目录按指定格式导出——JSON（domain 结构化）、
              目录（约定布局复制）、tar.gz/zip 压缩包。

    业务流程：1. 从上下文取 verbose/debug 调试标志；2. 解析源/目标
              路径为 Path 并回显导出参数；3. 按格式四路分派到对应
              实现函数；4. 成功打印完成横幅；异常统一兜底——普通
              模式打印错误退出码 1，debug 模式重新抛出保留堆栈。

    副作用：见各 _export_* 实现函数（写目标路径文件/目录/压缩包）。

    :param ctx: click 上下文（承载主组的 verbose/debug 标志）。
    :param model: 源项目目录路径（exists=True 已由 click 校验存在）。
    :param output: 输出路径（required=True，格式由 --format 决定）。
    :param format: 输出格式，取值 tar.gz/zip/dir/json 四选一。
    :param include_data: 是否携带 data/ 目录（True 携带）。
    :param include_config: 是否携带 config.yml/endpoints.yml（True 携带）。
    :return: 无返回；失败路径抛 SystemExit(1)（debug 时透传原异常）。
    """
    # 读取主组注入的调试标志：verbose 留作细粒度输出扩展位。
    verbose = ctx.obj.get("verbose", False)

    # debug 标志控制异常处理策略：False 转为退出码，True 透传堆栈。
    debug = ctx.obj.get("debug", False)

    # 解析源路径：字符串转 Path 对象（后续按 / 拼接子路径）。
    model_path = Path(model)

    # 解析目标路径：同上，导出实现按需创建父目录。
    output_path = Path(output)

    # 打印导出横幅：分隔线 + 标题。
    click.echo("=" * 50)
    click.echo("ICS AI - 导出")
    click.echo("=" * 50)

    # 回显关键参数：源/目标绝对路径与所选格式（用户可核对）。
    click.echo(f"源路径: {model_path.absolute()}")
    click.echo(f"输出路径: {output_path.absolute()}")
    click.echo(f"格式: {format}")
    click.echo()

    try:
        if format == "json":
            # 分支一：json —— 仅将 domain 导出为结构化 JSON。
            _export_json(model_path, output_path)
        elif format == "dir":
            # 分支二：dir —— 按约定布局复制为目录。
            _export_directory(
                model_path,
                output_path,
                include_data,
                include_config,
            )
        else:
            # 分支三：tar.gz/zip —— 目录导出后归档压缩。
            _export_archive(
                model_path,
                output_path,
                format,
                include_data,
                include_config,
            )

        # 成功收尾：空行分隔后打印完成横幅。
        click.echo()
        click.echo("=" * 50)
        click.echo(f"导出完成: {output_path}")
        click.echo("=" * 50)

    except Exception as e:
        # 异常兜底：打印错误到 stderr；debug 模式重新抛出保留堆栈，
        # 普通模式以退出码 1 结束进程（不静默吞错）。
        click.echo(f"导出失败: {e}", err=True)
        if debug:
            raise
        raise SystemExit(1)


def _export_json(source: Path, output: Path) -> None:
    """导出为JSON格式。

    核心功能：仅加载源目录的 domain 定义并序列化为 JSON 文件。

    业务流程：1. 依次探测 domain.yml/domain.yaml（兼容两种后缀）；
              2. 均不存在则抛 FileNotFoundError；3. Domain.load 加载
              域定义；4. to_dict 转字典后 ensure_ascii=False 写出
              （保留中文原文，缩进 2 美化）。

    副作用：读 domain yml 文件；向 output 写 JSON 文件（父目录自动
            创建）。

    :param source: 源项目目录（domain 文件所在）。
    :param output: JSON 输出文件路径（如 ./export/domain.json）。
    :return: 无返回；domain 缺失时抛 FileNotFoundError。
    """
    # 延迟导入 Domain：JSON 模式独占依赖，避免命令整体加载重模块。
    from app.ics_ai.core.domain import Domain

    # 优先探测 .yml 后缀的 domain 文件。
    domain_file = source / "domain.yml"

    # 回退探测 .yaml 后缀（两种后缀等价兼容）。
    if not domain_file.exists():
        domain_file = source / "domain.yaml"

    # 两种后缀均缺失：明确报错指出源目录，避免静默产出空导出。
    if not domain_file.exists():
        raise FileNotFoundError(f"未找到domain文件: {source}")

    # 提示加载进度。
    click.echo("加载domain...")

    # 经类方法加载域定义：解析 yml 为 Domain 对象（responses/actions 等齐备）。
    domain = Domain.load(str(domain_file))

    # 提示导出进度。
    click.echo("导出JSON...")

    # 确保输出父目录存在：parents=True 级联创建，exist_ok 幂等。
    output.parent.mkdir(parents=True, exist_ok=True)

    # 序列化写出：to_dict 转纯字典；ensure_ascii=False 保留中文；
    # indent=2 美化缩进便于人工查阅。
    with open(output, "w", encoding="utf-8") as f:
        json.dump(domain.to_dict(), f, ensure_ascii=False, indent=2)


def _export_directory(
    source: Path,
    output: Path,
    include_data: bool,
    include_config: bool,
) -> None:
    """导出为目录。

    核心功能：按"机器人项目约定布局"复制源目录要素到目标目录——
              domain 文件、config/endpoints（可选）、data/（可选）、
              actions/，并落 metadata.json 审计记录。

    业务流程：1. 创建目标目录；2. 探测并复制 domain 文件（yml/yaml
              取先命中者）；3. include_config 时复制 config 与
              endpoints；4. include_data 时整树复制 data/；5. actions/
              存在即整树复制；6. 写 metadata.json（时间戳/源路径/版本）。

    副作用：向 output 写目录树与文件；copy2 保留源文件元数据。

    :param source: 源项目目录。
    :param output: 目标目录（已存在则合并写入）。
    :param include_data: True 时复制 data/ 目录。
    :param include_config: True 时复制 config.yml/endpoints.yml。
    :return: 无返回；目录不可写等 IO 异常向上传播。
    """
    # 第一步：创建目标目录（幂等）。
    click.echo("创建导出目录...")
    output.mkdir(parents=True, exist_ok=True)

    # 第二步：复制 domain——遍历两种后缀，命中即复制并终止探测。
    for domain_file in ["domain.yml", "domain.yaml"]:
        # 拼接候选源路径。
        src = source / domain_file

        # 命中即复制：copy2 连同权限/时间戳等元数据一起保留。
        if src.exists():
            click.echo(f"  复制 {domain_file}")
            shutil.copy2(src, output / domain_file)
            break

    # 第三步：复制配置——受 include_config 开关控制。
    if include_config:
        # 探测 config.yml/config.yaml，命中即复制。
        for config_file in ["config.yml", "config.yaml"]:
            # 拼接候选源路径。
            src = source / config_file

            # 命中即复制并跳出探测。
            if src.exists():
                click.echo(f"  复制 {config_file}")
                shutil.copy2(src, output / config_file)
                break

        # 探测 endpoints.yml/endpoints.yaml，命中即复制。
        for endpoints_file in ["endpoints.yml", "endpoints.yaml"]:
            # 拼接候选源路径。
            src = source / endpoints_file

            # 命中即复制并跳出探测。
            if src.exists():
                click.echo(f"  复制 {endpoints_file}")
                shutil.copy2(src, output / endpoints_file)
                break

    # 第四步：复制训练数据——整树复制保留子目录结构。
    if include_data:
        # data 目录存在才复制（缺失时静默跳过，不视为错误）。
        data_dir = source / "data"
        if data_dir.exists():
            click.echo("  复制 data/")
            # copytree 递归复制；dirs_exist_ok 允许目标已存在时合并。
            shutil.copytree(data_dir, output / "data", dirs_exist_ok=True)

    # 第五步：复制自定义动作目录——导出物可部署的关键组成。
    actions_dir = source / "actions"
    if actions_dir.exists():
        click.echo("  复制 actions/")
        # 同 data/：整树递归 + 目标已存在时合并。
        shutil.copytree(actions_dir, output / "actions", dirs_exist_ok=True)

    # 第六步：写元数据——记录导出时间/来源路径/框架版本，便于追溯。
    metadata = {
        # 导出时刻：本地时区 ISO 8601 字符串。
        "exported_at": datetime.now().isoformat(),
        # 源项目绝对路径：还原导出上下文。
        "source": str(source.absolute()),
        # 框架版本：当前固定 0.1.0（与包 __version__ 同步维护）。
        "version": "0.1.0",
    }

    # 以 UTF-8 + 缩进 2 写出元数据文件。
    with open(output / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _export_archive(
    source: Path,
    output: Path,
    format: str,
    include_data: bool,
    include_config: bool,
) -> None:
    """导出为压缩包。

    核心功能：复用目录导出生成完整导出物，再归档为 tar.gz 或 zip。

    业务流程：1. 申请系统临时目录（with 块退出自动清理）；2. 先将
              导出物写入临时目录；3. 按格式调用 make_archive 归档到
              目标路径；4. 归档产物名与预期不符时移动纠正。

    副作用：创建并清理系统临时目录；向 output 写压缩包文件。

    :param source: 源项目目录。
    :param output: 压缩包目标路径（文件名含 .tar.gz/.zip 后缀）。
    :param format: 归档格式，取值 tar.gz 或 zip（json/dir 已上游分派）。
    :param include_data: 透传 _export_directory 的数据开关。
    :param include_config: 透传 _export_directory 的配置开关。
    :return: 无返回；格式不匹配 tar.gz/zip 时静默结束（上游 Choice
             已约束，防御性分支）。
    """
    # 临时目录上下文：块内创建、块退出自动清理，不留中间产物。
    with tempfile.TemporaryDirectory() as temp_dir:
        # 拼接临时导出根：所有目录导出物先落在这里。
        temp_path = Path(temp_dir) / "export"

        # 复用目录导出：与 --format dir 完全一致的产物布局。
        _export_directory(source, temp_path, include_data, include_config)

        # 确保压缩包父目录存在（用户给的 output 可能带深层路径）。
        output.parent.mkdir(parents=True, exist_ok=True)

        if format == "tar.gz":
            # 分支一：tar.gz 归档。
            click.echo("创建tar.gz压缩包...")

            # make_archive 要求基础名不带后缀：剥掉 .tar.gz 得基础名。
            archive_name = str(output).replace(".tar.gz", "")

            # gztar 格式归档临时目录（产物自动追加 .tar.gz 后缀）。
            shutil.make_archive(archive_name, "gztar", temp_path)

            # 计算实际产物路径；与目标不一致（如大小写差异）时移动对齐。
            created = Path(f"{archive_name}.tar.gz")
            if created != output:
                shutil.move(str(created), str(output))

        elif format == "zip":
            # 分支二：zip 归档（流程与 tar.gz 对称）。
            click.echo("创建zip压缩包...")

            # 剥掉 .zip 后缀得基础名。
            archive_name = str(output).replace(".zip", "")

            # zip 格式归档临时目录。
            shutil.make_archive(archive_name, "zip", temp_path)

            # 产物路径对齐：不一致时移动纠正。
            created = Path(f"{archive_name}.zip")
            if created != output:
                shutil.move(str(created), str(output))


# 导出清单：cli 聚合层按名导入本命令的唯一公开符号。
__all__ = ["export_command"]
