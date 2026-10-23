"""
__main__ - 程序命令行入口模块
核心功能：支持 `python -m app.ics_ai <command>` 方式启动 CLI——
          先加载当前工程目录的 .env 环境变量（API 密钥等敏感配置
          不硬编码），再把当前目录插入 sys.path 以支持导入工程内
          自定义模块，最后转交 cli 模块的主函数执行。
执行流程：1. python -m app.ics_ai 触发本模块（包入口协议）；2.
          main() 探测当前工作目录 .env——存在则显式加载（规避
          Windows 下默认搜索的路径歧义），缺失则回退 dotenv 默认
          搜索；3. 将 cwd 插入 sys.path[0]；4. 延迟导入 CLI 主函数
          并调用（进入命令分发）。
副作用：读取 .env 文件并写入进程环境变量（os.environ）；修改
        sys.path（进程级模块搜索路径）；下游 CLI 的副作用由
        cli.main 决定。
主要依赖：dotenv（.env 文件加载）、sys/pathlib（路径与运行时
          环境）、app.ics_ai.cli（命令行界面，延迟导入）。
"""
# 导入 sys：修改模块搜索路径（sys.path.insert）与退出码传递。
import sys
# 导入 Path：当前工作目录的路径构造（cwd / ".env" 拼接）。
from pathlib import Path
# 导入 dotenv：.env 文件加载器（键值对写入 os.environ）。
import dotenv
def main() -> None:
    """主入口函数。
    业务流程：1. 加载 .env 环境变量（cwd 优先、默认搜索兜底）；2.
              将 cwd 插入 sys.path（工程根目录的自定义模块可导入）；
              3. 延迟导入并执行 CLI 主函数。
    副作用：写进程环境变量（os.environ）、修改 sys.path；调用
            cli_main 后的副作用由 CLI 实现决定（启动服务等）。
    返回：
        无返回；CLI 内部正常结束时进程退出码为 0。
    """
    # 自动加载用户工程目录下的 .env 文件
    # 用户可在工程根目录放置 .env 文件配置 API KEY 等环境变量
    # 显式指定当前工作目录的 .env 文件路径，确保在 Windows 上正确加载
    cwd = Path.cwd()
    # 探测工程根目录的 .env：存在则显式按路径加载（确定性优先）。
    env_file = cwd / ".env"
    if env_file.exists():
        # 显式加载指定路径的 .env（键值写入进程环境变量）。
        dotenv.load_dotenv(env_file)
    else:
        # 如果当前目录没有，尝试默认搜索（dotenv 自身向上查找）。
        dotenv.load_dotenv()
    # 将当前目录添加到Python路径，以便导入自定义模块
    sys.path.insert(0, str(cwd))
    # 导入并运行CLI
    # 说明：CLI 已实现为 cli 子包（run/shell/init/export/inspect 五命令），
    # python -m app.ics_ai <command> 即可用。
    from app.ics_ai.cli import main as cli_main
    # 转交 CLI 主函数执行命令分发。
    cli_main()
if __name__ == "__main__":
    # 模块直跑协议：python -m app.ics_ai 时进入主入口。
    main()