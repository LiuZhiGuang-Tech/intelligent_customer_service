"""
db - 数据库连接模块

核心功能：创建 MySQL 数据库引擎与会话工厂，供各 Action 以
          SessionLocal() 开启独立数据库会话；另提供 __main__ 入口，
          经 sqlacodegen 反向工程将数据库表结构导出为 ORM 类文件
          db_table_class.py。

执行流程：1. 模块导入时即拼接连接 URL 并创建 engine（连接池惰性
          建连，此刻不发网络请求）；2. 以 engine 配置 sessionmaker
          得到 SessionLocal；3. 使用方（action_order/action_postsale/
          action_logistics 等）`with SessionLocal() as session` 开启
          会话操作表类；4. __main__ 直跑时调用 export_db_table_class
          执行 sqlacodegen 生成 db_table_class.py。

副作用：模块级创建引擎与会话工厂（全局单例状态，惰性不建连）；
        __main__ 分支会调用外部命令 sqlacodegen（子进程）并写
        db_table_class.py 文件；正常导入无网络 IO。

主要依赖：sqlalchemy（create_engine/sessionmaker，需 pymysql 驱动）、
          subprocess（__main__ 反向工程调用 sqlacodegen）、
          sqlacodegen（表结构反向工程工具，pip install sqlacodegen）。
"""

# 导入 subprocess：__main__ 分支中调用外部命令 sqlacodegen（子进程执行）。
import subprocess

# 导入 os：读取环境变量 MYSQL_PASSWORD（敏感凭据经 .env 注入）。
import os

# 从 sqlalchemy 导入引擎工厂与会话工厂：数据库连接层的核心构件。
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 数据库主机地址：本地开发环境
db_host = "localhost"

# 数据库端口号：MySQL 默认端口
db_port = 3306

# 数据库名称：电商客服示例库
db_name = "ecs"

# 数据库用户名：开发环境超级用户（生产环境应使用最小权限账号并改为
# 环境变量注入，避免凭据硬编码入库）
db_user_name = "root"

# 数据库密码：经环境变量 MYSQL_PASSWORD 注入（仓库根 .env 提供，由
# ics_ecs 入口的 dotenv 加载保证先于本模块导入），不再硬编码明文凭据
db_password = os.environ.get("MYSQL_PASSWORD", "")

# 拼接 SQLAlchemy 连接 URL：驱动选 pymysql（纯 Python 实现，免编译），
# charset=utf8 保证中文数据正常存取
url = f"mysql+pymysql://{db_user_name}:{db_password}@{db_host}:{db_port}/{db_name}?charset=utf8"

# 创建数据库引擎（此刻惰性，不真正建连）：进程内唯一连接池，
# 供 SessionLocal 复用
engine = create_engine(url)

# 配置会话工厂：绑定引擎，关闭自动提交与自动 flush——事务由使用方
# 显式 session.commit() 提交，保证写操作的原子性控制权在使用方
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


if __name__ == "__main__":

    def export_db_table_class(run=False):
        """将数据库表映射为Python类

        核心功能：以 sqlacodegen 反向工程生成 db_table_class.py。

        业务流程：1. run=False 直接返回（防误执行的守卫开关）；2.
                  构造 sqlacodegen 命令（模块方式调用 + 连接 URL）；
                  3. subprocess 子进程执行并捕获 stdout（ORM 类定义
                  源码）；4. 将输出写入当前目录的 db_table_class.py。

        副作用：启动子进程执行 sqlacodegen（读数据库表结构）；写
                db_table_class.py 文件（覆盖式）。

        @param run: 执行开关；仅显式传 True 才真正执行（默认 False
                    防止 import 触发副作用）。
        @return: 无返回；sqlacodegen 未安装或连接失败时由子进程
                 输出错误信息（不抛异常，错误内容写入结果文件）。
        """
        # 守卫开关：run=False 时静默返回，防止误触发反向工程
        if not run:
            return

        # 输出文件路径：相对当前工作目录
        output_path = "db_table_class.py"

        # 构造命令：python -m sqlacodegen <url>——以模块方式调用，
        # 避免依赖 PATH 中的可执行入口
        cmd = ["python", "-m", "sqlacodegen", url]

        # 子进程执行并捕获输出：stdout 即生成的 ORM 类源码，强制
        # UTF-8 解码防止中文表注释乱码
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")

        # 将生成的类定义源码写入文件
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.stdout)

    # 直跑入口：显式传 True 触发反向工程
    export_db_table_class(True)
