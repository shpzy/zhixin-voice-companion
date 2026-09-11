"""知心后端 · 启动入口（开发 / 宝塔 Python 项目管理器 / 手动 python 均可）。

单服务架构：/app 托管前端页面，/api/* 为接口，/static/audio 为语音文件。
本文件做了五层加固，确保任何环境都能启动且故障可查：
  1. 强制切换工作目录到本文件所在目录（.env、./chroma_db 不再依赖启动方 CWD）
  2. 强制把多个候选 site-packages 加入 sys.path（兼容 venv / 系统 Python / 容器内 Python）
  3. 全程输出落盘 startup.log（无论进程被谁杀，都能看到走到哪一步）
  4. pysqlite3 补丁（chromadb 要求 sqlite>=3.35，老系统自带 sqlite 通常更老）
  5. 显式加载同目录 .env（不靠 CWD 搜索）
"""
import faulthandler
import os
import sys
from pathlib import Path

# ---- 0) 一律切到本文件所在目录：所有相对路径不再依赖启动方 CWD ----
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
sys.path.insert(0, str(BASE_DIR))

# ---- 启动全程留痕（必须先打开 LOG_F）----
LOG_F = open(BASE_DIR / "startup.log", "a", buffering=1)  # 行缓冲，实时落盘
sys.stdout = LOG_F
sys.stderr = LOG_F
faulthandler.enable(file=LOG_F)

print(f"=== 启动开始 pid={os.getpid()} cwd={os.getcwd()} python={sys.executable} ===")

# ---- 1) 强制添加所有候选 site-packages（兼容 venv / 系统 Python / 容器）----
#    按"项目内 venv → 同级 venv → 系统 site-packages"顺序探测，谁存在就加谁
CANDIDATE_SITES = [
    BASE_DIR / "venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
    BASE_DIR / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
    BASE_DIR.parent / "venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
    Path(f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"),
    Path(f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"),
]
for p in CANDIDATE_SITES:
    if p.exists():
        sys.path.insert(0, str(p))
        print(f"sys.path += {p}")
print("sys.path 最终前5:", sys.path[:5])

# ---- 2) 老系统 sqlite 兼容补丁 ----
try:
    import pysqlite3  # noqa: F401

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    print("pysqlite3 补丁: OK")
except ImportError:
    print("警告: pysqlite3 未安装（系统 sqlite>=3.35 时可忽略）")

# ---- 3) 显式加载 .env ----
try:
    import dotenv

    dotenv.load_dotenv(BASE_DIR / ".env", override=False)
    print(".env 加载: OK")
except Exception as e:
    print(f"警告: .env 加载失败: {e}")

# ---- 4) 启动服务 ----
import uvicorn  # noqa: E402

print("uvicorn 导入 OK，开始监听 0.0.0.0:8000 ...")
uvicorn.run(
    "main:app",
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 8000)),
    log_level="info",
)
print("=== 服务已退出（正常情况不会执行到这，执行到说明异常）===")