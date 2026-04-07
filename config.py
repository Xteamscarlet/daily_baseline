# config.py
import os
from pathlib import Path

# =====================
# 1. 代理设置
# =====================
# 如果需要代理，请修改端口，不需要则注释掉下面两行
PROXY_HOST = "127.0.0.1"
PROXY_PORT = "7890"

# 设置环境变量
if 'PROXY_PORT' in dir():
    os.environ["HTTP_PROXY"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
    os.environ["HTTPS_PROXY"] = f"http://{PROXY_HOST}:{PROXY_PORT}"

# =====================
# 2. 项目路径
# =====================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# =====================
# 3. 股票池
# =====================
SYMBOL_POOL = ["600519", "000858", "601318", "000001", "300750"]
START_DATE = "20220101"
END_DATE = "20251231"
