# config.py
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # =====================
    # 1. 代理设置
    # =====================
    proxy_host: str = "127.0.0.1"
    proxy_port: str = "7890"  # 修改为你的端口，不需要则留空

    def __post_init__(self):
        if self.proxy_port:
            os.environ["HTTP_PROXY"] = f"http://{self.proxy_host}:{self.proxy_port}"
            os.environ["HTTPS_PROXY"] = f"http://{self.proxy_host}:{self.proxy_port}"
            print(f"[Config] Proxy enabled: {self.proxy_host}:{self.proxy_port}")

    # =====================
    # 2. 路径设置
    # =====================
    project_root: Path = Path(__file__).parent
    data_dir: Path = project_root / "data"
    model_dir: Path = data_dir / "models"

    def ensure_dirs(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    # =====================
    # 3. 股票池与时间
    # =====================
    # 这里可以直接读取外部文件，为了演示方便写死在代码里
    symbol_pool: List[str] = field(default_factory=lambda: ["600519", "000858", "601318", "000001", "300750"])
    start_date: str = "20220101"
    end_date: str = "20251231"


# 全局单例
cfg = Config()
cfg.ensure_dirs()
