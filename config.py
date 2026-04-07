# config.py
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # 1. 代理设置
    proxy_host: str = "127.0.0.1"
    proxy_port: str = "7890"  # 不需要请留空 ""

    def __post_init__(self):
        # 只有端口不为空时才设置代理
        if self.proxy_port:
            os.environ["HTTP_PROXY"] = f"http://{self.proxy_host}:{self.proxy_port}"
            os.environ["HTTPS_PROXY"] = f"http://{self.proxy_host}:{self.proxy_port}"
            print(f"[Config] Proxy enabled: {self.proxy_host}:{self.proxy_port}")

    # 2. 路径设置
    project_root: Path = field(default_factory=lambda: Path(__file__).parent)
    data_dir: Path = None
    model_dir: Path = None

    def setup_paths(self):
        self.data_dir = self.project_root / "data"
        self.model_dir = self.data_dir / "models"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    # 3. 股票池
    symbol_pool: List[str] = field(default_factory=lambda: ["600519", "000858", "601318", "000001", "300750"])
    start_date: str = "20220101"
    end_date: str = "20251231"


# 全局单例
cfg = Config()
cfg.setup_paths()
