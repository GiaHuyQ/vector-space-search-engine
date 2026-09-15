from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    RAW_DATA_PATH : Path = Path("data/raw/dataset.parquet")
    INDEX_PATH : Path = Path("data/index")
    DEFAULT_TOP_K : int = 5

    class Config:
        env_file = ".env"

settings = Settings()