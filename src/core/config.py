from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Xác định đường dẫn gốc của toàn bộ dự án
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """
    Quản lý toàn bộ cấu hình hệ thống AI.
    Tự động đọc từ biến môi trường hoặc file .env và kiểm tra kiểu dữ liệu.
    """
    # Thông tin chung
    APP_NAME: str = "Bookstore Hybrid RAG & Recommendation Engine"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Cấu hình Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Đường dẫn dữ liệu
    RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"

    # Cấu hình AI Core
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    TOP_K_RETRIEVAL: int = 20
    TOP_K_RERANK: int = 5

    # Kết nối bên ngoài
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION_NAME: str = "bookstore_knowledge"
    LLM_API_KEY: str = ""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Khởi tạo đối tượng Settings một lần duy nhất (Singleton Pattern)
    giúp tối ưu hóa tốc độ, không phải đọc lại file .env ở mỗi request.
    """
    return Settings()


settings = get_settings()