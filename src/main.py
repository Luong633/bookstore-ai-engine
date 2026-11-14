import sys
import numpy as np
import pandas as pd
from src.core.config import settings


def verify_environment() -> None:
    """Kiểm tra môi trường thực thi và cấu hình hệ thống Ngày 1."""
    print("=" * 60)
    print(f"🚀 KHỞI ĐỘNG HỆ THỐNG: {settings.APP_NAME}")
    print(f"📌 Phiên bản: {settings.APP_VERSION} | Môi trường: {settings.ENVIRONMENT}")
    print("=" * 60)
    print(f"✔ Python Interpreter : {sys.executable}")
    print(f"✔ Phiên bản Python   : {sys.version.split()[0]}")
    print(f"✔ NumPy Version      : {np.__version__}")
    print(f"✔ Pandas Version     : {pd.__version__}")
    print(f"✔ Thư mục dữ liệu thô: {settings.RAW_DATA_DIR}")
    print(f"✔ Mô hình Embedding  : {settings.EMBEDDING_MODEL_NAME}")
    print(f"✔ Cấu hình Retrieval : Top-{settings.TOP_K_RETRIEVAL} -> Rerank Top-{settings.TOP_K_RERANK} (Kiểu dữ liệu: {type(settings.TOP_K_RERANK).__name__})")
    print("=" * 60)
    print("✅ NGÀY 1 HOÀN TẤT: Môi trường và Kiến trúc dự án đã sẵn sàng!")


if __name__ == "__main__":
    verify_environment()