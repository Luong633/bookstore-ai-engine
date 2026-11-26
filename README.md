# 📚 Bookstore AI Engine

Hệ thống tìm kiếm thông minh (Hybrid Search & RAG Backend) dành cho nhà sách, kết hợp giữa tìm kiếm từ khóa truyền thống (**BM25**) và tìm kiếm ngữ nghĩa học sâu (**Dense Vector Embeddings**).

## 🚀 Tính năng đã hoàn thiện
- **Data Pipeline (`src/data/`):** Thu thập dữ liệu thô từ CSV, làm sạch thẻ HTML, khử trùng lặp, kiểm định hợp đồng dữ liệu bằng **Pydantic** và chia nhỏ văn bản đệ quy (**Recursive Text Chunking**).
- **Lexical Search (`src/core/lexical_search.py`):** Tự cài đặt thuật toán **BM25** từ gốc, hỗ trợ lưu/nạp chỉ mục JSON để khởi động nhanh.
- **Semantic Search (`src/core/semantic_search.py`):** Tích hợp mô hình đa ngôn ngữ `paraphrase-multilingual-MiniLM-L12-v2` mã hóa văn bản thành vector 384 chiều, tính độ tương đồng bằng **NumPy Vectorization**.
- **Quality Control (`tests/`):** 100% logic cốt lõi được kiểm soát tự động bằng **Pytest** và chuẩn hóa mã nguồn với **Ruff**.

## 🛠️ Công nghệ sử dụng
- **Ngôn ngữ:** Python 3.11+
- **AI / Data:** PyTorch, Sentence-Transformers, NumPy, Pandas, Pydantic
- **Testing & Linting:** Pytest, Ruff

## ⚙️ Hướng dẫn cài đặt & chạy kiểm thử

1. **Tạo và kích hoạt môi trường ảo:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate