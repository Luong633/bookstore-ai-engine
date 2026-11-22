import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.lexical_search import BM25Retriever
from src.data.chunker import RecursiveTextChunker
from src.data.cleaner import BookDataCleaner


class DataIngestionPipeline:
    """
    Điều phối toàn bộ luồng dữ liệu:
    CSV Thô -> Làm sạch (Cleaner) -> Cắt nhỏ (Chunker) -> Lập chỉ mục BM25 & Lưu đĩa.
    """

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50) -> None:
        self.cleaner = BookDataCleaner()
        self.chunker = RecursiveTextChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.bm25 = BM25Retriever()

    def run(
        self,
        raw_csv_path: str | Path = "data/raw/books_raw.csv",
        clean_json_path: str | Path = "data/processed/books_clean.json",
        chunks_json_path: str | Path = "data/processed/books_chunks.json",
        bm25_index_path: str | Path = "data/processed/bm25_index.json",
    ) -> dict[str, Any]:
        """Thực thi toàn bộ đường ống và trả về báo cáo hiệu năng chi tiết."""
        start_time = time.perf_counter()

        # Bước 1: Đọc và làm sạch dữ liệu thô
        df_raw = pd.read_csv(raw_csv_path)
        valid_docs, rejected_rows = self.cleaner.process_dataframe(df_raw)

        clean_path = Path(clean_json_path)
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        with clean_path.open("w", encoding="utf-8") as f:
            json.dump(
                [doc.model_dump() for doc in valid_docs],
                f,
                ensure_ascii=False,
                indent=2,
            )

        # Bước 2: Chia nhỏ văn bản thành các chunks chuẩn ngữ cảnh
        chunked_docs = self.chunker.chunk_documents(valid_docs)
        with Path(chunks_json_path).open("w", encoding="utf-8") as f:
            json.dump(
                [chunk.model_dump() for chunk in chunked_docs],
                f,
                ensure_ascii=False,
                indent=2,
            )

        # Bước 3: Lập chỉ mục từ khóa BM25 trên các chunks và lưu xuống ổ cứng
        self.bm25.fit(chunked_docs)
        self.bm25.save_index(bm25_index_path)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "status": "success",
            "raw_rows": len(df_raw),
            "valid_books": len(valid_docs),
            "rejected_rows": len(rejected_rows),
            "total_chunks": len(chunked_docs),
            "bm25_vocab_size": len(self.bm25.idf),
            "execution_time_ms": elapsed_ms,
        }


if __name__ == "__main__":
    pipeline = DataIngestionPipeline(chunk_size=250, chunk_overlap=40)
    summary = pipeline.run()
    print("=== BÁO CÁO CHẠY ĐƯỜNG ỐNG DỮ LIỆU HOÀN CHỈNH ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # Thử nghiệm nạp lại chỉ mục từ ổ cứng và tìm kiếm ngay lập tức
    loaded_bm25 = BM25Retriever.load_index("data/processed/bm25_index.json")
    hits = loaded_bm25.search("kiến trúc hệ thống phân tán", top_k=2)
    print("\n=== KIỂM TRA TÌM KIẾM TỪ CHỈ MỤC ĐÃ LƯU ===")
    for h in hits:
        print(f"- [{h.doc_id}] {h.title} (Điểm: {h.lexical_score})")