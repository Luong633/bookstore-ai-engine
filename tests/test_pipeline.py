from pathlib import Path
import pandas as pd
from src.core.lexical_search import BM25Retriever
from src.data.pipeline import DataIngestionPipeline


def test_end_to_end_pipeline_and_index_persistence(tmp_path: Path) -> None:
    """
    Kiểm thử toàn trình từ file CSV thô -> làm sạch -> cắt chunk -> lưu/nạp chỉ mục BM25.
    Sử dụng thư mục tạm tmp_path của pytest để không làm bẩn dữ liệu thật.
    """
    raw_csv = tmp_path / "raw.csv"
    clean_json = tmp_path / "clean.json"
    chunks_json = tmp_path / "chunks.json"
    index_json = tmp_path / "bm25_index.json"

    df = pd.DataFrame(
        [
            {
                "doc_id": "S01",
                "title": "  Hệ thống Phân tán Thực chiến  ",
                "author": "Martin Kleppmann",
                "category": "Architecture",
                "price": 350000,
                "content": "<p>Hướng dẫn chi tiết về nhân bản dữ liệu và chịu lỗi hệ thống.</p>",
                "isbn": "111-222",
            },
            {
                "doc_id": "S02",
                "title": "Sách Lỗi",
                "author": "Ai Đó",
                "category": "Test",
                "price": -100,  # Giá âm -> Bị loại bỏ
                "content": "Mô tả hợp lệ nhưng giá âm.",
                "isbn": "333-444",
            },
        ]
    )
    df.to_csv(raw_csv, index=False)

    pipeline = DataIngestionPipeline(chunk_size=200, chunk_overlap=30)
    report = pipeline.run(
        raw_csv_path=raw_csv,
        clean_json_path=clean_json,
        chunks_json_path=chunks_json,
        bm25_index_path=index_json,
    )

    assert report["valid_books"] == 1
    assert report["rejected_rows"] == 1
    assert index_json.exists()

    # Nạp lại chỉ mục từ file JSON và kiểm tra kết quả tìm kiếm
    restored_retriever = BM25Retriever.load_index(index_json)
    results = restored_retriever.search("nhân bản dữ liệu", top_k=3)

    assert len(results) == 1
    assert results[0].doc_id == "S01_chunk_0"
    assert results[0].metadata["parent_doc_id"] == "S01"