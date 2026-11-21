import pytest
from src.core.schemas import DocumentSchema
from src.data.chunker import RecursiveTextChunker


def test_invalid_chunker_config() -> None:
    """Đảm bảo báo lỗi nếu cấu hình overlap lớn hơn hoặc bằng chunk_size."""
    with pytest.raises(ValueError):
        RecursiveTextChunker(chunk_size=100, chunk_overlap=100)


def test_short_text_is_not_split() -> None:
    """Văn bản ngắn hơn chunk_size phải được giữ nguyên vẹn thành 1 chunk."""
    chunker = RecursiveTextChunker(chunk_size=200, chunk_overlap=30)
    text = "Đây là đoạn mô tả ngắn gọn về cuốn sách lập trình Python."
    chunks = chunker.split_text(text)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_splitting_with_overlap_and_metadata() -> None:
    """Kiểm tra chia nhỏ tài liệu dài thành nhiều chunk và lưu đúng parent_doc_id."""
    long_content = (
        "Chương 1 giới thiệu tổng quan về kiến trúc hệ thống phân tán hiện đại. "
        "Chương 2 phân tích sâu về các mô hình lưu trữ dữ liệu quan hệ và NoSQL. "
        "Chương 3 hướng dẫn cách nhân bản dữ liệu và xử lý chịu lỗi khi máy chủ sập. "
        "Chương 4 trình bày thuật toán đồng thuận và xử lý luồng dữ liệu thời gian thực."
    )
    doc = DocumentSchema(
        doc_id="B01",
        title="Designing Data-Intensive Applications",
        author="Martin Kleppmann",
        category="System Design",
        price=450000.0,
        content=long_content,
        metadata={"isbn": "978-1449373320"},
    )

    chunker = RecursiveTextChunker(chunk_size=150, chunk_overlap=75)
    chunked_docs = chunker.chunk_documents([doc])

    assert len(chunked_docs) >= 2
    assert chunked_docs[0].doc_id == "B01_chunk_0"
    assert chunked_docs[0].metadata["parent_doc_id"] == "B01"
    assert chunked_docs[0].metadata["isbn"] == "978-1449373320"
    assert chunked_docs[1].metadata["chunk_index"] == 1
    # Kiểm tra cơ chế gối đầu: câu cuối của chunk 0 xuất hiện ở đầu chunk 1
    assert "Chương 2" in chunked_docs[0].content
    assert "Chương 2" in chunked_docs[1].content