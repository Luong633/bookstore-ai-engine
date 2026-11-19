from src.core.lexical_search import BM25Retriever
from src.core.schemas import DocumentSchema


def _get_sample_books() -> list[DocumentSchema]:
    """Tạo danh sách sách mẫu để phục vụ kiểm thử thuật toán BM25."""
    return [
        DocumentSchema(
            doc_id="B01",
            title="Designing Data-Intensive Applications",
            author="Martin Kleppmann",
            category="System Design",
            price=450000.0,
            content="Cuốn sách kinh điển về kiến trúc hệ thống phân tán và xử lý dữ liệu lớn.",
            metadata={"isbn": "978-1449373320"},
        ),
        DocumentSchema(
            doc_id="B02",
            title="Clean Code",
            author="Robert C. Martin",
            category="Software Engineering",
            price=220000.0,
            content="Cuốn sách hướng dẫn kỹ thuật viết mã sạch, dễ đọc và dễ bảo trì.",
            metadata={"isbn": "978-0132350884"},
        ),
        DocumentSchema(
            doc_id="B03",
            title="Lập trình Python và Khoa học Dữ liệu",
            author="Nguyễn Văn A",
            category="AI & Data",
            price=180000.0,
            content="Cuốn sách nhập môn ngôn ngữ Python, thư viện NumPy và Pandas.",
            metadata={"isbn": "978-6041234567"},
        ),
    ]


def test_bm25_exact_author_and_keyword_search() -> None:
    """Kiểm tra tìm kiếm chính xác tên tác giả và từ khóa chuyên ngành."""
    retriever = BM25Retriever()
    retriever.fit(_get_sample_books())

    # Tìm theo tên tác giả hiếm gặp
    results = retriever.search("sách của Martin Kleppmann", top_k=3)
    assert len(results) >= 1
    assert results[0].doc_id == "B01"
    assert results[0].lexical_score > 0.0


def test_bm25_category_filter() -> None:
    """Kiểm tra bộ lọc thể loại hoạt động chính xác."""
    retriever = BM25Retriever()
    retriever.fit(_get_sample_books())

    # Từ "cuốn sách" xuất hiện ở cả 3 quyển, nhưng lọc theo "AI & Data" chỉ trả về B03
    results = retriever.search("cuốn sách", top_k=5, category_filter="AI & Data")
    assert len(results) == 1
    assert results[0].doc_id == "B03"


def test_bm25_no_match_returns_empty() -> None:
    """Kiểm tra khi truy vấn từ khóa hoàn toàn không tồn tại trong kho sách."""
    retriever = BM25Retriever()
    retriever.fit(_get_sample_books())

    results = retriever.search("ẩm thực nấu ăn", top_k=5)
    assert results == []