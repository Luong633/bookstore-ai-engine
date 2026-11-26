from pathlib import Path
import pytest
from src.core.schemas import DocumentSchema
from src.core.semantic_search import SemanticRetriever


@pytest.fixture(scope="module")
def shared_retriever() -> SemanticRetriever:
    """Khởi tạo 1 lần duy nhất cho toàn bộ file test để tối ưu tốc độ tải mô hình."""
    books = [
        DocumentSchema(
            doc_id="B01",
            title="Designing Data-Intensive Applications",
            author="Martin Kleppmann",
            category="System Design",
            price=450000.0,
            content="Kiến trúc hệ thống phân tán, cơ sở dữ liệu lớn và khả năng chịu lỗi.",
            metadata={"isbn": "978-1449373320"},
        ),
        DocumentSchema(
            doc_id="B02",
            title="Clean Code",
            author="Robert C. Martin",
            category="Software Engineering",
            price=220000.0,
            content="Kỹ thuật viết mã nguồn sạch, tái cấu trúc hàm và tổ chức phần mềm.",
            metadata={"isbn": "978-0132350884"},
        ),
        DocumentSchema(
            doc_id="B03",
            title="Nhà Giả Kim",
            author="Paulo Coelho",
            category="Văn học",
            price=79000.0,
            content="Hành trình theo đuổi ước mơ và khám phá vận mệnh của chàng chăn cừu.",
            metadata={"isbn": "978-6045312345"},
        ),
    ]
    retriever = SemanticRetriever()
    retriever.fit(books)
    return retriever


def test_semantic_understanding_without_exact_keywords(
    shared_retriever: SemanticRetriever,
) -> None:
    """
    Kiểm tra khả năng hiểu ngữ nghĩa: truy vấn không hề chứa từ khóa 'Clean Code'
    hay 'mã nguồn sạch' nhưng vẫn tìm đúng cuốn B02.
    """
    results = shared_retriever.search("cách lập trình viên viết code đẹp và dễ sửa", top_k=2)
    assert len(results) >= 1
    assert results[0].doc_id == "B02"
    assert results[0].vector_score > 0.4


def test_semantic_cross_lingual_vietnamese_to_english(
    shared_retriever: SemanticRetriever,
) -> None:
    """Kiểm tra tìm kiếm xuyên ngôn ngữ (hỏi tiếng Việt trúng sách tiêu đề tiếng Anh)."""
    results = shared_retriever.search("thiết kế hệ thống xử lý dữ liệu phân tán", top_k=2)
    assert results[0].doc_id == "B01"


def test_semantic_save_and_load_index(
    shared_retriever: SemanticRetriever, tmp_path: Path
) -> None:
    """Kiểm tra lưu ma trận vector (.npy) và nạp lại hoạt động chính xác."""
    index_dir = tmp_path / "semantic_index"
    shared_retriever.save_index(index_dir)

    loaded = SemanticRetriever.load_index(index_dir)
    results = loaded.search("câu chuyện phiêu lưu theo đuổi ước mơ", top_k=1)

    assert len(results) == 1
    assert results[0].doc_id == "B03"