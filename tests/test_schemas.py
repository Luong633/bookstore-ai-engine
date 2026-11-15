import pytest
from pydantic import ValidationError
from src.core.schemas import DocumentSchema, SearchMode, QuerySchema


def test_valid_document() -> None:
    """Kiểm tra khởi tạo Document hợp lệ và tự động cắt khoảng trắng."""
    doc = DocumentSchema(
        doc_id="DOC001",
        title="  Clean Code  ",
        author="Robert C. Martin",
        category="Programming",
        price=150000.0,
        content="Sách hướng dẫn viết mã sạch và dễ bảo trì."
    )
    # Kiểm tra xem khoảng trắng thừa đã bị cắt chưa
    assert doc.title == "Clean Code"
    assert doc.price == 150000.0


def test_invalid_document_negative_price() -> None:
    """Đảm bảo hệ thống bắt lỗi (ValidationError) khi giá sách bị âm."""
    with pytest.raises(ValidationError):
        DocumentSchema(
            doc_id="DOC002",
            title="Test Book",
            category="AI",
            price=-5000.0,
            content="Mô tả hợp lệ dài hơn mười ký tự."
        )


def test_query_schema_validation() -> None:
    """Kiểm tra QuerySchema chặn câu hỏi rỗng và giới hạn top_k."""
    # Hợp lệ
    q = QuerySchema(query_text="Sách AI", top_k=10)
    assert q.mode == SearchMode.HYBRID

    # Bắt lỗi: Câu hỏi chỉ chứa khoảng trắng
    with pytest.raises(ValidationError):
        QuerySchema(query_text="     ")

    # Bắt lỗi: top_k vượt quá ngưỡng cho phép (50)
    with pytest.raises(ValidationError):
        QuerySchema(query_text="Sách AI", top_k=100)