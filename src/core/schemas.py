from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator


class SearchMode(str, Enum):
    """Các chế độ tìm kiếm hỗ trợ trong hệ thống AI."""
    LEXICAL = "lexical"   # Chỉ dùng từ khóa (BM25)
    VECTOR = "vector"     # Chỉ dùng ngữ nghĩa (Vector Embedding)
    HYBRID = "hybrid"     # Tìm kiếm lai kết hợp cả 2 thuật toán


class DocumentSchema(BaseModel):
    """Định nghĩa một cuốn sách/tài liệu gốc sau khi làm sạch."""
    doc_id: str = Field(..., min_length=1, description="Mã định danh duy nhất của tài liệu")
    title: str = Field(..., min_length=1, description="Tiêu đề tài liệu")
    author: str = Field(default="Unknown", description="Tên tác giả")
    category: str = Field(..., min_length=1, description="Thể loại")
    price: float = Field(default=0.0, ge=0.0, description="Giá (không được âm)")
    content: str = Field(..., min_length=10, description="Nội dung chính hoặc mô tả")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Các thông tin bổ sung")

    @field_validator("title", "content")
    @classmethod
    def strip_whitespace(cls, value: str) -> str:
        """Custom Validator: Tự động cắt bỏ khoảng trắng thừa ở 2 đầu văn bản."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Trường văn bản không được chỉ chứa khoảng trắng.")
        return cleaned


class QuerySchema(BaseModel):
    """Định nghĩa cấu trúc kiểm soát câu hỏi đầu vào từ người dùng."""
    query_text: str = Field(..., min_length=2, max_length=500, description="Câu hỏi hoặc từ khóa")
    top_k: int = Field(default=5, ge=1, le=50, description="Số lượng kết quả (1-50)")
    mode: SearchMode = Field(default=SearchMode.HYBRID, description="Chế độ tìm kiếm")
    category_filter: str | None = Field(default=None, description="Bộ lọc theo thể loại")

    @field_validator("query_text")
    @classmethod
    def validate_query_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 2:
            raise ValueError("Câu truy vấn sau khi làm sạch phải có ít nhất 2 ký tự.")
        return cleaned


class SearchResultSchema(BaseModel):
    """Định nghĩa cấu trúc cho một kết quả tìm kiếm trả về."""
    doc_id: str
    title: str
    content: str
    lexical_score: float = Field(default=0.0, description="Điểm thuật toán từ khóa")
    vector_score: float = Field(default=0.0, description="Điểm tương đồng Vector")
    final_score: float = Field(..., description="Điểm tổng hợp sau cùng để xếp hạng")
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGResponseSchema(BaseModel):
    """Cấu trúc hoàn chỉnh của hệ thống Hỏi-Đáp AI trả về cho Frontend."""
    query: str
    answer: str = Field(..., description="Câu trả lời từ LLM")
    sources: list[SearchResultSchema] = Field(default_factory=list, description="Danh sách tài liệu tham khảo")
    process_time_ms: float = Field(..., ge=0.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))