import pandas as pd
from src.data.cleaner import BookDataCleaner


def test_clean_text_removes_html_and_extra_spaces() -> None:
    """Kiểm tra hàm làm sạch văn bản xóa đúng thẻ HTML và khoảng trắng thừa."""
    dirty = "  <p>Sách   <b>AI</b> &amp; Dữ liệu</p>  "
    assert BookDataCleaner.clean_text(dirty) == "Sách AI & Dữ liệu"


def test_process_dataframe_deduplication_and_validation() -> None:
    """Kiểm tra pipeline khử trùng lặp, điền tác giả thiếu và chặn dữ liệu bẩn."""
    raw_df = pd.DataFrame(
        [
            {
                "doc_id": "D01",
                "title": "  Sách Hợp Lệ  ",
                "author": None,
                "category": "Tech",
                "price": 120000,
                "content": "<p>Nội dung hợp lệ dài hơn mười ký tự.</p>",
                "isbn": "123-456",
            },
            {
                "doc_id": "D01",  # Trùng ID
                "title": "Sách Trùng ID",
                "author": "Ai Đó",
                "category": "Tech",
                "price": 120000,
                "content": "Nội dung trùng lặp cần bị loại bỏ.",
                "isbn": "123-456",
            },
            {
                "doc_id": "D02",  # Giá âm -> Bị Pydantic chặn
                "title": "Sách Giá Âm",
                "author": "Ai Đó",
                "category": "Tech",
                "price": -1000,
                "content": "Nội dung hợp lệ nhưng giá âm.",
                "isbn": "999-999",
            },
        ]
    )

    cleaner = BookDataCleaner()
    valid_docs, rejected = cleaner.process_dataframe(raw_df)

    assert len(valid_docs) == 1
    assert valid_docs[0].doc_id == "D01"
    assert valid_docs[0].author == "Unknown"
    assert valid_docs[0].content == "Nội dung hợp lệ dài hơn mười ký tự."
    assert len(rejected) == 1
    assert rejected[0]["doc_id"] == "D02"