import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from src.core.schemas import DocumentSchema


class BookDataCleaner:
    """
    Đường ống thu thập, làm sạch và chuẩn hóa dữ liệu sách từ file thô
    sang định dạng DocumentSchema chuẩn của hệ thống AI.
    """

    @staticmethod
    def clean_text(text: Any) -> str:
        """Loại bỏ thẻ HTML, giải mã ký tự đặc biệt và chuẩn hóa khoảng trắng."""
        if pd.isna(text) or text is None:
            return ""
        raw_str = str(text)
        # Giải mã HTML entities (ví dụ: &amp; -> &, &quot; -> ")
        unescaped = html.unescape(raw_str)
        # Xóa toàn bộ thẻ HTML (ví dụ: <p>, <br>, <div>...)
        no_html = re.sub(r"<[^>]+>", " ", unescaped)
        # Gộp nhiều khoảng trắng/xuống dòng liên tiếp thành 1 dấu cách
        normalized = re.sub(r"\s+", " ", no_html).strip()
        return normalized

    def process_dataframe(self, df: pd.DataFrame) -> tuple[list[DocumentSchema], list[dict[str, Any]]]:
        """
        Làm sạch DataFrame và kiểm định từng dòng bằng Pydantic DocumentSchema.
        Trả về: (danh_sách_sách_hợp_lệ, danh_sách_lỗi_bị_loại)
        """
        # 1. Loại bỏ các dòng trùng lặp dựa trên mã sách doc_id (giữ lại bản ghi đầu tiên)
        df_dedup = df.drop_duplicates(subset=["doc_id"], keep="first").copy()

        valid_docs: list[DocumentSchema] = []
        rejected_rows: list[dict[str, Any]] = []

        for idx, row in df_dedup.iterrows():
            raw_author = self.clean_text(row.get("author"))
            author = raw_author if raw_author else "Unknown"

            # Đóng gói các trường phụ trợ (như ISBN) vào metadata
            metadata: dict[str, Any] = {}
            isbn = self.clean_text(row.get("isbn"))
            if isbn:
                metadata["isbn"] = isbn

            # Xử lý giá tiền an toàn
            raw_price = row.get("price", 0.0)
            price = 0.0 if pd.isna(raw_price) else float(raw_price)

            try:
                doc = DocumentSchema(
                    doc_id=self.clean_text(row.get("doc_id")),
                    title=self.clean_text(row.get("title")),
                    author=author,
                    category=self.clean_text(row.get("category")),
                    price=price,
                    content=self.clean_text(row.get("content")),
                    metadata=metadata,
                )
                valid_docs.append(doc)
            except ValidationError as exc:
                rejected_rows.append(
                    {
                        "row_index": idx,
                        "doc_id": row.get("doc_id"),
                        "reason": str(exc.errors()[0]["msg"]),
                    }
                )

        return valid_docs, rejected_rows

    def run_pipeline(self, input_csv_path: str | Path, output_json_path: str | Path) -> dict[str, Any]:
        """Chạy toàn bộ quy trình từ file CSV thô xuất ra file JSON sạch và trả về báo cáo EDA."""
        df = pd.read_csv(input_csv_path)
        total_raw = len(df)

        valid_docs, rejected_rows = self.process_dataframe(df)

        # Lưu danh sách tài liệu sạch xuống file JSON
        output_path = Path(output_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serialized_data = [doc.model_dump() for doc in valid_docs]

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(serialized_data, f, ensure_ascii=False, indent=2)

        # Thống kê mô tả cơ bản (EDA) trên tập dữ liệu sạch
        avg_content_len = (
            round(sum(len(d.content) for d in valid_docs) / len(valid_docs), 1)
            if valid_docs
            else 0.0
        )

        return {
            "total_raw_rows": total_raw,
            "valid_documents": len(valid_docs),
            "rejected_or_duplicate_rows": total_raw - len(valid_docs),
            "avg_content_length_chars": avg_content_len,
            "rejected_details": rejected_rows,
        }


if __name__ == "__main__":
    cleaner = BookDataCleaner()
    report = cleaner.run_pipeline(
        input_csv_path="data/raw/books_raw.csv",
        output_json_path="data/processed/books_clean.json",
    )
    print("=== BÁO CÁO LÀM SẠCH & THỐNG KÊ DỮ LIỆU (EDA) ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))