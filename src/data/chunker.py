import re
from src.core.schemas import DocumentSchema


class RecursiveTextChunker:
    """
    Bộ chia nhỏ văn bản thông minh theo cấu trúc câu/đoạn (Recursive Splitting),
    giúp bảo toàn ngữ nghĩa và giữ liên kết với tài liệu gốc qua metadata.
    """

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap phải nhỏ hơn chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        """Chia một chuỗi văn bản dài thành danh sách các đoạn nhỏ có gối đầu."""
        cleaned = text.strip()
        if not cleaned:
            return []

        # Nếu văn bản đã ngắn hơn ngưỡng tối đa thì giữ nguyên
        if len(cleaned) <= self.chunk_size:
            return [cleaned]

        # Tách văn bản thành các câu hoàn chỉnh dựa trên dấu kết thúc câu (. ! ?) hoặc xuống dòng
        sentences = re.split(r"(?<=[.!?\n])\s+", cleaned)

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Trường hợp hy hữu: bản thân 1 câu đơn lẻ dài hơn cả chunk_size -> cắt theo từ
            if len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                chunks.extend(self._split_long_sentence(sentence))
                continue

            # Tính độ dài nếu ghép thêm câu này (cộng 1 ký tự khoảng trắng nối câu)
            added_len = len(sentence) + (1 if current_chunk else 0)

            if current_length + added_len <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += added_len
            else:
                # Đóng gói chunk hiện tại
                chunks.append(" ".join(current_chunk))

                # Giữ lại phần gối đầu (overlap) từ cuối chunk vừa đóng gói
                overlap_sentences: list[str] = []
                overlap_len = 0
                for prev_sent in reversed(current_chunk):
                    if overlap_len + len(prev_sent) <= self.chunk_overlap:
                        overlap_sentences.insert(0, prev_sent)
                        overlap_len += len(prev_sent) + 1
                    else:
                        break

                current_chunk = [*overlap_sentences, sentence]
                current_length = sum(len(s) for s in current_chunk) + len(current_chunk) - 1

        if current_chunk:
            last_text = " ".join(current_chunk)
            # Nếu mảnh cuối cùng quá vụn (< 15 ký tự) và đã có mảnh trước đó, gộp vào mảnh trước
            if len(last_text) < 15 and chunks:
                chunks[-1] = f"{chunks[-1]} {last_text}"
            else:
                chunks.append(last_text)

        return chunks

    def _split_long_sentence(self, sentence: str) -> list[str]:
        """Cắt một câu quá dài theo ranh giới từ (khoảng trắng) để không làm rách chữ."""
        words = sentence.split()
        sub_chunks: list[str] = []
        current_words: list[str] = []
        current_len = 0

        for word in words:
            added_len = len(word) + (1 if current_words else 0)
            if current_len + added_len <= self.chunk_size:
                current_words.append(word)
                current_len += added_len
            else:
                if current_words:
                    sub_chunks.append(" ".join(current_words))
                current_words = [word]
                current_len = len(word)

        if current_words:
            sub_chunks.append(" ".join(current_words))
        return sub_chunks

    def chunk_documents(self, documents: list[DocumentSchema]) -> list[DocumentSchema]:
        """Chuyển đổi danh sách sách gốc thành danh sách các DocumentSchema dạng chunk."""
        chunked_docs: list[DocumentSchema] = []

        for doc in documents:
            text_chunks = self.split_text(doc.content)
            total_chunks = len(text_chunks)

            for idx, chunk_text in enumerate(text_chunks):
                # Sao chép metadata gốc và bổ sung thông tin truy vết chunk
                chunk_metadata = {
                    **doc.metadata,
                    "parent_doc_id": doc.doc_id,
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                }
                chunked_docs.append(
                    DocumentSchema(
                        doc_id=f"{doc.doc_id}_chunk_{idx}",
                        title=doc.title,
                        author=doc.author,
                        category=doc.category,
                        price=doc.price,
                        content=chunk_text,
                        metadata=chunk_metadata,
                    )
                )

        return chunked_docs