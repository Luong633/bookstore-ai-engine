import math
import re
from collections import Counter
from src.core.schemas import DocumentSchema, SearchResultSchema


class BM25Retriever:
    """
    Bộ tìm kiếm từ khóa sử dụng thuật toán BM25 (Best Matching 25).
    Tự cài đặt từ gốc để kiểm soát hoàn toàn công thức toán và điểm số.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        # k1: Tham số kiểm soát độ bão hòa tần suất từ (thường chọn từ 1.2 đến 2.0)
        # b: Tham số điều chỉnh mức độ phạt theo độ dài văn bản (từ 0.0 đến 1.0)
        self.k1 = k1
        self.b = b
        self.documents: list[DocumentSchema] = []
        self.doc_freqs: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0
        self.tokenized_corpus: list[list[str]] = []

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Tách văn bản thành danh sách các từ (tokens) viết thường, bỏ dấu câu."""
        # Giữ lại chữ cái (bao gồm tiếng Việt có dấu) và chữ số
        return re.findall(r"\w+", text.lower())

    def fit(self, documents: list[DocumentSchema]) -> None:
        """
        Xây dựng chỉ mục (Index) và tính toán trọng số IDF cho toàn bộ kho tài liệu.
        """
        self.documents = documents
        self.tokenized_corpus = []
        self.doc_len = []
        self.doc_freqs = {}
        self.idf = {}

        if not documents:
            self.avgdl = 0.0
            return

        for doc in documents:
            # Gộp tiêu đề, tác giả, thể loại và nội dung để tìm kiếm toàn diện
            full_text = f"{doc.title} {doc.author} {doc.category} {doc.content}"
            tokens = self.tokenize(full_text)
            self.tokenized_corpus.append(tokens)
            self.doc_len.append(len(tokens))

            # Đếm số tài liệu chứa mỗi từ (mỗi từ chỉ tính 1 lần trên mỗi tài liệu)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        num_docs = len(documents)
        self.avgdl = sum(self.doc_len) / num_docs

        # Tính chỉ số IDF (Độ hiếm của từ) theo công thức chuẩn của BM25
        for token, freq in self.doc_freqs.items():
            idf_score = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[token] = idf_score

    def _score_document(self, query_tokens: list[str], doc_idx: int) -> float:
        """Tính tổng điểm BM25 của một tài liệu đối với danh sách từ khóa truy vấn."""
        score = 0.0
        doc_tokens = self.tokenized_corpus[doc_idx]
        doc_length = self.doc_len[doc_idx]
        term_counts = Counter(doc_tokens)

        for token in query_tokens:
            if token not in self.idf:
                continue

            tf = term_counts[token]
            if tf == 0:
                continue

            idf = self.idf[token]
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * (
                1.0 - self.b + self.b * (doc_length / self.avgdl)
            )
            score += idf * (numerator / denominator)

        return score

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        category_filter: str | None = None,
    ) -> list[SearchResultSchema]:
        """Tìm kiếm và trả về Top-K tài liệu có điểm BM25 cao nhất."""
        if not self.documents:
            return []

        query_tokens = self.tokenize(query_text)
        if not query_tokens:
            return []

        scored_docs: list[tuple[float, DocumentSchema]] = []

        for idx, doc in enumerate(self.documents):
            # Lọc theo thể loại nếu người dùng có truyền bộ lọc
            if (
                category_filter
                and doc.category.lower() != category_filter.lower()
            ):
                continue

            score = self._score_document(query_tokens, idx)
            # Chỉ giữ lại những tài liệu có ít nhất 1 từ khóa khớp (điểm > 0)
            if score > 0.0:
                scored_docs.append((score, doc))

        # Sắp xếp giảm dần theo điểm số BM25
        scored_docs.sort(key=lambda item: item[0], reverse=True)

        results: list[SearchResultSchema] = []
        for score, doc in scored_docs[:top_k]:
            results.append(
                SearchResultSchema(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    content=doc.content,
                    lexical_score=round(score, 4),
                    vector_score=0.0,
                    final_score=round(score, 4),
                    metadata=doc.metadata,
                )
            )

        return results