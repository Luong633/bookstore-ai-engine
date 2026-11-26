import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from src.core.schemas import DocumentSchema, SearchResultSchema


class SemanticRetriever:
    """
    Bộ tìm kiếm ngữ nghĩa (Semantic Search) sử dụng mô hình Deep Learning Bi-Encoder.
    Mã hóa văn bản thành Vector 384 chiều và tính độ tương đồng bằng NumPy Vectorization.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self.documents: list[DocumentSchema] = []
        self.embeddings: np.ndarray | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Tải mô hình theo cơ chế Lazy Loading (chỉ nạp vào RAM khi thực sự cần dùng)."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Chuyển danh sách chuỗi văn bản thành ma trận Vector đã chuẩn hóa L2."""
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Chuẩn hóa độ dài vector = 1 để dùng phép nhân dot
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def fit(self, documents: list[DocumentSchema]) -> None:
        """Mã hóa toàn bộ kho tài liệu thành ma trận Vector lưu trên RAM."""
        self.documents = documents
        if not documents:
            self.embeddings = None
            return

        corpus_texts = [
            f"{doc.title} - {doc.author} ({doc.category}): {doc.content}"
            for doc in documents
        ]
        self.embeddings = self.encode_texts(corpus_texts)

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        category_filter: str | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResultSchema]:
        """Tìm kiếm Top-K tài liệu gần nhất về mặt ngữ nghĩa."""
        if not self.documents or self.embeddings is None or not query_text.strip():
            return []

        # Mã hóa câu truy vấn thành vector 1 chiều (384,)
        query_vec = self.encode_texts([query_text])[0]

        # Vì các vector đã chuẩn hóa L2, tích vô hướng (dot product) chính là Cosine Similarity
        similarities: np.ndarray = np.dot(self.embeddings, query_vec)

        # Lấy danh sách chỉ số sắp xếp giảm dần theo điểm tương đồng
        sorted_indices = np.argsort(similarities)[::-1]

        results: list[SearchResultSchema] = []
        for idx in sorted_indices:
            doc = self.documents[int(idx)]
            score = float(similarities[int(idx)])

            if score < min_score:
                break

            if (
                category_filter
                and doc.category.lower() != category_filter.lower()
            ):
                continue

            results.append(
                SearchResultSchema(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    content=doc.content,
                    lexical_score=0.0,
                    vector_score=round(score, 4),
                    final_score=round(score, 4),
                    metadata=doc.metadata,
                )
            )

            if len(results) >= top_k:
                break

        return results

    def save_index(self, dir_path: str | Path) -> None:
        """Lưu ma trận vector (.npy) và danh sách tài liệu (.json) xuống ổ cứng."""
        if self.embeddings is None:
            raise ValueError("Chưa có dữ liệu embeddings để lưu. Hãy gọi fit() trước.")

        folder = Path(dir_path)
        folder.mkdir(parents=True, exist_ok=True)

        np.save(folder / "embeddings.npy", self.embeddings)
        meta_payload = {
            "model_name": self.model_name,
            "documents": [doc.model_dump() for doc in self.documents],
        }
        with (folder / "semantic_docs.json").open("w", encoding="utf-8") as f:
            json.dump(meta_payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_index(cls, dir_path: str | Path) -> "SemanticRetriever":
        """Khôi phục bộ tìm kiếm ngữ nghĩa từ ổ cứng."""
        folder = Path(dir_path)
        with (folder / "semantic_docs.json").open("r", encoding="utf-8") as f:
            meta_payload = json.load(f)

        retriever = cls(model_name=meta_payload["model_name"])
        retriever.documents = [
            DocumentSchema(**doc_data) for doc_data in meta_payload["documents"]
        ]
        retriever.embeddings = np.load(folder / "embeddings.npy")
        return retriever