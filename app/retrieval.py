from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Literal

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams

from app.budget import BudgetLedger
from app.chunking import Chunk, SUPPORTED_EXTENSIONS, chunk_document
from app.config import Settings
from app.models import DocumentSummary, SourceChunk


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
COLLECTION_NAME = "evidence_chunks"
EMBEDDING_SIZE = 1536
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "what", "when", "which", "who", "with", "must", "should",
}


def tokenize(text: str) -> list[str]:
    return [
        token for token in TOKEN_PATTERN.findall(text.lower())
        if token not in STOP_WORDS and len(token) > 1
    ]


class CorpusStore:
    def __init__(self, config: Settings, budget: BudgetLedger):
        self.config = config
        self.budget = budget
        self.config.ensure_directories()
        self.chunks: list[Chunk] = []
        self._openai = (
            OpenAI(api_key=config.openai_api_key, max_retries=0)
            if config.llm_enabled
            else None
        )
        self._qdrant: QdrantClient | None = None
        self._load_chunks()

    def _load_chunks(self) -> None:
        if not self.config.chunks_file.exists():
            return
        data = json.loads(self.config.chunks_file.read_text(encoding="utf-8"))
        self.chunks = [Chunk(**item) for item in data]

    def _save_chunks(self) -> None:
        payload = [chunk.to_dict() for chunk in self.chunks]
        self.config.chunks_file.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def rebuild(self) -> int:
        paths = sorted(
            path
            for directory in (self.config.sample_dir, self.config.upload_dir)
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        self.chunks = [chunk for path in paths for chunk in chunk_document(path)]
        self._save_chunks()
        if self.config.llm_enabled and self.chunks:
            self._build_dense_index()
        return len(self.chunks)

    def list_documents(self) -> list[DocumentSummary]:
        counts: Counter[tuple[str, str]] = Counter(
            (chunk.document, chunk.source_type) for chunk in self.chunks
        )
        return [
            DocumentSummary(name=name, chunks=count, source_type=source_type)
            for (name, source_type), count in sorted(counts.items())
        ]

    def _build_dense_index(self) -> None:
        assert self._openai is not None
        texts = [chunk.text for chunk in self.chunks]
        reservation = self.budget.reserve(
            "document_embeddings",
            self.config.embedding_model,
            self.budget.estimate_embedding(texts),
        )
        try:
            response = self._openai.embeddings.create(
                model=self.config.embedding_model,
                input=texts,
            )
        except Exception as exc:
            self.budget.mark_uncertain(reservation, type(exc).__name__)
            raise
        input_tokens = int(response.usage.total_tokens)
        self.budget.commit(
            reservation,
            self.budget.prices.embedding_cost(input_tokens),
            input_tokens,
        )
        vectors = [item.embedding for item in response.data]
        vector_size = len(vectors[0])
        self.close()
        self._qdrant = QdrantClient(path=str(self.config.qdrant_dir))
        if self._qdrant.collection_exists(COLLECTION_NAME):
            self._qdrant.delete_collection(COLLECTION_NAME)
        self._qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        self._qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=chunk.source_id,
                    vector=vector,
                    payload=chunk.to_dict(),
                )
                for chunk, vector in zip(self.chunks, vectors, strict=True)
            ],
        )
        self._prune_dense_index()

    def _prune_dense_index(self) -> None:
        if self._qdrant is None:
            return
        known_ids = {chunk.source_id for chunk in self.chunks}
        stale_ids: list[str] = []
        offset = None
        while True:
            points, offset = self._qdrant.scroll(
                collection_name=COLLECTION_NAME,
                offset=offset,
                limit=100,
                with_payload=False,
                with_vectors=False,
            )
            stale_ids.extend(
                str(point.id) for point in points if str(point.id) not in known_ids
            )
            if offset is None:
                break
        if stale_ids:
            self._qdrant.delete(
                collection_name=COLLECTION_NAME,
                points_selector=PointIdsList(points=stale_ids),
                wait=True,
            )

    def _bm25_rank(self, question: str) -> list[tuple[str, float]]:
        query = tokenize(question)
        documents = [tokenize(chunk.text) for chunk in self.chunks]
        if not query or not documents:
            return []
        average_length = sum(map(len, documents)) / len(documents)
        document_frequency = Counter(
            token for document in documents for token in set(document)
        )
        scores: list[tuple[str, float]] = []
        k1, b = 1.5, 0.75
        for chunk, tokens in zip(self.chunks, documents, strict=True):
            frequencies = Counter(tokens)
            score = 0.0
            for token in query:
                frequency = frequencies[token]
                if not frequency:
                    continue
                df = document_frequency[token]
                idf = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
                denominator = frequency + k1 * (
                    1 - b + b * len(tokens) / max(average_length, 1)
                )
                score += idf * frequency * (k1 + 1) / denominator
            if score > 0:
                scores.append((chunk.source_id, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)

    def _dense_rank(self, question: str, limit: int) -> list[tuple[str, float]]:
        if not self.config.llm_enabled or self._openai is None:
            return []
        if self._qdrant is None:
            if not self.config.qdrant_dir.exists():
                self._build_dense_index()
            else:
                self._qdrant = QdrantClient(path=str(self.config.qdrant_dir))
                self._prune_dense_index()
        reservation = self.budget.reserve(
            "query_embedding",
            self.config.embedding_model,
            self.budget.estimate_embedding([question]),
        )
        try:
            response = self._openai.embeddings.create(
                model=self.config.embedding_model, input=[question]
            )
        except Exception as exc:
            self.budget.mark_uncertain(reservation, type(exc).__name__)
            raise
        input_tokens = int(response.usage.total_tokens)
        self.budget.commit(
            reservation,
            self.budget.prices.embedding_cost(input_tokens),
            input_tokens,
        )
        vector = response.data[0].embedding
        result = self._qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=limit,
            with_payload=False,
        )
        return [(str(point.id), float(point.score)) for point in result.points]

    def search(
        self,
        question: str,
        top_k: int = 5,
        retrieval_mode: Literal["lexical", "hybrid"] = "hybrid",
    ) -> list[SourceChunk]:
        bm25 = self._bm25_rank(question)
        dense = (
            self._dense_rank(question, max(top_k * 2, 10))
            if retrieval_mode == "hybrid"
            else []
        )
        bm25_max = bm25[0][1] if bm25 else 1.0
        bm25_scores = {source_id: score / bm25_max for source_id, score in bm25}
        dense_scores = {
            source_id: max(0.0, min(1.0, score)) for source_id, score in dense
        }
        known_ids = {chunk.source_id for chunk in self.chunks}
        source_ids = (set(bm25_scores) | set(dense_scores)) & known_ids
        fused = []
        for source_id in source_ids:
            methods = []
            if source_id in bm25_scores:
                methods.append("bm25")
            if source_id in dense_scores:
                methods.append("dense")
            if bm25_scores and dense_scores:
                score = 0.5 * bm25_scores.get(source_id, 0.0) + 0.5 * dense_scores.get(source_id, 0.0)
            else:
                score = bm25_scores.get(source_id, dense_scores.get(source_id, 0.0))
            if score >= 0.25:
                fused.append((source_id, score, methods))
        fused.sort(key=lambda item: item[1], reverse=True)
        chunk_by_id = {chunk.source_id: chunk for chunk in self.chunks}
        # Preserve the two strongest passages, then diversify the remaining
        # slots so comparison questions can receive evidence from other documents.
        selected: list[tuple[str, float, list[str]]] = fused[: min(2, top_k)]
        seen_documents = {
            chunk_by_id[item[0]].document for item in selected
        }
        for item in fused[min(2, top_k):]:
            document = chunk_by_id[item[0]].document
            if document not in seen_documents and len(selected) < top_k:
                selected.append(item)
                seen_documents.add(document)
        if len(selected) < top_k:
            selected_ids = {item[0] for item in selected}
            selected.extend(
                item
                for item in fused
                if item[0] not in selected_ids
            )
        return [
            SourceChunk(
                source_id=source_id,
                document=chunk_by_id[source_id].document,
                chunk_index=chunk_by_id[source_id].chunk_index,
                text=chunk_by_id[source_id].text,
                score=round(score, 4),
                retrieval_methods=methods,
            )
            for source_id, score, methods in selected[:top_k]
            if source_id in chunk_by_id
        ]

    def close(self) -> None:
        if self._qdrant is not None:
            self._qdrant.close()
            self._qdrant = None
