from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    name: str
    chunks: int
    source_type: str


class BudgetStatus(BaseModel):
    cap_usd: float
    spent_usd: float
    remaining_usd: float
    recorded_calls: int


class SourceChunk(BaseModel):
    source_id: str
    document: str
    chunk_index: int
    text: str
    score: float
    retrieval_methods: list[str]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1_000)
    top_k: int = Field(default=5, ge=1, le=10)
    role_description: str | None = Field(default=None, max_length=12_000)


class TraceStep(BaseModel):
    name: str
    detail: str
    status: Literal["complete", "skipped", "warning"] = "complete"


class AskResponse(BaseModel):
    answer: str
    status: Literal["answered", "insufficient_evidence"]
    mode: Literal["openai", "evidence_preview"]
    sources: list[SourceChunk]
    trace: list[TraceStep]
    latency_ms: int


class EvaluationCaseResult(BaseModel):
    case_id: str
    category: str
    question: str
    expected_documents: list[str]
    retrieved_documents: list[str]
    retrieval_recall: float
    reciprocal_rank: float
    expected_status: Literal["answered", "insufficient_evidence"]
    actual_status: Literal["answered", "insufficient_evidence"]
    status_correct: bool
    expected_terms_found: bool | None = None
    forbidden_terms_absent: bool | None = None
    citation_valid: bool | None = None
    latency_ms: int


class CategoryMetric(BaseModel):
    category: str
    cases: int
    retrieval_recall_at_k: float | None = None
    status_accuracy: float | None = None


class EvaluationResponse(BaseModel):
    mode: Literal["retrieval", "full"]
    retrieval_mode: Literal["lexical", "hybrid"]
    top_k: int
    total_cases: int
    retrieval_recall_at_k: float
    mean_reciprocal_rank: float
    abstention_accuracy: float | None = None
    answer_key_term_accuracy: float | None = None
    citation_validity_rate: float | None = None
    safety_pass_rate: float | None = None
    average_latency_ms: float
    categories: list[CategoryMetric]
    cases: list[EvaluationCaseResult]
