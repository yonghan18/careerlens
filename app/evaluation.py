from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal

from app.models import CategoryMetric, EvaluationCaseResult, EvaluationResponse
from app.workflow import ResearchWorkflow


def _mean(values: list[float | bool]) -> float:
    return round(sum(float(value) for value in values) / max(len(values), 1), 3)


def run_evaluation(
    workflow: ResearchWorkflow,
    dataset_path: Path,
    mode: Literal["retrieval", "full"] = "retrieval",
    retrieval_mode: Literal["lexical", "hybrid"] = "hybrid",
    top_k: int = 5,
) -> EvaluationResponse:
    cases = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results: list[EvaluationCaseResult] = []
    for case in cases:
        started = time.perf_counter()
        role_description = (
            (dataset_path.parent / case["role_fixture"]).read_text(encoding="utf-8")
            if case.get("role_fixture")
            else None
        )
        if mode == "full":
            response = workflow.ask(
                case["question"], top_k, role_description=role_description
            )
            sources = response.sources
            answer = response.answer.lower()
            actual_status = response.status
            cited = [int(value) for value in re.findall(r"\[S(\d+)\]", response.answer)]
            cited_role = "[JD]" in response.answer
            citation_valid = (
                (bool(cited) or cited_role)
                and all(1 <= value <= len(sources) for value in cited)
                and (not cited_role or bool(role_description))
                if actual_status == "answered"
                else None
            )
            terms_found = all(
                term.lower() in answer for term in case.get("expected_terms", [])
            ) if case["expected_status"] == "answered" else None
            forbidden_absent = all(
                term.lower() not in answer for term in case.get("forbidden_terms", [])
            ) if case.get("forbidden_terms") else None
            latency_ms = response.latency_ms
        else:
            sources = workflow.store.search(
                case["question"], top_k, retrieval_mode=retrieval_mode
            )
            actual_status = "answered" if sources else "insufficient_evidence"
            citation_valid = None
            terms_found = None
            forbidden_absent = None
            latency_ms = round((time.perf_counter() - started) * 1_000)

        retrieved_documents = list(dict.fromkeys(source.document for source in sources))
        expected_documents = case.get("expected_documents", [])
        relevant_ranks = [
            index
            for index, source in enumerate(sources, start=1)
            if source.document in expected_documents
        ]
        recall = (
            len(set(retrieved_documents) & set(expected_documents))
            / len(expected_documents)
            if expected_documents
            else (1.0 if not sources else 0.0)
        )
        reciprocal_rank = 1 / min(relevant_ranks) if relevant_ranks else 0.0
        results.append(
            EvaluationCaseResult(
                case_id=case["id"],
                category=case["category"],
                question=case["question"],
                expected_documents=expected_documents,
                retrieved_documents=retrieved_documents,
                retrieval_recall=round(recall, 3),
                reciprocal_rank=round(reciprocal_rank, 3),
                expected_status=case["expected_status"],
                actual_status=actual_status,
                status_correct=actual_status == case["expected_status"],
                expected_terms_found=terms_found,
                forbidden_terms_absent=forbidden_absent,
                citation_valid=citation_valid,
                latency_ms=latency_ms,
            )
        )

    answer_cases = [item for item in results if item.expected_terms_found is not None]
    citation_cases = [item for item in results if item.citation_valid is not None]
    safety_cases = [item for item in results if item.forbidden_terms_absent is not None]
    answerable_retrieval_cases = [item for item in results if item.expected_documents]
    abstention_cases = [
        item for item in results if item.expected_status == "insufficient_evidence"
    ]
    grouped: dict[str, list[EvaluationCaseResult]] = defaultdict(list)
    for item in results:
        grouped[item.category].append(item)

    return EvaluationResponse(
        mode=mode,
        retrieval_mode="hybrid" if mode == "full" else retrieval_mode,
        top_k=top_k,
        total_cases=len(results),
        retrieval_recall_at_k=_mean(
            [item.retrieval_recall for item in answerable_retrieval_cases]
        ),
        mean_reciprocal_rank=_mean(
            [item.reciprocal_rank for item in answerable_retrieval_cases]
        ),
        abstention_accuracy=(
            _mean([item.status_correct for item in abstention_cases])
            if mode == "full" and abstention_cases
            else None
        ),
        answer_key_term_accuracy=(
            _mean([bool(item.expected_terms_found) for item in answer_cases])
            if answer_cases else None
        ),
        citation_validity_rate=(
            _mean([bool(item.citation_valid) for item in citation_cases])
            if citation_cases else None
        ),
        safety_pass_rate=(
            _mean([bool(item.forbidden_terms_absent) for item in safety_cases])
            if safety_cases else None
        ),
        average_latency_ms=round(
            sum(item.latency_ms for item in results) / max(len(results), 1), 1
        ),
        categories=[
            CategoryMetric(
                category=category,
                cases=len(items),
                retrieval_recall_at_k=(
                    _mean([item.retrieval_recall for item in items if item.expected_documents])
                    if any(item.expected_documents for item in items)
                    else None
                ),
                status_accuracy=(
                    _mean([item.status_correct for item in items])
                    if mode == "full"
                    else None
                ),
            )
            for category, items in sorted(grouped.items())
        ],
        cases=results,
    )
