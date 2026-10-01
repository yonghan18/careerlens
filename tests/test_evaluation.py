import json
from pathlib import Path

from app.evaluation import run_evaluation
from tests.test_workflow import build_workflow


def test_retrieval_evaluation_scores_answerable_and_unknown_cases(
    tmp_path: Path,
) -> None:
    workflow = build_workflow(tmp_path)
    dataset = tmp_path / "evaluation.jsonl"
    cases = [
        {
            "id": "known",
            "category": "factual",
            "question": "When must P1 incidents be acknowledged?",
            "expected_documents": ["policy.md"],
            "expected_terms": ["10 minutes"],
            "expected_status": "answered",
        },
        {
            "id": "unknown",
            "category": "unanswerable",
            "question": "What is the cafeteria lunch menu?",
            "expected_documents": [],
            "expected_terms": [],
            "expected_status": "insufficient_evidence",
        },
    ]
    dataset.write_text(
        "\n".join(json.dumps(case) for case in cases), encoding="utf-8"
    )

    result = run_evaluation(workflow, dataset, mode="retrieval", top_k=5)

    assert result.total_cases == 2
    assert result.retrieval_recall_at_k == 1.0
    assert result.mean_reciprocal_rank == 1.0
    assert result.abstention_accuracy is None
    assert result.answer_key_term_accuracy is None
