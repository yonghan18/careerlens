from pathlib import Path
from types import SimpleNamespace

from app.config import ROOT, Settings
from app.budget import BudgetLedger, PriceBook
from app.retrieval import CorpusStore
from app.workflow import ResearchWorkflow


def make_settings(tmp_path: Path) -> Settings:
    data = tmp_path / "data"
    return Settings(
        root=tmp_path,
        data_dir=data,
        sample_dir=data / "sample",
        upload_dir=data / "uploads",
        index_dir=data / "index",
        chunks_file=data / "index" / "chunks.json",
        qdrant_dir=data / "index" / "qdrant",
        openai_api_key=None,
        openai_model="unused",
        embedding_model="unused",
        origins=("http://localhost:5173",),
    )


def build_workflow(tmp_path: Path) -> ResearchWorkflow:
    config = make_settings(tmp_path)
    config.ensure_directories()
    (config.sample_dir / "policy.md").write_text(
        "# Response policy\n\nP1 incidents must be acknowledged within 10 minutes.\n\n"
        "Production changes require a rollback plan and peer approval.",
        encoding="utf-8",
    )
    budget = BudgetLedger(
        tmp_path / "budget.json",
        4.5,
        PriceBook(0.2, 1.2, 0.02),
    )
    store = CorpusStore(config, budget)
    store.rebuild()
    return ResearchWorkflow(store, config, budget)


def test_retrieval_returns_expected_evidence(tmp_path: Path) -> None:
    workflow = build_workflow(tmp_path)
    response = workflow.ask("How quickly must a P1 incident be acknowledged?")
    assert response.status == "answered"
    assert response.mode == "evidence_preview"
    assert response.sources[0].document == "policy.md"
    assert "10 minutes" in response.answer
    assert "[S1]" in response.answer


def test_unknown_question_abstains(tmp_path: Path) -> None:
    workflow = build_workflow(tmp_path)
    response = workflow.ask("What is the cafeteria lunch menu?")
    assert response.status == "insufficient_evidence"
    assert response.sources == []


def test_job_description_is_request_scoped_and_cited(tmp_path: Path) -> None:
    workflow = build_workflow(tmp_path)
    captured: dict[str, object] = {}

    def create(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            output_text="P1 response is documented [S1] and fits the role [JD].",
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )

    workflow.client = SimpleNamespace(
        responses=SimpleNamespace(create=create)
    )
    role_description = "The role requires P1 incident response experience."
    response = workflow.ask(
        "How does the documented P1 response match this role?",
        role_description=role_description,
    )

    assert response.status == "answered"
    assert "[JD]" in response.answer
    assert role_description in str(captured["input"])
    assert "reasoning" not in captured
    assert all(role_description not in chunk.text for chunk in workflow.store.chunks)
    assert not any(tmp_path.rglob("*role*"))


def test_only_personal_documents_are_indexed(tmp_path: Path) -> None:
    config = make_settings(tmp_path)
    config = Settings(
        **{
            **config.__dict__,
            "sample_dir": ROOT / "data" / "sample",
        }
    )
    budget = BudgetLedger(
        tmp_path / "budget.json", 4.5, PriceBook(0.2, 1.2, 0.02)
    )
    store = CorpusStore(config, budget)
    store.rebuild()

    assert {item.name for item in store.list_documents()} == {
        "candidate-profile.md",
        "automation-case-study.md",
        "engineering-case-study.md",
        "careerlens-project.md",
    }
