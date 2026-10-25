from pathlib import Path
from types import SimpleNamespace

import pytest

from app.retrieval import COLLECTION_NAME
from tests.test_workflow import build_workflow


def test_hybrid_search_ignores_obsolete_dense_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow = build_workflow(tmp_path)
    store = workflow.store
    current_id = store.chunks[0].source_id
    obsolete_id = "00000000-0000-4000-8000-000000000001"
    monkeypatch.setattr(
        store,
        "_dense_rank",
        lambda question, limit: [(obsolete_id, 1.0), (current_id, 0.8)],
    )

    sources = store.search("How quickly must a P1 incident be acknowledged?")

    assert sources
    assert sources[0].document == "policy.md"
    assert "10 minutes" in sources[0].text
    assert current_id in {source.source_id for source in sources}
    assert obsolete_id not in {source.source_id for source in sources}
    assert all(source.source_id == current_id for source in sources)


def test_pruning_removes_only_obsolete_ids_across_pages(tmp_path: Path) -> None:
    store = build_workflow(tmp_path).store
    current_id = store.chunks[0].source_id
    obsolete_ids = [
        "00000000-0000-4000-8000-000000000001",
        "00000000-0000-4000-8000-000000000002",
    ]
    offsets: list[object] = []
    deletions: list[dict[str, object]] = []

    def scroll(**kwargs: object) -> tuple[list[SimpleNamespace], object]:
        assert kwargs["collection_name"] == COLLECTION_NAME
        assert kwargs["with_payload"] is False
        assert kwargs["with_vectors"] is False
        offsets.append(kwargs["offset"])
        if kwargs["offset"] is None:
            return [
                SimpleNamespace(id=current_id),
                SimpleNamespace(id=obsolete_ids[0]),
            ], "next-page"
        assert kwargs["offset"] == "next-page"
        return [SimpleNamespace(id=obsolete_ids[1])], None

    def delete(**kwargs: object) -> None:
        deletions.append(kwargs)

    store._qdrant = SimpleNamespace(scroll=scroll, delete=delete)
    store._prune_dense_index()

    assert offsets == [None, "next-page"]
    assert len(deletions) == 1
    deletion = deletions[0]
    assert deletion["collection_name"] == COLLECTION_NAME
    assert deletion["wait"] is True
    assert set(deletion["points_selector"].points) == set(obsolete_ids)
    assert current_id not in deletion["points_selector"].points
