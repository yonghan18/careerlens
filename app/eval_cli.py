from __future__ import annotations

import json

from app.config import settings
from app.evaluation import run_evaluation
from app.main import workflow


if __name__ == "__main__":
    try:
        result = run_evaluation(
            workflow,
            settings.root / "evals" / "golden.jsonl",
            mode="retrieval",
            retrieval_mode="lexical",
        )
        print(json.dumps(result.model_dump(), indent=2))
    finally:
        workflow.store.close()
