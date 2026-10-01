from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAIError

from app.budget import BudgetExceeded, BudgetLedger, PriceBook
from app.chunking import SUPPORTED_EXTENSIONS
from app.config import settings
from app.evaluation import run_evaluation
from app.models import AskRequest, AskResponse, BudgetStatus, DocumentSummary, EvaluationResponse
from app.retrieval import CorpusStore
from app.workflow import ResearchWorkflow


settings.ensure_directories()
budget = BudgetLedger(
    settings.root / "data" / "budget" / "ledger.json",
    settings.budget_usd,
    PriceBook(
        input_per_million=settings.input_usd_per_million,
        output_per_million=settings.output_usd_per_million,
        embedding_per_million=settings.embedding_usd_per_million,
    ),
)
store = CorpusStore(settings, budget)
if not store.chunks:
    store.rebuild()
workflow = ResearchWorkflow(store, settings, budget)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    store.close()

app = FastAPI(title="CareerLens", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(BudgetExceeded)
def budget_exceeded_handler(_, exc: BudgetExceeded) -> JSONResponse:
    return JSONResponse(status_code=402, content={"detail": str(exc)})


@app.exception_handler(OpenAIError)
def openai_error_handler(_, __: OpenAIError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "detail": "The AI provider request failed. No automatic retry was made; "
            "check connectivity and the budget ledger before trying again."
        },
    )


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "llm_enabled": settings.llm_enabled,
        "mode": "hybrid" if settings.llm_enabled else "bm25",
        "documents": len(store.list_documents()),
        "chunks": len(store.chunks),
        "budget": budget.status(),
    }


@app.get("/api/budget", response_model=BudgetStatus)
def budget_status() -> BudgetStatus:
    return BudgetStatus(**budget.status())


@app.get("/api/evaluation/latest")
def latest_evaluation() -> dict[str, object]:
    path = settings.root / "evals" / "results-demo.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/documents", response_model=list[DocumentSummary])
def documents() -> list[DocumentSummary]:
    return store.list_documents()


@app.post("/api/documents", response_model=list[DocumentSummary])
async def upload_documents(files: list[UploadFile] = File(...)) -> list[DocumentSummary]:
    for upload in files:
        original = Path(upload.filename or "document.txt")
        suffix = original.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=415, detail=f"Unsupported file type: {suffix}")
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", original.stem).strip(".-") or "document"
        destination = settings.upload_dir / f"{safe_stem}{suffix}"
        destination.write_bytes(await upload.read())
    store.rebuild()
    return store.list_documents()


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return workflow.ask(request.question, request.top_k, request.role_description)


@app.post("/api/evaluate", response_model=EvaluationResponse)
def evaluate(
    mode: Literal["retrieval", "full"] = "retrieval",
    retrieval_mode: Literal["lexical", "hybrid"] = "hybrid",
) -> EvaluationResponse:
    return run_evaluation(
        workflow,
        settings.root / "evals" / "golden.jsonl",
        mode=mode,
        retrieval_mode=retrieval_mode,
    )
