from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass(frozen=True)
class Chunk:
    source_id: str
    document: str
    chunk_index: int
    text: str
    source_type: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document type: {suffix}")
    if suffix == ".pdf":
        reader = PdfReader(path)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def chunk_document(path: Path, max_chars: int = 650) -> list[Chunk]:
    text = re.sub(r"\r\n?", "\n", extract_text(path)).strip()
    paragraphs = [re.sub(r"\s+", " ", part).strip() for part in re.split(r"\n\s*\n", text)]
    paragraphs = [part for part in paragraphs if part]
    grouped: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > max_chars:
            grouped.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}".strip()
    if current:
        grouped.append(current)

    chunks: list[Chunk] = []
    for index, content in enumerate(grouped):
        stable_id = str(uuid5(NAMESPACE_URL, f"{path.name}:{index}:{content}"))
        chunks.append(
            Chunk(
                source_id=stable_id,
                document=path.name,
                chunk_index=index,
                text=content,
                source_type=path.suffix.lower().lstrip("."),
            )
        )
    return chunks
