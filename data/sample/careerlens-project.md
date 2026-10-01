# CareerLens project record

CareerLens is a local personal evidence assistant built with FastAPI and React/TypeScript. It ingests Markdown, text, and text-based PDF documents and creates stable paragraph chunks. BM25 supplies lexical retrieval; OpenAI embeddings and local Qdrant add dense retrieval when an API key is configured.

A LangGraph workflow retrieves evidence, generates a cited answer or abstains, and validates citation identifiers. An optional job description is supplied for one question only and is not persisted in the document index. The workflow exposes retrieved passages and execution steps for inspection.

The public demo uses fictional candidate documents and a 30-case evaluation dataset. It measures Recall@5, MRR@5, and, in full mode, abstention, answer key terms, citation identifiers, latency, and forbidden-output avoidance. Identifier validity does not establish factual support for every claim.

CareerLens has no web search fallback, autonomous tool selection, MCP server, or deployment with real users. Its budget ledger reserves estimated costs before provider calls and records reported token costs afterward. Uploads, generated indexes, credentials, and budget records remain outside version control.
