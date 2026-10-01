# Public synthetic retrieval baseline

Measured on 2026-10-01 using the included four fictional/demo documents, 30 questions, BM25-only retrieval, and k=5. No OpenAI calls were made.

- Answerable document Recall@5: 0.938.
- Answerable MRR@5: 0.861.
- Factual and role-comparison document recall: 1.000.
- Cross-document synthesis recall: 0.500.

This baseline exposes a retrieval weakness: some synthesis questions retrieve only one of the two labelled documents. Unknown questions can also retrieve overlapping words from irrelevant passages. Returning excerpts in offline mode is not evidence that the requested fact was found. LLM abstention, answer quality, citation support, and injection resistance have not been measured for this public corpus.

Reproduce with `python -m app.eval_cli` with `OPENAI_API_KEY` unset. Small latency values may round to zero milliseconds in this tiny in-memory retrieval experiment; they are not full-pipeline latency results.
