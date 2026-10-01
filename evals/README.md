# Public demo evaluation

`golden.jsonl` contains 30 labelled synthetic cases: 12 factual, 3 synthesis, 3 request-scoped role comparisons, 5 evidence-boundary, 5 unanswerable, and 2 adversarial cases. The candidate and employer records are fictional. The JD fixtures are test inputs outside the personal corpus.

Run `python -m app.eval_cli` without an API key for lexical retrieval. This reports Recall@5 and MRR@5 over answerable cases. Recall is the fraction of labelled relevant documents found among retrieved passages; MRR is the reciprocal rank of the first relevant passage. Labels are at document level, not proof of passage-level completeness.

Full mode is available through `POST /api/evaluate?mode=full`; with a key it makes paid calls. It measures abstention accuracy on cases labelled insufficient evidence, strict answer key-term accuracy, valid citation identifiers, forbidden-output avoidance, and latency. Full mode without a key evaluates evidence previews rather than model answers, so those results must not be reported as LLM performance.

Citation validity checks reference syntax/range and the presence of request-scoped JD context, not support for each claim. The two adversarial cases exercise JD injection and requests to invent metrics. They do not establish comprehensive prompt-injection robustness.

Private historical results are excluded. Generated results are ignored by Git. Review labels and inspect per-case failures before reporting any score. The public dataset differs from the original personal benchmark.
