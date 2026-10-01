import { useEffect, useState } from "react";
import {
  BookOpen,
  Check,
  ChevronDown,
  Clock3,
  FileText,
  FlaskConical,
  Search,
  ShieldCheck,
  Upload,
} from "lucide-react";

type Health = {
  status: string;
  llm_enabled: boolean;
  mode: string;
  documents: number;
  chunks: number;
  budget: { cap_usd: number; spent_usd: number; remaining_usd: number; recorded_calls: number };
};

type Document = { name: string; chunks: number; source_type: string };
type Source = {
  source_id: string;
  document: string;
  chunk_index: number;
  text: string;
  score: number;
  retrieval_methods: string[];
};
type Trace = { name: string; detail: string; status: string };
type Answer = {
  answer: string;
  status: string;
  mode: string;
  sources: Source[];
  trace: Trace[];
  latency_ms: number;
};
type Evaluation = {
  mode: "retrieval" | "full";
  retrieval_mode: "lexical" | "hybrid";
  total_cases: number;
  retrieval_recall_at_k: number;
  mean_reciprocal_rank: number;
  abstention_accuracy: number | null;
  average_latency_ms: number;
};
type Benchmark = {
  dataset_cases: number;
  full_pipeline: {
    retrieval_recall_at_5: number;
    mean_reciprocal_rank_at_5: number;
    abstention_accuracy: number;
    answer_key_term_accuracy: number;
    citation_validity_rate: number;
    retrieved_content_safety_pass_rate: number;
    average_latency_ms: number;
  };
};

const starterQuestions = [
  "What did Alex build at Northstar Media?",
  "Which projects demonstrate Generative AI work?",
  "Which achievements are documented?",
];

async function getJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = payload && typeof payload.detail === "string" ? payload.detail : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  return payload as T;
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [question, setQuestion] = useState(starterQuestions[0]);
  const [roleDescription, setRoleDescription] = useState("");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [benchmark, setBenchmark] = useState<Benchmark | null>(null);
  const [answerUsedRole, setAnswerUsedRole] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const refresh = async () => {
    const [nextHealth, nextDocuments, nextBenchmark] = await Promise.all([
      getJson<Health>("/api/health"),
      getJson<Document[]>("/api/documents"),
      getJson<Benchmark | Record<string, never>>("/api/evaluation/latest"),
    ]);
    setHealth(nextHealth);
    setDocuments(nextDocuments);
    setBenchmark("full_pipeline" in nextBenchmark ? nextBenchmark as Benchmark : null);
  };

  useEffect(() => {
    refresh().catch((reason) => setError(reason.message));
  }, []);

  const ask = async () => {
    if (!question.trim()) return;
    setLoading(true);
    setError("");
    try {
      const nextAnswer = await getJson<Answer>("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          top_k: 5,
          role_description: roleDescription.trim() || null,
        }),
      });
      setAnswer(nextAnswer);
      setAnswerUsedRole(Boolean(roleDescription.trim()));
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to answer");
    } finally {
      setLoading(false);
    }
  };

  const upload = async (files: FileList | null) => {
    if (!files?.length) return;
    setLoading(true);
    const body = new FormData();
    Array.from(files).forEach((file) => body.append("files", file));
    try {
      await getJson<Document[]>("/api/documents", { method: "POST", body });
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  const evaluate = async () => {
    setLoading(true);
    try {
      const nextEvaluation = await getJson<Evaluation>("/api/evaluate", { method: "POST" });
      setEvaluation(nextEvaluation);
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Evaluation failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-mark"><BookOpen size={19} /></div>
        <div>
          <h1>CareerLens</h1>
          <p>Grounded application intelligence</p>
        </div>
        <div className="system-status">
          <span className="status-dot" />
          {health?.llm_enabled ? "Hybrid RAG · OpenAI enabled" : "Lexical retrieval · Model disabled"}
          {health && <span className="budget-chip">${health.budget.remaining_usd.toFixed(4)} remaining</span>}
        </div>
      </header>

      <main className="workspace">
        <aside className="library panel">
          <div className="panel-heading">
            <div><span className="eyebrow">Personal knowledge base</span><h2>Your evidence</h2></div>
            <label className="icon-button" title="Upload personal evidence">
              <Upload size={18} />
              <input type="file" multiple accept=".txt,.md,.pdf" onChange={(event) => upload(event.target.files)} />
            </label>
          </div>
          <p className="helper">Your profile, work, and projects are indexed here. Upload more personal evidence when you have it.</p>
          <div className="document-list">
            {documents.map((document) => (
              <div className="document-row" key={document.name}>
                <div className="file-icon"><FileText size={17} /></div>
                <div><strong>{document.name}</strong><span>{document.chunks} chunks · {document.source_type.toUpperCase()}</span></div>
              </div>
            ))}
          </div>
          <div className="library-footer">
            <ShieldCheck size={16} />
            <span>{health?.chunks ?? 0} indexed chunks</span>
          </div>
        </aside>

        <section className="research panel">
          <div className="panel-heading">
            <div><span className="eyebrow">Grounded career research</span><h2>Ask your evidence</h2></div>
            <button className="secondary-button" onClick={evaluate} disabled={loading}>
              <FlaskConical size={16} /> Evaluate retrieval
            </button>
          </div>

          <div className="role-input">
            <div className="role-input-heading">
              <label htmlFor="role-description">Target job description <span>optional · current question only</span></label>
              {roleDescription && <button type="button" onClick={() => setRoleDescription("")}>Clear</button>}
            </div>
            <textarea
              id="role-description"
              value={roleDescription}
              onChange={(event) => setRoleDescription(event.target.value.slice(0, 12000))}
              placeholder="Paste a job description to compare its requirements with your evidence."
              rows={4}
              maxLength={12000}
            />
            <p>This text is sent to the AI model for this question, not added to your indexed sources. Role requirements are cited as [JD].</p>
          </div>

          <div className="question-box">
            <Search size={20} />
            <textarea value={question} onChange={(event) => setQuestion(event.target.value)} rows={2} />
            <button onClick={ask} disabled={loading}>{loading ? "Working…" : "Research"}</button>
          </div>
          <div className="suggestions">
            {starterQuestions.map((item) => <button key={item} onClick={() => setQuestion(item)}>{item}</button>)}
            {roleDescription.trim() && (
              <button onClick={() => setQuestion("How does my experience match this job description?")}>
                Compare with this role
              </button>
            )}
          </div>

          {benchmark && (
            <section className="benchmark-card">
              <div className="benchmark-heading">
                <div><span className="eyebrow">Latest fixed benchmark</span><h3>{benchmark.dataset_cases}-case evaluation snapshot</h3></div>
                <span>v2 · hybrid · k=5</span>
              </div>
              <div className="benchmark-metrics">
                <div><span>Recall@5</span><strong>{Math.round(benchmark.full_pipeline.retrieval_recall_at_5 * 100)}%</strong></div>
                <div><span>MRR@5</span><strong>{benchmark.full_pipeline.mean_reciprocal_rank_at_5.toFixed(3)}</strong></div>
                <div><span>Abstention</span><strong>{Math.round(benchmark.full_pipeline.abstention_accuracy * 100)}%</strong></div>
                <div><span>Citation IDs</span><strong>{Math.round(benchmark.full_pipeline.citation_validity_rate * 100)}%</strong></div>
              </div>
              <p>Strict answer terms {Math.round(benchmark.full_pipeline.answer_key_term_accuracy * 100)}% · safety {Math.round(benchmark.full_pipeline.retrieved_content_safety_pass_rate * 100)}% · average full-pipeline latency {(benchmark.full_pipeline.average_latency_ms / 1000).toFixed(2)}s</p>
            </section>
          )}

          {error && <div className="error-message">{error}</div>}
          {evaluation && (
            <div className="evaluation-strip">
              <div><span>{evaluation.retrieval_mode} recall@5</span><strong>{Math.round(evaluation.retrieval_recall_at_k * 100)}%</strong></div>
              <div><span>MRR@5</span><strong>{evaluation.mean_reciprocal_rank.toFixed(2)}</strong></div>
              <div><span>Abstention</span><strong>{evaluation.abstention_accuracy === null ? "Full eval" : `${Math.round(evaluation.abstention_accuracy * 100)}%`}</strong></div>
              <div><span>Cases</span><strong>{evaluation.total_cases}</strong></div>
            </div>
          )}

          {!answer ? (
            <div className="empty-state">
              <div className="empty-symbol"><Search size={28} /></div>
              <h3>Turn experience into defensible evidence</h3>
              <p>Explore your documented experience, or paste a job description to compare it with a target role.</p>
            </div>
          ) : (
            <div className="result-grid">
              <article className="answer-card">
                <div className="result-meta">
                  <span className={`badge ${answer.status}`}><Check size={13} /> {answer.status.replace("_", " ")}</span>
                  <span><Clock3 size={14} /> {answer.latency_ms} ms</span>
                  <span>{answer.mode.replace("_", " ")}</span>
                </div>
                <h3>Answer</h3>
                <div className="answer-text">{answer.answer}</div>
                {answerUsedRole && <p className="citation-key">[S] Your evidence · [JD] Job description supplied for this question</p>}
              </article>

              <div className="evidence-section">
                <h3>Retrieved evidence <span>{answer.sources.length}</span></h3>
                {answer.sources.map((source, index) => (
                  <details className="source-card" key={source.source_id} open={index === 0}>
                    <summary>
                      <span className="source-number">S{index + 1}</span>
                      <div><strong>{source.document}</strong><span>Chunk {source.chunk_index + 1}</span></div>
                      <div className="score">{Math.round(source.score * 100)}%</div>
                      <ChevronDown size={16} />
                    </summary>
                    <p>{source.text}</p>
                    <div className="method-row">{source.retrieval_methods.map((method) => <span key={method}>{method}</span>)}</div>
                  </details>
                ))}
              </div>

              <div className="trace-section">
                <h3>Workflow trace</h3>
                {answer.trace.map((step, index) => (
                  <div className="trace-row" key={`${step.name}-${index}`}>
                    <span className={`trace-node ${step.status}`}><Check size={12} /></span>
                    <div><strong>{step.name}</strong><p>{step.detail}</p></div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
