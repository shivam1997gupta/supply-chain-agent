# Supply Chain Agentic Insights

A multi-agent, natural-language analytics assistant for supply-chain data, built from scratch on **LangGraph**. Ask a question in plain English — *"which products are at risk of stockout?"*, *"show revenue by region and tell me which is highest"* — and a supervisor agent orchestrates specialist agents to query a database, plot the result, and explain it.

This is a true multi-agent system: a supervisor decides, step by step, which specialist agent runs next, rather than following a fixed script. It is exposed as a containerized HTTP API.

---

## Architecture

```
                         ┌─────────────┐
   question ──────────▶  │ SUPERVISOR  │ ◀──────────────┐
                         │  (LLM loop) │                │
                         └──────┬──────┘                │
              decides next worker each iteration        │ workers loop back
                                │                        │
           ┌────────────┬───────┴────────┬──────────────┤
           ▼            ▼                 ▼              │
     ┌──────────┐ ┌───────────┐   ┌─────────────┐       │
     │ SQL agent│ │ viz agent │   │  summarize  │───────┘
     │ (RAG +   │ │ (chart    │   │  (grounded  │
     │  Text2SQL)│ │  spec +   │   │   NL answer)│
     └──────────┘ │ matplotlib)│  └─────────────┘
                  └───────────┘
        │
        ▼   (retrieval-augmented)
   ┌─────────────────────────────┐
   │ FAISS + sentence-transformers│  glossary · KPI defs · few-shot SQL
   └─────────────────────────────┘
```

The supervisor is re-invoked after every worker, tracks which workers have run
(accumulated in graph state via a reducer), and dispatches the next one until
the question's needs are met — so different questions run different sequences
(`sql → summarize`, `sql → visualize`, or `sql → visualize → summarize`).

---

## Key features

- **Supervisor orchestration** — an LLM supervisor in a loop chooses the next agent and decides when the task is done, with guardrails that guarantee termination.
- **Text2SQL agent** — turns a question into a `SELECT`, runs it, returns rows. Read-only guard rejects any non-SELECT.
- **Visualization agent** — the LLM emits a structured chart *spec* (JSON); deterministic matplotlib code renders it. The model decides, code executes.
- **Summarization agent** — writes a grounded natural-language answer constrained to the numbers actually returned (no hallucinated figures).
- **RAG layer (FAISS)** — retrieves a business glossary, KPI definitions, and few-shot question→SQL examples by semantic similarity and injects them into the SQL prompt, so the model follows domain conventions instead of guessing.
- **Provider-agnostic LLM** — one `get_llm()` factory; switch OpenAI / Anthropic / Gemini via env vars, no code change.
- **FastAPI service** — typed request/response schemas, auto-generated `/docs`, a `/chart` image endpoint.
- **Dockerized** — model and seeded database baked into the image for reproducible, offline-capable startup; secrets injected at runtime.
- **No-API-key mode** — a fake graph (`graph_fake.py`) runs the entire pipeline with zero LLM calls, for development and demos.

---

## Tech stack

LangGraph · LangChain · FastAPI · FAISS · sentence-transformers · matplotlib · SQLite · Docker · Google Gemini (swappable)

---

## Quickstart

```bash
pip install -r requirements.txt
python scripts/seed_db.py          # builds the synthetic supply-chain SQLite DB
```

**Run the agent from the CLI**
```bash
# no API key needed (deterministic fake graph):
python -m app.graph_fake "show revenue by region and tell me which is highest"

# with a real LLM — set your key in .env first (see below):
python -m app.graph "which products are at risk of stockout"
```

**Run as an API**
```bash
uvicorn app.api:api --reload            # http://localhost:8000/docs
# no-key demo:  SCAI_FAKE=1 uvicorn app.api:api
```

**Run in Docker**
```bash
docker compose up --build               # http://localhost:8000/docs
```

### Configuration

Create a `.env` file (gitignored):
```
LLM_PROVIDER=google_genai
LLM_MODEL=gemini-2.5-flash-lite
GOOGLE_API_KEY=your-key-here
```
A free key comes from Google AI Studio. Switch providers by changing
`LLM_PROVIDER` / `LLM_MODEL` and supplying that vendor's key.

---

## Project structure

```
app/
  graph.py        the LangGraph supervisor loop (real LLM)
  graph_fake.py   same graph, no API calls (dev/demo)
  state.py        shared AgentState (typed; uses a reducer for the loop)
  llm.py          provider-agnostic get_llm() factory
  db.py           SQLite helpers (schema introspection, read-only query)
  api.py          FastAPI service
  agents/
    supervisor.py  the orchestrator (routing loop + termination guardrails)
    text2sql.py    question -> SQL -> rows (RAG-augmented)
    visualize.py   chart spec (LLM) -> chart (matplotlib)
    summarize.py   grounded natural-language answer
  rag/
    corpus.py      glossary + KPI defs + few-shot Q->SQL examples
    retriever.py   sentence-transformers embeddings + FAISS index
scripts/
  seed_db.py      generates the synthetic database (fixed seed, reproducible)
Dockerfile · docker-compose.yml · requirements.txt
```

---

## Design decisions

- **Why a supervisor loop, not a fixed pipeline?** Different questions need
  different agents in different combinations. A loop lets the system compose
  them per-question; a fixed chain cannot. Guardrails (SQL first, no repeats)
  make the loop provably terminate.
- **The LLM decides, code executes.** Agents emit structured specs (SQL, a
  chart spec) that deterministic code runs. LLM output never executes directly
  — e.g. the DB layer rejects non-SELECT statements.
- **Local embeddings over an embedding API.** sentence-transformers runs
  offline, costs nothing, and has no rate limit — ideal for a small retrieval
  corpus. FAISS scales the same code to millions of vectors.
- **Provider-agnostic by design.** Nothing outside `llm.py` names a vendor, so
  the model is an env-var change. Cross-provider quirks (e.g. Gemini returning
  content as a list vs a string) are normalized in one place.
- **Secrets and config at runtime, never in the image.** `.env` is gitignored
  and `.dockerignore`d; the key is injected when the container runs — the same
  principle as Kubernetes Secrets.

---

## Roadmap

- [ ] Conversation checkpointing (multi-turn follow-ups)
- [ ] LangSmith evaluation harness (the few-shot bank doubles as the eval set)
- [ ] MCP server interface
- [ ] Postgres backend option (SQLite is used for portability)

---

*Built as a hands-on study of production agentic systems: orchestration,
retrieval-augmented generation, and deployment.*
