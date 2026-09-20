"""
FastAPI service — turns the agent graph into a callable HTTP API.

This is what makes the project a *system* rather than a script: anything that
can make an HTTP request (a web UI, another service, curl) can now ask the
supply-chain agent a question and get structured JSON back.

Endpoints:
  GET  /health        -> liveness check
  POST /ask  {question}-> runs the graph, returns answer + sql + chart info
  GET  /chart         -> returns the most recently rendered chart image

Run the server:
  uvicorn app.api:api --reload
Then POST to it:
  curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
       -d '{"question": "which products are at risk of stockout"}'

Set SCAI_FAKE=1 to run against the no-LLM fake graph (handy for demos):
  SCAI_FAKE=1 uvicorn app.api:api --reload
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Pick which graph to serve: real (LLM) or fake (no key). One switch, whole app.
if os.getenv("SCAI_FAKE") == "1":
    from app.graph_fake import app as graph_app
else:
    from app.graph import app as graph_app


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm: load the embedding model at startup, BEFORE any request arrives,
    # so the first user doesn't eat the cold-start load. In a scaled deployment
    # every replica warms itself on boot this way. (Skip in fake mode — no LLM
    # path, so no retriever is needed.)
    if os.getenv("SCAI_FAKE") != "1":
        from app.rag.retriever import get_retriever
        get_retriever()
    yield


api = FastAPI(title="Supply Chain Agent API", version="1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str | None = None
    sql: str | None = None
    chart_path: str | None = None
    workers_used: list[str] = []


@api.get("/health")
def health():
    return {"status": "ok"}


@api.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    # run the whole agent graph on the question
    try:
        result = graph_app.invoke({"question": req.question, "completed": []})
    except Exception as e:
        # Transient errors are retried inside the graph (see RetryPolicy); if we
        # still land here the failure is real. Return a clean 503 instead of a
        # raw 500 stack trace so callers get a usable signal. Rate-limit/quota
        # errors surface here once retries are exhausted.
        msg = str(e).lower()
        if any(m in msg for m in ["429", "rate", "resource_exhausted", "quota"]):
            raise HTTPException(
                status_code=503,
                detail="LLM rate limit or quota reached — please retry shortly.",
            )
        raise HTTPException(status_code=500, detail=f"agent error: {e}")
    return AskResponse(
        question=result["question"],
        answer=result.get("answer"),
        sql=result.get("sql"),
        chart_path=result.get("chart_path"),
        workers_used=result.get("completed", []),
    )


@api.get("/chart")
def chart():
    path = os.path.join(os.path.dirname(__file__), "..", "charts", "chart.png")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="no chart has been generated yet")
    return FileResponse(path, media_type="image/png")
