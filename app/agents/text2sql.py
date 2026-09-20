"""
The Text2SQL agent — two nodes.

  write_sql : (question + schema) --LLM--> SQL string   -> state["sql"]
  run_sql   : SQL string          --DB--->  rows         -> state["rows"]

This is the first node that calls an LLM. Notice the shape is identical to
the nodes you already wrote: take state, do work, return the slots you changed.
The only new thing is that "do work" now means "ask a language model."
"""

import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.db import run_query
from app.llm import get_llm, to_text
from app.rag.retriever import retrieve_context
from app.state import AgentState

SYSTEM_PROMPT = """You are an expert data analyst who writes SQLite SQL.
You are given a database schema, some retrieved reference knowledge (business
definitions, KPI formulas, and example question->SQL pairs), and a question.
Return ONE SQL query that answers the question. Rules:
- Output ONLY the SQL. No explanation, no markdown fences.
- Use only tables and columns that exist in the schema.
- Follow the conventions in the reference knowledge when it applies
  (e.g. how a term is defined, or how a similar question was answered).
- The query must be a SELECT.
"""


def _clean_sql(text: str) -> str:
    """LLMs love to wrap SQL in ```sql ... ``` fences or add a stray word.
    Strip that so we hand the database a bare query."""
    text = to_text(text).strip()
    # remove ```sql ... ``` or ``` ... ``` fences if present
    fence = re.match(r"^```(?:sql)?\s*(.*?)\s*```$", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    return text


def write_sql(state: AgentState) -> dict:
    # RAG: retrieve relevant glossary/KPI/example knowledge for THIS question
    context = retrieve_context(state["question"], k=4)
    print(f"[write_sql] retrieved context:\n{context}")

    llm = get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"Schema:\n{state['schema']}\n\n"
                    f"Reference knowledge:\n{context}\n\n"
                    f"Question: {state['question']}"
        ),
    ]
    response = llm.invoke(messages)          # <-- the LLM call
    sql = _clean_sql(response.content)
    print(f"[write_sql] {sql}")
    return {"sql": sql}


def run_sql(state: AgentState) -> dict:
    columns, rows = run_query(state["sql"])
    print(f"[run_sql] {len(rows)} row(s)")
    # store as a list of dicts so it's easy to read/plot later
    result = [dict(zip(columns, r)) for r in rows]
    return {"rows": result}
