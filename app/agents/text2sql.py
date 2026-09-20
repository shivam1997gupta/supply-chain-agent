"""
The Text2SQL agent — with error-feedback self-correction.

  write_sql : (question + schema + retrieved knowledge) --LLM--> SQL
  run_sql   : SQL --DB--> rows

  generate_and_run : the SELF-CORRECTION LOOP. Generate SQL, try to run it,
                     and if the database rejects it (syntax error, bad column,
                     etc.), feed the ERROR MESSAGE back to the model and ask it
                     to fix the query. Retry up to max_attempts.

Why a loop and not a plain retry: LLM output is non-deterministic, so a bad
query often becomes valid on a second try -- but only if the model is TOLD what
went wrong. Blindly re-running the same prompt might repeat the mistake;
feeding the error back lets the model actually converge on a working query.
This is a standard, high-value pattern for Text2SQL systems.
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

# Extra instruction appended when a previous attempt failed, so the model can fix it.
REPAIR_TEMPLATE = """
Your previous query FAILED. Fix it.

Previous SQL:
{bad_sql}

Database error:
{error}

Return a corrected SQL query that avoids this error. Output ONLY the SQL.
"""


def _clean_sql(text: str) -> str:
    """LLMs love to wrap SQL in ```sql ... ``` fences or add a stray word.
    Strip that so we hand the database a bare query."""
    text = to_text(text).strip()
    fence = re.match(r"^```(?:sql)?\s*(.*?)\s*```$", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    return text


def _generate_sql(question, schema, context, bad_sql=None, error=None) -> str:
    """Ask the LLM for SQL. If bad_sql+error are given, ask it to REPAIR instead."""
    user = (f"Schema:\n{schema}\n\n"
            f"Reference knowledge:\n{context}\n\n"
            f"Question: {question}")
    if error:
        user += "\n" + REPAIR_TEMPLATE.format(bad_sql=bad_sql, error=error)
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user),
    ])
    return _clean_sql(response.content)


def generate_and_run(state: AgentState, max_attempts: int = 3) -> dict:
    """Self-correcting Text2SQL: generate -> run -> on DB error, feed the error
    back and regenerate, up to max_attempts. Returns {sql, rows, sql_attempts}."""
    context = retrieve_context(state["question"], k=4)
    print(f"[sql] retrieved context ({len(context.splitlines())} lines)")

    bad_sql, error = None, None
    for attempt in range(1, max_attempts + 1):
        sql = _generate_sql(state["question"], state["schema"], context, bad_sql, error)
        print(f"[sql] attempt {attempt}: {sql}")
        try:
            columns, rows = run_query(sql)
            result = [dict(zip(columns, r)) for r in rows]
            print(f"[sql] OK on attempt {attempt}: {len(result)} row(s)")
            return {"sql": sql, "rows": result, "sql_attempts": attempt}
        except Exception as e:
            error = str(e)
            bad_sql = sql
            print(f"[sql] attempt {attempt} FAILED: {error} -> feeding back to model")

    # all attempts failed: return the last error instead of crashing the graph
    print(f"[sql] gave up after {max_attempts} attempts")
    return {"sql": bad_sql, "rows": [],
            "answer": f"Could not produce a valid SQL query after "
                      f"{max_attempts} attempts. Last error: {error}",
            "sql_attempts": max_attempts}


# --- kept for the no-key fake graph and the retriever demo ---------------
def write_sql(state: AgentState) -> dict:
    context = retrieve_context(state["question"], k=4)
    sql = _generate_sql(state["question"], state["schema"], context)
    print(f"[write_sql] {sql}")
    return {"sql": sql}


def run_sql(state: AgentState) -> dict:
    columns, rows = run_query(state["sql"])
    print(f"[run_sql] {len(rows)} row(s)")
    return {"rows": [dict(zip(columns, r)) for r in rows]}
