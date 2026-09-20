"""
The supply-chain agent graph — TRUE looping supervisor.

    START -> supervisor --(next?)--> sql_agent -------+
                 ^                    viz_agent -------+--> back to supervisor
                 |                    summarize_agent -+
                 +------------------------------------+
             (FINISH) -> END

The shape is a LOOP, not a line. The supervisor picks the next worker; that
worker runs and returns to the supervisor; the supervisor picks again, until
it says FINISH. Different questions therefore run different sequences of
workers. This is the difference between "a fixed pipeline" and "an orchestrator".

Each worker is a COMPOSITE node that bundles its two steps (write+run SQL,
plan+render chart) and reports {"completed": [...]} so the supervisor knows
it's done. We compose the existing agent functions here without changing them.

Run:  python -m app.graph "show revenue by region and tell me which is highest"
      python -m app.graph_fake "..."   (no key)
"""

import sys
import time

from langgraph.graph import START, END, StateGraph
from langgraph.types import RetryPolicy

from app.agents.supervisor import supervise
from app.agents.summarize import summarize
from app.agents.text2sql import generate_and_run
from app.agents.visualize import plan_chart, render_chart
from app.db import get_schema_text
from app.state import AgentState


# --- retry policy for LLM nodes -------------------------------------------
# Only TRANSIENT failures should be retried: rate limits (429), timeouts, and
# 5xx/connection blips clear on their own after a short wait. A logic error
# (e.g. malformed SQL) will fail identically every time, so retrying it just
# wastes calls — we fail fast on those. The predicate below decides which is
# which by inspecting the error, kept provider-agnostic (matches on message,
# not on a specific vendor's exception class).
def _is_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    markers = ["429", "rate limit", "resource_exhausted", "timeout", "timed out",
               "temporarily", "unavailable", "503", "502", "500", "connection"]
    return any(m in msg for m in markers)


# Exponential backoff: wait ~1s, then ~2s, then ~4s (capped), with jitter so
# many concurrent requests don't retry in lockstep. Max 3 attempts total.
LLM_RETRY = RetryPolicy(
    max_attempts=3,
    initial_interval=1.0,
    backoff_factor=2.0,
    max_interval=8.0,
    jitter=True,
    retry_on=_is_transient,
)


def timed(name, fn):
    def wrapper(state):
        t = time.perf_counter()
        result = fn(state)
        print(f"[time] {name:<16} {time.perf_counter() - t:6.2f}s")
        return result
    return wrapper


# --- composite worker nodes -----------------------------------------------
# Each bundles its sub-steps and stamps {"completed": [...]} so the supervisor
# (which reads state["completed"]) knows this worker has run.

def sql_agent(state: AgentState) -> dict:
    schema = get_schema_text()
    # generate_and_run handles the self-correction loop internally
    out = generate_and_run({**state, "schema": schema})
    out["schema"] = schema
    out["completed"] = ["sql"]
    return out


def viz_agent(state: AgentState) -> dict:
    s = dict(state)
    s.update(plan_chart(s))
    s.update(render_chart(s))
    return {"chart_spec": s.get("chart_spec"), "chart_path": s.get("chart_path"),
            "completed": ["visualize"]}


def summarize_agent(state: AgentState) -> dict:
    upd = summarize(state)
    return {"answer": upd["answer"], "completed": ["summarize"]}


# --- the router: turn the supervisor's decision into the next node ---------
def route_from_supervisor(state: AgentState) -> str:
    return state["next"]


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", timed("supervisor", supervise), retry_policy=LLM_RETRY)
    graph.add_node("sql_agent", timed("sql_agent", sql_agent), retry_policy=LLM_RETRY)
    graph.add_node("viz_agent", timed("viz_agent", viz_agent), retry_policy=LLM_RETRY)
    graph.add_node("summarize_agent", timed("summarize_agent", summarize_agent), retry_policy=LLM_RETRY)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "sql": "sql_agent",
            "visualize": "viz_agent",
            "summarize": "summarize_agent",
            "FINISH": END,
        },
    )
    # every worker loops back to the supervisor for the next decision
    graph.add_edge("sql_agent", "supervisor")
    graph.add_edge("viz_agent", "supervisor")
    graph.add_edge("summarize_agent", "supervisor")
    return graph.compile()


app = build_graph()


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else \
        "Show total revenue by region and tell me which region is highest."
    t0 = time.perf_counter()
    result = app.invoke({"question": q, "completed": []})
    print(f"\n[time] TOTAL           {time.perf_counter() - t0:6.2f}s")
    print("\n--- final state ---")
    print("question:", result["question"])
    print("workers used:", result.get("completed"))
    print("answer:", result.get("answer"))
    print("chart:", result.get("chart_path"))
