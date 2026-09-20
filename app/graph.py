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

from app.agents.supervisor import supervise
from app.agents.summarize import summarize
from app.agents.text2sql import write_sql, run_sql
from app.agents.visualize import plan_chart, render_chart
from app.db import get_schema_text
from app.state import AgentState


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
    s = dict(state)
    s.update({"schema": get_schema_text()})
    s.update(write_sql(s))
    s.update(run_sql(s))
    return {"schema": s["schema"], "sql": s["sql"], "rows": s["rows"],
            "completed": ["sql"]}


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
    graph.add_node("supervisor", timed("supervisor", supervise))
    graph.add_node("sql_agent", timed("sql_agent", sql_agent))
    graph.add_node("viz_agent", timed("viz_agent", viz_agent))
    graph.add_node("summarize_agent", timed("summarize_agent", summarize_agent))

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
