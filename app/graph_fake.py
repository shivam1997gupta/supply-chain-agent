"""
Looping supervisor graph WITHOUT an LLM / API key.

Same LOOP topology as app.graph, but the supervisor and the two "thinking"
sub-steps are deterministic so you can watch the orchestration with zero API
calls. The fake supervisor uses keyword rules to decide which workers a
question needs, then dispatches them one per loop.

Try:
  python -m app.graph_fake "how many suppliers are there"
        -> sql -> summarize -> FINISH
  python -m app.graph_fake "chart revenue by region"
        -> sql -> visualize -> FINISH
  python -m app.graph_fake "show revenue by region and tell me which is highest"
        -> sql -> visualize -> summarize -> FINISH
"""

import sys

from langgraph.graph import START, END, StateGraph

from app.agents.text2sql import run_sql
from app.agents.visualize import render_chart
from app.db import get_schema_text
from app.state import AgentState


# --- deterministic supervisor (keyword planner) ---------------------------
def supervise_fake(state: AgentState) -> dict:
    completed = state.get("completed", [])
    q = state["question"].lower()
    wants_viz = any(w in q for w in
                    ["show", "plot", "chart", "graph", "compare", "trend",
                     "visual", "top ", " by ", "over time"])
    wants_sum = any(w in q for w in
                    ["how many", "average", "which", "what is", "explain",
                     "tell me", "summar", "total"])

    if "sql" not in completed:
        nxt = "sql"
    elif wants_viz and "visualize" not in completed:
        nxt = "visualize"
    elif (wants_sum or not wants_viz) and "summarize" not in completed:
        nxt = "summarize"
    else:
        nxt = "FINISH"
    print(f"[supervise_fake] done={completed}  ->  next={nxt}")
    return {"next": nxt}


# --- composite workers (canned "thinking" + real execution) ---------------
def sql_agent(state: AgentState) -> dict:
    sql = ("SELECT w.region, SUM(s.revenue) AS total_revenue "
           "FROM sales s JOIN warehouses w ON w.warehouse_id = s.warehouse_id "
           "GROUP BY w.region ORDER BY total_revenue DESC")
    s = dict(state); s["schema"] = get_schema_text(); s["sql"] = sql
    s.update(run_sql(s))
    return {"schema": s["schema"], "sql": sql, "rows": s["rows"], "completed": ["sql"]}


def viz_agent(state: AgentState) -> dict:
    s = dict(state)
    s["chart_spec"] = {"chart_type": "bar", "x": "region",
                       "y": "total_revenue", "title": "Revenue by Region"}
    s.update(render_chart(s))
    return {"chart_spec": s["chart_spec"], "chart_path": s.get("chart_path"),
            "completed": ["visualize"]}


def summarize_agent(state: AgentState) -> dict:
    rows = state.get("rows") or []
    top = rows[0] if rows else {}
    answer = f"Highest region: {top}. Full results: {rows}"
    print(f"[summarize_fake] {answer}")
    return {"answer": answer, "completed": ["summarize"]}


def route_from_supervisor(state: AgentState) -> str:
    return state["next"]


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervise_fake)
    graph.add_node("sql_agent", sql_agent)
    graph.add_node("viz_agent", viz_agent)
    graph.add_node("summarize_agent", summarize_agent)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor", route_from_supervisor,
        {"sql": "sql_agent", "visualize": "viz_agent",
         "summarize": "summarize_agent", "FINISH": END},
    )
    graph.add_edge("sql_agent", "supervisor")
    graph.add_edge("viz_agent", "supervisor")
    graph.add_edge("summarize_agent", "supervisor")
    return graph.compile()


app = build_graph()


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else \
        "Show total revenue by region and tell me which region is highest."
    result = app.invoke({"question": q, "completed": []})
    print("\n--- final state ---")
    print("workers used:", result.get("completed"))
    print("answer:", result.get("answer"))
    print("chart:", result.get("chart_path"))
