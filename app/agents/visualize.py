"""
The visualization agent — two nodes.

  plan_chart   : (question + result columns) --LLM--> a JSON chart spec
  render_chart : chart spec + rows            --matplotlib--> a PNG file

Same shape as the SQL agent, and the same safety split: the LLM only DECIDES
how to plot (it returns a small JSON spec); our code does the actual drawing.
We never execute LLM-generated plotting code.

Spec format the LLM must return:
    {"chart_type": "bar" | "line" | "none", "x": "<column>", "y": "<column>", "title": "<text>"}
"chart_type": "none" means the data isn't worth charting (e.g. a single number).
"""

import json
import os
import re

import matplotlib
matplotlib.use("Agg")  # render to a file, no GUI window needed
import matplotlib.pyplot as plt

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_llm, to_text
from app.state import AgentState

CHART_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "charts")

PLAN_PROMPT = """You choose how to visualize a small SQL result set.
You are given the user's question and the columns available (with one sample row).
Return ONLY a JSON object, no markdown, with these keys:
  chart_type: "bar", "line", or "none"
  x: the column for the x-axis (a label/category column)
  y: the column for the y-axis (a numeric column)
  title: a short chart title
Use "bar" for comparing categories, "line" for trends over time/order.
Use "none" if the result is a single value or has no sensible chart.
"""


def _parse_json(text: str) -> dict:
    """Pull a JSON object out of the model's reply, tolerating ``` fences."""
    text = to_text(text).strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def plan_chart(state: AgentState) -> dict:
    rows = state.get("rows") or []
    if not rows:
        return {"chart_spec": {"chart_type": "none"}}

    columns = list(rows[0].keys())
    sample = rows[0]
    llm = get_llm()
    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(
            content=f"Question: {state['question']}\n"
                    f"Columns: {columns}\nSample row: {sample}"
        ),
    ]
    response = llm.invoke(messages)
    try:
        spec = _parse_json(response.content)
    except (json.JSONDecodeError, ValueError):
        spec = {"chart_type": "none"}  # if the model returns junk, just skip
    print(f"[plan_chart] {spec}")
    return {"chart_spec": spec}


def render_chart(state: AgentState) -> dict:
    spec = state.get("chart_spec") or {}
    rows = state.get("rows") or []
    ctype = spec.get("chart_type", "none")

    if ctype == "none" or not rows:
        print("[render_chart] nothing to plot")
        return {"chart_path": None}

    x_col, y_col = spec.get("x"), spec.get("y")
    xs = [r.get(x_col) for r in rows]
    ys = [r.get(y_col) for r in rows]

    os.makedirs(CHART_DIR, exist_ok=True)
    path = os.path.join(CHART_DIR, "chart.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    if ctype == "line":
        ax.plot(xs, ys, marker="o")
    else:  # default to bar
        ax.bar([str(x) for x in xs], ys)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(spec.get("title", ""))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"[render_chart] saved -> {path}")
    return {"chart_path": os.path.abspath(path)}
