"""
The shared state for the supply-chain agent graph.

Everything the graph knows flows through this one object. Two fields power the
supervisor LOOP:

  next       - the supervisor's decision of which worker runs next
               ("sql" | "visualize" | "summarize" | "FINISH")
  completed  - which workers have already run. This uses a REDUCER
               (operator.add): when a node returns {"completed": ["sql"]},
               LangGraph APPENDS it to the existing list instead of replacing
               it. That accumulation is how the supervisor remembers what's
               been done across loop iterations.
"""

import operator
from typing import Annotated, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # --- input ---
    question: str

    # --- supervisor loop control ---
    next: Optional[str]                       # who runs next / FINISH
    completed: Annotated[list, operator.add]  # append-only log of finished workers

    # --- produced by the workers ---
    schema: Optional[str]
    sql: Optional[str]
    rows: Optional[list]
    chart_spec: Optional[dict]
    chart_path: Optional[str]
    answer: Optional[str]
