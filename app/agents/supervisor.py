"""
The supervisor — now a TRUE looping orchestrator (not a one-shot router).

It is called REPEATEDLY. Each time, it looks at the question and which workers
have already run, and decides who runs NEXT — or that we're FINISHED. After a
worker runs, control returns to the supervisor and it decides again. That loop
is what lets one question use several workers in whatever order fits:

  "how many suppliers?"                 -> sql -> summarize -> FINISH
  "chart revenue by region"             -> sql -> visualize -> FINISH
  "show revenue by region AND tell me
   which is highest"                    -> sql -> visualize -> summarize -> FINISH

Workers:
  sql        - query the database (MUST run first; others need its rows)
  visualize  - draw a chart from the rows
  summarize  - write a short text answer from the rows
  FINISH     - the question's needs are met
"""

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_llm, to_text
from app.state import AgentState

CHOICES = ["sql", "visualize", "summarize", "finish"]

SUPERVISOR_PROMPT = """You are the supervisor of a supply-chain analytics team.
You coordinate three workers by choosing which one runs NEXT, or finishing.

  sql        - queries the database to get data. MUST run before the others.
  visualize  - draws a chart. Use when the user wants to SEE/plot/compare data
               or view a trend or a ranking.
  summarize  - writes a short TEXT answer. Use when the user wants a number,
               a fact, or an explanation in words.
  finish     - everything the question needs has been produced.

You are given the question and which workers have ALREADY run.
Rules:
- Never choose a worker that already ran.
- A question can need BOTH visualize and summarize (e.g. "show X and tell me Y").
- Choose finish once the question's needs are met.

IMPORTANT: A question often contains MORE THAN ONE request. Read the WHOLE
question for separate intents before deciding.
- "show"/"plot"/"chart"/"graph"/"visualize"/"trend" -> visualize is needed.
- "tell me"/"explain"/"which"/"what"/"summarize"/"describe" -> summarize is needed.
If BOTH kinds of words appear, BOTH workers must run before you finish.

Example:
  Question: "show revenue by region and tell me which is highest"
  Already done: ['sql', 'visualize']  -> choose summarize (not finish yet).

Reply with EXACTLY ONE word: sql, visualize, summarize, or finish.
"""


def supervise(state: AgentState) -> dict:
    completed = state.get("completed", [])
    llm = get_llm()
    messages = [
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(
            content=f"Question: {state['question']}\n"
                    f"Already done: {completed or 'nothing yet'}"
        ),
    ]
    said = to_text(llm.invoke(messages).content).strip().lower()
    decision = next((w for w in CHOICES if w in said), "finish")

    # --- guardrails: a safety net around the LLM's choice ---
    # These both keep the loop correct AND guarantee it always terminates.
    if "sql" not in completed:
        decision = "sql"            # always fetch data first, no matter what
    elif decision in completed:
        decision = "finish"         # never repeat a worker -> loop must end
    elif decision == "sql":
        decision = "finish"         # sql already done; nothing left to do

    # --- deterministic backstop for compound questions ---
    # A small model sometimes says "finish" while the question clearly still
    # asks for a worker that hasn't run. Don't fully trust the model: if the
    # wording obviously wants a chart or a summary that's still missing, run it.
    if decision == "finish":
        q = state["question"].lower()
        wants_viz = any(w in q for w in
                        ["show", "plot", "chart", "graph", "visual", "trend"])
        wants_sum = any(w in q for w in
                        ["tell me", "explain", "which", "what", "summar", "describe"])
        if wants_viz and "visualize" not in completed:
            decision = "visualize"
        elif wants_sum and "summarize" not in completed:
            decision = "summarize"

    nxt = "FINISH" if decision == "finish" else decision
    print(f"[supervisor] done={completed}  ->  next={nxt}   (model said: {said!r})")
    return {"next": nxt}