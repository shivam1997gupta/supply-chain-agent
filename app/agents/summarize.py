"""
The summarization agent — one node.

  summarize : (question + rows) --LLM--> a short natural-language answer -> state["answer"]

It closes the "answer" route: instead of dumping raw rows like
{'SUM(forecasted_units)': 654}, it replies "The forecasted demand for
product 8 over the next 30 days is 654 units."

The important idea here is GROUNDING. The prompt tells the model to use ONLY
the numbers in the provided rows and never invent figures. This is how you
stop an LLM from hallucinating data it wasn't given — you hand it the facts
and constrain it to them. It's the same discipline behind RAG.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_llm, to_text
from app.state import AgentState

SUMMARIZE_PROMPT = """You summarize SQL query results for a supply-chain analyst.
You are given the user's question and the exact rows returned by the database.
Write a concise, direct answer (1-3 sentences).

Rules:
- Use ONLY the numbers and values present in the rows. Never invent figures.
- If the rows are empty, say the query returned no results.
- Answer the question directly; don't describe the table or restate the SQL.
"""


def summarize(state: AgentState) -> dict:
    rows = state.get("rows") or []
    llm = get_llm()
    messages = [
        SystemMessage(content=SUMMARIZE_PROMPT),
        HumanMessage(
            content=f"Question: {state['question']}\nRows: {rows}"
        ),
    ]
    answer = to_text(llm.invoke(messages).content).strip()
    print(f"[summarize] {answer}")
    return {"answer": answer}
