"""
Latency probe — find out WHERE the time goes.

Separates three things that "the pipeline is slow" lumps together:
  1. how long get_llm() takes to CONSTRUCT the model
  2. how long a TRIVIAL call takes (raw network + account round-trip)
  3. how long a REALISTIC call takes (our actual prompt size)

How to read it:
  - trivial call ~1-3s  -> the API is fine; slowness is our prompts/thinking
  - trivial call ~10s+  -> it's network/account/region latency, not our code
  - construct slow each time -> we're rebuilding the model on every node (fixable)

Run:  python -m scripts.probe_llm
"""

import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_llm


def timeit(label, fn):
    t = time.perf_counter()
    out = fn()
    print(f"{label:<32} {time.perf_counter() - t:6.2f}s")
    return out


print("provider/model from .env:")
import os
print("  LLM_PROVIDER =", os.getenv("LLM_PROVIDER"))
print("  LLM_MODEL    =", os.getenv("LLM_MODEL"))
print()

llm = timeit("construct get_llm() (1st)", get_llm)
timeit("trivial call 'reply OK' (1st)",
       lambda: llm.invoke("Reply with the single word: OK"))
timeit("trivial call 'reply OK' (2nd)",
       lambda: llm.invoke("Reply with the single word: OK"))

# a realistic-size call: system prompt + a chunk of schema-like text
big_system = SystemMessage(content="You write SQLite SQL. Output only SQL." * 3)
big_user = HumanMessage(content="Schema: " + ("products(id, name, cost) " * 20) +
                        "\nQuestion: top 5 products by cost")
timeit("realistic call (sql-sized prompt)",
       lambda: llm.invoke([big_system, big_user]))

llm2 = timeit("construct get_llm() (2nd)", get_llm)
print("\nDone. Compare the trivial vs realistic numbers.")
