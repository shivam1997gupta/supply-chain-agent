"""
Tiny database helpers. For now we only need to read the schema so a node
can load it into state. Later the Text2SQL agent will also run queries here.
"""

import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "supply_chain.db")


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def get_schema_text() -> str:
    """Return a compact text description of every table and its columns.

    This is exactly what an LLM needs in its prompt to write correct SQL:
    table names + column names + types. We build it from sqlite_master so
    it stays in sync with the real database automatically.
    """
    conn = get_connection()
    cur = conn.cursor()
    tables = [
        r[0]
        for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]
    lines = []
    for t in tables:
        cols = cur.execute(f"PRAGMA table_info({t})").fetchall()
        col_defs = ", ".join(f"{c[1]} {c[2]}" for c in cols)
        lines.append(f"{t}({col_defs})")
    conn.close()
    return "\n".join(lines)


def run_query(sql: str):
    """Run a read-only SQL query and return (column_names, rows).

    Kept deliberately simple. The one guard rail: we only allow SELECT, so a
    hallucinated DROP/DELETE from the LLM can't damage the database.
    """
    if not sql.lstrip().lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")
    conn = get_connection()
    try:
        cur = conn.execute(sql)
        columns = [d[0] for d in cur.description]
        rows = cur.fetchall()
        return columns, rows
    finally:
        conn.close()


if __name__ == "__main__":
    print(get_schema_text())
