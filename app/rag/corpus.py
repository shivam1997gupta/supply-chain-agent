"""
The RAG knowledge corpus — the "house rules" we retrieve and feed to the SQL
agent so it follows OUR conventions instead of guessing on ambiguous questions.

Three kinds of knowledge:
  GLOSSARY  - what a business term means (e.g. "stockout risk")
  KPIS      - how a metric is computed, with a SQL hint
  EXAMPLES  - question -> correct SQL pairs (few-shot; also our eval set later)

Each entry is turned into a short text string that we EMBED. At query time we
embed the user's question and retrieve the closest entries by meaning.
"""

GLOSSARY = [
    {"term": "stockout risk",
     "definition": "A product-warehouse is at stockout risk when its stock_level "
                   "is below its reorder_point. Always evaluate PER (product, "
                   "warehouse) row, never summed across warehouses."},
    {"term": "lead time",
     "definition": "Days between a purchase order's order_date and its "
                   "actual_delivery (or expected_delivery if not yet delivered)."},
    {"term": "supplier reliability",
     "definition": "How dependably a supplier delivers on time. Use "
                   "suppliers.reliability_score (0-1, higher is better) and/or the "
                   "share of that supplier's purchase_orders with status='delayed'."},
    {"term": "net position",
     "definition": "stock_level minus reorder_point for an inventory row. Negative "
                   "means below reorder point."},
]

KPIS = [
    {"name": "total revenue",
     "definition": "Sum of sales.revenue.",
     "sql_hint": "SELECT SUM(revenue) AS total_revenue FROM sales"},
    {"name": "revenue by region",
     "definition": "Total sales revenue grouped by warehouse region.",
     "sql_hint": "SELECT w.region, SUM(s.revenue) AS total_revenue FROM sales s "
                 "JOIN warehouses w ON w.warehouse_id = s.warehouse_id "
                 "GROUP BY w.region"},
    {"name": "on-time delivery rate",
     "definition": "Share of delivered purchase orders where actual_delivery <= "
                   "expected_delivery.",
     "sql_hint": "SELECT AVG(CASE WHEN actual_delivery <= expected_delivery THEN 1.0 "
                 "ELSE 0 END) AS on_time_rate FROM purchase_orders "
                 "WHERE status='delivered'"},
    {"name": "inventory coverage",
     "definition": "How current stock compares to reorder point, as a ratio "
                   "stock_level / reorder_point per product-warehouse.",
     "sql_hint": "SELECT product_id, warehouse_id, "
                 "stock_level * 1.0 / reorder_point AS coverage FROM inventory"},
]

EXAMPLES = [
    {"question": "Which products are at risk of stockout?",
     "sql": "SELECT p.product_name, w.location, i.stock_level, i.reorder_point "
            "FROM inventory i JOIN products p ON p.product_id = i.product_id "
            "JOIN warehouses w ON w.warehouse_id = i.warehouse_id "
            "WHERE i.stock_level < i.reorder_point "
            "ORDER BY (i.reorder_point - i.stock_level) DESC"},
    {"question": "What is the total revenue by warehouse region?",
     "sql": "SELECT w.region, SUM(s.revenue) AS total_revenue FROM sales s "
            "JOIN warehouses w ON w.warehouse_id = s.warehouse_id "
            "GROUP BY w.region ORDER BY total_revenue DESC"},
    {"question": "Show total sales revenue per day over time.",
     "sql": "SELECT sale_date, SUM(revenue) AS total_revenue FROM sales "
            "GROUP BY sale_date ORDER BY sale_date"},
    {"question": "Which suppliers are least reliable?",
     "sql": "SELECT s.supplier_name, s.reliability_score, "
            "SUM(CASE WHEN po.status='delayed' THEN 1 ELSE 0 END) AS delayed_orders "
            "FROM suppliers s JOIN purchase_orders po ON po.supplier_id = s.supplier_id "
            "GROUP BY s.supplier_id ORDER BY delayed_orders DESC"},
    {"question": "How many purchase orders are in each status?",
     "sql": "SELECT status, COUNT(*) AS order_count FROM purchase_orders "
            "GROUP BY status"},
]


def build_documents():
    """Flatten the three corpora into a single list of retrievable documents.

    Each doc has:
      text    - the string we EMBED and search against
      kind    - "glossary" | "kpi" | "example"
      content - the string we INJECT into the SQL prompt when retrieved
    """
    docs = []
    for g in GLOSSARY:
        docs.append({
            "text": f"{g['term']}: {g['definition']}",
            "kind": "glossary",
            "content": f"- {g['term']}: {g['definition']}",
        })
    for k in KPIS:
        docs.append({
            "text": f"{k['name']}: {k['definition']}",
            "kind": "kpi",
            "content": f"- KPI '{k['name']}': {k['definition']}\n  SQL: {k['sql_hint']}",
        })
    for e in EXAMPLES:
        docs.append({
            "text": e["question"],  # embed the QUESTION so similar questions match
            "kind": "example",
            "content": f"- Q: {e['question']}\n  SQL: {e['sql']}",
        })
    return docs
