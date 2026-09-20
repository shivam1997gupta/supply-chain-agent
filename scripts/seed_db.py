"""
Seed a synthetic supply-chain SQLite database.

Generates a small but realistic dataset that supports the kinds of
questions a supply-chain insights chatbot should answer:
  - stockout risk (inventory vs reorder point vs demand)
  - overspending (purchase orders, supplier cost/reliability)
  - sales trends and demand-vs-forecast accuracy

Run:  python scripts/seed_db.py
Output: data/supply_chain.db
"""

import os
import random
import sqlite3
from datetime import date, timedelta

random.seed(42)  # reproducible: same DB every run

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "supply_chain.db")

SCHEMA = """
DROP TABLE IF EXISTS sales;
DROP TABLE IF EXISTS purchase_orders;
DROP TABLE IF EXISTS demand_forecast;
DROP TABLE IF EXISTS inventory;
DROP TABLE IF EXISTS suppliers;
DROP TABLE IF EXISTS warehouses;
DROP TABLE IF EXISTS products;

CREATE TABLE products (
    product_id   INTEGER PRIMARY KEY,
    product_name TEXT    NOT NULL,
    category     TEXT    NOT NULL,
    unit_cost    REAL    NOT NULL,   -- what we pay
    unit_price   REAL    NOT NULL    -- what we sell for
);

CREATE TABLE warehouses (
    warehouse_id INTEGER PRIMARY KEY,
    location     TEXT NOT NULL,
    region       TEXT NOT NULL
);

CREATE TABLE suppliers (
    supplier_id      INTEGER PRIMARY KEY,
    supplier_name    TEXT NOT NULL,
    lead_time_days   INTEGER NOT NULL,   -- avg days to deliver
    reliability_score REAL NOT NULL      -- 0..1, higher is better
);

CREATE TABLE inventory (
    inventory_id  INTEGER PRIMARY KEY,
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    warehouse_id  INTEGER NOT NULL REFERENCES warehouses(warehouse_id),
    stock_level   INTEGER NOT NULL,
    reorder_point INTEGER NOT NULL,
    snapshot_date TEXT NOT NULL          -- ISO date of this stock snapshot
);

CREATE TABLE demand_forecast (
    forecast_id      INTEGER PRIMARY KEY,
    product_id       INTEGER NOT NULL REFERENCES products(product_id),
    forecast_date    TEXT NOT NULL,
    forecasted_units INTEGER NOT NULL
);

CREATE TABLE purchase_orders (
    po_id             INTEGER PRIMARY KEY,
    product_id        INTEGER NOT NULL REFERENCES products(product_id),
    supplier_id       INTEGER NOT NULL REFERENCES suppliers(supplier_id),
    order_date        TEXT NOT NULL,
    quantity          INTEGER NOT NULL,
    unit_cost         REAL NOT NULL,
    status            TEXT NOT NULL,    -- delivered | in_transit | delayed
    expected_delivery TEXT NOT NULL,
    actual_delivery   TEXT              -- NULL if not yet delivered
);

CREATE TABLE sales (
    sale_id      INTEGER PRIMARY KEY,
    product_id   INTEGER NOT NULL REFERENCES products(product_id),
    warehouse_id INTEGER NOT NULL REFERENCES warehouses(warehouse_id),
    sale_date    TEXT NOT NULL,
    quantity     INTEGER NOT NULL,
    revenue      REAL NOT NULL
);
"""

PRODUCTS = [
    # name, category, unit_cost, unit_price
    ("Aluminum Sheet 2mm",   "Raw Material", 12.0, 19.0),
    ("Copper Wire Spool",    "Raw Material", 28.0, 41.0),
    ("Steel Bolt M8",        "Component",     0.4,  0.9),
    ("Plastic Casing A",     "Component",     3.2,  6.0),
    ("Lithium Cell 18650",   "Component",     2.1,  4.5),
    ("Circuit Board v3",     "Component",    14.0, 27.0),
    ("Power Adapter 65W",    "Finished Good", 9.5, 22.0),
    ("Smart Sensor Unit",    "Finished Good", 18.0, 49.0),
    ("Cooling Fan 120mm",    "Finished Good", 5.5, 14.0),
    ("Cable Assembly Kit",   "Finished Good", 7.0, 17.0),
]

WAREHOUSES = [
    ("Mumbai",     "West"),
    ("Delhi",      "North"),
    ("Bengaluru",  "South"),
    ("Kolkata",    "East"),
]

SUPPLIERS = [
    # name, lead_time_days, reliability_score
    ("Apex Components Ltd",   7,  0.96),
    ("Global Metals Co",     14,  0.88),
    ("Speedline Parts",       4,  0.74),  # fast but flaky
    ("Reliance Industrial",  10,  0.93),
    ("BudgetSource Traders",  21,  0.62),  # cheap but unreliable
]

STATUSES = ["delivered", "in_transit", "delayed"]


def daterange(start: date, days: int):
    for i in range(days):
        yield start + timedelta(days=i)


def main():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript(SCHEMA)

    # products
    cur.executemany(
        "INSERT INTO products(product_id, product_name, category, unit_cost, unit_price) "
        "VALUES (?,?,?,?,?)",
        [(i + 1, n, c, cost, price) for i, (n, c, cost, price) in enumerate(PRODUCTS)],
    )

    # warehouses
    cur.executemany(
        "INSERT INTO warehouses(warehouse_id, location, region) VALUES (?,?,?)",
        [(i + 1, loc, reg) for i, (loc, reg) in enumerate(WAREHOUSES)],
    )

    # suppliers
    cur.executemany(
        "INSERT INTO suppliers(supplier_id, supplier_name, lead_time_days, reliability_score) "
        "VALUES (?,?,?,?)",
        [(i + 1, n, lt, rs) for i, (n, lt, rs) in enumerate(SUPPLIERS)],
    )

    product_ids = list(range(1, len(PRODUCTS) + 1))
    warehouse_ids = list(range(1, len(WAREHOUSES) + 1))
    supplier_ids = list(range(1, len(SUPPLIERS) + 1))

    # inventory: one current snapshot per product per warehouse
    today = date(2026, 6, 1)
    inv_rows = []
    inv_id = 1
    for pid in product_ids:
        for wid in warehouse_ids:
            reorder = random.randint(80, 200)
            # ~30% of rows deliberately below reorder point -> stockout risk
            if random.random() < 0.30:
                stock = random.randint(0, reorder - 1)
            else:
                stock = random.randint(reorder, reorder + 400)
            inv_rows.append((inv_id, pid, wid, stock, reorder, today.isoformat()))
            inv_id += 1
    cur.executemany(
        "INSERT INTO inventory(inventory_id, product_id, warehouse_id, stock_level, "
        "reorder_point, snapshot_date) VALUES (?,?,?,?,?,?)",
        inv_rows,
    )

    # demand forecast: next 30 days per product
    fc_rows = []
    fc_id = 1
    for pid in product_ids:
        base = random.randint(20, 120)
        for d in daterange(today, 30):
            units = max(0, int(random.gauss(base, base * 0.2)))
            fc_rows.append((fc_id, pid, d.isoformat(), units))
            fc_id += 1
    cur.executemany(
        "INSERT INTO demand_forecast(forecast_id, product_id, forecast_date, forecasted_units) "
        "VALUES (?,?,?,?)",
        fc_rows,
    )

    # purchase orders: last 120 days
    po_rows = []
    po_id = 1
    start = today - timedelta(days=120)
    for _ in range(300):
        pid = random.choice(product_ids)
        sid = random.choice(supplier_ids)
        order_dt = start + timedelta(days=random.randint(0, 120))
        lead = SUPPLIERS[sid - 1][1]
        expected = order_dt + timedelta(days=lead)
        status = random.choices(STATUSES, weights=[0.7, 0.15, 0.15])[0]
        actual = None
        if status == "delivered":
            slip = random.randint(-2, 6)  # can be early or late
            actual = (expected + timedelta(days=slip)).isoformat()
        elif status == "delayed":
            actual = None
        qty = random.randint(50, 500)
        # PO unit cost wiggles around catalog cost
        cost = round(PRODUCTS[pid - 1][2] * random.uniform(0.95, 1.20), 2)
        po_rows.append(
            (po_id, pid, sid, order_dt.isoformat(), qty, cost, status,
             expected.isoformat(), actual)
        )
        po_id += 1
    cur.executemany(
        "INSERT INTO purchase_orders(po_id, product_id, supplier_id, order_date, quantity, "
        "unit_cost, status, expected_delivery, actual_delivery) VALUES (?,?,?,?,?,?,?,?,?)",
        po_rows,
    )

    # sales: last 90 days
    sale_rows = []
    sale_id = 1
    start = today - timedelta(days=90)
    for _ in range(1500):
        pid = random.choice(product_ids)
        wid = random.choice(warehouse_ids)
        sale_dt = start + timedelta(days=random.randint(0, 90))
        qty = random.randint(1, 60)
        price = PRODUCTS[pid - 1][3]
        revenue = round(qty * price * random.uniform(0.9, 1.0), 2)  # small discounts
        sale_rows.append((sale_id, pid, wid, sale_dt.isoformat(), qty, revenue))
        sale_id += 1
    cur.executemany(
        "INSERT INTO sales(sale_id, product_id, warehouse_id, sale_date, quantity, revenue) "
        "VALUES (?,?,?,?,?,?)",
        sale_rows,
    )

    conn.commit()

    # quick summary
    print(f"Database written to: {os.path.abspath(DB_PATH)}\n")
    for tbl in ["products", "warehouses", "suppliers", "inventory",
                "demand_forecast", "purchase_orders", "sales"]:
        n = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl:<18} {n:>6} rows")

    conn.close()


if __name__ == "__main__":
    main()
