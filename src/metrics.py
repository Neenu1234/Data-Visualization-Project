from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def guess_column(df: pd.DataFrame, candidates) -> Optional[str]:
    for name in df.columns:
        lower = name.lower()
        for cand in candidates:
            if cand in lower:
                return name
    return None


def infer_schema(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    schema = {
        "date": guess_column(df, ["date", "order_date", "invoice_date", "created"]),
        "revenue": guess_column(df, ["revenue", "amount", "sales", "total", "price", "unitprice", "subtotal"]),
        "quantity": guess_column(df, ["qty", "quantity", "units"]),
        "category": guess_column(df, ["category", "segment", "productline", "department"]),
        "product": guess_column(df, ["product", "item", "sku", "stockcode", "description"]),
        "customer": guess_column(df, ["customer", "cust", "client", "userid"]),
        "country": guess_column(df, ["country", "region", "market"]),
        "channel": guess_column(df, ["channel", "source", "platform"]),
        "cost": guess_column(df, ["cost", "cogs"]),
    }
    # If revenue missing but price*quantity present, synthesize
    if schema["revenue"] is None and schema["quantity"] and schema["product"]:
        # Try to detect unit price column
        unit_price = guess_column(df, ["unitprice", "unit_price", "price"])
        if unit_price and pd.api.types.is_numeric_dtype(df[unit_price]) and pd.api.types.is_numeric_dtype(df[schema["quantity"]]):
            # Create a synthetic revenue column name hint; the caller can compute it
            schema["unit_price"] = unit_price
    return schema


def ensure_revenue_column(df: pd.DataFrame, schema: Dict[str, Optional[str]]) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    if schema.get("revenue") is not None:
        return df, schema
    quantity_col = schema.get("quantity")
    unit_price = schema.get("unit_price")
    if quantity_col and unit_price:
        tmp = df.copy()
        try:
            tmp["_revenue"] = pd.to_numeric(tmp[quantity_col], errors="coerce") * pd.to_numeric(tmp[unit_price], errors="coerce")
            schema["revenue"] = "_revenue"
            return tmp, schema
        except Exception:
            return df, schema
    return df, schema


def compute_kpis(df: pd.DataFrame, schema: Dict[str, Optional[str]]) -> Dict[str, Optional[float]]:
    revenue_col = schema.get("revenue")
    quantity_col = schema.get("quantity")
    date_col = schema.get("date")
    customer_col = schema.get("customer")
    cost_col = schema.get("cost")

    total_revenue = float(pd.to_numeric(df[revenue_col], errors="coerce").sum()) if revenue_col else None
    total_orders = int(len(df)) if len(df) else 0
    total_units = int(pd.to_numeric(df[quantity_col], errors="coerce").sum()) if quantity_col else None
    aov = (total_revenue / total_orders) if (revenue_col and total_orders > 0) else None
    unique_customers = int(df[customer_col].nunique()) if customer_col and customer_col in df.columns else None
    total_cost = float(pd.to_numeric(df[cost_col], errors="coerce").sum()) if cost_col else None
    gross_margin = (total_revenue - total_cost) if (total_revenue is not None and total_cost is not None) else None
    gross_margin_pct = ((gross_margin / total_revenue) if (gross_margin is not None and total_revenue) else None)

    # MoM revenue growth if date present
    mom_growth = None
    if date_col and date_col in df.columns:
        s = df[[date_col]].copy()
        s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
        rev = pd.to_numeric(df[revenue_col], errors="coerce") if revenue_col else pd.Series(np.ones(len(df)))
        grp = rev.groupby(s[date_col].dt.to_period("M")).sum().sort_index()
        if len(grp) >= 2:
            last, prev = grp.iloc[-1], grp.iloc[-2]
            if prev != 0:
                mom_growth = float((last - prev) / prev)

    return {
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "total_units": total_units,
        "avg_order_value": aov,
        "unique_customers": unique_customers,
        "mom_growth": mom_growth,
        "total_cost": total_cost,
        "gross_margin": gross_margin,
        "gross_margin_pct": gross_margin_pct,
    }


def build_cohort_table(df: pd.DataFrame, schema: Dict[str, Optional[str]]) -> Optional[pd.DataFrame]:
    date_col = schema.get("date")
    customer_col = schema.get("customer")
    if not date_col or not customer_col or date_col not in df.columns or customer_col not in df.columns:
        return None
    d = df[[customer_col, date_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col, customer_col])
    d["order_month"] = d[date_col].dt.to_period("M")
    # cohort is the first purchase month per customer
    firsts = d.groupby(customer_col)["order_month"].min().rename("cohort")
    d = d.join(firsts, on=customer_col)
    # cohort period index (0 for first month, 1 for next, etc.)
    d["period_index"] = (d["order_month"] - d["cohort"]).apply(lambda p: p.n)
    pivot = (
        d.groupby(["cohort", "period_index"])[customer_col]
        .nunique()
        .reset_index()
        .pivot(index="cohort", columns="period_index", values=customer_col)
        .fillna(0)
        .astype(int)
    )
    # Convert to percentage retention by dividing each row by its 0th column
    base = pivot.iloc[:, 0].replace(0, 1)
    retention = (pivot.divide(base, axis=0) * 100).round(1)
    retention.index = retention.index.astype(str)
    return retention


def compute_rfm(df: pd.DataFrame, schema: Dict[str, Optional[str]]) -> Optional[pd.DataFrame]:
    date_col = schema.get("date")
    customer_col = schema.get("customer")
    revenue_col = schema.get("revenue")
    if not date_col or not customer_col or date_col not in df.columns or customer_col not in df.columns:
        return None
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col, customer_col])
    if revenue_col and revenue_col in tmp.columns:
        tmp[revenue_col] = pd.to_numeric(tmp[revenue_col], errors="coerce").fillna(0)
    else:
        tmp["_rev_fallback"] = 1.0
        revenue_col = "_rev_fallback"
    snapshot_date = tmp[date_col].max() + pd.Timedelta(days=1)
    rfm = tmp.groupby(customer_col).agg(
        recency_days=(date_col, lambda s: (snapshot_date - s.max()).days),
        frequency=(customer_col, "count"),
        monetary=(revenue_col, "sum"),
    )
    # Score 1-5 (5 best)
    rfm["R"] = pd.qcut(rfm["recency_days"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["RFM_Score"] = rfm["R"] * 100 + rfm["F"] * 10 + rfm["M"]
    # Simple segments
    def segment(row):
        if row["R"] >= 4 and row["F"] >= 4 and row["M"] >= 4:
            return "Champions"
        if row["R"] >= 4 and row["F"] >= 3:
            return "Loyal"
        if row["R"] <= 2 and row["F"] <= 2:
            return "At Risk"
        if row["R"] >= 3 and row["M"] >= 4:
            return "Big Spenders"
        return "Others"
    rfm["Segment"] = rfm.apply(segment, axis=1)
    return rfm.reset_index()
