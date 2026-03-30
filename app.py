import os
from typing import Dict, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import plotly.express as px

from src.data_loader import load_from_path, load_from_zip, memory_usage_mb
from src.charts import get_numeric_columns, get_categorical_columns, get_datetime_columns, histogram, bar_agg, timeseries, gauge, geo_map, funnel_chart, heatmap
from src.ai_qna import DataframeAssistant
from src.metrics import infer_schema, ensure_revenue_column, compute_kpis, compute_rfm


load_dotenv()
st.set_page_config(page_title="Online Sales Analytics + AI Q&A", layout="wide")


def sidebar_dataset_section() -> Optional[pd.DataFrame]:
    st.sidebar.header("Dataset")
    default_downloads = os.path.expanduser("~/Downloads/online_sales.zip")
    path = st.sidebar.text_input("Path to file (.zip/.csv/.xlsx)", value=default_downloads)
    uploaded = st.sidebar.file_uploader("Or upload CSV/ZIP/XLSX", type=["csv", "zip", "xlsx", "xls"])
    df = None
    details: Dict[str, pd.DataFrame] = {}

    try:
        if uploaded is not None:
            tmp_path = f"/tmp/{uploaded.name}"
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            df, details = load_from_path(tmp_path)
        elif path:
            if os.path.exists(path):
                df, details = load_from_path(path)
    except Exception as exc:
        st.sidebar.error(f"Failed to load dataset: {exc}")

    if df is not None:
        st.sidebar.success(f"Loaded rows={len(df):,}, cols={len(df.columns)} | mem ~{memory_usage_mb(df)} MB")
        if len(details) > 1:
            with st.sidebar.expander("Files detected"):
                for name, sdf in details.items():
                    st.write(f"{name}: {len(sdf):,} rows, {len(sdf.columns)} cols")
    return df


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    filtered = df.copy()
    cat_cols = get_categorical_columns(df)
    dt_cols = get_datetime_columns(df)

    # Date filter
    if dt_cols:
        date_col = st.sidebar.selectbox("Date column", options=dt_cols)
        min_date = pd.to_datetime(filtered[date_col], errors="coerce").min()
        max_date = pd.to_datetime(filtered[date_col], errors="coerce").max()
        if pd.notna(min_date) and pd.notna(max_date):
            start, end = st.sidebar.date_input("Date range", value=(min_date, max_date))
            if start and end:
                mask = (pd.to_datetime(filtered[date_col], errors="coerce") >= pd.to_datetime(start)) & (
                    pd.to_datetime(filtered[date_col], errors="coerce") <= pd.to_datetime(end)
                )
                filtered = filtered.loc[mask]

    # Category filter (up to 3)
    max_filters = 3
    if cat_cols:
        selected_cols = st.sidebar.multiselect("Category filters (max 3)", options=cat_cols, default=cat_cols[:1])
        for col_name in selected_cols[:max_filters]:
            vals = ["(All)"] + sorted([str(v) for v in filtered[col_name].dropna().astype(str).unique()])[:200]
            choice = st.sidebar.selectbox(f"Filter: {col_name}", options=vals)
            if choice != "(All)":
                filtered = filtered[filtered[col_name].astype(str) == choice]
    return filtered


def top_filters(df: pd.DataFrame) -> pd.DataFrame:
    # Reverted: date filter + up to 3 category filters (like the old sidebar behavior),
    # but rendered at the top of the dashboard.
    filtered = df.copy()
    cat_cols = get_categorical_columns(df)
    dt_cols = get_datetime_columns(df)

    # Date filter
    if dt_cols:
        date_col = st.selectbox("Date column", options=dt_cols, index=0, key="flt_date_col")
        min_date = pd.to_datetime(filtered[date_col], errors="coerce").min()
        max_date = pd.to_datetime(filtered[date_col], errors="coerce").max()
        if pd.notna(min_date) and pd.notna(max_date):
            start, end = st.date_input("Date range", value=(min_date, max_date), key="flt_date_range")
            if start and end:
                mask = (pd.to_datetime(filtered[date_col], errors="coerce") >= pd.to_datetime(start)) & (
                    pd.to_datetime(filtered[date_col], errors="coerce") <= pd.to_datetime(end)
                )
                filtered = filtered.loc[mask]

    # Category filter (up to 3)
    max_filters = 3
    if cat_cols:
        selected_cols = st.multiselect(
            "Category filters (max 3)",
            options=cat_cols,
            default=cat_cols[:1],
            key="flt_cat_cols",
        )
        for col_name in selected_cols[:max_filters]:
            vals = ["(All)"] + sorted(
                [str(v) for v in filtered[col_name].dropna().astype(str).unique()]
            )[:200]
            choice = st.selectbox(f"Filter: {col_name}", options=vals, index=0, key=f"flt_{col_name}")
            if choice != "(All)":
                filtered = filtered[filtered[col_name].astype(str) == choice]

    return filtered


def load_dataset_silently() -> Optional[pd.DataFrame]:
    # Prefer the user's local default without showing a left sidebar UI.
    default_downloads = os.path.expanduser("~/Downloads/online_sales.zip")
    try:
        if os.path.exists(default_downloads):
            df, _details = load_from_path(default_downloads)
            return df
    except Exception:
        pass

    return None


def show_overview(df: pd.DataFrame):
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rows 📦", f"{len(df):,}")
    with c2:
        st.metric("Columns 🧱", f"{len(df.columns)}")
    with c3:
        st.metric("Memory (MB) 💾", f"{memory_usage_mb(df)}")
    with c4:
        dt_cols = get_datetime_columns(df)
        st.metric("Datetime Cols ⏱️", f"{len(dt_cols)}")
    st.dataframe(df.head(100), use_container_width=True)


def show_charts(df: pd.DataFrame):
    st.subheader("Quick Charts 📊")
    num_cols = get_numeric_columns(df)
    cat_cols = get_categorical_columns(df)
    dt_cols = get_datetime_columns(df)

    tab_hist, tab_bar, tab_ts = st.tabs(["Histogram", "Bar (Agg)", "Time Series"])

    with tab_hist:
        if num_cols:
            col = st.selectbox("Numeric column", options=num_cols, key="hist_num")
            st.plotly_chart(histogram(df, col), use_container_width=True)
        else:
            st.info("No numeric columns available.")

    with tab_bar:
        if cat_cols and num_cols:
            cat = st.selectbox("Category", options=cat_cols, key="bar_cat")
            metric = st.selectbox("Metric", options=num_cols, key="bar_metric")
            agg = st.selectbox("Aggregation", options=["sum", "mean", "count", "median"], index=0)
            topn = st.slider("Top N", min_value=5, max_value=50, value=20, step=5)
            st.plotly_chart(bar_agg(df, cat, metric, agg, topn), use_container_width=True)
        else:
            st.info("Need both categorical and numeric columns.")

    with tab_ts:
        if dt_cols:
            dt = st.selectbox("Date column", options=dt_cols, key="ts_dt")
            metric_opt = [None] + num_cols
            metric = st.selectbox("Metric (optional)", options=metric_opt, index=0, format_func=lambda x: x or "(count)")
            freq = st.selectbox("Frequency", options=["D", "W", "M"], index=0)
            st.plotly_chart(timeseries(df, dt, metric, freq=freq), use_container_width=True)
        else:
            st.info("No datetime column available.")


def show_kpis_and_advanced(df: pd.DataFrame):
    schema = infer_schema(df)
    df2, schema = ensure_revenue_column(df, schema)
    kpis = compute_kpis(df2, schema)

    # KPI tiles (heading intentionally omitted per request)
    st.markdown(
        """
        <style>
        .kpi-tile {
          aspect-ratio: 1 / 1;
          width: 100%;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
          text-align: center;
          padding: 14px;
          border-radius: 14px;
          border: 1px solid rgba(255, 255, 255, 0.08);
          background: rgba(15, 23, 42, 0.85);
          overflow: hidden;
        }
        .kpi-title {
          font-size: 14px;
          color: #9CA3AF;
          margin-bottom: 8px;
          line-height: 1.2;
          word-break: break-word;
          text-align: center;
        }
        .kpi-value {
          font-size: 28px;
          font-weight: 700;
          color: #E5E7EB;
          line-height: 1.1;
          word-break: break-word;
          text-align: center;
        }
        .kpi-sub {
          margin-top: 6px;
          font-size: 12px;
          color: #6B7280;
          line-height: 1.2;
          text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def render_kpi_tile(title: str, value: str, subtitle: str = "") -> None:
        st.markdown(
            f"""
            <div class="kpi-tile">
              <div class="kpi-title">{title}</div>
              <div class="kpi-value">{value}</div>
              {f'<div class="kpi-sub">{subtitle}</div>' if subtitle else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )

    a, b, c, d, e = st.columns(5)
    with a:
        render_kpi_tile("Revenue 💸", f"{kpis['total_revenue']:,.0f}" if kpis["total_revenue"] is not None else "—")
    with b:
        render_kpi_tile("Orders 🧾", f"{kpis['total_orders']:,.0f}")
    with c:
        render_kpi_tile("Units 📦", f"{kpis['total_units']:,.0f}" if kpis["total_units"] is not None else "—")
    with d:
        render_kpi_tile("Avg Order Value 🛒", f"{kpis['avg_order_value']:,.2f}" if kpis["avg_order_value"] is not None else "—")
    with e:
        render_kpi_tile("Customers 👥", f"{kpis['unique_customers']:,.0f}" if kpis["unique_customers"] is not None else "—")

    # KPI gallery with icons/emojis
    gcols = st.columns(5)
    if kpis.get("gross_margin") is not None:
        with gcols[0]:
            render_kpi_tile("Gross Margin 🧮", f"{kpis['gross_margin']:,.0f}")
    if kpis.get("gross_margin_pct") is not None:
        with gcols[1]:
            render_kpi_tile("Margin % 📈", f"{kpis['gross_margin_pct']*100:,.1f}%")
    with gcols[2]:
        # heuristic: customers with freq >= 2 / all customers
        # heuristic: customers with freq >= 2 / all customers
        if schema.get("customer") and schema["customer"] in df2.columns:
            freq = df2.groupby(df2[schema["customer"]]).size()
            repeat_ratio = float((freq >= 2).mean() * 100)
            render_kpi_tile("Repeat Buyers 🔁", f"{repeat_ratio:,.1f}%")
        else:
            render_kpi_tile("Repeat Buyers 🔁", "—")
    with gcols[3]:
        if schema.get("quantity") and schema["quantity"] in df2.columns:
            qty_per_order = float(pd.to_numeric(df2[schema['quantity']], errors='coerce').mean())
            render_kpi_tile("Units/Order 🚚", f"{qty_per_order:,.2f}")
        else:
            render_kpi_tile("Units/Order 🚚", "—")
    with gcols[4]:
        if schema.get("country") and schema["country"] in df2.columns:
            render_kpi_tile("Markets 🌍", f"{df2[schema['country']].nunique():,}")
        else:
            render_kpi_tile("Markets 🌍", "—")
    # Latest Data 🕒 tile removed per request

    # Business gauges
    st.subheader("Business Health Indicators 🧠")
    repeat_ratio = None
    if schema.get("customer") and schema["customer"] in df2.columns:
        freq = df2.groupby(df2[schema["customer"]]).size()
        repeat_ratio = float((freq >= 2).mean() * 100)

    mom_growth_pct = kpis["mom_growth"] * 100 if kpis["mom_growth"] is not None else None
    margin_pct = kpis["gross_margin_pct"] * 100 if kpis.get("gross_margin_pct") is not None else None
    g1, g2, g3 = st.columns(3)
    with g1:
        if mom_growth_pct is not None:
            st.plotly_chart(
                gauge(mom_growth_pct, "Revenue Momentum", suffix="%", min_val=-50, max_val=50),
                use_container_width=True,
            )
        else:
            st.info("Revenue momentum needs a date column.")
    with g2:
        if margin_pct is not None:
            st.plotly_chart(
                gauge(margin_pct, "Profitability (GM%)", suffix="%", min_val=-100, max_val=100),
                use_container_width=True,
            )
        else:
            st.info("Profitability needs a `cost`/COGS column to compute gross margin.")
    with g3:
        if repeat_ratio is not None:
            st.plotly_chart(
                gauge(repeat_ratio, "Customer Loyalty (Repeat%)", suffix="%", min_val=0, max_val=100),
                use_container_width=True,
            )
        else:
            st.info("Customer loyalty needs a `customer` column.")

    # Advanced visuals
    st.subheader("Advanced Visuals 🚀")
    tabs = st.tabs(["Top Categories", "Geo Map", "Funnel", "Time Trend (Revenue)"])

    with tabs[0]:
        if schema.get("category") and schema.get("revenue"):
            st.plotly_chart(bar_agg(df2, schema["category"], schema["revenue"], agg="sum", top_n=15), use_container_width=True)
        else:
            st.info("Needs category and revenue columns.")

    with tabs[1]:
        if schema.get("country") and schema.get("revenue"):
            top_n = st.slider("Show top N countries", min_value=10, max_value=100, value=30, step=5)
            st.plotly_chart(
                geo_map(df2, schema["country"], schema["revenue"], top_n=top_n),
                use_container_width=True,
            )
        else:
            st.info("Needs country and revenue columns.")

    with tabs[2]:
        # Simple synthetic funnel: sessions -> add to cart -> checkout -> orders
        # If quantity present, treat orders as count of rows; else use rows
        orders = len(df2)
        funnel = [("Visits", orders * 5.0), ("Add to Cart", orders * 2.5), ("Checkout", orders * 1.5), ("Orders", float(orders))]
        st.plotly_chart(funnel_chart(funnel, "Sales Funnel"), use_container_width=True)

    with tabs[3]:
        if schema.get("date"):
            st.plotly_chart(timeseries(df2, schema["date"], schema.get("revenue"), freq="M"), use_container_width=True)
        else:
            st.info("Needs a date column.")


def show_ai_chat(df: pd.DataFrame):
    st.subheader("Ask questions about your data (AI)")
    secret_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = os.getenv("OPENAI_API_KEY", "") or secret_key

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "monthwise_revenue_df" not in st.session_state:
        st.session_state.monthwise_revenue_df = None

    def maybe_answer_monthwise_revenue(user_question: str) -> Optional[pd.DataFrame]:
        q = user_question.lower()
        is_revenue = "revenue" in q or "sales" in q
        is_month = any(k in q for k in ["month", "monthwise", "momth", "mounth", "monht"])
        if (not is_revenue) or (not is_month):
            return None
        schema = infer_schema(df)
        df2, schema = ensure_revenue_column(df, schema)
        date_col = schema.get("date")
        revenue_col = schema.get("revenue")
        if not date_col or not revenue_col:
            return None
        if date_col not in df2.columns or revenue_col not in df2.columns:
            return None
        tmp = df2[[date_col, revenue_col]].copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp[revenue_col] = pd.to_numeric(tmp[revenue_col], errors="coerce")
        tmp = tmp.dropna(subset=[date_col, revenue_col])
        if tmp.empty:
            return None
        tmp["month"] = tmp[date_col].dt.to_period("M").astype(str)
        out = tmp.groupby("month")[revenue_col].sum().reset_index().sort_values("month")
        out = out.rename(columns={revenue_col: "revenue"})
        return out

    user_input = st.chat_input("Ask a question, e.g., 'What is total revenue by month?'")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            try:
                # Fast deterministic answer for common question; avoids LLM errors for simple aggregations.
                monthwise = maybe_answer_monthwise_revenue(user_input)
                if monthwise is not None:
                    st.session_state.monthwise_revenue_df = monthwise
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": "✅ Total revenue by month (computed from your dataset).",
                        }
                    )
                else:
                    if not api_key:
                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": "AI is not available because `OPENAI_API_KEY` is missing. Add it to environment or `/.streamlit/secrets.toml`.",
                            }
                        )
                        return
                    assistant = DataframeAssistant(df, openai_api_key=api_key or None)
                    answer = assistant.answer(user_input)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
            except Exception as exc:
                st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {exc}"})

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg["content"].startswith("✅ Total revenue by month"):
                if st.session_state.monthwise_revenue_df is not None:
                    md_df = st.session_state.monthwise_revenue_df
                    st.plotly_chart(
                        px.bar(md_df, x="month", y="revenue", title="Monthly Revenue"),
                        use_container_width=True,
                    )
                    st.dataframe(md_df, use_container_width=True)


def main():
    st.title("Online Sales Analytics Dashboard")
    st.caption("Load your online sales dataset, explore with charts, KPI cards, and ask AI questions. ✨")

    df = load_dataset_silently()
    if df is None:
        st.error("Dataset not found at `~/Downloads/online_sales.zip`. Upload a dataset to continue.")
        uploaded = st.file_uploader("Upload CSV/ZIP/XLSX", type=["csv", "zip", "xlsx", "xls"])
        if uploaded is None:
            return
        tmp_path = f"/tmp/{uploaded.name}"
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        df, _details = load_from_path(tmp_path)
    df_filtered = top_filters(df)
    show_kpis_and_advanced(df_filtered)
    show_charts(df_filtered)
    show_ai_chat(df_filtered)
    show_retention_and_rfm(df_filtered)


def show_retention_and_rfm(df: pd.DataFrame):
    st.subheader("Retention & RFM 🎯")
    schema = infer_schema(df)
    df2, schema = ensure_revenue_column(df, schema)
    tabs = st.tabs(["RFM Segments", "R×F Matrix", "Revenue by Segment (Monthly)"])

    rfm = compute_rfm(df2, schema)
    if rfm is None:
        st.info("Needs `customer` and `date` columns to compute RFM.")
        return

    date_col = schema.get("date")
    revenue_col = schema.get("revenue")
    customer_col = schema.get("customer")

    with tabs[0]:
        left, right = st.columns([2, 1])
        with left:
            st.dataframe(
                rfm.sort_values(["Segment", "RFM_Score"], ascending=[True, False]).head(500),
                use_container_width=True,
            )
        with right:
            seg_counts = rfm.groupby("Segment").size().reset_index(name="count").sort_values("count", ascending=False)
            st.plotly_chart(px.bar(seg_counts, x="Segment", y="count", title="Customers by Segment"), use_container_width=True)

    with tabs[1]:
        # business-friendly heatmap: how customers distribute across R (recency) and F (frequency)
        pivot = rfm.pivot_table(
            index="R",
            columns="F",
            values=customer_col,
            aggfunc="count",
            fill_value=0,
        )
        pivot = pivot.reindex(index=[1, 2, 3, 4, 5], columns=[1, 2, 3, 4, 5])
        pivot_pct = (pivot / max(pivot.values.sum(), 1) * 100).round(1)
        # Data labels inside each cell (business users usually want exact values)
        text_labels = pivot_pct.applymap(lambda v: f"{v:.1f}%")
        fig = px.imshow(
            pivot_pct,
            labels=dict(x="Frequency (F)", y="Recency (R)", color="% of customers"),
            x=pivot_pct.columns.astype(str),
            y=pivot_pct.index.astype(str),
            color_continuous_scale="Blues",
            aspect="auto",
            title="Customer distribution by Recency (R) and Frequency (F)",
        )
        fig.update_traces(
            text=text_labels.to_numpy(),
            texttemplate="%{text}",
            textfont={"size":12, "color":"black"},
            hovertemplate="Recency (R): %{y}<br>Frequency (F): %{x}<br>% of customers: %{z:.1f}%<extra></extra>",
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Cells show % of customers. Higher R = more recent; higher F = purchased more often.")

    with tabs[2]:
        if date_col and revenue_col and customer_col and date_col in df2.columns and revenue_col in df2.columns:
            tmp = df2[[date_col, revenue_col, customer_col]].copy()
            tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
            tmp[revenue_col] = pd.to_numeric(tmp[revenue_col], errors="coerce")
            tmp = tmp.dropna(subset=[date_col, customer_col, revenue_col])
            if tmp.empty:
                st.info("No data available to compute revenue by segment.")
            else:
                tmp["month"] = tmp[date_col].dt.to_period("M").astype(str)
                tmp = tmp.groupby(["month", customer_col])[revenue_col].sum().reset_index()
                tmp = tmp.merge(rfm[[customer_col, "Segment"]], on=customer_col, how="left")
                seg_month = tmp.groupby(["month", "Segment"])[revenue_col].sum().reset_index()
                seg_month = seg_month.sort_values(["month", revenue_col], ascending=[True, False])

                fig = px.bar(
                    seg_month,
                    x="month",
                    y=revenue_col,
                    color="Segment",
                    barmode="stack",
                    title="Monthly Revenue by RFM Segment",
                )
                fig.update_layout(height=480, margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Needs `date` and `revenue` columns to compute revenue by segment.")


if __name__ == "__main__":
    main()
