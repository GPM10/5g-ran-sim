import os
from typing import Dict, List, Optional

import altair as alt
import httpx
import pandas as pd
import streamlit as st


DEFAULT_API = os.getenv("API_BASE", "http://127.0.0.1:8000")


def fetch_json(path: str, params: Optional[Dict] = None, api_base: str = DEFAULT_API) -> Dict:
    url = f"{api_base.rstrip('/')}{path}"
    with httpx.Client(timeout=5.0) as client:
        response = client.get(url, params=params or {})
        response.raise_for_status()
        return response.json()


def build_history_frame(history: Dict[str, List[float]]) -> pd.DataFrame:
    if not history:
        return pd.DataFrame()
    df = pd.DataFrame(history)
    df["step"] = range(len(df))
    return df.set_index("step")


def render_metrics(snapshot: Dict):
    aggregates = snapshot.get("aggregates", {})
    cols = st.columns(4)
    cols[0].metric("Avg Latency (ms)", f"{aggregates.get('avg_latency_ms', 0.0):.1f}")
    cols[1].metric("Avg Throughput (Mbps)", f"{aggregates.get('avg_throughput_mbps', 0.0):.2f}")
    cols[2].metric("Max Cell Load", f"{aggregates.get('max_cell_load', 0.0):.2f}")
    cols[3].metric("Avg SINR (dB)", f"{aggregates.get('avg_sinr_db', 0.0):.1f}")


def render_top_latency(snapshot: Dict, limit: int = 5):
    ue_stats = snapshot.get("ue_stats", [])
    if not ue_stats:
        st.info("No UE stats available yet.")
        return
    sorted_ues = sorted(
        ue_stats,
        key=lambda u: (u["latency_ms"] - u["latency_budget_ms"]),
        reverse=True,
    )
    st.subheader("Top latency-challenged UEs")
    st.table(
        [
            {
                "UE": item["ue_id"],
                "Serving BS": item["serving_bs"],
                "Latency (ms)": f"{item['latency_ms']:.1f}",
                "Budget (ms)": item["latency_budget_ms"],
                "Queue (Mbits)": f"{item.get('queue_mbits', 0.0):.2f}",
            }
            for item in sorted_ues[:limit]
        ]
    )


def render_topology(snapshot: Dict):
    cell_stats = snapshot.get("cell_stats", [])
    ue_stats = snapshot.get("ue_stats", [])
    if not cell_stats:
        st.info("No topology data yet.")
        return
    cells_df = pd.DataFrame(cell_stats)
    cells_df["type"] = "Base Station"
    cells_df["x"] = pd.to_numeric(cells_df["x"], errors="coerce")
    cells_df["y"] = pd.to_numeric(cells_df["y"], errors="coerce")
    cells_df = cells_df.dropna(subset=["x", "y"])

    ue_df = pd.DataFrame(ue_stats) if ue_stats else pd.DataFrame(columns=["x", "y"])
    ue_df["type"] = "UE"
    ue_df["traffic_profile"] = ue_df.get("traffic_profile", "unknown")
    ue_df["x"] = pd.to_numeric(ue_df.get("x"), errors="coerce")
    ue_df["y"] = pd.to_numeric(ue_df.get("y"), errors="coerce")
    ue_df["sinr_db"] = pd.to_numeric(ue_df.get("sinr_db"), errors="coerce")
    ue_df = ue_df.dropna(subset=["x", "y"])

    chart_cells = (
        alt.Chart(cells_df)
        .mark_point(size=200, shape="square")
        .encode(
            x=alt.X("x:Q", title="X (m)"),
            y=alt.Y("y:Q", title="Y (m)"),
            color=alt.Color("tier", title="Tier"),
            tooltip=["bs_id", "tier", "load", "capacity_mbps"],
        )
    )
    if not ue_df.empty:
        chart_ues = (
            alt.Chart(ue_df)
            .mark_point(filled=True, opacity=0.6)
            .encode(
                x=alt.X("x:Q"),
                y=alt.Y("y:Q"),
                color=alt.Color("traffic_profile", title="UE profile"),
                shape=alt.Shape("traffic_profile", title="UE profile"),
                tooltip=["ue_id", "serving_bs", "traffic_profile", "latency_ms"],
            )
        )
        links_df = build_link_df(ue_df, cells_df)
        if not links_df.empty:
            chart_links = (
                alt.Chart(links_df)
                .mark_rule(opacity=0.5)
                .encode(
                    x=alt.X("ue_x:Q"),
                    x2=alt.X2("bs_x:Q"),
                    y=alt.Y("ue_y:Q"),
                    y2=alt.Y2("bs_y:Q"),
                    color=alt.Color(
                        "sinr_db:Q", title="SINR (dB)", scale=alt.Scale(scheme="viridis")
                    ),
                    tooltip=["ue_id", "bs_id", "sinr_db"],
                )
            )
            chart = chart_cells + chart_ues + chart_links
        else:
            chart = chart_cells + chart_ues
    else:
        chart = chart_cells
    st.subheader("Topology Layout")
    st.altair_chart(chart.interactive().properties(height=400), use_container_width=True)


def build_link_df(ue_df: pd.DataFrame, cells_df: pd.DataFrame) -> pd.DataFrame:
    if ue_df.empty or cells_df.empty:
        return pd.DataFrame()
    bs_lookup = cells_df.set_index("bs_id")[["x", "y"]]
    records = []
    for row in ue_df.itertuples():
        bs_id = getattr(row, "serving_bs", None)
        if bs_id not in bs_lookup.index:
            continue
        bs_coords = bs_lookup.loc[bs_id]
        sinr = getattr(row, "sinr_db", None)
        records.append(
            {
                "ue_id": row.ue_id,
                "bs_id": bs_id,
                "ue_x": row.x,
                "ue_y": row.y,
                "bs_x": float(bs_coords.x),
                "bs_y": float(bs_coords.y),
                "sinr_db": sinr,
            }
        )
    return pd.DataFrame(records)


def main():
    st.set_page_config(page_title="RIC Metrics Dashboard", layout="wide")
    st.title("Latency-Aware RIC Metrics Dashboard")

    api_base = st.sidebar.text_input("Metrics API base URL", value=DEFAULT_API)
    history_tail = st.sidebar.slider("History window (steps)", 10, 200, 60, step=10)
    if st.sidebar.button("Refresh now"):
        st.experimental_rerun()

    placeholder = st.empty()
    try:
        snapshot = fetch_json("/metrics", params={"advance": False}, api_base=api_base)
        history = fetch_json("/history", params={"tail": history_tail}, api_base=api_base)
    except Exception as exc:  # noqa: BLE001
        placeholder.error(f"Failed to reach API at {api_base}: {exc}")
        return

    render_metrics(snapshot)
    render_topology(snapshot)
    render_top_latency(snapshot)

    history_df = build_history_frame(history)
    if not history_df.empty:
        st.subheader("Latency / Throughput trends")
        st.line_chart(history_df[["avg_latency_ms", "avg_throughput"]])
        st.subheader("Cell load + queues")
        st.line_chart(history_df[["max_load", "avg_queue_mbits"]])
    else:
        st.info("Waiting for history samples...")


if __name__ == "__main__":
    main()
