# app.py
# ─────────────────────────────────────────────────────────────────
# Feedback Analysis System — Streamlit Application
#
# Run with:   streamlit run app.py
# ─────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import sys
import os

# ── Path setup so backend modules are importable ─────────────────
sys.path.insert(0, os.path.dirname(__file__))

from backend.nlp.auto_clustering    import run_clustering_pipeline
from backend.nlp.format_responses   import get_all_clusters_table, generate_formatted_responses
from backend.nlp.analysis_modules   import (
    label_all_clusters,
    analyze_sentiment,
    cluster_themes,
    extract_actionable_insights,
)
from backend.utils.get_system_prompt import get_system_prompt
from backend.utils.document_reader   import load_file
from backend.nlp.llm_client          import get_active_backend

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Feedback Analysis System",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — matches the dark analytical aesthetic ────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;0,9..144,900;1,9..144,300;1,9..144,600&display=swap');

:root {
  --bg:      #0b0d12; --surface: #111318; --surface2: #181c25;
  --border:  #232840; --gold: #e8c468;    --teal: #5ebfb5;
  --coral:   #e07b6a; --violet: #8b7fe8;  --mint: #76d9a8;
  --text-1:  #eeeae0; --text-2: #8a8f9e;  --text-3: #4a5068;
}

/* Global overrides */
html, body, [class*="css"] {
  font-family: 'DM Mono', monospace !important;
  background-color: var(--bg) !important;
  color: var(--text-1) !important;
}
.stApp { background-color: var(--bg); }

/* Headers */
h1, h2, h3 { font-family: 'Fraunces', serif !important; letter-spacing: -.3px; }
h1 { color: var(--text-1); }
h2 { color: var(--text-1); font-size: 1.5rem !important; }
h3 { color: var(--text-2); font-size: 1.1rem !important; }

/* Metric cards */
[data-testid="metric-container"] {
  background: var(--surface); border: 1px solid var(--border);
  padding: 16px; border-radius: 3px;
}
[data-testid="metric-container"] label { color: var(--text-3) !important; font-size: 10px !important; text-transform: uppercase; letter-spacing: .12em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--gold) !important; font-family: 'Fraunces', serif !important; font-size: 2.2rem !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text-2) !important; }

/* Buttons */
.stButton > button {
  background: var(--gold) !important; color: #0b0d12 !important;
  border: none !important; border-radius: 3px !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 12px !important; font-weight: 500 !important;
  text-transform: uppercase !important; letter-spacing: .1em !important;
  padding: 10px 24px !important;
}
.stButton > button:hover { background: #f0d080 !important; }

/* Inputs */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > div {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-1) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 12px !important;
  border-radius: 3px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
  background: var(--surface) !important;
  border: 1px dashed var(--border) !important;
  border-radius: 3px !important;
}

/* Dataframe */
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 3px !important; }

/* Expander */
.streamlit-expanderHeader {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-2) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 12px !important;
}
.streamlit-expanderContent {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
}

/* Status / info boxes */
.stAlert { border-radius: 3px !important; font-size: 12px !important; }
.stSuccess { background: rgba(94,191,181,.1) !important; border-color: var(--teal) !important; color: var(--teal) !important; }
.stWarning { background: rgba(232,196,104,.08) !important; border-color: var(--gold) !important; color: var(--gold) !important; }
.stError   { background: rgba(224,123,106,.08) !important; border-color: var(--coral) !important; color: var(--coral) !important; }
.stInfo    { background: rgba(139,127,232,.08) !important; border-color: var(--violet) !important; color: var(--violet) !important; }

/* Progress bar */
.stProgress > div > div > div > div { background: var(--gold) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: var(--surface) !important; border-bottom: 1px solid var(--border) !important; }
.stTabs [data-baseweb="tab"] { color: var(--text-3) !important; font-family: 'DM Mono', monospace !important; font-size: 11px !important; }
.stTabs [aria-selected="true"] { color: var(--gold) !important; border-bottom-color: var(--gold) !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ═══════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "pipeline_done":     False,
        "labeled_df":        None,
        "best_k":            None,
        "likert_cols":       [],
        "text_cols":         [],
        "cluster_labels":    {},
        "sentiment_data":    {},
        "theme_data":        {},
        "action_data":       {},
        "pca_coords":        None,
        "program_name":      "",
        "document_text":     "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()


# ═══════════════════════════════════════════════════════════════════
# COLOUR HELPERS
# ═══════════════════════════════════════════════════════════════════
CLUSTER_COLORS = ["#e8c468", "#5ebfb5", "#e07b6a", "#8b7fe8", "#76d9a8"]

def cluster_color(i: int) -> str:
    return CLUSTER_COLORS[i % len(CLUSTER_COLORS)]

def sentiment_color(s: str) -> str:
    return {"positive": "#5ebfb5", "negative": "#e07b6a", "neutral": "#8a8f9e", "mixed": "#8b7fe8"}.get(s, "#8a8f9e")


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ◉ Feedback Analysis")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        options=["Upload & Config", "Run Pipeline", "Cluster Profiles", "Dashboard", "Respondent Table", "Ask AI"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Pipeline Status**")
    if st.session_state.pipeline_done:
        st.success(f"✓ {st.session_state.best_k} clusters · {len(st.session_state.labeled_df)} respondents")
        urgent = sum(1 for r in (st.session_state.sentiment_data.get("results") or []) if r.get("flag_urgent"))
        if urgent:
            st.warning(f"⚠ {urgent} urgent flags")
    else:
        st.info("No analysis loaded")

    st.markdown("---")
    st.caption(f"Backend: {get_active_backend() or 'none'}")


# ═══════════════════════════════════════════════════════════════════
# PAGE: UPLOAD & CONFIG
# ═══════════════════════════════════════════════════════════════════
if page == "Upload & Config":
    st.title("Configure Your Analysis")
    st.markdown("Upload your survey CSV and configure how the AI should cluster and analyse responses.")

    col1, col2 = st.columns(2)
    with col1:
        csv_file = st.file_uploader("Survey CSV", type=["csv"], help="Likert rating columns + free-text response columns")
    with col2:
        doc_file = st.file_uploader("Program Context (optional)", type=["pdf","docx","txt"], help="Program brief to ground the AI's analysis")

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Clustering Settings")
        auto_cluster = st.toggle("Auto-discover clusters (recommended)", value=True)
        k_val = None
        if not auto_cluster:
            k_val = st.number_input("Number of clusters (k)", min_value=2, max_value=10, value=4)
        themes_raw = st.text_area(
            "Predefined themes — one per line (leave blank for auto-discovery)",
            placeholder="Facilitator Effectiveness\nModule Pacing\nContent Relevance\nFollow-up Support",
            height=120,
        )

    with col4:
        st.markdown("#### Analysis Scope")
        program_name = st.text_input("Programme / evaluation name", placeholder="e.g. Leadership Development Series — Q4 2024")
        flag_urgent  = st.toggle("Flag urgent negative responses", value=True)
        gen_insights = st.toggle("Extract actionable insights",    value=True)

    st.markdown("---")

    if st.button("⚡ Run Analysis Pipeline", disabled=(csv_file is None), use_container_width=True):
        # ── Store config ─────────────────────────────────────────
        st.session_state.program_name = program_name

        # ── Load CSV ──────────────────────────────────────────────
        try:
            csv_result = load_file(csv_file, csv_file.name)
            df_raw     = csv_result["dataframe"]
            # Drop first column if it looks like an auto-index
            if df_raw.columns[0].lower() in ("", "unnamed: 0", "index"):
                df_raw = df_raw.iloc[:, 1:]
            st.success(f"✓ Loaded {len(df_raw)} rows × {len(df_raw.columns)} columns")
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")
            st.stop()

        # ── Load context document ─────────────────────────────────
        doc_text = ""
        if doc_file:
            try:
                doc_result = load_file(doc_file, doc_file.name)
                doc_text   = doc_result.get("raw_text") or ""
                st.success(f"✓ Loaded context document ({doc_file.name})")
            except Exception as e:
                st.warning(f"Could not read context document: {e}")

        st.session_state.document_text = doc_text
        system_prompt = get_system_prompt(doc_text, program_name)

        # ── STAGE 1-2: Clustering ─────────────────────────────────
        with st.spinner("Stage 1–2: Embedding & clustering responses…"):
            try:
                pipeline_out = run_clustering_pipeline(
                    dataframe  = df_raw,
                    force_k    = None if auto_cluster else k_val,
                    n_pca_dims = 30,
                )
                st.session_state.labeled_df  = pipeline_out["labeled_df"]
                st.session_state.best_k      = pipeline_out["best_k"]
                st.session_state.likert_cols = pipeline_out["likert_cols"]
                st.session_state.text_cols   = pipeline_out["text_cols"]
                st.session_state.pca_coords  = pipeline_out["pca_coords"]
                st.success(f"✓ Clustering complete — {pipeline_out['best_k']} clusters")
            except Exception as e:
                st.error(f"Clustering failed: {e}")
                st.stop()

        # Build shared context tables
        all_clusters_table = get_all_clusters_table(
            st.session_state.labeled_df,
            st.session_state.best_k,
            st.session_state.likert_cols,
            st.session_state.text_cols,
        )

        # ── STAGE 3: Cluster Labelling ────────────────────────────
        with st.spinner("Stage 3: Labelling clusters with AI…"):
            try:
                labels = label_all_clusters(
                    best_k             = st.session_state.best_k,
                    labeled_df         = st.session_state.labeled_df,
                    likert_cols        = st.session_state.likert_cols,
                    text_cols          = st.session_state.text_cols,
                    all_clusters_table = all_clusters_table,
                    system_prompt      = system_prompt,
                )
                st.session_state.cluster_labels = labels
                st.success(f"✓ {len(labels)} clusters labelled")
            except Exception as e:
                st.warning(f"Cluster labelling failed: {e}")
                st.session_state.cluster_labels = {}

        # ── STAGE 4: Sentiment ────────────────────────────────────
        with st.spinner("Stage 4: Analysing sentiment…"):
            try:
                sent = analyze_sentiment(all_clusters_table, system_prompt)
                st.session_state.sentiment_data = sent
                n_urgent = sum(1 for r in (sent.get("results") or []) if r.get("flag_urgent"))
                st.success(f"✓ {sent.get('total_classified', 0)} responses classified · {n_urgent} urgent")
            except Exception as e:
                st.warning(f"Sentiment analysis failed: {e}")
                st.session_state.sentiment_data = {}

        # ── STAGE 5: Themes ───────────────────────────────────────
        with st.spinner("Stage 5: Clustering themes…"):
            try:
                predefined = [t.strip() for t in themes_raw.split("\n") if t.strip()]
                themes = cluster_themes(all_clusters_table, system_prompt, predefined or None)
                st.session_state.theme_data = themes
                st.success(f"✓ {len(themes.get('themes', []))} themes identified")
            except Exception as e:
                st.warning(f"Theme clustering failed: {e}")
                st.session_state.theme_data = {}

        # ── STAGE 6: Insights ─────────────────────────────────────
        if gen_insights:
            with st.spinner("Stage 6: Extracting actionable insights…"):
                try:
                    actions = extract_actionable_insights(all_clusters_table, system_prompt)
                    st.session_state.action_data = actions
                    st.success(f"✓ {actions.get('total_insights', 0)} actionable insights")
                except Exception as e:
                    st.warning(f"Insight extraction failed: {e}")
                    st.session_state.action_data = {}

        st.session_state.pipeline_done = True
        st.balloons()
        st.success("✓ Analysis complete! Navigate to Cluster Profiles or Dashboard.")


# ═══════════════════════════════════════════════════════════════════
# PAGE: RUN PIPELINE (status view)
# ═══════════════════════════════════════════════════════════════════
elif page == "Run Pipeline":
    st.title("Pipeline Status")
    if st.session_state.pipeline_done:
        st.success("Pipeline complete. Navigate to Cluster Profiles or Dashboard.")
        r = st.session_state
        col1, col2, col3 = st.columns(3)
        col1.metric("Clusters", r.best_k)
        col2.metric("Respondents", len(r.labeled_df))
        col3.metric("Themes", len((r.theme_data or {}).get("themes", [])))
    else:
        st.info("Run the pipeline from the Upload & Config page.")


# ═══════════════════════════════════════════════════════════════════
# PAGE: CLUSTER PROFILES
# ═══════════════════════════════════════════════════════════════════
elif page == "Cluster Profiles":
    st.title("Cluster Profiles")

    if not st.session_state.pipeline_done:
        st.info("Run the pipeline first from Upload & Config.")
        st.stop()

    df        = st.session_state.labeled_df
    k         = st.session_state.best_k
    labels    = st.session_state.cluster_labels
    sent_data = st.session_state.sentiment_data
    sent_summ = (sent_data or {}).get("cluster_summary", {})

    # Export button
    col_hdr, col_btn = st.columns([3, 1])
    with col_btn:
        if st.button("↓ Export Labels JSON"):
            json_bytes = json.dumps(labels, indent=2).encode()
            st.download_button("Download cluster_labels.json", json_bytes, "cluster_labels.json", "application/json")

    for ci in range(k):
        cl  = labels.get(str(ci), labels.get(ci, {}))
        col = cluster_color(ci)
        sc  = sent_summ.get(str(ci), sent_summ.get(ci, {}))

        # Colour-coded expander per cluster
        with st.expander(f"{'●'} Cluster {ci} — {cl.get('label', f'Cluster {ci}')}", expanded=(ci == 0)):
            left, right = st.columns([3, 2])

            with left:
                # Label + n
                cluster_rows = df[df["cluster"] == ci]
                st.markdown(
                    f"<div style='font-family:Fraunces,serif;font-size:22px;font-weight:900;"
                    f"color:{col};margin-bottom:4px'>{cl.get('label', f'Cluster {ci}')}</div>"
                    f"<div style='font-size:10px;color:#4a5068;text-transform:uppercase;letter-spacing:.12em'>"
                    f"n = {len(cluster_rows)}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("---")

                # Respondent profile
                st.markdown("**Respondent Profile**")
                st.markdown(
                    f"<div style='font-family:Fraunces,serif;font-style:italic;font-size:13px;"
                    f"color:#8a8f9e;border-left:3px solid {col};padding-left:12px;line-height:1.7'>"
                    f"{cl.get('respondent_profile', '—')}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("")

                # Key drivers
                st.markdown("**Key Drivers**")
                drivers = cl.get("key_drivers", [])
                driver_html = " ".join(
                    f"<span style='display:inline-block;background:#181c25;border:1px solid #232840;"
                    f"padding:4px 10px;border-radius:20px;font-size:10px;color:#8a8f9e;margin:2px'>"
                    f"<span style='display:inline-block;width:5px;height:5px;border-radius:50%;"
                    f"background:{col};margin-right:5px;vertical-align:middle'></span>{d}</span>"
                    for d in drivers
                )
                st.markdown(driver_html or "—", unsafe_allow_html=True)
                st.markdown("")

                # Distinguishing feature
                st.markdown("**Distinguishing Feature**")
                st.markdown(
                    f"<div style='font-size:11px;color:#8a8f9e;background:#181c25;padding:10px 14px;"
                    f"border-radius:3px;border:1px solid #232840'>"
                    f"{cl.get('distinguishing_features', '—')}</div>",
                    unsafe_allow_html=True,
                )

            with right:
                # Average ratings bars
                if st.session_state.likert_cols:
                    st.markdown("**Average Ratings**")
                    avg = df[df["cluster"] == ci][st.session_state.likert_cols].mean().round(2)
                    for col_name, val in avg.items():
                        if pd.notna(val):
                            pct = int((val / 5) * 100)
                            st.markdown(
                                f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:10px;color:#4a5068'>"
                                f"<div style='width:120px;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{col_name}</div>"
                                f"<div style='flex:1;height:3px;background:#232840;border-radius:2px;overflow:hidden'>"
                                f"<div style='width:{pct}%;height:100%;background:{col};border-radius:2px'></div></div>"
                                f"<div style='width:36px;text-align:right;color:#8a8f9e'>{val}/5</div></div>",
                                unsafe_allow_html=True,
                            )

                # Sentiment summary
                if sc:
                    st.markdown("**Sentiment**")
                    s_pos = sc.get("positive", 0)
                    s_neg = sc.get("negative", 0)
                    s_neu = sc.get("neutral", 0)
                    cols_s = st.columns(3)
                    cols_s[0].metric("Positive", f"{s_pos}%")
                    cols_s[1].metric("Negative", f"{s_neg}%")
                    cols_s[2].metric("Neutral",  f"{s_neu}%")


# ═══════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════
elif page == "Dashboard":
    st.title("Insights Dashboard")

    if not st.session_state.pipeline_done:
        st.info("Run the pipeline first from Upload & Config.")
        st.stop()

    df        = st.session_state.labeled_df
    k         = st.session_state.best_k
    labels    = st.session_state.cluster_labels
    sent_data = st.session_state.sentiment_data or {}
    theme_data = st.session_state.theme_data or {}
    action_data = st.session_state.action_data or {}
    sent_summ = sent_data.get("cluster_summary", {})
    urgent    = [r for r in (sent_data.get("results") or []) if r.get("flag_urgent")]

    # ── KPIs ──────────────────────────────────────────────────────
    all_pos = [s.get("positive", 0) for s in sent_summ.values()]
    avg_pos = round(sum(all_pos) / len(all_pos)) if all_pos else 0
    avg_sat = 0.0
    if st.session_state.likert_cols:
        avg_sat = df[st.session_state.likert_cols].mean().mean()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Overall Positive",  f"{avg_pos}%",   help="% of responses classified as positive")
    kpi2.metric("Urgent Flags",       len(urgent),     help="Responses flagged for immediate attention")
    kpi3.metric("Avg Satisfaction",   f"{avg_sat:.1f}", help="Mean Likert rating across all questions")
    kpi4.metric("Action Items",       len(action_data.get("insights", [])), help="Improvement suggestions extracted")

    st.markdown("---")

    # ── Scatter + Sentiment ───────────────────────────────────────
    try:
        import plotly.graph_objects as go
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**Cluster Map** — PCA 2D projection")
            fig_scatter = go.Figure()
            pca = st.session_state.pca_coords
            for ci in range(k):
                mask = df["cluster"].values == ci
                if pca is not None and len(pca) >= len(df):
                    xs = pca[mask, 0]
                    ys = pca[mask, 1]
                else:
                    center = [(-2.5 + ci * 1.8), (1.8 - ci * 1.2)]
                    xs = center[0] + np.random.randn(mask.sum()) * 0.8
                    ys = center[1] + np.random.randn(mask.sum()) * 0.8
                cl = labels.get(str(ci), labels.get(ci, {}))
                fig_scatter.add_trace(go.Scatter(
                    x=xs, y=ys, mode="markers",
                    name=cl.get("label", f"C{ci}"),
                    marker=dict(color=cluster_color(ci), size=8, opacity=0.8),
                ))
            fig_scatter.update_layout(
                paper_bgcolor="#111318", plot_bgcolor="#111318",
                font=dict(family="DM Mono", color="#8a8f9e", size=10),
                legend=dict(font=dict(size=10), bgcolor="#111318"),
                margin=dict(l=0, r=0, t=10, b=0),
                height=280,
                xaxis=dict(gridcolor="#232840"), yaxis=dict(gridcolor="#232840"),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_right:
            st.markdown("**Sentiment by Cluster**")
            cl_names = [labels.get(str(i), labels.get(i, {})).get("label", f"C{i}") for i in range(k)]
            pos_vals = [sent_summ.get(str(i), {}).get("positive", 0) for i in range(k)]
            neu_vals = [sent_summ.get(str(i), {}).get("neutral",  0) for i in range(k)]
            neg_vals = [sent_summ.get(str(i), {}).get("negative", 0) for i in range(k)]
            fig_sent = go.Figure(data=[
                go.Bar(name="Positive", x=cl_names, y=pos_vals, marker_color="#5ebfb5cc"),
                go.Bar(name="Neutral",  x=cl_names, y=neu_vals, marker_color="#8a8f9e55"),
                go.Bar(name="Negative", x=cl_names, y=neg_vals, marker_color="#e07b6acc"),
            ])
            fig_sent.update_layout(
                barmode="stack", paper_bgcolor="#111318", plot_bgcolor="#111318",
                font=dict(family="DM Mono", color="#8a8f9e", size=10),
                legend=dict(font=dict(size=10), bgcolor="#111318"),
                margin=dict(l=0, r=0, t=10, b=0), height=280,
                xaxis=dict(gridcolor="#232840"), yaxis=dict(gridcolor="#232840", ticksuffix="%"),
            )
            st.plotly_chart(fig_sent, use_container_width=True)

    except ImportError:
        st.info("Install plotly for charts: `pip install plotly`")

    st.markdown("---")

    # ── Themes + Actions ──────────────────────────────────────────
    themes = theme_data.get("themes", [])
    insights = action_data.get("insights", [])

    col_t, col_a = st.columns(2)

    with col_t:
        st.markdown("**Theme Frequency**")
        if themes:
            try:
                import plotly.graph_objects as go
                fig_theme = go.Figure(go.Bar(
                    x=[t["count"] for t in themes],
                    y=[t["name"]  for t in themes],
                    orientation="h",
                    marker_color=[CLUSTER_COLORS[i % 5] + "88" for i in range(len(themes))],
                ))
                fig_theme.update_layout(
                    paper_bgcolor="#111318", plot_bgcolor="#111318",
                    font=dict(family="DM Mono", color="#8a8f9e", size=10),
                    margin=dict(l=0, r=0, t=10, b=0), height=260,
                    xaxis=dict(gridcolor="#232840"), yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_theme, use_container_width=True)
            except ImportError:
                for t in themes:
                    st.write(f"**{t['name']}** — {t['count']} mentions")
        else:
            st.info("No themes available.")

    with col_a:
        st.markdown("**Actionable Suggestions**")
        if insights:
            priority_colors = {"high": "#e07b6a", "medium": "#e8c468", "low": "#5ebfb5"}
            for ins in insights[:6]:
                pc = priority_colors.get(ins.get("priority",""), "#e8c468")
                st.markdown(
                    f"<div style='border-left:3px solid {pc};padding:10px 14px;background:#181c25;"
                    f"border-radius:3px;margin-bottom:8px;border:1px solid #232840'>"
                    f"<span style='font-size:9px;text-transform:uppercase;letter-spacing:.1em;color:{pc}'>"
                    f"{ins.get('priority','').upper()}</span> "
                    f"<span style='font-size:9px;color:#4a5068'>· {ins.get('category','')}</span>"
                    f"<div style='font-size:11px;color:#eeeae0;margin-top:6px'>{ins.get('insight','')}</div>"
                    f"<div style='font-size:10px;color:#4a5068;margin-top:3px'>"
                    f"Clusters {ins.get('source_clusters',[])} · {ins.get('breadth','')}</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No actionable insights extracted.")

    # ── Urgent Flags ───────────────────────────────────────────────
    if urgent:
        st.markdown("---")
        st.markdown("**⚠ Urgent Flags**")
        for f in urgent:
            st.error(
                f"🚨 **{f['respondent_id']}** (Cluster {f['cluster']}) — "
                f"{f.get('flag_reason','Flagged for urgent review')}"
            )


# ═══════════════════════════════════════════════════════════════════
# PAGE: RESPONDENT TABLE
# ═══════════════════════════════════════════════════════════════════
elif page == "Respondent Table":
    st.title("Respondent Table")

    if not st.session_state.pipeline_done:
        st.info("Run the pipeline first from Upload & Config.")
        st.stop()

    df        = st.session_state.labeled_df
    k         = st.session_state.best_k
    labels    = st.session_state.cluster_labels
    sent_data = st.session_state.sentiment_data or {}
    results   = sent_data.get("results", [])
    text_cols = st.session_state.text_cols

    # ── Build flat table ──────────────────────────────────────────
    sent_map = {r["respondent_id"]: r for r in results}

    rows = []
    for ci in range(k):
        cluster_rows = df[df["cluster"] == ci]
        cl = labels.get(str(ci), labels.get(ci, {}))
        for idx, row in cluster_rows.iterrows():
            rid  = f"R{str(idx + 1).zfill(3)}"
            sent = sent_map.get(rid, {})
            # First text column for preview
            preview = ""
            question_text = ""
            if text_cols:
                preview = str(row.get(text_cols[0], "") or "").strip()[:120]
                question_text = text_cols[0]
            rows.append({
                "ID":            rid,
                "Cluster":       f"C{ci}: {cl.get('label', f'Cluster {ci}')}",
                "Sentiment":     sent.get("sentiment", "—"),
                "Confidence":    sent.get("confidence", "—"),
                "Urgent":        "🚨" if sent.get("flag_urgent") else "",
                "Key Phrases":   " · ".join(sent.get("key_phrases", [])),
                "Urgent Reason": sent.get("flag_reason") or "",
                f"Q: {question_text}" if question_text else "Response": preview,
            })

    flat_df = pd.DataFrame(rows)

    # ── Filters ───────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        cluster_options = ["All"] + [f"C{i}: {labels.get(str(i),{}).get('label','')}" for i in range(k)]
        filter_cluster  = st.selectbox("Filter by Cluster", cluster_options)
    with col2:
        filter_sentiment = st.selectbox("Filter by Sentiment", ["All","positive","neutral","negative","mixed"])
    with col3:
        filter_urgent = st.selectbox("Filter by Urgent", ["All","Urgent Only"])

    if filter_cluster   != "All":       flat_df = flat_df[flat_df["Cluster"] == filter_cluster]
    if filter_sentiment != "All":       flat_df = flat_df[flat_df["Sentiment"] == filter_sentiment]
    if filter_urgent    == "Urgent Only": flat_df = flat_df[flat_df["Urgent"] == "🚨"]

    # ── Export ────────────────────────────────────────────────────
    csv_bytes = flat_df.to_csv(index=False).encode()
    st.download_button("↓ Export CSV", csv_bytes, "respondent_table.csv", "text/csv")

    st.markdown(f"**{len(flat_df)} respondents**")

    # ── Table ─────────────────────────────────────────────────────
    st.dataframe(
        flat_df,
        use_container_width=True,
        column_config={
            "Sentiment": st.column_config.TextColumn("Sentiment"),
            "Urgent":    st.column_config.TextColumn("⚑"),
        },
        hide_index=True,
    )

    # ── Respondent Detail Expander ────────────────────────────────
    st.markdown("---")
    st.markdown("**View Full Response Detail**")
    selected_id = st.text_input("Enter Respondent ID (e.g. R001)", placeholder="R001")
    if selected_id:
        sent_rec = sent_map.get(selected_id, {})
        # Find the DataFrame row
        idx = None
        try:
            idx = int(selected_id.replace("R","")) - 1
        except Exception:
            pass
        if idx is not None and idx < len(df):
            row = df.iloc[idx]
            ci  = int(row.get("cluster", 0))
            col = cluster_color(ci)
            cl  = labels.get(str(ci), {})

            left_d, right_d = st.columns([2, 1])
            with left_d:
                st.markdown(
                    f"<div style='color:{col};font-size:9px;text-transform:uppercase;letter-spacing:.12em'>"
                    f"Cluster {ci}: {cl.get('label','—')}</div>",
                    unsafe_allow_html=True,
                )
                if text_cols:
                    for qcol in text_cols:
                        answer = str(row.get(qcol, "") or "").strip()
                        if answer:
                            st.markdown(
                                f"<div style='font-size:9px;color:#4a5068;text-transform:uppercase;letter-spacing:.1em;margin-top:12px'>"
                                f"Q: {qcol}</div>"
                                f"<div style='font-family:Fraunces,serif;font-style:italic;font-size:13px;color:#eeeae0;"
                                f"border-left:3px solid {col};padding:10px 14px;background:#181c25;margin-top:4px;line-height:1.7'>"
                                f"\u201c{answer}\u201d</div>",
                                unsafe_allow_html=True,
                            )

            with right_d:
                s = sent_rec.get("sentiment","—")
                st.markdown(f"**Sentiment:** {s}")
                st.markdown(f"**Confidence:** {sent_rec.get('confidence','—')}")
                if sent_rec.get("flag_urgent"):
                    st.error(f"🚨 {sent_rec.get('flag_reason','Urgent')}")
                if sent_rec.get("key_phrases"):
                    st.markdown("**Key Phrases**")
                    for p in sent_rec["key_phrases"]:
                        st.markdown(
                            f"<span style='display:inline-block;background:#1e2330;border:1px solid #232840;"
                            f"padding:3px 10px;border-radius:10px;font-size:10px;color:#4a5068;margin:2px'>{p}</span>",
                            unsafe_allow_html=True,
                        )
                if st.session_state.likert_cols:
                    st.markdown("**Ratings**")
                    for lc in st.session_state.likert_cols:
                        val = row.get(lc)
                        if pd.notna(val):
                            st.markdown(f"`{lc[:20]}` → **{val}/5**")
        else:
            st.warning(f"Respondent {selected_id} not found.")


# ═══════════════════════════════════════════════════════════════════
# PAGE: ASK AI
# ═══════════════════════════════════════════════════════════════════
elif page == "Ask AI":
    st.title("Ask AI")
    st.markdown("Ask Claude anything about your survey results.")

    if not st.session_state.pipeline_done:
        st.info("Run the pipeline first.")
        st.stop()

    # Build context summary for the LLM
    k         = st.session_state.best_k
    labels    = st.session_state.cluster_labels
    sent_data = st.session_state.sentiment_data or {}
    action_data = st.session_state.action_data or {}
    theme_data  = st.session_state.theme_data or {}

    cl_desc = "\n".join(
        f"Cluster {i} '{labels.get(str(i),{}).get('label','—')}': "
        f"n={len(st.session_state.labeled_df[st.session_state.labeled_df['cluster']==i])}, "
        f"{(sent_data.get('cluster_summary',{}).get(str(i),{}) or {}).get('positive',0)}% positive. "
        f"Profile: {labels.get(str(i),{}).get('respondent_profile','—')}"
        for i in range(k)
    )
    themes_list = ", ".join(t["name"] for t in (theme_data.get("themes") or []))
    top_actions = "\n".join(
        f"[{ins.get('priority','').upper()}] {ins.get('insight','')}"
        for ins in (action_data.get("insights") or [])[:4]
    )
    urgent_n = sum(1 for r in (sent_data.get("results") or []) if r.get("flag_urgent"))

    context = (
        f"Programme: {st.session_state.program_name or 'unnamed'}\n"
        f"{k} clusters identified.\n\nClusters:\n{cl_desc}\n\n"
        f"Themes: {themes_list}\n\nTop actions:\n{top_actions}\n\n"
        f"Urgent flags: {urgent_n}"
    )
    sys_prompt = (
        "You are an expert in learning & development evaluation analytics. "
        "You have completed a full analysis of a post-program survey.\n"
        f"ANALYSIS CONTEXT:\n{context}\n"
        "Answer questions precisely. Reference specific clusters, sentiment data, and themes. "
        "Keep responses to 2-5 sentences unless the user asks for more detail."
    )

    # Quick questions
    st.markdown("**Quick Questions**")
    quick_qs = [
        "Which cluster needs the most urgent attention?",
        "What are the top 3 improvement priorities?",
        "How does sentiment differ across clusters?",
        "Which themes appear in multiple clusters?",
    ]
    cols_q = st.columns(4)
    for i, q in enumerate(quick_qs):
        if cols_q[i].button(q, key=f"qq_{i}"):
            st.session_state["chat_input_prefill"] = q

    st.markdown("---")

    # Chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! I have full context of your survey analysis. What would you like to explore?"}
        ]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prefill = st.session_state.pop("chat_input_prefill", "")
    user_input = st.chat_input("Ask about your results…", key="chat_main")
    if not user_input and prefill:
        user_input = prefill

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        from backend.nlp.llm_client import call_llm_with_retry
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    reply = call_llm_with_retry(sys_prompt, user_input, max_tokens=600)
                except Exception as e:
                    reply = f"Error: {e}"
            st.write(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
