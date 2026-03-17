"""
Enterprise IT Spend Analyzer — Streamlit Application
Main UI entry point. Multi-agent AI system for TBM-aligned IT financial analysis.
"""

import os
import io
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise IT Spend Analyzer",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .agent-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-intake      { background:#dbeafe; color:#1e40af; }
    .badge-cost        { background:#dcfce7; color:#166534; }
    .badge-tbm         { background:#fef9c3; color:#854d0e; }
    .badge-optimization{ background:#fce7f3; color:#9d174d; }
    .badge-report      { background:#ede9fe; color:#5b21b6; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; border-radius: 6px 6px 0 0; }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ─────────────────────────────────────────────
def init_session_state():
    defaults = {
        "df": None,
        "analytics": {},
        "quality_report": {},
        "agent_results": {},
        "chat_history": [],
        "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "analysis_run": False,
        "pending_prompt": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ─── Helper Functions (must be defined before sidebar calls them) ─────────────

def _load_and_analyze(uploaded_file, use_sample: bool):
    """Load data from file or sample, run initial analysis."""
    from src.analytics.deterministic import validate_and_clean, compute_full_analytics

    try:
        if use_sample:
            df_raw = _load_sample_data()
        else:
            df_raw = pd.read_csv(uploaded_file)

        df, quality_report = validate_and_clean(df_raw)
        analytics = compute_full_analytics(df)

        st.session_state.df = df
        st.session_state.analytics = analytics
        st.session_state.quality_report = quality_report
        st.session_state.agent_results = {}
        st.session_state.analysis_run = False

        st.rerun()

    except Exception as e:
        st.error(f"Error loading data: {e}")


def _load_sample_data() -> pd.DataFrame:
    """Load the bundled sample enterprise IT spend CSV."""
    sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_it_spend.csv")
    return pd.read_csv(sample_path)


def _call_agent_api(agent_name: str, question: str) -> bool:
    """Execute agent API call, store results. No st.rerun()."""
    from src.orchestrator import Orchestrator
    from src.models.schemas import SpendContext

    question_for_context = (
        "run a comprehensive full analysis of all IT spend data"
        if agent_name == "full" else question
    )
    context = SpendContext(
        df=st.session_state.df,
        analytics=st.session_state.analytics,
        prior_results={k: {"summary": v.summary, "metadata": v.metadata}
                       for k, v in st.session_state.agent_results.items()},
        conversation_history=st.session_state.chat_history,
        user_question=question_for_context,
    )
    try:
        orchestrator = Orchestrator(api_key=st.session_state.api_key)
        results = orchestrator.route_and_run(context)
        st.session_state.agent_results.update(results)
        for name, resp in results.items():
            if resp.success:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": resp.summary,
                    "agent": name,
                })
        return True
    except Exception as e:
        st.error(f"Analysis error: {e}")
        return False


def _run_agent(agent_name: str, question: str):
    """Run an agent from a button click — shows status, then reruns."""
    if not st.session_state.api_key:
        st.error("ANTHROPIC_API_KEY not set. Add it to your .env file.")
        return
    if st.session_state.df is None:
        st.error("Please load data first.")
        return
    with st.status("Running AI analysis — this takes 15–30 seconds...", expanded=True):
        success = _call_agent_api(agent_name, question)
    if success:
        st.rerun()


def _fmt_usd(v: float) -> str:
    if not v or pd.isna(v):
        return "$0"
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"${v/1_000:.0f}K"
    return f"${v:.0f}"


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/180x40/1f2937/ffffff?text=IT+Spend+Analyzer", width=180)
    st.markdown("---")

    st.subheader("Data Upload")

    uploaded_file = st.file_uploader(
        "Upload IT Spend CSV",
        type=["csv"],
        help="Upload your enterprise IT spend data in CSV format.",
    )

    use_sample = st.button("Load Sample Data", use_container_width=True)

    if uploaded_file or use_sample:
        _load_and_analyze(uploaded_file, use_sample)

    if st.session_state.df is not None:
        st.success(f"Data loaded: {len(st.session_state.df)} records")
        st.markdown("---")
        st.subheader("Quick Actions")
        if st.button("Run Full Analysis", use_container_width=True, type="primary"):
            _run_agent("full", "Run a comprehensive full analysis of all IT spend data")
        if st.button("Find Optimizations", use_container_width=True):
            _run_agent("optimization", "Generate all cost optimization recommendations with ROI estimates")
        if st.button("Generate Report", use_container_width=True):
            _run_agent("report", "Generate a complete executive summary report")
        if st.button("TBM Analysis", use_container_width=True):
            _run_agent("tbm", "Perform a TBM framework analysis and business capability mapping")

    st.markdown("---")
    st.caption("Powered by Claude Opus 4.6 | Multi-Agent Architecture")


# ─── Main Content Area ────────────────────────────────────────────────────────
st.markdown('<p class="main-header">Enterprise IT Spend Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">TBM-Aligned Multi-Agent Analysis Platform — Powered by Claude Opus 4.6</p>',
            unsafe_allow_html=True)

if st.session_state.df is None:
    # Landing state
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1** — Upload your IT spend CSV or load the sample dataset from the sidebar.")
    with col2:
        st.info("**Step 2** — Use Quick Actions or the chat interface to analyze spend.")
    with col3:
        st.info("**Step 3** — Export structured insights to CSV for Excel, Power BI, or Tableau.")

    st.markdown("---")
    st.subheader("Expected CSV Schema")
    schema_df = pd.DataFrame([
        {"column": "vendor", "type": "text", "required": "Yes", "example": "Amazon Web Services"},
        {"column": "service_name", "type": "text", "required": "Yes", "example": "EC2 Compute"},
        {"column": "department", "type": "text", "required": "Yes", "example": "Engineering"},
        {"column": "cost_category", "type": "text", "required": "Yes", "example": "Cloud"},
        {"column": "annual_cost", "type": "number", "required": "Yes", "example": "540000"},
        {"column": "business_unit", "type": "text", "required": "No", "example": "Product"},
        {"column": "spend_type", "type": "text", "required": "No", "example": "OPEX"},
        {"column": "monthly_cost", "type": "number", "required": "No", "example": "45000"},
        {"column": "contract_type", "type": "text", "required": "No", "example": "Enterprise"},
        {"column": "contract_end_date", "type": "date", "required": "No", "example": "2024-12-31"},
        {"column": "utilization_pct", "type": "number", "required": "No", "example": "78"},
        {"column": "region", "type": "text", "required": "No", "example": "US-East"},
        {"column": "headcount_supported", "type": "number", "required": "No", "example": "250"},
        {"column": "notes", "type": "text", "required": "No", "example": "Primary compute cluster"},
    ])
    st.dataframe(schema_df, use_container_width=True, hide_index=True)

else:
    # ─── KPI Metrics Row ──────────────────────────────────────────────────────
    s = st.session_state.analytics.get("summary", {})
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Annual Spend", _fmt_usd(s.get("total_annual_spend", 0)))
    with col2:
        st.metric("Unique Vendors", s.get("unique_vendors", 0))
    with col3:
        st.metric("Avg Utilization", f"{s.get('avg_utilization', 0):.1f}%")
    with col4:
        waste = st.session_state.analytics.get("waste_estimate_total", 0)
        st.metric("Est. Waste", _fmt_usd(waste), delta=f"-{_fmt_usd(waste)} potential", delta_color="inverse")
    with col5:
        renewals = st.session_state.analytics.get("renewals", pd.DataFrame())
        st.metric("Upcoming Renewals", len(renewals))

    st.markdown("---")

    # ─── Main Tabs ────────────────────────────────────────────────────────────
    tab_overview, tab_cost, tab_tbm, tab_optimize, tab_report, tab_chat = st.tabs([
        "Overview", "Cost Analysis", "TBM Insights", "Optimizations", "Executive Report", "AI Chat"
    ])

    # ── TAB 1: Overview ───────────────────────────────────────────────────────
    with tab_overview:
        st.subheader("Data Overview")
        col_l, col_r = st.columns(2)

        with col_l:
            by_cat = st.session_state.analytics.get("by_category", pd.DataFrame())
            if not by_cat.empty:
                fig = px.pie(
                    by_cat,
                    names="cost_category",
                    values="total_spend",
                    title="IT Spend by Category",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                fig.update_layout(showlegend=False, margin=dict(t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)

        with col_r:
            by_dept = st.session_state.analytics.get("by_department", pd.DataFrame())
            if not by_dept.empty:
                fig = px.bar(
                    by_dept.head(10),
                    x="total_spend",
                    y="department",
                    orientation="h",
                    title="Top 10 Departments by Spend",
                    color="total_spend",
                    color_continuous_scale="Blues",
                    text_auto=".2s",
                )
                fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"},
                                  margin=dict(t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Raw Data Preview")
        df_display = st.session_state.df.copy()
        if "annual_cost" in df_display.columns:
            df_display["annual_cost"] = df_display["annual_cost"].apply(
                lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
            )
        st.dataframe(df_display, use_container_width=True, height=400)

    # ── TAB 2: Cost Analysis ──────────────────────────────────────────────────
    with tab_cost:
        if "cost_analysis" in st.session_state.agent_results:
            result = st.session_state.agent_results["cost_analysis"]
            st.markdown(result.summary)
            if result.data is not None and not result.data.empty:
                st.dataframe(result.data, use_container_width=True)
        else:
            col_l, col_r = st.columns(2)

            with col_l:
                top_v = st.session_state.analytics.get("top_vendors", pd.DataFrame())
                if not top_v.empty:
                    fig = px.bar(
                        top_v.head(10),
                        x="total_spend",
                        y="vendor",
                        orientation="h",
                        title="Top 10 Vendors by Spend",
                        color="pct_of_total",
                        color_continuous_scale="Reds",
                        text_auto=".2s",
                    )
                    fig.update_layout(yaxis={"categoryorder": "total ascending"},
                                      coloraxis_showscale=False, margin=dict(t=40, b=0))
                    st.plotly_chart(fig, use_container_width=True)

            with col_r:
                underutil = st.session_state.analytics.get("underutilized", pd.DataFrame())
                if not underutil.empty and "utilization_pct" in underutil.columns:
                    fig = px.scatter(
                        underutil,
                        x="utilization_pct",
                        y="annual_cost",
                        color="cost_category",
                        size="annual_cost",
                        hover_data=["vendor", "service_name"] if "service_name" in underutil.columns else ["vendor"],
                        title="Underutilized Services (<50% utilization)",
                        labels={"utilization_pct": "Utilization %", "annual_cost": "Annual Cost ($)"},
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.subheader("Underutilized Services")
            underutil = st.session_state.analytics.get("underutilized", pd.DataFrame())
            if not underutil.empty:
                st.dataframe(underutil, use_container_width=True)
            else:
                st.success("No significantly underutilized services detected.")

            st.subheader("Contract Renewals (Next 180 Days)")
            renewals = st.session_state.analytics.get("renewals", pd.DataFrame())
            if not renewals.empty:
                st.dataframe(renewals, use_container_width=True)
            else:
                st.success("No contracts renewing in the next 180 days.")

            if st.button("Run AI Cost Analysis", type="primary"):
                _run_agent("cost_analysis", "Perform a comprehensive cost analysis")

    # ── TAB 3: TBM Insights ───────────────────────────────────────────────────
    with tab_tbm:
        col_l, col_r = st.columns(2)

        with col_l:
            tbm_pools = st.session_state.analytics.get("tbm_pools", pd.DataFrame())
            if not tbm_pools.empty:
                fig = px.bar(
                    tbm_pools,
                    x="tbm_cost_pool",
                    y="total_spend",
                    color="tbm_cost_pool",
                    title="Spend by TBM Cost Pool",
                    text_auto=".2s",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                fig.update_layout(showlegend=False, margin=dict(t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)

        with col_r:
            vs = st.session_state.analytics.get("value_streams", pd.DataFrame())
            if not vs.empty:
                fig = px.pie(
                    vs,
                    names="value_stream",
                    values="total_spend",
                    title="Spend by Value Stream",
                    color_discrete_sequence=px.colors.qualitative.Pastel2,
                )
                fig.update_layout(margin=dict(t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)

        if "tbm" in st.session_state.agent_results:
            result = st.session_state.agent_results["tbm"]
            st.subheader("TBM Analysis")
            st.markdown(result.summary)
            if result.data is not None and not result.data.empty:
                st.dataframe(result.data, use_container_width=True)
        else:
            if st.button("Run TBM Analysis", type="primary"):
                _run_agent("tbm", "Perform a comprehensive TBM framework analysis")

    # ── TAB 4: Optimizations ──────────────────────────────────────────────────
    with tab_optimize:
        if "optimization" in st.session_state.agent_results:
            result = st.session_state.agent_results["optimization"]
            st.markdown(result.summary)

            if result.data is not None and not result.data.empty:
                st.subheader("Optimization Opportunities")
                recs = result.data.copy()

                # Color-code priority
                def highlight_priority(val):
                    colors = {"High": "background-color: #fef2f2",
                              "Medium": "background-color: #fefce8",
                              "Low": "background-color: #f0fdf4"}
                    return colors.get(val, "")

                if "priority" in recs.columns:
                    st.dataframe(
                        recs.style.applymap(highlight_priority, subset=["priority"]),
                        use_container_width=True,
                    )
                else:
                    st.dataframe(recs, use_container_width=True)

                # Savings waterfall
                if "annual_savings_midpoint" in recs.columns and "opportunity" in recs.columns:
                    fig = px.bar(
                        recs.head(10),
                        x="opportunity",
                        y="annual_savings_midpoint",
                        color="category" if "category" in recs.columns else "priority",
                        title="Top 10 Optimization Opportunities by Savings Potential",
                        labels={"annual_savings_midpoint": "Annual Savings ($)", "opportunity": ""},
                        text_auto=".2s",
                    )
                    fig.update_xaxes(tickangle=45)
                    fig.update_layout(margin=dict(t=40, b=120))
                    st.plotly_chart(fig, use_container_width=True)

                # Export
                csv = recs.to_csv(index=False).encode()
                st.download_button(
                    "Download Recommendations CSV",
                    data=csv,
                    file_name="it_optimization_recommendations.csv",
                    mime="text/csv",
                )
        else:
            st.info("Click the button below to generate AI-powered optimization recommendations with ROI estimates.")
            if st.button("Generate Optimization Recommendations", type="primary", use_container_width=True):
                _run_agent("optimization", "Generate all cost optimization recommendations with ROI estimates")

            st.markdown("---")
            st.subheader("Identified Waste & Inefficiencies (Pre-Analysis)")
            col_l, col_r = st.columns(2)
            with col_l:
                underutil = st.session_state.analytics.get("underutilized", pd.DataFrame())
                if not underutil.empty:
                    waste_total = underutil.get("waste_estimate", pd.Series([0])).sum()
                    st.metric("Estimated Annual Waste", _fmt_usd(waste_total))
                    st.dataframe(underutil.head(10), use_container_width=True)
            with col_r:
                dups = st.session_state.analytics.get("duplicates", pd.DataFrame())
                if not dups.empty:
                    st.metric("Duplicate Tool Instances", len(dups))
                    st.dataframe(dups, use_container_width=True)

    # ── TAB 5: Executive Report ───────────────────────────────────────────────
    with tab_report:
        if "report" in st.session_state.agent_results:
            result = st.session_state.agent_results["report"]
            st.markdown(result.summary)

            if result.data is not None and not result.data.empty:
                st.subheader("Full Data Export (Enriched with TBM Dimensions)")
                st.dataframe(result.data.head(20), use_container_width=True)

                col_l, col_r = st.columns(2)
                with col_l:
                    csv = result.data.to_csv(index=False).encode()
                    st.download_button(
                        "Download Full Analysis CSV",
                        data=csv,
                        file_name="it_spend_analysis_export.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with col_r:
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                        result.data.to_excel(writer, sheet_name="IT Spend Analysis", index=False)
                        if "optimization" in st.session_state.agent_results:
                            opt_data = st.session_state.agent_results["optimization"].data
                            if opt_data is not None and not opt_data.empty:
                                opt_data.to_excel(writer, sheet_name="Optimizations", index=False)
                    st.download_button(
                        "Download Excel Workbook",
                        data=excel_buffer.getvalue(),
                        file_name="it_spend_analysis.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
        else:
            st.info("Click the button below to generate a C-suite ready executive report with export options.")
            if st.button("Generate Executive Report", type="primary", use_container_width=True):
                _run_agent("report", "Generate a comprehensive executive IT spend analysis report")

    # ── TAB 6: AI Chat ────────────────────────────────────────────────────────
    with tab_chat:
        st.subheader("Ask the AI Analyst")
        st.caption("Ask questions about your IT spend data. The system routes your question to the most relevant specialist agent.")

        # Render full chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                agent = msg.get("agent", "assistant")
                badge_class = {
                    "intake": "badge-intake",
                    "cost_analysis": "badge-cost",
                    "tbm": "badge-tbm",
                    "optimization": "badge-optimization",
                    "report": "badge-report",
                }.get(agent, "badge-cost")
                with st.chat_message("assistant"):
                    st.markdown(
                        f'<span class="agent-badge {badge_class}">{agent.replace("_", " ").title()} Agent</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(msg["content"])

        # Process pending prompt (stored from previous rerun)
        if st.session_state.get("pending_prompt"):
            prompt = st.session_state.pending_prompt
            st.session_state.pending_prompt = None
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your IT spend data..."):
                    _call_agent_api("auto", prompt)
            st.rerun()

        # Chat input — must be the very last element
        if prompt := st.chat_input("Ask a question about your IT spend..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.pending_prompt = prompt
            st.rerun()
