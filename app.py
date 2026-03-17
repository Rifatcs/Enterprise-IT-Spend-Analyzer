"""
Enterprise IT Spend Analyzer — Chat-First Multi-Agent Interface
The Orchestrator routes every question to the right specialist agent.
"""

import os
import io
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="FinOps AI — IT Spend Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .agent-pill {
        display: inline-block;
        padding: 2px 12px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .pill-intake       { background:#dbeafe; color:#1e40af; }
    .pill-cost_analysis{ background:#dcfce7; color:#166534; }
    .pill-tbm          { background:#fef9c3; color:#854d0e; }
    .pill-optimization { background:#fce7f3; color:#9d174d; }
    .pill-report       { background:#ede9fe; color:#5b21b6; }
    .pill-orchestrator { background:#f1f5f9; color:#475569; }
    div[data-testid="stChatMessage"] { margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ─── Session State ────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "df": None,
        "analytics": {},
        "agent_results": {},
        "messages": [],          # {role, content, agent, df, fig}
        "pending": None,
        "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _fmt(v):
    if not v or pd.isna(v): return "$0"
    if v >= 1_000_000: return f"${v/1_000_000:.2f}M"
    if v >= 1_000: return f"${v/1_000:.0f}K"
    return f"${v:.0f}"


def _load_data(file_or_path):
    from src.analytics.deterministic import validate_and_clean, compute_full_analytics
    if isinstance(file_or_path, str):
        df_raw = pd.read_csv(file_or_path)
    else:
        df_raw = pd.read_csv(file_or_path)
    df, report = validate_and_clean(df_raw)
    analytics = compute_full_analytics(df)
    st.session_state.df = df
    st.session_state.analytics = analytics
    st.session_state.agent_results = {}
    st.session_state.messages = []
    # Auto-greet
    s = analytics.get("summary", {})
    total = s.get("total_annual_spend", 0)
    vendors = s.get("unique_vendors", 0)
    waste = analytics.get("waste_estimate_total", 0)
    renewals = len(analytics.get("renewals", pd.DataFrame()))
    greeting = (
        f"**Data loaded successfully!** I can see **{len(df)} services** across "
        f"**{vendors} vendors** totaling **{_fmt(total)}/year**.\n\n"
        f"Quick findings: ~**{_fmt(waste)}** in estimated waste, "
        f"**{renewals}** contracts renewing soon.\n\n"
        f"Ask me anything — e.g. *\"What are the top cost drivers?\"*, "
        f"*\"Find optimization opportunities\"*, *\"Run a full TBM analysis\"*, "
        f"or *\"Generate an executive report\"*."
    )
    st.session_state.messages.append({
        "role": "assistant", "content": greeting,
        "agent": "orchestrator", "df": None, "fig": None,
    })


def _call_api(question: str):
    """Route question through orchestrator, return agent name + response."""
    from src.orchestrator import Orchestrator
    from src.models.schemas import SpendContext

    # Build conversation history for context (last 6 turns)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-6:]
        if m["role"] in ("user", "assistant") and m.get("content")
    ]

    context = SpendContext(
        df=st.session_state.df,
        analytics=st.session_state.analytics,
        prior_results={k: {"summary": v.summary, "metadata": v.metadata}
                       for k, v in st.session_state.agent_results.items()},
        conversation_history=history,
        user_question=question,
    )

    orchestrator = Orchestrator(api_key=st.session_state.api_key)
    results = orchestrator.route_and_run(context)
    st.session_state.agent_results.update(results)

    responses = []
    for name, resp in results.items():
        if resp.success and resp.summary:
            responses.append((name, resp.summary, resp.data))
        elif not resp.success and resp.error:
            # Surface agent errors so the user sees what went wrong
            responses.append((name, f"⚠️ {resp.error}", None))
    return responses


def _make_chart(agent_name: str, data: pd.DataFrame):
    """Auto-generate a chart from agent result data if possible."""
    if data is None or data.empty:
        return None
    try:
        if agent_name == "optimization" and "annual_savings_midpoint" in data.columns:
            top = data.head(8)
            fig = px.bar(
                top, x="annual_savings_midpoint",
                y="opportunity" if "opportunity" in top.columns else top.columns[0],
                orientation="h",
                title="Top Optimization Savings ($/year)",
                color="priority" if "priority" in top.columns else None,
                color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"},
                text_auto=".2s",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"},
                              showlegend=True, margin=dict(t=40, b=0), height=350)
            return fig
        if "total_spend" in data.columns and "cost_category" in data.columns:
            fig = px.pie(data.head(8), names="cost_category", values="total_spend",
                         title="Spend by Category",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(margin=dict(t=40, b=0), height=300)
            return fig
        if "total_spend" in data.columns and "vendor" in data.columns:
            fig = px.bar(data.head(10), x="total_spend", y="vendor", orientation="h",
                         title="Top Vendors by Spend", text_auto=".2s",
                         color_discrete_sequence=["#3b82f6"])
            fig.update_layout(yaxis={"categoryorder": "total ascending"},
                              margin=dict(t=40, b=0), height=320)
            return fig
    except Exception:
        pass
    return None


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 FinOps AI")
    st.caption("Multi-Agent IT Spend Intelligence")
    st.markdown("---")

    # File upload
    st.markdown("### Load Data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"],
                                 help="Any IT spend CSV — columns are auto-detected")
    if uploaded:
        with st.spinner("Processing data..."):
            try:
                _load_data(uploaded)
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("Load Sample Data", use_container_width=True):
        path = os.path.join(os.path.dirname(__file__), "data", "sample_it_spend.csv")
        with st.spinner("Loading sample data..."):
            _load_data(path)
        st.rerun()

    if st.session_state.df is not None:
        s = st.session_state.analytics.get("summary", {})
        st.success(f"{len(st.session_state.df)} records loaded")
        st.metric("Total Spend", _fmt(s.get("total_annual_spend", 0)))
        st.metric("Est. Waste", _fmt(st.session_state.analytics.get("waste_estimate_total", 0)))

        st.markdown("---")
        st.markdown("### Quick Actions")
        quick = {
            "Cost Analysis": "Analyze the top cost drivers, trends, and anomalies",
            "Find Waste": "Identify underutilized and duplicate services",
            "TBM Mapping": "Map spend to TBM cost pools and value streams",
            "Optimize Spend": "Generate prioritized optimization recommendations with ROI",
            "Executive Report": "Generate a full executive summary report",
            "Full Analysis": "run a comprehensive full analysis of all IT spend data",
        }
        for label, prompt in quick.items():
            if st.button(label, use_container_width=True, key=f"qa_{label}"):
                st.session_state.messages.append({"role": "user", "content": prompt,
                                                   "agent": None, "df": None, "fig": None})
                st.session_state.pending = prompt
                st.rerun()

        st.markdown("---")
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.agent_results = {}
            st.rerun()

        # Export
        if st.session_state.df is not None:
            csv = st.session_state.df.to_csv(index=False).encode()
            st.download_button("Export Data CSV", data=csv,
                               file_name="it_spend_export.csv", mime="text/csv",
                               use_container_width=True)

    st.markdown("---")
    st.caption("Powered by Claude Opus 4.6")


# ─── Main Chat Area ───────────────────────────────────────────────────────────
st.markdown("## FinOps AI — IT Spend Intelligence")

if st.session_state.df is None:
    st.info("Upload your IT spend CSV or click **Load Sample Data** in the sidebar to get started.")
    st.markdown("### What this system can do")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Cost Analysis Agent**\nIdentifies top spend drivers, anomalies, vendor concentration, and utilization issues.")
    with cols[1]:
        st.markdown("**TBM Agent**\nMaps spend to IT cost pools, business capabilities, and value streams using the TBM framework.")
    with cols[2]:
        st.markdown("**Optimization Agent**\nGenerates specific cost reduction recommendations with ROI estimates and implementation guidance.")

else:
    # ── Render all messages ───────────────────────────────────────────────────
    for msg in st.session_state.messages:
        role = msg["role"]
        agent = msg.get("agent", "")
        content = msg.get("content", "")
        df = msg.get("df")
        fig = msg.get("fig")

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant"):
                if agent:
                    pill_class = f"pill-{agent}"
                    label = agent.replace("_", " ").title() + " Agent"
                    st.markdown(
                        f'<span class="agent-pill {pill_class}">{label}</span>',
                        unsafe_allow_html=True,
                    )
                st.markdown(content)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                if df is not None and not df.empty:
                    st.dataframe(df, use_container_width=True, height=300)

    # ── Process pending message ───────────────────────────────────────────────
    if st.session_state.pending:
        prompt = st.session_state.pending
        st.session_state.pending = None

        with st.chat_message("assistant"):
            with st.spinner("Routing to specialist agent..."):
                try:
                    responses = _call_api(prompt)
                    if responses:
                        for agent_name, summary, data in responses:
                            fig = _make_chart(agent_name, data)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": summary,
                                "agent": agent_name,
                                "df": data,
                                "fig": fig,
                            })
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "⚠️ No response returned. Please check your ANTHROPIC_API_KEY in the .env file.",
                            "agent": "orchestrator", "df": None, "fig": None,
                        })
                except Exception as e:
                    err = f"⚠️ Error: {e}"
                    st.session_state.messages.append({
                        "role": "assistant", "content": err,
                        "agent": "orchestrator", "df": None, "fig": None,
                    })
        st.rerun()

    # ── Chat input ────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask about your IT spend — e.g. 'What are the top vendors?' or 'Find optimization opportunities'"):
        st.session_state.messages.append({
            "role": "user", "content": prompt,
            "agent": None, "df": None, "fig": None,
        })
        st.session_state.pending = prompt
        st.rerun()
