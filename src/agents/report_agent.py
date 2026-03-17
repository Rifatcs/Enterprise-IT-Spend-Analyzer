"""
Report Agent — Executive Summary & Structured Output Generation
Responsibility: Produce C-suite ready reports combining all agent insights.
"""

import pandas as pd
from src.agents.base_agent import BaseAgent
from src.models.schemas import SpendContext, AgentResponse
from src.analytics.deterministic import format_analytics_for_llm
from config.prompts import REPORT_AGENT_SYSTEM_PROMPT


class ReportAgent(BaseAgent):

    agent_name = "report"
    system_prompt = REPORT_AGENT_SYSTEM_PROMPT

    def run(self, context: SpendContext) -> AgentResponse:
        try:
            user_message = self._build_user_message(context)
            messages = self._build_messages(context, user_message)

            # Use streaming for long-form report generation
            response_text = self._call_claude(messages, stream=True)

            export_df = self._build_export_dataframe(context)

            return AgentResponse(
                agent_name=self.agent_name,
                summary=response_text,
                data=export_df,
                suggested_actions=self._default_suggested_actions(),
                metadata={"report_type": "executive_summary"},
            )
        except Exception as e:
            return AgentResponse(
                agent_name=self.agent_name,
                summary="",
                error=f"Report Agent error: {str(e)}",
            )

    def _build_user_message(self, context: SpendContext) -> str:
        analytics_text = format_analytics_for_llm(context.analytics)
        s = context.analytics.get("summary", {})

        def fmt(v):
            if not v: return "$0"
            if v >= 1_000_000: return f"${v/1_000_000:.2f}M"
            return f"${v/1_000:.0f}K"

        # Gather prior agent results for synthesis
        prior_results_text = ""
        for agent_name, result in context.prior_results.items():
            if result.get("summary"):
                label = agent_name.replace("_", " ").title()
                prior_results_text += f"\n--- {label} Agent Findings ---\n{result['summary'][:1500]}\n"

        # Key risk signals
        underutilized = context.analytics.get("underutilized", pd.DataFrame())
        anomalies = context.analytics.get("anomalies", pd.DataFrame())
        renewals = context.analytics.get("renewals", pd.DataFrame())
        waste = context.analytics.get("waste_estimate_total", 0)

        msg = f"""Generate a comprehensive executive IT Spend Analysis Report for this enterprise.

FINANCIAL SNAPSHOT:
- Total Annual IT Spend: {fmt(s.get('total_annual_spend', 0))}
- Active Services/Vendors: {s.get('record_count', 0)} services across {s.get('unique_vendors', 0)} vendors
- Average Utilization: {s.get('avg_utilization', 'N/A')}%
- Estimated Waste (underutilized assets): {fmt(waste)}
- Services with <50% utilization: {len(underutilized)}
- Statistical Anomalies: {len(anomalies)}
- Contracts Renewing (180 days): {len(renewals)}

FULL ANALYTICS:
{analytics_text}

PRIOR ANALYSIS FINDINGS:
{prior_results_text if prior_results_text else "No prior agent results available — generate comprehensive analysis from raw data."}

Generate a full executive report following this structure:
1. EXECUTIVE SUMMARY (5 key bullets)
2. IT SPEND OVERVIEW (totals, distribution, key metrics)
3. COST ANALYSIS FINDINGS (major drivers, anomalies, trends)
4. TBM ALIGNMENT ASSESSMENT (cost pool health, value stream alignment)
5. TOP 5 OPTIMIZATION OPPORTUNITIES (with specific savings estimates)
6. QUICK WINS (90-day action plan with owners suggested)
7. RISK AREAS (financial, operational, contractual)
8. RECOMMENDED NEXT STEPS (3-5 specific actions)

Use professional C-suite language. Include specific dollar amounts throughout.
This report will be presented to the CIO and CFO."""

        if context.user_question:
            msg += f"\n\nSpecific focus requested: {context.user_question}"

        return msg

    def _build_export_dataframe(self, context: SpendContext) -> pd.DataFrame:
        """
        Build the master export DataFrame combining all analytics dimensions.
        This is what gets downloaded as CSV for Excel/Power BI/Tableau.
        """
        from src.models.schemas import TBM_COST_POOL_MAP, TBM_VALUE_STREAM_MAP

        df = context.df.copy()

        # Add TBM dimensions
        if "cost_category" in df.columns:
            df["tbm_cost_pool"] = df["cost_category"].map(TBM_COST_POOL_MAP).fillna("IT Applications")
        if "department" in df.columns:
            df["tbm_value_stream"] = df["department"].map(TBM_VALUE_STREAM_MAP).fillna("Operational Efficiency")

        # Add utilization classification
        if "utilization_pct" in df.columns:
            df["utilization_tier"] = pd.cut(
                df["utilization_pct"].fillna(50),
                bins=[0, 25, 50, 75, 100],
                labels=["Critical (<25%)", "Low (25-50%)", "Adequate (50-75%)", "High (75-100%)"],
                include_lowest=True,
            )

        # Add waste estimate
        if "utilization_pct" in df.columns and "annual_cost" in df.columns:
            df["waste_estimate"] = (
                df["annual_cost"] * (1 - df["utilization_pct"].fillna(50) / 100)
            ).round(0)

        # Clean up for export
        df.sort_values("annual_cost", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    def _default_suggested_actions(self) -> list:
        return [
            "Download the full analysis as CSV",
            "Ask for deep-dive on a specific finding",
            "Request optimization implementation roadmap",
            "Analyze a specific department or vendor",
            "Run a fresh analysis with updated data",
        ]
