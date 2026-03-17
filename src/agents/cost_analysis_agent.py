"""
Cost Analysis Agent — Financial Insights & Trend Analysis
Responsibility: Identify cost drivers, trends, anomalies, vendor concentration.
"""

import pandas as pd
from src.agents.base_agent import BaseAgent
from src.models.schemas import SpendContext, AgentResponse
from src.analytics.deterministic import format_analytics_for_llm
from config.prompts import COST_ANALYSIS_AGENT_SYSTEM_PROMPT


class CostAnalysisAgent(BaseAgent):

    agent_name = "cost_analysis"
    system_prompt = COST_ANALYSIS_AGENT_SYSTEM_PROMPT

    def run(self, context: SpendContext) -> AgentResponse:
        try:
            user_message = self._build_user_message(context)
            messages = self._build_messages(context, user_message)

            response_text = self._call_claude(messages)
            insights_df = self._build_insights_dataframe(context.analytics)

            return AgentResponse(
                agent_name=self.agent_name,
                summary=response_text,
                data=insights_df,
                suggested_actions=self._default_suggested_actions(),
                metadata=self._compute_metadata(context.analytics),
            )
        except Exception as e:
            return AgentResponse(
                agent_name=self.agent_name,
                summary="",
                error=f"Cost Analysis Agent error: {str(e)}",
            )

    def _build_user_message(self, context: SpendContext) -> str:
        analytics_text = format_analytics_for_llm(context.analytics)
        s = context.analytics.get("summary", {})

        def fmt(v):
            if not v:
                return "$0"
            if v >= 1_000_000:
                return f"${v/1_000_000:.2f}M"
            return f"${v/1_000:.0f}K"

        # Highlight key risk signals for the LLM
        risk_signals = []
        underutilized = context.analytics.get("underutilized", pd.DataFrame())
        if not underutilized.empty:
            waste = context.analytics.get("waste_estimate_total", 0)
            risk_signals.append(f"- {len(underutilized)} underutilized services representing ~{fmt(waste)} in potential waste")

        anomalies = context.analytics.get("anomalies", pd.DataFrame())
        if not anomalies.empty:
            risk_signals.append(f"- {len(anomalies)} statistical spend anomalies detected")

        renewals = context.analytics.get("renewals", pd.DataFrame())
        if not renewals.empty:
            urgent = renewals[renewals.get("urgency", pd.Series()).astype(str).str.contains("URGENT", na=False)]
            risk_signals.append(f"- {len(renewals)} contracts renewing in next 180 days ({len(urgent)} urgent)")

        duplicates = context.analytics.get("duplicates", pd.DataFrame())
        if not duplicates.empty:
            risk_signals.append(f"- {len(duplicates)} potential duplicate tool instances across departments")

        risk_section = "\n".join(risk_signals) if risk_signals else "- No critical risk signals detected"

        msg = f"""Perform a comprehensive IT cost analysis for this enterprise.

Total Annual IT Spend: {fmt(s.get('total_annual_spend', 0))}
Average Utilization: {s.get('avg_utilization', 'N/A')}%
Top-3 Vendor Concentration: {s.get('top_vendor_concentration', 'N/A')}%

KEY RISK SIGNALS:
{risk_section}

FULL ANALYTICS DATA:
{analytics_text}

Please provide:
1. Executive-level summary of the spend profile (2-3 sentences)
2. Top 3 cost drivers and their financial significance
3. Vendor concentration risk assessment
4. Utilization efficiency analysis
5. Anomalies and spend irregularities
6. Department spend analysis — who are the biggest consumers?
7. CAPEX vs OPEX health check (if data available)
8. Top 3 immediate financial concerns for IT leadership

Be specific with dollar amounts and percentages throughout."""

        if context.user_question:
            msg += f"\n\nSpecific question from user: {context.user_question}"

        return msg

    def _build_insights_dataframe(self, analytics: dict) -> pd.DataFrame:
        """Build a key insights summary table."""
        rows = []
        s = analytics.get("summary", {})

        def fmt(v):
            if not v:
                return "$0"
            if v >= 1_000_000:
                return f"${v/1_000_000:.2f}M"
            return f"${v/1_000:.0f}K"

        # Spend by category
        by_cat = analytics.get("by_category", pd.DataFrame())
        if not by_cat.empty:
            for _, row in by_cat.iterrows():
                rows.append({
                    "metric": "Spend by Category",
                    "dimension": row.get("cost_category", ""),
                    "value": fmt(row.get("total_spend", 0)),
                    "pct_of_total": f"{row.get('pct_of_total', 0):.1f}%",
                })

        # Top vendors
        top_v = analytics.get("top_vendors", pd.DataFrame())
        if not top_v.empty:
            for _, row in top_v.head(5).iterrows():
                rows.append({
                    "metric": "Top Vendor",
                    "dimension": row.get("vendor", ""),
                    "value": fmt(row.get("total_spend", 0)),
                    "pct_of_total": f"{row.get('pct_of_total', 0):.1f}%",
                })

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _compute_metadata(self, analytics: dict) -> dict:
        s = analytics.get("summary", {})
        return {
            "total_spend": s.get("total_annual_spend", 0),
            "underutilized_count": len(analytics.get("underutilized", [])),
            "anomaly_count": len(analytics.get("anomalies", [])),
            "renewal_count": len(analytics.get("renewals", [])),
        }

    def _default_suggested_actions(self) -> list:
        return [
            "Map spend to TBM cost pools and business capabilities",
            "Generate optimization recommendations with ROI estimates",
            "Review underutilized services for immediate cost reduction",
            "Analyze upcoming contract renewals for renegotiation leverage",
            "Produce an executive cost report",
        ]
