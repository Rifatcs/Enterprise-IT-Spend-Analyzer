"""
TBM Agent — Technology Business Management Framework Analysis
Responsibility: Map spend to TBM towers, business capabilities, value streams, governance.
"""

import pandas as pd
from src.agents.base_agent import BaseAgent
from src.models.schemas import SpendContext, AgentResponse
from src.analytics.deterministic import format_analytics_for_llm
from config.prompts import TBM_AGENT_SYSTEM_PROMPT


class TBMAgent(BaseAgent):

    agent_name = "tbm"
    system_prompt = TBM_AGENT_SYSTEM_PROMPT

    def run(self, context: SpendContext) -> AgentResponse:
        try:
            user_message = self._build_user_message(context)
            messages = self._build_messages(context, user_message)

            response_text = self._call_claude(messages)
            tbm_df = self._build_tbm_dataframe(context)

            return AgentResponse(
                agent_name=self.agent_name,
                summary=response_text,
                data=tbm_df,
                suggested_actions=self._default_suggested_actions(),
                metadata=self._compute_metadata(context.analytics),
            )
        except Exception as e:
            return AgentResponse(
                agent_name=self.agent_name,
                summary="",
                error=f"TBM Agent error: {str(e)}",
            )

    def _build_user_message(self, context: SpendContext) -> str:
        analytics = context.analytics
        analytics_text = format_analytics_for_llm(analytics)

        tbm_pools = analytics.get("tbm_pools", pd.DataFrame())
        vs = analytics.get("value_streams", pd.DataFrame())
        s = analytics.get("summary", {})
        total = s.get("total_annual_spend", 0)

        def fmt(v):
            if not v: return "$0"
            if v >= 1_000_000: return f"${v/1_000_000:.2f}M"
            return f"${v/1_000:.0f}K"

        pool_summary = tbm_pools.to_string(index=False) if not tbm_pools.empty else "Not computed"
        vs_summary = vs.to_string(index=False) if not vs.empty else "Not computed"

        msg = f"""Apply the Technology Business Management (TBM) framework to analyze this enterprise IT portfolio.

TOTAL ANNUAL IT SPEND: {fmt(total)}

TBM COST POOL DISTRIBUTION (pre-computed):
{pool_summary}

VALUE STREAM SPEND DISTRIBUTION (pre-computed):
{vs_summary}

FULL ANALYTICS DATA:
{analytics_text}

Please provide a comprehensive TBM analysis covering:

1. **TBM Cost Pool Assessment**
   - How is spend distributed across IT Infrastructure, IT Applications, IT Management?
   - Is the distribution healthy compared to TBM benchmarks? (Industry: ~60% Infrastructure, ~30% Applications, ~10% Management)
   - Which cost pools are over/under-invested?

2. **Business Capability Mapping**
   - Map the top 10 spend items to specific business capabilities they enable
   - Identify capabilities that are over-funded vs. under-funded
   - Highlight any gaps in capability coverage

3. **Value Stream Alignment**
   - How is spend distributed across Revenue Generation, Operational Efficiency, Risk Mitigation, Innovation?
   - Is the investment mix aligned with business strategy?
   - Recommend rebalancing if needed

4. **IT Governance Assessment**
   - Business unit accountability gaps
   - Shared services vs. dedicated service allocation
   - Shadow IT risk indicators
   - Contract governance maturity

5. **TBM Maturity Recommendations**
   - What steps should this organization take to improve TBM maturity?
   - Prioritize 3 governance improvements with business justification

Be specific, use TBM terminology, and frame all findings in business value terms."""

        if context.user_question:
            msg += f"\n\nSpecific question: {context.user_question}"

        return msg

    def _build_tbm_dataframe(self, context: SpendContext) -> pd.DataFrame:
        """Build a TBM-mapped view of the spend data."""
        from src.models.schemas import TBM_COST_POOL_MAP, TBM_VALUE_STREAM_MAP

        df = context.df.copy()
        if "cost_category" not in df.columns or "annual_cost" not in df.columns:
            return pd.DataFrame()

        df["tbm_cost_pool"] = df["cost_category"].map(TBM_COST_POOL_MAP).fillna("IT Applications")
        if "department" in df.columns:
            df["tbm_value_stream"] = df["department"].map(TBM_VALUE_STREAM_MAP).fillna("Operational Efficiency")
        else:
            df["tbm_value_stream"] = "Operational Efficiency"

        cols = [c for c in ["vendor", "service_name", "department", "cost_category",
                              "tbm_cost_pool", "tbm_value_stream", "annual_cost",
                              "utilization_pct"] if c in df.columns]

        result = df[cols].sort_values("annual_cost", ascending=False)
        return result.reset_index(drop=True)

    def _compute_metadata(self, analytics: dict) -> dict:
        tbm = analytics.get("tbm_pools", pd.DataFrame())
        vs = analytics.get("value_streams", pd.DataFrame())
        return {
            "cost_pool_count": len(tbm),
            "value_stream_count": len(vs),
        }

    def _default_suggested_actions(self) -> list:
        return [
            "Generate optimization recommendations based on TBM findings",
            "Ask about a specific business capability or value stream",
            "Identify governance improvement priorities",
            "Produce a TBM-aligned executive report",
            "Analyze cost pool rebalancing opportunities",
        ]
