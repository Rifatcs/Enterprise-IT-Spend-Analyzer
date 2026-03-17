"""
Optimization Agent — Cost Reduction Recommendations & ROI Estimates
Responsibility: Generate specific, prioritized recommendations with financial impact.
"""

import json
import re
import pandas as pd
from src.agents.base_agent import BaseAgent
from src.models.schemas import SpendContext, AgentResponse
from src.analytics.deterministic import format_analytics_for_llm
from config.prompts import OPTIMIZATION_AGENT_SYSTEM_PROMPT


class OptimizationAgent(BaseAgent):

    agent_name = "optimization"
    system_prompt = OPTIMIZATION_AGENT_SYSTEM_PROMPT

    def run(self, context: SpendContext) -> AgentResponse:
        try:
            user_message = self._build_user_message(context)
            messages = self._build_messages(context, user_message)
            response_text = self._call_claude(messages)

            # Parse structured recommendations from the response
            recs_df = self._parse_recommendations(response_text)

            return AgentResponse(
                agent_name=self.agent_name,
                summary=response_text,
                data=recs_df,
                suggested_actions=self._default_suggested_actions(),
                metadata=self._compute_metadata(recs_df),
            )
        except Exception as e:
            return AgentResponse(
                agent_name=self.agent_name,
                summary="",
                error=f"Optimization Agent error: {str(e)}",
            )

    def _build_user_message(self, context: SpendContext) -> str:
        analytics_text = format_analytics_for_llm(context.analytics)
        s = context.analytics.get("summary", {})

        def fmt(v):
            if not v: return "$0"
            if v >= 1_000_000: return f"${v/1_000_000:.2f}M"
            return f"${v/1_000:.0f}K"

        waste = context.analytics.get("waste_estimate_total", 0)
        underutilized = context.analytics.get("underutilized", pd.DataFrame())
        duplicates = context.analytics.get("duplicates", pd.DataFrame())

        msg = f"""Identify the top 10-15 IT cost optimization opportunities for this enterprise.

SPEND CONTEXT:
- Total Annual IT Spend: {fmt(s.get('total_annual_spend', 0))}
- Underutilized Services: {len(underutilized)} services | Estimated Waste: {fmt(waste)}
- Duplicate Tools Detected: {len(duplicates)} instances
- Avg Utilization: {s.get('avg_utilization', 'N/A')}%

FULL ANALYTICS DATA:
{analytics_text}

Provide:
1. A concise executive narrative (300-400 words) covering total savings potential, top 3 priorities, quick wins, and recommended sequencing.
2. A JSON array of recommendations in a ```json code block with this structure:
[
  {{
    "rank": 1,
    "opportunity": "description",
    "category": "Cloud Optimization|SaaS Rationalization|License Management|Vendor Consolidation|Contract Optimization|Infrastructure|Other",
    "affected_vendor": "vendor name",
    "affected_department": "department name",
    "annual_savings_low": 50000,
    "annual_savings_high": 100000,
    "implementation_effort": "Low|Medium|High",
    "risk_level": "Low|Medium|High",
    "time_to_value": "Immediate|Short-term|Long-term",
    "priority": "High|Medium|Low",
    "action_required": "specific next step"
  }}
]

Base all savings estimates on the actual spend data — be precise with dollar amounts.
Focus on highest ROI opportunities first."""

        if context.user_question:
            msg += f"\n\nFocus area from user: {context.user_question}"

        return msg

    def _parse_recommendations(self, response_text: str) -> pd.DataFrame:
        """Extract JSON recommendations from the response text."""
        try:
            # Look for a ```json code block
            match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
            if match:
                json_str = match.group(1)
            else:
                # Try to find a JSON array directly
                match = re.search(r'\[\s*\{[\s\S]*?\}\s*\]', response_text)
                if not match:
                    return pd.DataFrame()
                json_str = match.group(0)

            recs = json.loads(json_str)
            if not isinstance(recs, list) or not recs:
                return pd.DataFrame()

            df = pd.DataFrame(recs)
            # Add savings midpoint for sorting/charting
            if "annual_savings_low" in df.columns and "annual_savings_high" in df.columns:
                df["annual_savings_midpoint"] = (
                    (df["annual_savings_low"] + df["annual_savings_high"]) / 2
                ).round(0)
                df = df.sort_values("annual_savings_midpoint", ascending=False)
            return df.reset_index(drop=True)

        except Exception:
            return pd.DataFrame()

    def _compute_metadata(self, recs_df: pd.DataFrame) -> dict:
        if recs_df.empty:
            return {"total_opportunities": 0}

        low_col = recs_df.get("annual_savings_low", pd.Series([0])) if "annual_savings_low" in recs_df.columns else pd.Series([0])
        high_col = recs_df.get("annual_savings_high", pd.Series([0])) if "annual_savings_high" in recs_df.columns else pd.Series([0])
        total_low = low_col.sum()
        total_high = high_col.sum()
        high_count = len(recs_df[recs_df["priority"] == "High"]) if "priority" in recs_df.columns else 0

        return {
            "total_opportunities": len(recs_df),
            "total_savings_low": total_low,
            "total_savings_high": total_high,
            "high_priority_count": high_count,
        }

    def _default_suggested_actions(self) -> list:
        return [
            "Generate a full executive report with all findings",
            "Deep-dive into cloud optimization opportunities",
            "Analyze SaaS rationalization in more detail",
            "Review upcoming contract renewals for renegotiation",
            "Ask about a specific vendor or department",
        ]
