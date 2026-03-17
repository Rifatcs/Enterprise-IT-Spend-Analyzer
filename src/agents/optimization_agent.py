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


# Tool definition for structured recommendations
RECOMMENDATION_TOOL = {
    "name": "record_recommendations",
    "description": "Record structured optimization recommendations in a machine-readable format",
    "input_schema": {
        "type": "object",
        "properties": {
            "recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "rank": {"type": "integer", "description": "Priority rank 1-15"},
                        "opportunity": {"type": "string", "description": "Clear description of the opportunity"},
                        "category": {"type": "string", "enum": ["Cloud Optimization", "SaaS Rationalization",
                                                                   "License Management", "Vendor Consolidation",
                                                                   "Contract Optimization", "Infrastructure", "Other"]},
                        "affected_vendor": {"type": "string"},
                        "affected_department": {"type": "string"},
                        "annual_savings_low": {"type": "number", "description": "Conservative savings estimate in USD"},
                        "annual_savings_high": {"type": "number", "description": "Aggressive savings estimate in USD"},
                        "implementation_effort": {"type": "string", "enum": ["Low", "Medium", "High"]},
                        "risk_level": {"type": "string", "enum": ["Low", "Medium", "High"]},
                        "time_to_value": {"type": "string", "enum": ["Immediate", "Short-term", "Long-term"]},
                        "priority": {"type": "string", "enum": ["High", "Medium", "Low"]},
                        "action_required": {"type": "string", "description": "Specific next step"},
                    },
                    "required": ["rank", "opportunity", "category", "annual_savings_low",
                                 "annual_savings_high", "implementation_effort", "priority", "action_required"],
                },
            }
        },
        "required": ["recommendations"],
    },
}


class OptimizationAgent(BaseAgent):

    agent_name = "optimization"
    system_prompt = OPTIMIZATION_AGENT_SYSTEM_PROMPT

    def run(self, context: SpendContext) -> AgentResponse:
        try:
            user_message = self._build_user_message(context)
            messages = self._build_messages(context, user_message)

            # First pass: get structured recommendations via tool use
            recs_df = self._get_structured_recommendations(messages)

            # Second pass: get narrative analysis
            narrative = self._get_narrative_analysis(context, recs_df)

            return AgentResponse(
                agent_name=self.agent_name,
                summary=narrative,
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

Generate specific, quantified recommendations using the record_recommendations tool.
Base savings estimates on actual spend data — be precise.
Focus on highest ROI opportunities first.
Every recommendation must have a specific dollar savings range and a concrete action step."""

        if context.user_question:
            msg += f"\n\nFocus area from user: {context.user_question}"

        return msg

    def _get_structured_recommendations(self, messages: list) -> pd.DataFrame:
        """Use tool use to get structured, machine-readable recommendations."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                thinking={"type": "adaptive"},
                system=self.system_prompt,
                tools=[RECOMMENDATION_TOOL],
                tool_choice={"type": "auto"},
                messages=messages,
            )

            # Extract tool use block
            for block in response.content:
                if hasattr(block, "type") and block.type == "tool_use":
                    recs = block.input.get("recommendations", [])
                    if recs:
                        df = pd.DataFrame(recs)
                        # Add savings midpoint for sorting
                        if "annual_savings_low" in df.columns and "annual_savings_high" in df.columns:
                            df["annual_savings_midpoint"] = ((df["annual_savings_low"] + df["annual_savings_high"]) / 2).round(0)
                        return df.sort_values("annual_savings_midpoint", ascending=False).reset_index(drop=True)

        except Exception:
            pass

        # Fallback: return empty DataFrame
        return pd.DataFrame()

    def _get_narrative_analysis(self, context: SpendContext, recs_df: pd.DataFrame) -> str:
        """Generate a narrative explanation of the recommendations."""
        analytics_text = format_analytics_for_llm(context.analytics)
        s = context.analytics.get("summary", {})

        def fmt(v):
            if not v: return "$0"
            if v >= 1_000_000: return f"${v/1_000_000:.2f}M"
            return f"${v/1_000:.0f}K"

        # Build recommendations summary for narrative
        recs_summary = ""
        if not recs_df.empty:
            total_low = recs_df.get("annual_savings_low", pd.Series([0])).sum()
            total_high = recs_df.get("annual_savings_high", pd.Series([0])).sum()
            recs_summary = f"\n\nSUMMARY OF IDENTIFIED OPPORTUNITIES:\n"
            recs_summary += f"Total Savings Potential: {fmt(total_low)} to {fmt(total_high)} annually\n"
            recs_summary += f"Number of Opportunities: {len(recs_df)}\n\n"

            high_priority = recs_df[recs_df.get("priority", pd.Series()) == "High"] if "priority" in recs_df.columns else pd.DataFrame()
            if not high_priority.empty:
                recs_summary += "HIGH PRIORITY OPPORTUNITIES:\n"
                for _, row in high_priority.head(5).iterrows():
                    lo = row.get("annual_savings_low", 0)
                    hi = row.get("annual_savings_high", 0)
                    recs_summary += f"  - {row.get('opportunity', '')}: {fmt(lo)}-{fmt(hi)} annually\n"

        narrative_msg = f"""Based on the optimization analysis, provide a concise executive narrative covering:

1. Opening: Total savings potential summary (1-2 sentences)
2. Top 3 priority recommendations with specific financial impact
3. Quick wins that can be achieved in 30-90 days
4. Strategic consolidation opportunities (longer-term)
5. Recommended execution sequencing — what to do first and why

ANALYTICS:
{analytics_text}
{recs_summary}

Keep the narrative concise (300-400 words), financially precise, and action-oriented.
Write for a CFO/CTO who will make the final decision."""

        messages = [{"role": "user", "content": narrative_msg}]
        return self._call_claude(messages)

    def _compute_metadata(self, recs_df: pd.DataFrame) -> dict:
        if recs_df.empty:
            return {"total_opportunities": 0}

        total_low = recs_df.get("annual_savings_low", pd.Series([0])).sum()
        total_high = recs_df.get("annual_savings_high", pd.Series([0])).sum()
        high_count = len(recs_df[recs_df.get("priority", pd.Series()) == "High"]) if "priority" in recs_df.columns else 0

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
