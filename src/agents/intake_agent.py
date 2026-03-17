"""
Intake Agent — Data Validation & Quality Assessment
Responsibility: Validate uploaded data, assess quality, confirm TBM readiness.
"""

import pandas as pd
from src.agents.base_agent import BaseAgent
from src.models.schemas import SpendContext, AgentResponse
from src.analytics.deterministic import compute_data_quality_score
from config.prompts import INTAKE_AGENT_SYSTEM_PROMPT


class IntakeAgent(BaseAgent):

    agent_name = "intake"
    system_prompt = INTAKE_AGENT_SYSTEM_PROMPT

    def run(self, context: SpendContext) -> AgentResponse:
        """
        Validate data quality and generate a structured quality assessment.
        """
        try:
            df = context.df
            analytics = context.analytics

            # Build a focused quality assessment message
            quality_data = self._compute_quality_details(df, analytics)
            user_message = self._build_user_message(context, quality_data)
            messages = [{"role": "user", "content": user_message}]

            response_text = self._call_claude(messages)
            quality_df = self._build_quality_dataframe(df)

            return AgentResponse(
                agent_name=self.agent_name,
                summary=response_text,
                data=quality_df,
                suggested_actions=self._default_suggested_actions(),
                metadata={
                    "quality_score": quality_data.get("score", 0),
                    "record_count": len(df),
                    "column_count": len(df.columns),
                },
            )
        except Exception as e:
            return AgentResponse(
                agent_name=self.agent_name,
                summary="",
                error=f"Intake Agent error: {str(e)}",
            )

    def _compute_quality_details(self, df: pd.DataFrame, analytics: dict) -> dict:
        """Generate a detailed quality report from the DataFrame."""
        from src.analytics.deterministic import compute_data_quality_score

        quality_report = analytics.get("quality_report", {
            "issues": [], "warnings": [], "original_rows": len(df)
        })
        return compute_data_quality_score(df, quality_report)

    def _build_user_message(self, context: SpendContext, quality_data: dict = None) -> str:
        df = context.df
        analytics = context.analytics

        # Column presence summary
        col_list = "\n".join([f"  - {col}: {df[col].notna().sum()}/{len(df)} non-null ({df[col].dtype})"
                               for col in df.columns])

        # Sample data
        sample = df.head(3).to_string(index=False)

        # Spend summary
        summary = analytics.get("summary", {})
        total = summary.get("total_annual_spend", 0)

        msg = f"""Please perform a comprehensive data quality assessment on the following enterprise IT spend dataset.

DATASET OVERVIEW:
- Total Records: {len(df)}
- Columns Present ({len(df.columns)}):
{col_list}

SPEND METRICS:
- Total Annual Spend in dataset: ${total:,.0f}
- Records with valid annual_cost: {df['annual_cost'].notna().sum() if 'annual_cost' in df.columns else 'N/A'}

SAMPLE DATA (first 3 rows):
{sample}

DUPLICATE DETECTION:
- Exact duplicate rows: {df.duplicated().sum()}

NULL ANALYSIS:
{df.isnull().sum().to_string()}

Please assess:
1. Data completeness and quality score
2. Critical missing fields for TBM analysis
3. Any data integrity issues
4. Recommendations for enrichment
5. Confirmation of TBM readiness"""

        if context.user_question:
            msg += f"\n\nAdditional question: {context.user_question}"

        return msg

    def _build_quality_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build a structured DataFrame showing column-by-column quality metrics."""
        rows = []
        for col in df.columns:
            non_null = df[col].notna().sum()
            null_count = df[col].isna().sum()
            completeness = round(non_null / len(df) * 100, 1) if len(df) > 0 else 0
            rows.append({
                "column": col,
                "data_type": str(df[col].dtype),
                "non_null_count": non_null,
                "null_count": null_count,
                "completeness_pct": completeness,
                "unique_values": df[col].nunique(),
                "sample_value": str(df[col].dropna().iloc[0]) if non_null > 0 else "N/A",
            })
        return pd.DataFrame(rows)

    def _default_suggested_actions(self) -> list:
        return [
            "Run cost analysis to identify top spend categories",
            "Perform TBM framework mapping",
            "Detect optimization opportunities",
            "Check for contract renewal alerts",
        ]
