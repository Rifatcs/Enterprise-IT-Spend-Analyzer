"""
Data models and schemas for the Enterprise IT Spend Analyzer.
Pydantic models ensure type safety and validation across all agents.
"""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


# ─── Required CSV columns for full TBM analysis ───────────────────────────────
REQUIRED_COLUMNS = [
    "vendor",
    "service_name",
    "department",
    "cost_category",
    "annual_cost",
]

OPTIONAL_COLUMNS = [
    "business_unit",
    "spend_type",
    "monthly_cost",
    "contract_type",
    "contract_start_date",
    "contract_end_date",
    "utilization_pct",
    "region",
    "headcount_supported",
    "notes",
]

ALL_EXPECTED_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

# ─── TBM Taxonomy Mappings ─────────────────────────────────────────────────────
TBM_COST_POOL_MAP = {
    "Cloud": "IT Infrastructure",
    "Infrastructure": "IT Infrastructure",
    "Telecom": "IT Infrastructure",
    "SaaS": "IT Applications",
    "Software License": "IT Applications",
    "Professional Services": "IT Management",
    "Security": "IT Management",
}

TBM_VALUE_STREAM_MAP = {
    "Sales": "Revenue Generation",
    "Marketing": "Revenue Generation",
    "Customer Success": "Revenue Generation",
    "Engineering": "Innovation Enablement",
    "Data & Analytics": "Innovation Enablement",
    "Finance": "Operational Efficiency",
    "HR": "Operational Efficiency",
    "Legal": "Operational Efficiency",
    "IT Operations": "Risk Mitigation",
    "IT Security": "Risk Mitigation",
    "All": "Operational Efficiency",
    "Corporate": "Operational Efficiency",
}

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


# ─── Context object passed between agents ─────────────────────────────────────
@dataclass
class SpendContext:
    """
    Central context object passed to every agent.
    Carries the raw data, pre-computed analytics, prior agent results,
    and the current conversation history.
    """
    df: pd.DataFrame
    analytics: dict = field(default_factory=dict)
    prior_results: dict = field(default_factory=dict)
    conversation_history: list = field(default_factory=list)
    user_question: str = ""

    def add_result(self, agent_name: str, result: "AgentResponse") -> None:
        """Store a completed agent result for downstream agents to reference."""
        self.prior_results[agent_name] = {
            "summary": result.summary,
            "metadata": result.metadata,
        }


# ─── Standardized agent response ──────────────────────────────────────────────
@dataclass
class AgentResponse:
    """
    Standardized response returned by every agent.
    - summary: Human-readable narrative text
    - data: Optional structured DataFrame (e.g., recommendations table)
    - suggested_actions: 3-5 follow-up actions presented to the user
    - metadata: Flexible dict for agent-specific metrics
    """
    agent_name: str
    summary: str
    data: Optional[pd.DataFrame] = None
    suggested_actions: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


# ─── Orchestrator routing result ──────────────────────────────────────────────
@dataclass
class RoutingDecision:
    """Result of the orchestrator's routing logic."""
    agent: str          # Agent name or "all"
    intent: str         # What the user wants
    confidence: str     # high | medium | low
    suggested_actions: list = field(default_factory=list)
