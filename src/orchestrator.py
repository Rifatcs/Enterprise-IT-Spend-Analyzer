"""
Orchestrator — Central routing and coordination layer.
Interprets user intent, routes to the appropriate specialist agent,
maintains conversation state, and suggests next best actions.
"""

import json
import re
import os
from typing import Optional
import anthropic

from src.models.schemas import SpendContext, AgentResponse, RoutingDecision
from src.agents.intake_agent import IntakeAgent
from src.agents.cost_analysis_agent import CostAnalysisAgent
from src.agents.tbm_agent import TBMAgent
from src.agents.optimization_agent import OptimizationAgent
from src.agents.report_agent import ReportAgent
from config.prompts import ORCHESTRATOR_SYSTEM_PROMPT


# ─── Keyword-based fallback routing (fast, no API call needed) ────────────────
KEYWORD_ROUTES = {
    "intake": ["validate", "quality", "data check", "clean", "missing", "null", "schema"],
    "cost_analysis": ["cost", "spend", "trend", "top vendor", "breakdown", "category",
                      "anomaly", "concentration", "department", "analysis", "expensive",
                      "biggest", "highest", "growth"],
    "tbm": ["tbm", "capability", "value stream", "governance", "cost pool", "tower",
            "business alignment", "it management", "business value", "apptio"],
    "optimization": ["optimize", "save", "reduce", "waste", "recommend", "roi",
                     "rightsize", "underutilized", "duplicate", "consolidate",
                     "renegotiate", "cut", "lower"],
    "report": ["report", "summary", "executive", "export", "download", "present",
               "deck", "comprehensive", "full analysis", "complete"],
}

ALL_AGENTS_TRIGGERS = [
    "everything", "all agents", "full analysis", "complete analysis",
    "run all", "analyze everything", "comprehensive report",
]


class Orchestrator:
    """
    Central coordinator for the multi-agent IT spend analysis system.
    Manages routing, state, and agent execution.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-opus-4-6"

        # Instantiate all specialist agents
        self._agents = {
            "intake": IntakeAgent(api_key=self.api_key),
            "cost_analysis": CostAnalysisAgent(api_key=self.api_key),
            "tbm": TBMAgent(api_key=self.api_key),
            "optimization": OptimizationAgent(api_key=self.api_key),
            "report": ReportAgent(api_key=self.api_key),
        }

    # ─── Public API ───────────────────────────────────────────────────────────

    def route_and_run(self, context: SpendContext) -> dict[str, AgentResponse]:
        """
        Main entry point. Route the user's question and run the appropriate agent(s).
        Returns a dict of agent_name → AgentResponse.
        """
        decision = self._route(context)
        results = {}

        if decision.agent == "all":
            results = self._run_all_agents(context)
        elif decision.agent in self._agents:
            agent = self._agents[decision.agent]
            results[decision.agent] = agent.run(context)
        else:
            # Fallback to cost analysis
            results["cost_analysis"] = self._agents["cost_analysis"].run(context)

        # Store results in context for downstream agents
        for name, resp in results.items():
            context.add_result(name, resp)

        # Add suggested actions to results metadata
        first_result = next(iter(results.values()), None)
        if first_result:
            first_result.metadata["routing_decision"] = decision.agent
            first_result.metadata["routing_intent"] = decision.intent
            first_result.metadata["suggested_actions"] = decision.suggested_actions

        return results

    def get_routing_decision(self, context: SpendContext) -> RoutingDecision:
        """Public accessor for the routing decision (useful for UI display)."""
        return self._route(context)

    # ─── Routing Logic ────────────────────────────────────────────────────────

    def _route(self, context: SpendContext) -> RoutingDecision:
        """
        Determine which agent(s) should handle the user's request.
        Uses keyword matching first (fast), falls back to LLM classification.
        """
        question = context.user_question.lower().strip()

        if not question:
            return RoutingDecision(
                agent="cost_analysis",
                intent="General spend overview",
                confidence="high",
                suggested_actions=self._default_suggested_actions(),
            )

        # Check for "run all" triggers
        for trigger in ALL_AGENTS_TRIGGERS:
            if trigger in question:
                return RoutingDecision(
                    agent="all",
                    intent="Full multi-agent analysis",
                    confidence="high",
                    suggested_actions=self._default_suggested_actions(),
                )

        # Keyword-based routing (no API cost, instant)
        keyword_match = self._keyword_route(question)
        if keyword_match:
            return RoutingDecision(
                agent=keyword_match,
                intent=f"User asked about {keyword_match.replace('_', ' ')}",
                confidence="medium",
                suggested_actions=self._default_suggested_actions(),
            )

        # LLM-based routing for ambiguous queries
        return self._llm_route(context)

    def _keyword_route(self, question: str) -> Optional[str]:
        """Fast keyword-based routing. Returns agent name or None."""
        scores = {agent: 0 for agent in KEYWORD_ROUTES}
        for agent, keywords in KEYWORD_ROUTES.items():
            for kw in keywords:
                if kw in question:
                    scores[agent] += 1
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else None

    def _llm_route(self, context: SpendContext) -> RoutingDecision:
        """Use Claude to classify the user's intent for ambiguous queries."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                system=ORCHESTRATOR_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"Route this user request: {context.user_question}"
                }],
            )
            text = ""
            for block in response.content:
                if hasattr(block, "type") and block.type == "text":
                    text += block.text

            # Parse JSON response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return RoutingDecision(
                    agent=data.get("agent", "cost_analysis"),
                    intent=data.get("intent", "User query"),
                    confidence=data.get("confidence", "medium"),
                    suggested_actions=data.get("suggested_actions", self._default_suggested_actions()),
                )
        except Exception:
            pass

        # Final fallback
        return RoutingDecision(
            agent="cost_analysis",
            intent="General question",
            confidence="low",
            suggested_actions=self._default_suggested_actions(),
        )

    def _run_all_agents(self, context: SpendContext) -> dict[str, AgentResponse]:
        """
        Run all agents in the recommended sequence.
        Each agent has access to prior agents' results.
        """
        sequence = ["intake", "cost_analysis", "tbm", "optimization", "report"]
        results = {}

        for agent_name in sequence:
            agent = self._agents[agent_name]
            result = agent.run(context)
            results[agent_name] = result
            # Store in context so later agents can reference earlier findings
            context.add_result(agent_name, result)

        return results

    # ─── Initial Analysis (called on CSV upload) ──────────────────────────────

    def run_initial_analysis(self, context: SpendContext) -> dict[str, AgentResponse]:
        """
        Run the intake + cost analysis agents automatically on upload.
        Gives users immediate value without requiring them to ask a question.
        """
        results = {}

        intake_result = self._agents["intake"].run(context)
        results["intake"] = intake_result
        context.add_result("intake", intake_result)

        cost_result = self._agents["cost_analysis"].run(context)
        results["cost_analysis"] = cost_result
        context.add_result("cost_analysis", cost_result)

        return results

    def _default_suggested_actions(self) -> list:
        return [
            "Map IT spend to TBM cost pools and business capabilities",
            "Generate cost optimization recommendations with ROI estimates",
            "Identify underutilized services and waste reduction opportunities",
            "Review upcoming contract renewals for renegotiation leverage",
            "Produce a full executive report for C-suite presentation",
        ]
