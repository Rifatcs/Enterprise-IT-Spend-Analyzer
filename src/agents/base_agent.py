"""
Base agent class — all specialist agents inherit from this.
Handles Claude API interaction, streaming, error handling, and conversation management.
"""

import os
import anthropic
from typing import Optional
from src.models.schemas import SpendContext, AgentResponse
from src.analytics.deterministic import format_analytics_for_llm


class BaseAgent:
    """
    Abstract base for all specialist agents.
    Subclasses override `system_prompt` and `_build_user_message()`.
    """

    agent_name: str = "base"
    system_prompt: str = ""

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        self.client = anthropic.Anthropic(api_key=key)
        self.model = "claude-opus-4-6"

    def run(self, context: SpendContext) -> AgentResponse:
        """
        Main entry point. Builds the message, calls Claude, returns AgentResponse.
        Subclasses can override this for custom behavior.
        """
        try:
            user_message = self._build_user_message(context)
            messages = self._build_messages(context, user_message)

            response_text = self._call_claude(messages)
            data = self._parse_structured_output(response_text, context)
            actions = self._default_suggested_actions()

            return AgentResponse(
                agent_name=self.agent_name,
                summary=response_text,
                data=data,
                suggested_actions=actions,
                metadata={"model": self.model},
            )
        except anthropic.AuthenticationError:
            return AgentResponse(
                agent_name=self.agent_name,
                summary="",
                error="Invalid API key. Please check your ANTHROPIC_API_KEY.",
            )
        except anthropic.RateLimitError:
            return AgentResponse(
                agent_name=self.agent_name,
                summary="",
                error="Rate limit exceeded. Please wait a moment and try again.",
            )
        except Exception as e:
            return AgentResponse(
                agent_name=self.agent_name,
                summary="",
                error=f"Agent error: {str(e)}",
            )

    def _build_user_message(self, context: SpendContext) -> str:
        """Override in subclasses to customize the user message content."""
        analytics_text = format_analytics_for_llm(context.analytics)
        base = f"Here is the enterprise IT spend analytics data:\n\n{analytics_text}"
        if context.user_question:
            base += f"\n\nUser question: {context.user_question}"
        return base

    def _build_messages(self, context: SpendContext, user_message: str) -> list:
        """
        Build the messages array, including conversation history for follow-up questions.
        Only the last N turns of history are included to manage token usage.
        """
        MAX_HISTORY_TURNS = 6  # 3 user + 3 assistant
        messages = []

        # Include recent conversation history (skip system-level messages)
        recent_history = context.conversation_history[-MAX_HISTORY_TURNS:] if context.conversation_history else []
        for msg in recent_history:
            if msg.get("role") in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_message})
        return messages

    def _call_claude(self, messages: list, stream: bool = False) -> str:
        """
        Call the Claude API. Uses adaptive thinking for complex reasoning.
        Returns the complete response text.
        """
        if stream:
            return self._call_claude_streaming(messages)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=self.system_prompt,
            messages=messages,
        )
        return self._extract_text(response)

    def _call_claude_streaming(self, messages: list) -> str:
        """
        Streaming variant for long-form outputs (reports, full analyses).
        """
        with self.client.messages.stream(
            model=self.model,
            max_tokens=8192,
            thinking={"type": "adaptive"},
            system=self.system_prompt,
            messages=messages,
        ) as stream:
            final = stream.get_final_message()
        return self._extract_text(final)

    def _extract_text(self, response) -> str:
        """Extract text content from a Claude API response object."""
        text_parts = []
        for block in response.content:
            if hasattr(block, "type") and block.type == "text":
                text_parts.append(block.text)
        return "\n".join(text_parts).strip()

    def _parse_structured_output(self, response_text: str, context: SpendContext):
        """
        Override in subclasses to parse structured data (DataFrames) from the response.
        Base implementation returns None.
        """
        return None

    def _default_suggested_actions(self) -> list:
        """Override in subclasses to provide context-aware suggestions."""
        return [
            "Ask a follow-up question about the analysis",
            "Generate optimization recommendations",
            "Produce an executive report",
        ]
