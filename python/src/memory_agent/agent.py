"""
Cogito Agent - Symbolic Memory-Augmented Agent

The Cogito Patter extends ReAct by making Symbolic memory intrinsic to reasoning.
The agent thinks through the graph, not just the prompt.
"""

import json
import logging

from datetime import datetime

from enum import Enum
from dataclasses import dataclass, field
from typing import Any

from memory_graph import (
    MemoryGraph,
    Node,
    Edge,
    EdgeType,
    SymmetricConnections,
    AsymmetricConnections,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PathType(Enum):
    """The reasoning path the agent can take."""

    REACT = "react"  # Simple ReAct: think -> act -> respond
    COGITO = "cogito"  # Full Cogito: comprehend -> retrieve -> consolidate -> reason -> update -> respond


class Confidence(Enum):
    """Confidence levels for knowledge."""

    CERTAIN = 1.0
    HIGH = 0.9
    MEDIUM = 0.7
    LOW = 0.5
    SPECULATIVE = 0.3


@dataclass
class Primitive:
    """A graph primitive extracted from text."""

    primitive_type: str  # "entity", "relation", "attribute"
    data: dict[str, Any]
    confidence: float = 0.7
    source: str = "llm_extraction"


@dataclass
class intent:
    """The intent and complexity of a user query."""

    action: str  # "ask", "task", "clarify", "correct"
    entities: list[str] = field(default_factory=list)
    relations: list[dict[str, str]] = field(default_factory=list)
    requires_tools: bool = False
    complexity: str = "simple"  # "simple", "moderate", "complex"


@dataclass
class ReasoningResult:
    """Result of the reasoning step."""

    reasoning_trace: str
    solution: str
    tool_results: list[dict[str, Any]] | None = None
    confidence: float = 1.0


class LLMInterface:
    """Abstract interface for LLM providers."""

    def __init__(self, model: str = "gpt-4", temprature: float = 0.7):
        self.model = model
        self.temprature = temprature

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Send a chat message to the LLM"""
        raise NotImplementedError

    def structured_output(
        self, messages: list[dict[str, str]], schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Get structured output from the LLM."""
        raise NotImplementedError


class Prompts:
    """Prompt templates for the Cogito Agent."""

    @staticmethod
    def reconstruct_query(query: str) -> str:
        return f"""
        You are a precise analyzer. Reconstruct the user's query to ensure full comprehnsion.

        User query: 
        ---
        {query}
        ---

        Reconstruct the query by:
        1. Indentifying the core intent
        2. Clarifying any ambiguities
        3. Rephrasing for clarity

        Return the reconstructed query.
        """

    @staticmethod
    def extract_primitives(query: str) -> str:
        return f"""
        Extract all graph primitives from the user query.

        User query:
        ---
        {query}
        ---

        Return a JSON object with:
        {{
            "entities": [{{"id": "...", "label": "...", "metadata": {{"type": "..."}}}}],
            "relations": [{{"source": "...", "target": "...", "label": "...", "direction": "symmetric|asymmetric"}}],
            "attributes": [{{"entity": "...", "key": "...", "value": "..."}}],
            "intent": "ask|task|clarify|correct"
        }}
        """

    @staticmethod
    def classify_complexity(primitives: str, query: str) -> str:
        return f"""
        Classify the complexity of this query.

        Query:
        ---
        {query}
        ---
        primitives: {primitives}

        Complexity levels:
        - simple: Single fact retrieval or simple question
        - moderate: Multi-step reasoning or comparison
        - complex: Requires building understanding, detecting contradictions, or multi-turn reasoning
        
        Return only: "simple", "moderate", "complex"
        """

    @staticmethod
    def consolidate(context: str, new_primitives: str, query: str) -> str:
        return f"""
        Analyze what you know versus what is new

        Current knowledge:
        {context}

        New information:
        {new_primitives}

        User query:
        {query}

        Analyze:
        1. What do I already know?
        2. What is new information?
        3. What conflicts with existing knowledge?
        4. What gaps need filling?
        5. What should I update in my memory?

        Return as JSON with:
        - "knowledge_summary": str
        - "new_information": [{{"type": "entity|relation|attribute", "data": {{...}}, "confidence": 0.0-1.0}}]
        - "gaps": [str]
        - "updates": [{{"type": "add|modify|conflict", "data": {{...}}}}]
        """

    @staticmethod
    def reason(
        query: str, context: str, gaps: str, tool_results: str | None = None
    ) -> str:
        return f"""
        You are a reasoning agent with access to a symbolic knowledge graph.

        User query: {query}

        Knowledge graph context:
        {context}

        Identified gaps:
        {gaps}

        {f"Tool results: {tool_results}" if tool_results else "No tool results needed."}

        Think step by step:
        1. What do I need to solve this?
        2. How does the graph context help?
        3. What gaps must I fill?
        4. What is the solution?

        Provide your reasoning and solution.
        """

    @staticmethod
    def synthesize_response(
        reasoning: str, solution: str, confidence: float, graph_context: str
    ) -> str:
        return f"""
        Synthesize a clear, natural response.

        Reasoning: {reasoning}
        Solution: {solution}
        Confidence: {confidence}
        Graph context: {graph_context}

        Respond to the user with:
        1. The answer or solution
        2. Any relevant caveats or confidence notes
        3. Suggested next steps (if applicable)

        Keep it concise, helpful, and friendly.
        """

    @staticmethod
    def update_decision(new_information: str, current_knowledge: str) -> str:
        return f"""
        Decide what to store in memory.

        Current knowledge: {current_knowledge}
        New information: {new_information}

        For each piece of new information:
        1. Should it be stored? (yes/no)
        2. What confidence level? (0.0-1.0)
        3. What source attribution?

        Return as JSON:
        {{
            "decisions": [
                {{"id": "...", "store": true|false, "confidence": 0.0-1.0, "source": "..."}}
            ]
        }}
        """
