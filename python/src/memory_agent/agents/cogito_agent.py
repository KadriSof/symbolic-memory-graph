"""
Cogito Agent - Symbolic Memory-Augmented Agent

The Cogito Patter extends ReAct by making Symbolic memory intrinsic to reasoning.
The agent thinks through the graph, not just the prompt.
"""

import json
import logging

from datetime import datetime

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from memory_graph import (
    MemoryGraph,
    Node,
    Edge,
    EdgeType,
)

from memory_agent.llm.base import BaseLLM

from memory_agent.agents.react_agent import ReactAgent, ReactState

from memory_agent.tools import Tool

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# Constants and Markers
# 2. Cogito Markers
MARKER_COMPREHEND = "COMPREHEND"
MARKER_RETRIEVE = "RETRIEVE"
MARKER_CONSOLIDATE = "CONSOLIDATE"
MARKER_REASON = "REASON"
MARKER_UPDATE = "UPDATE"
MARKER_RESPOND = "RESPOND"
MARKER_ERROR = "ERROR"


class PathType(Enum):
    """The reasoning path the agent can take."""

    REACT = "react"  # Simple ReAct: think -> act -> respond
    COGITO = "cogito"  # Full Cogito: comprehend -> retrieve -> consolidate -> reason -> update -> respond


@dataclass
class CogitoState(ReactState):
    """Single source of truth for agent state."""

    active_context: str | None = None  # "lore_building", "coding_project", etc.
    session_entities: list[str] = field(default_factory=list)
    graph_snapshots: list[dict[str, Any]] = field(default_factory=list)

    confidence_threshold: float = 0.6
    confidence_history: list[float] = field(default_factory=list)

    current_mode: str = "REACT"
    mode_history: list[dict[str, str]] = field(
        default_factory=list
    )  # Track mode changes

    llm_calls: int = 0

    def add_session_entity(self, entity_id: str) -> None:
        """Record an entity to the session tracking"""
        if entity_id not in self.session_entities:
            self.session_entities.append(entity_id)

    def add_graph_snapshot(self, nodes: int, edges: int) -> None:
        """Record a graph snapshot for traceability."""
        self.graph_snapshots.append(
            {"nodes": nodes, "edges": edges, "timestamp": datetime.now().isoformat()}
        )

    def set_mode(self, new_mode: str, context: str) -> None:
        """Switch mode with a reason for traceability"""
        self.mode_history.append(
            {
                "from": self.current_mode,
                "to": new_mode,
                "context": context,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.current_mode = new_mode
        self.active_context = context

    def increment_llm_calls(self) -> None:
        """Increment the LLM call counter"""
        self.llm_calls += 1


@dataclass
class intent:
    """The intent and complexity of a user query."""

    action: str  # "ask", "task", "clarify", "correct"
    entities: list[str] = field(default_factory=list)
    relations: list[dict[str, str]] = field(default_factory=list)
    requires_tools: bool = False
    complexity: str = "simple"  # "simple", "moderate", "complex"


@dataclass
class ConsolidationResult:
    """Result of the consolidation step."""

    current_knowledge: str
    new_information: list[dict[str, Any]]
    gaps: list[str]
    updates: list[dict[str, Any]]
    conflicts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ReasoningResult:
    """Result of the reasoning step."""

    reasoning_trace: str
    solution: str
    confidence: float = 1.0
    tool_results: list[dict[str, Any]] | None = None


class Prompts:
    """Prompt templates for the Cogito Agent."""

    @staticmethod
    def comprehend_query(query: str) -> str:
        return f"""
        You are a precise query analyzer for a cognitive agent with symbolic memory.

        **IMPORTANT: Your goal is to choose the SIMPLEST path that correctly handles the query.**
        - REACT is fast, cheap, and sufficient for most queries.
        - COGITO is powerful but expensive — use it ONLY when absolutely necessary.

        Original user query:
        ---
        {query}
        ---

        Analyze the query and return a JSON object with the following fields:

        1. **reconstructed_query** (string): Rephrase the query for full comprehension.
        2. **user_intent** (string): "ask" | "task" | "clarify" | "correct"
        3. **modus_operandi** (string): "REACT" or "COGITO"
        4. **active_context** (string): Classify the overall context:
        - "question_answering": Simple Q&A, no ongoing project
        - "lore_building": World-building, story creation, character development
        - "coding_project": Software development, API design, debugging
        - "research": Information gathering, analysis, comparison
        - "planning": Strategy, roadmap, task breakdown
        - "analysis": Deep dive into a topic, relationship mapping
        - "casual": General chat, no specific context

        **DECISION RULES for modus_operandi:**

        ## USE "REACT" (Simple, fast, cheap) when:
        - Factual questions: "What is the capital of France?"
        - Direct calculations: "What is 2 + 2?"
        - Simple translations: "Say hello in Spanish"
        - Single-step tool calls: "Get the weather in Paris"
        - Database queries: "Return the number of rows in the students column"
        - Direct lookups: "Find the user with ID 123"
        - No relationships between multiple entities
        - No need for memory or context from previous conversations

        ## USE "COGITO" (Complex, powerful, expensive) ONLY when:
        - The query explicitly asks about RELATIONSHIPS between entities
        - The query requires building UNDERSTANDING, not just retrieval
        - The query is about CONNECTIONS: "How is X connected to Y?"
        - The query requires DETECTING CONTRADICTIONS
        - The query involves MULTI-STEP reasoning with memory
        - The query is EXPLICITLY about the agent's memory: "What do you know about X?"
        - The query involves BUILDING a mental model: "Tell me about X and Y"

        **Key principle: When in doubt, choose REACT.** COGITO is a powerful tool, but it should be used sparingly.

        Return ONLY valid JSON, no additional text.

        Example outputs:
        1. Simple query → REACT:
        {{
            "reconstructed_query": "What is the capital of Tunisia?",
            "user_intent": "ask",
            "modus_operandi": "REACT",
            "active_context": "question_answering"
        }}

        2. Complex query → COGITO:
        {{
            "reconstructed_query": "What is the relationship between Geralt and Yennefer?",
            "user_intent": "ask",
            "modus_operandi": "COGITO",
            "active_context" "lore_building"
        }}
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


class CogitoAgent(ReactAgent):
    """
    Cogito Agent — Symbolic Memory-Augmented Agent.

    Extends ReAct with intrinsic symbolic memory. The agent thinks through
    the graph, not just the prompt. Memory is always active, not optional.

    Core Flow:
    1. Comprehend → Understand the query and determine complexity
    2. Retrieve → Query the memory graph for context
    3. Consolidate → Analyze new vs existing knowledge
    4. Reason → Think using the consolidated graph
    5. Update → Integrate new knowledge into memory
    6. Respond → Synthesize and return the final answer
    """

    def __init__(
        self,
        llm: BaseLLM,
        memory_graph: MemoryGraph | None = None,
        tools: list[Tool] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(llm, tools, config)
        self._CLSNAME = self.__class__.__name__

        self.memory = memory_graph or MemoryGraph({"type": "cogito_agent"})

        # State tracking
        self.state = CogitoState()
        self.confidence_threshold = self.config.get("confidence_threshold", 0.6)

        self._initialize_state()

        self.logger = logging.getLogger(name=f"[{self._CLSNAME}]")

    # State Management
    def _initialize_state(self) -> CogitoState:
        """Create a CogitoState instance"""
        return CogitoState(confidence_threshold=0.6)

    # PUBLIC API
    def process_query(self, user_query: str) -> str:
        """
        Process a user query through the Cogito loop.

        Args:
            user_query: The user's input

        Returns:
            The agent's response
        """
        self.logger.info(f"Processing query:\n{user_query[:100]}\n---")

        # Step 1: COMPREHEND
        comprehension_ = self._comprehend(user_query)
        reconstructed_query = comprehension_.get("reconstructed", user_query)

        # Step 2: Determine path
        if comprehension_.get("modus_operandi", "REACT") == PathType.REACT:
            return super().process_query(query=reconstructed_query)

        # Step 3: RETRIEVE
        context = self._retrieve(comprehension_.get("primitives", {}))

        # Step 4: CONSOLIDATE
        consolidated = self._consolidate(comprehension_.get("primitives", {}), context)

        # Step 5: REASON
        reasoning_result = self._reason(
            query=reconstructed_query,
            context=context,
            consolidated=consolidated,
        )

        # Step 6: UPDATE
        if consolidated.updates or consolidated.conflicts:
            self._update_memory(consolidated)

        # Step 7: RESPOND
        response = self._respond(reasoning_result, context)

        # Record history
        self.state.add_message("assistant", response)

        return response

    def _comprehend(self, query: str) -> dict[str, Any]:
        """Comprehend the user query and determine complexity."""
        try:
            # 1. Comprehend the query for better understanding
            comprehension = self._llm_structured(query=Prompts.comprehend_query(query))

            # 2. Reconstruct the user query and retrieve context
            modus_operandi = comprehension.get("modus_operandi", "REACT")
            reconstructed_query = comprehension.get("reconstructed_query", query)
            active_context = comprehension.get("active_context", "")

            # 3. Update state
            self.state.add_message(role="USER", content=query)
            self.state.set_mode(new_mode=modus_operandi, context=active_context)

            # 2. Extract primitives
            if modus_operandi == "COGITO":
                primitives = self._extract_primitives(query)
                for entity in primitives.get("entities", []):
                    entity_id = entity.get("id")
                    if entity_id:
                        self.state.add_session_entity(entity_id)

            else:
                primitives = {}

            return {
                "modus_operandi": modus_operandi,
                "primitives": primitives,
                "intent": primitives.get("intent", "ask"),
                "reconstructed": reconstructed_query,
            }

        except Exception as e:
            self.logger.error(f"Comprehension failed:\n{e}\n---")
            # Fallback to simple ReAct
            return {
                "modus_operandi": PathType.REACT,
                "primitives": {},
            }

    # Step 2: RETRIEVE
    def _retrieve(self, primitives: dict[str, Any]) -> str:
        """Retrieve relevant context from the memory graph"""
        entities = primitives.get("entities", [])
        context_parts = []

        for entity in entities:
            entity_id = entity.get("id")
            if not entity_id or not self.memory.has_node(entity_id):
                continue

            node = self.memory.get_node(entity_id)
            neighbors = self.memory.get_neighbors(entity_id)

            # TODO: [QUESTION] Why are we building the context parts as list?
            context_parts.append("=" * 40)
            context_parts.append(f"NODE: {node.get_id()}")
            context_parts.append(f"  Label: {node.get_label()}")
            context_parts.append(f"  Metadata: {json.dumps(node.get_metadata())}")

            if neighbors:
                context_parts.append("  Connections:")
                for n in neighbors[:5]:
                    context_parts.append(f"    → {n.get_label()} ({n.get_id()})")

            context_parts.append("")

        # Add graph summary if manageable
        node_count = len(self.memory.get_nodes())
        if node_count < 50:
            graph_summary = f"GRAPH SUMMARY: {node_count} nodes, {len(self.memory.get_edges())} edges"
            context_parts.append(graph_summary)
            context_parts.append("=" * 40)

        return (
            "\n".join(context_parts) if context_parts else "No relevant context found."
        )

    # Step 3: CONSOLIDATE
    def _consolidate(
        self, primitives: dict[str, Any], context: str
    ) -> ConsolidationResult:
        """Consolidate new information with existing knowledge"""
        # 1. Check for direct contradictions
        conflicts = self._detect_conflicts(primitives)

        try:
            # Analyze and consolidate
            result = self._llm_structured(
                query=Prompts.consolidate(
                    context=context,
                    new_primitives=json.dumps(primitives),
                    query=context[:500],  # Context as a query
                )
            )

            auto_updates = []
            for entity in primitives.get("entities", []):
                entity_id = entity.get("id")
                if entity_id and not self.memory.has_node(entity_id):
                    auto_updates.append(
                        {
                            "type": "add",
                            "data": {"type": "entity", **entity},
                            "confidence": 0.7,
                        }
                    )

            all_updates = result.get("updates", []) + auto_updates

            return ConsolidationResult(
                current_knowledge=result.get("knowledge_summary", ""),
                new_information=result.get("new_information", []),
                gaps=result.get("gaps", []),
                updates=all_updates,
                conflicts=conflicts,
            )

        except Exception as e:
            self.logger.error(f"[CogitoAgent] Consolidation failed:\n---\n{e}\n---")
            return ConsolidationResult(
                current_knowledge="",
                new_information=[],
                gaps=[],
                updates=[],
                conflicts=conflicts,
            )

    # Step 4: REASON
    def _reason(
        self,
        query: str,
        context: str,
        consolidated: ConsolidationResult,
    ) -> ReasoningResult:
        """Reason using the consolidated graph."""
        try:
            # Fill gaps using tools if needed
            tool_results = None
            if consolidated.gaps and self.tools:
                tool_results = self._call_tools(consolidated.gaps)

            # Generate reasoning
            reasoning_text = self._llm_call(
                query=Prompts.reason(
                    query=query,
                    context=context,
                    gaps=json.dumps(consolidated.gaps),
                    tool_results=json.dumps(tool_results) if tool_results else None,
                )
            )

            # Parse reasoning
            # (For now, we will treat it as text. Otherwise, it should parse the structure from the LLM output)
            # Extract solution (simplified)
            solution = self._extract_solution(reasoning_text)

            # Estrimate confidence
            confidence = self._estimate_confidence(reasoning_text, consolidated)

            return ReasoningResult(
                reasoning_trace=reasoning_text,
                solution=solution,
                tool_results=tool_results,
                confidence=confidence,
            )

        except Exception as e:
            self.logger.error(f"[CogitoAgent] Reasoning failed:\n---\n{e}\n---")
            return ReasoningResult(
                reasoning_trace="",
                solution="I encountered an issue while reasoning. Could you rephrase your question?",
                confidence=0.3,
            )

    # Step 5: UPDATE
    def _update_memory(self, consolidated: ConsolidationResult) -> None:
        """Update the memory graph with new information"""
        for update in consolidated.updates:
            confidence = update.get("confidence", 0.5)

            # Skip low confidence updates
            if confidence < self.confidence_threshold:
                self.logger.debug(f"Skipping low-confidence update: {confidence}")
                continue

            try:
                self._apply_update(update.get("data", {}), confidence)

            except Exception as e:
                self.logger.error(f"[CogitoAgent] Failed to appy update: {e}")

        # Store conflicts for review
        for conflict in consolidated.conflicts:
            self._store_conflict(conflict)

    # Step 6: RESPOND
    def _respond(self, reasoning_result: ReasoningResult, context: str) -> str:
        """Synthesize a response from the reasoning."""
        try:
            response = self._llm_call(
                Prompts.synthesize_response(
                    reasoning=reasoning_result.reasoning_trace,
                    solution=reasoning_result.solution,
                    confidence=reasoning_result.confidence,
                    graph_context=context[:1000],  # Limit context
                )
            )

            return response.strip()

        except Exception as e:
            self.logger.error(f"Response synthesis failed: {e}")
            return reasoning_result.solution

    # TODO: [PRIORITY] Requires adjustment or removal if not needed.
    def _llm_call(self, query: str) -> str:
        """Call the LLM with a prompt."""
        messages = [
            {
                "role": "system",
                "content": "You are the Cogito Agent, a reasoning agent with symbolic memory.",
            },
            {"role": "user", "content": query},
        ]

        response = self.llm.chat(messages=messages)
        self.state.increment_llm_calls()
        return response

    # TODO: [PRIORITY] Requires adjustements or removal if not needed.
    def _llm_structured(self, query: str) -> dict[str, Any]:
        """Call the LLM and parse structured output."""
        # TODO: [PERSPECTIVE] Prompt can be improved and retrieved from a separate prompts script/module.
        messages = [
            {
                "role": "system",
                "content": "You are the Cogito Agent, a reasoning agent with symbolic memory. Return valid JSON.",
            },
            {"role": "user", "content": query},
        ]
        response = self.llm.chat(messages)

        try:
            # Try to parse JSON from response
            # Find JSON block
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
            return {"raw": response}
        except json.JSONDecodeError:
            return {"raw": response}

    # Helpers
    def _extract_primitives(self, query: str) -> dict[str, Any]:
        """Extract graph primitives from query."""
        response = self._llm_structured(Prompts.extract_primitives(query))
        return {
            "entities": response.get("entities", []),
            "relations": response.get("relations", []),
            "attributes": response.get("attributes", []),
            "intent": response.get("intent", "ask"),
        }

    def _detect_conflicts(self, primitives: dict[str, Any]) -> list[dict[str, Any]]:
        """Detect contradictions between new and existing knowledge."""
        conflicts = []

        for entity in primitives.get("entities", []):
            entity_id = entity.get("id")
            if not entity_id or not self.memory.has_node(entity_id):
                continue

            existing = self.memory.get_node(entity_id)

            # Check type conflict
            existing_type = existing.get_metadata().get("type")
            new_type = entity.get("metadata", {}).get("type")
            if existing_type and new_type and existing_type != new_type:
                conflicts.append(
                    {
                        "entity": entity_id,
                        "attribute": "type",
                        "existing": existing_type,
                        "new": new_type,
                    }
                )

            # Check label conflict
            existing_label = existing.get_label()
            new_label = entity.get("label")
            if existing_label and new_label and existing_label != new_label:
                conflicts.append(
                    {
                        "entity": entity_id,
                        "attribute": "label",
                        "existing": existing_label,
                        "new": new_label,
                    }
                )

        return conflicts

    def _apply_update(self, data: dict[str, Any], confidence: float) -> None:
        """Apply a single update to the graph."""
        primitive_type = data.get("type")

        if primitive_type == "entity":
            entity_id = data.get("id", "")
            label = data.get("label", entity_id)
            metadata = data.get("metadata", {})
            metadata["confidence"] = confidence
            metadata["source"] = "cogito_agent"
            metadata["updated_at"] = datetime.now().isoformat()

            if self.memory.has_node(entity_id):
                node = self.memory.get_node(entity_id)
                for key, value in metadata.items():
                    node.update_metadata(key, value)
                if label != node.get_label():
                    node.set_label(label)
            else:
                node = Node(entity_id, label, metadata)
                self.memory.add_node(node)

        elif primitive_type == "relation":
            source = data.get("source", "")
            target = data.get("target", "")
            label = data.get("label", "")
            direction = data.get("direction", "asymmetric")
            weight = data.get("weight", 0.8)
            metadata = data.get("metadata", {})
            metadata["confidence"] = confidence

            edge_id = f"{source}_{target}"

            if direction == "symmetric":
                self.memory.add_group_edge(
                    id=edge_id,
                    label=label,
                    node_ids={source, target},
                    weight=weight,
                    metadata=metadata,
                )
            else:
                edge = Edge(
                    id=edge_id,
                    label=label,
                    type=EdgeType.ASYMMETRIC,
                    connections=(source, target),
                    weight=weight,
                    metadata=metadata,
                )
                self.memory.add_edge(edge)

    def _store_conflict(self, conflict: dict[str, Any]) -> None:
        """Store a conflict in the graph for later resolution."""
        conflict_node = Node(
            id=f"conflict_{datetime.now().timestamp()}",
            label="Conflict",
            metadata={
                "type": "conflict",
                "data": conflict,
                "timestamp": datetime.now().isoformat(),
            },
        )
        self.memory.add_node(conflict_node)

    def _call_tools(self, gaps: list[str]) -> list[dict[str, Any]]:
        """Call tools to fill gaps."""
        results = []
        for gap in gaps:
            for tool_name, tool in self.tools.items():
                if tool_name.lower() in gap.lower():
                    try:
                        result = tool.execute(gap=gap)
                        results.append(
                            {
                                "tool": tool_name,
                                "gap": gap,
                                "result": result,
                                "success": True,
                            }
                        )
                        break
                    except Exception as e:
                        results.append(
                            {
                                "tool": tool_name,
                                "gap": gap,
                                "error": str(e),
                                "success": False,
                            }
                        )
        return results

    def _extract_solution(self, text: str) -> str:
        """Extract solution from reasoning text."""
        if "Solution:" in text:
            parts = text.split("Solution:", 1)
            return parts[1].strip() if len(parts) > 1 else text
        return text

    def _estimate_confidence(
        self, text: str, consolidated: ConsolidationResult
    ) -> float:
        """Estimate confidence in the reasoning."""
        uncertainty_markers = [
            "not sure",
            "I think",
            "maybe",
            "possibly",
            "could be",
            "might",
        ]
        uncertainty_count = sum(1 for m in uncertainty_markers if m in text.lower())

        confidence = max(0.3, 0.9 - (uncertainty_count * 0.1))

        if consolidated.new_information:
            avg = sum(
                item.get("confidence", 0.5) for item in consolidated.new_information
            )
            avg /= (
                len(consolidated.new_information) if consolidated.new_information else 1
            )
            confidence = (confidence + avg) / 2

        return min(1.0, confidence)

    def _log_cogito_init(self) -> None:
        """Log Cogito-specific initialization."""
        self.logger.info("[X] Initialized")
        self.logger.info(
            f"---Memory: {len(self.memory.get_nodes())} nodes, {len(self.memory.get_edges())} edges"
        )
        self.logger.info(f"---Confidence threshold: {self.confidence_threshold}")

    def save_memory(self, filepath: str) -> None:
        """Save memory graph to file."""
        import json

        with open(filepath, "w") as f:
            json.dump(self.memory.to_json(), f, indent=2)
        self.logger.info(f"Memory saved to {filepath}")

    def load_memory(self, filepath: str) -> None:
        """Load memory graph from file."""
        import json

        with open(filepath, "r") as f:
            data = json.load(f)
            self.memory = MemoryGraph.from_json(data)
        self.logger.info(f"Memory loaded from {filepath}")
