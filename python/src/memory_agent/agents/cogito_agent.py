"""
Cogito Agent — Compact, Graph-Centric Reasoning Agent.

Core Principles:
1. Think through the graph, not just the prompt.
2. One LLM call per step (optimized).
3. Confidence is tracked for every piece of knowledge.
4. Memory is intrinsic, not optional.
5. Compact and maintainable.

Flow: Comprehend → Retrieve → Consolidate → Reason → Update → Respond
"""

import json
import logging
from datetime import datetime
from typing import Any

from memory_graph import MemoryGraph, Node, Edge, EdgeType
from memory_graph.core import GraphRepresentation
from memory_graph_core import (
    bfs,
    find_nodes_by_label,
    find_nodes_by_metadata,
    has_cycle,
    shortest_path,
    subgraph,
)

from memory_agent.agents.react_agent import ReactAgent
from memory_agent.llm.base import BaseLLM
from memory_agent.tools import Tool

# Import from core
from memory_agent.core import (
    CogitoState,
    Entity,
    Relation,
    KnowledgeDelta,
    Confidence,
    ComprehensionResult,
    ReasoningResult,
    Prompts,
)

# Import Pydantic schemas
from memory_agent.core.schemas import (
    ComprehensionSchema,
    ConsolidationSchema,
)

logger = logging.getLogger(__name__)


class CogitoAgent(ReactAgent):
    """
    Compact Cogito Agent — Graph-Centric Reasoning.

    Key Design Decisions:
    - Single comprehension call per query (optimized).
    - Graph-first retrieval with smart formatting.
    - Knowledge delta for minimal updates.
    - Confidence scoring for all knowledge.
    - Automatic memory persistence.
    - Uses generic structured output parser (no project coupling).
    """

    def __init__(
        self,
        llm: BaseLLM,
        memory_graph: MemoryGraph | None = None,
        tools: list[Tool] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._CLSNAME = self.__class__.__name__
        super().__init__(llm, tools, config)

        self.memory = memory_graph or MemoryGraph({"type": "cogito_agent"})
        self.state = CogitoState()
        self.current_mode = "REACT"
        self.llm_calls = 0

        # Configuration
        self.confidence_threshold = self.config.get("confidence_threshold", 0.6)
        self.context_format = self.config.get("context_format", "linearized")
        self.max_graph_nodes = self.config.get("max_graph_nodes", 50)
        self.max_edges_per_node = self.config.get("max_edges_per_node", 5)

        # Initialize state
        self.state.set_context_format(self.context_format)
        self.state.set_confidence_threshold(self.confidence_threshold)

        self._log_init()

    # PUBLIC API
    def process_query(self, query: str) -> str:
        """
        Process a user query through the Cogito loop.

        Single entry point for all queries.
        Automatically falls back to ReAct for simple queries.
        """
        logger.info(f"[{self._CLSNAME}] Processing: {query[:100]}...")

        # Step 1: COMPREHEND (Unified)
        comprehension = self._comprehend(query)

        # Fallback to ReAct for simple queries
        if comprehension.is_react():
            logger.info(f"[{self._CLSNAME}] Falling back to ReAct")
            self.current_mode = "REACT"
            return super().process_query(query)

        self.current_mode = "COGITO"

        # Step 2: RETRIEVE (Graph-First)
        context = self._retrieve(comprehension.entities)

        # Step 3: CONSOLIDATE (Delta)
        delta = self._consolidate(comprehension, context)

        # Step 4: REASON (Graph-Augmented)
        reasoning = self._cogito_reason(query, context, delta)

        # Step 5: UPDATE (Automatic)
        if delta.has_changes:
            self._update_memory(delta)

        # Step 6: RESPOND (Synthesized)
        response = self._respond(reasoning, context)

        # Record history
        self.state.add_message("assistant", response)

        return response

    # Step 1: COMPREHEND (Unified)
    def _comprehend(self, query: str) -> ComprehensionResult:
        """
        Unified comprehension: one LLM call for everything.
        Uses the generic structured output parser with ComprehensionSchema.
        """
        try:
            # Get structured output using the generic parser
            result = self.llm.get_structured(
                messages=[{"role": "user", "content": Prompts.comprehend(query)}],
                model=ComprehensionSchema,
            )

            if result is None:
                logger.warning("Comprehension returned None, using fallback")
                return ComprehensionResult.fallback(query)

            # Convert schema result to internal Entity/Relation objects
            entities = []
            for e in result.entities:
                entities.append(
                    Entity(
                        id=e.id,
                        label=e.label,
                        type=e.type,
                        metadata=e.metadata,
                        confidence=Confidence(
                            score=e.confidence,
                            source="llm_comprehension",
                        ),
                    )
                )

            relations = []
            for r in result.relations:
                relations.append(
                    Relation(
                        source=r.source,
                        target=r.target,
                        label=r.label,
                        direction=r.direction,
                        weight=r.weight,
                        confidence=Confidence(
                            score=r.confidence,
                            source="llm_comprehension",
                        ),
                    )
                )

            # Graph-Aware Override: Check if the query references entities in the graph
            path = result.modus_operandi
            graph_has_info = self._graph_has_info(entities, relations)

            if graph_has_info and path == "REACT":
                logger.info(f"[{self._CLSNAME}] Graph override: REACT -> COGITO")
                path = "COGITO"

            # Update state
            if hasattr(self.state, "set_context"):
                self.state.set_context(result.active_context)

            if hasattr(self.state, "add_session_entity"):
                for entity in entities:
                    self.state.add_session_entity(entity.id)

            return ComprehensionResult(
                reconstructed_query=result.reconstructed_query,
                intent=result.user_intent,
                path=path,
                context=result.active_context,
                entities=entities,
                relations=relations,
                confidence=Confidence(
                    score=result.confidence,
                    source="llm_comprehension",
                ),
            )

        except Exception as e:
            logger.error(f"Comprehension failed: {e}")
            return ComprehensionResult.fallback(query)

    def _graph_has_info(
        self, entities: list[Entity], relations: list[Relation]
    ) -> bool:
        """Check if the graph contains information relevant to the query."""
        for entity in entities:
            if entity and entity.id:
                if self.memory.has_node(entity.id):
                    return True

        for relation in relations:
            if relation:
                if relation.source and self.memory.has_node(relation.source):
                    return True
                if relation.target and self.memory.has_node(relation.target):
                    return True

        return False

    # Step 2: RETRIEVE (Graph-First)
    def _retrieve(self, entities: list[Entity]) -> str:
        """Retrieve graph context optimized for LLM consumption."""
        node_count = len(self.memory.get_nodes())

        if node_count == 0:
            return "The memory graph is empty. No previous knowledge available."

        if node_count < self.max_graph_nodes:
            return GraphRepresentation.to_llm_context(
                graph=self.memory,
                format=self.context_format,
                max_nodes=self.max_graph_nodes,
                max_edges=100,
                include_metadata=True,
            )

        if entities:
            entity_ids = [e.id for e in entities if e.id]
            subgraph = self._extract_subgraph(entity_ids)
            return GraphRepresentation.to_llm_context(
                graph=subgraph,
                format=self.context_format,
                max_nodes=20,
                max_edges=50,
                include_metadata=True,
            )

        return GraphRepresentation.to_llm_context(
            graph=self.memory,
            format="linearized",
            max_nodes=20,
            max_edges=30,
            include_metadata=False,
        )

    def _extract_subgraph(self, entity_ids: list[str]) -> MemoryGraph:
        """Extract a subgraph centered on the given entities."""
        nodes = set()

        for eid in entity_ids:
            if self.memory.has_node(eid):
                nodes.add(eid)
                neighbors = self.memory.get_neighbors(eid)
                for neighbor in neighbors[: self.max_edges_per_node]:
                    nodes.add(neighbor.get_id())

        subgraph = MemoryGraph({"type": "subgraph", "source": "retrieval"})

        for nid in nodes:
            if self.memory.has_node(nid):
                subgraph.add_node(self.memory.get_node(nid))

        for edge in self.memory.get_edges():
            if edge.get_type() == EdgeType.SYMMETRIC:
                conn = set(edge.get_connections())
                if conn.issubset(nodes):
                    subgraph.add_edge(edge)
            else:
                source, target = edge.get_connections()
                if source in nodes and target in nodes:
                    subgraph.add_edge(edge)

        return subgraph

    # Step 3: CONSOLIDATE (Delta)
    def _consolidate(
        self, comprehension: ComprehensionResult, context: str
    ) -> KnowledgeDelta:
        """Consolidate new information into a knowledge delta."""
        try:
            # Prepare primitives for the LLM
            primitives = {
                "entities": [e.to_dict() for e in comprehension.entities],
                "relations": [r.to_dict() for r in comprehension.relations],
            }

            logger.info(
                f"[{self._CLSNAME}] Consolidating with {len(primitives['entities'])} entities, "
                f"{len(primitives['relations'])} relations"
            )

            messages = [
                {
                    "role": "user",
                    "content": Prompts.consolidate(
                        context=context,
                        new_primitives=json.dumps(primitives, indent=2),
                        query=comprehension.reconstructed_query,
                    ),
                }
            ]
            result = self.llm.get_structured(
                messages=messages,
                model=ConsolidationSchema,
            )

            if result is None:
                logger.warning("Consolidation returned None, using empty delta")
                return KnowledgeDelta.empty()

            # Convert schema nodes/edges to graph nodes/edges
            delta = KnowledgeDelta(
                new_nodes=[Node.from_json(n.model_dump()) for n in result.new_nodes],
                new_edges=[Edge.from_json(e.model_dump()) for e in result.new_edges],
                modified_nodes=[
                    Node.from_json(n.model_dump()) for n in result.modified_nodes
                ],
                modified_edges=[
                    Edge.from_json(e.model_dump()) for e in result.modified_edges
                ],
                conflicts=result.conflicts,
                gaps=result.gaps,
                confidence=Confidence(
                    score=result.confidence, source="llm_consolidation"
                ),
                source="llm_consolidation",
            )

            logger.info(f"[{self._CLSNAME}] Delta built successfully")
            return delta

        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
            return KnowledgeDelta.empty()

    # Step 4: REASON (Graph-Augmented)
    def _cogito_reason(
        self, query: str, context: str, delta: KnowledgeDelta
    ) -> ReasoningResult:
        """Reason using the consolidated graph context."""
        try:
            # Call tools to fill gaps if needed
            tool_results = None
            if delta.gaps and self.tools:
                tool_results = self._call_tools(delta.gaps)

            prompt = Prompts.reason(
                query=query,
                context=context,
                gaps=json.dumps(delta.gaps, indent=2),
                tool_results=(
                    json.dumps(tool_results, indent=2) if tool_results else None
                ),
            )

            logger.info(
                f"[{self._CLSNAME}] Sending reason prompt to LLM (length: {len(prompt)})"
            )

            # Reason step uses natural language, not structured output
            reasoning_text = self._llm_call(query=prompt)

            logger.info(
                f"[{self._CLSNAME}] Received reasoning response (length: {len(reasoning_text)})"
            )

            # Extract solution
            solution = self._extract_solution(reasoning_text)

            # Estimate confidence
            confidence_score = self._estimate_confidence(reasoning_text)
            confidence = Confidence(
                score=confidence_score,
                source="llm_reasoning",
                explanation="Estimated from reasoning text",
            )

            return ReasoningResult(
                reasoning_trace=reasoning_text,
                solution=solution,
                confidence=confidence,
                tool_results=tool_results,
            )

        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return ReasoningResult(
                reasoning_trace="",
                solution="I encountered an issue while reasoning. Could you rephrase your question?",
                confidence=Confidence(score=0.3, source="error_fallback"),
            )

    # Step 5: UPDATE (Automatic)
    def _update_memory(self, delta: KnowledgeDelta) -> None:
        """Update the memory graph with the knowledge delta."""
        if delta.confidence.score < self.confidence_threshold:
            logger.debug(
                f"Skipping update: confidence {delta.confidence.score} < {self.confidence_threshold}"
            )
            return

        logger.info(
            f"[{self._CLSNAME}] Updating memory with {len(delta.new_nodes)} new nodes, "
            f"{len(delta.new_edges)} new edges"
        )

        added_nodes = 0
        added_edges = 0

        # Add new nodes
        for node in delta.new_nodes:
            if not self.memory.has_node(node.get_id()):
                try:
                    self.memory.add_node(node)
                    added_nodes += 1
                except Exception as e:
                    logger.error(f"Failed to add node {node.get_id()}: {e}")

        # Add new edges
        for edge in delta.new_edges:
            if not self.memory.has_edge(edge.get_id()):
                try:
                    self.memory.add_edge(edge)
                    added_edges += 1
                except Exception as e:
                    logger.error(f"Failed to add edge {edge.get_id()}: {e}")

        # Update modified nodes
        for node in delta.modified_nodes:
            if self.memory.has_node(node.get_id()):
                existing = self.memory.get_node(node.get_id())
                if node.get_label() != existing.get_label():
                    existing.set_label(node.get_label())
                for key, value in node.get_metadata().items():
                    existing.update_metadata(key, value)

        # Store conflicts
        for conflict in delta.conflicts:
            self._store_conflict(conflict)

        # Record snapshot
        if hasattr(self.state, "add_graph_snapshot"):
            self.state.add_graph_snapshot(
                len(self.memory.get_nodes()),
                len(self.memory.get_edges()),
            )

        logger.info(
            f"[{self._CLSNAME}] Memory updated: +{added_nodes} nodes, +{added_edges} edges"
        )

    # Step 6: RESPOND (Synthesized)
    def _respond(self, reasoning: ReasoningResult, context: str) -> str:
        """Synthesize a response from the reasoning result."""
        try:
            prompt = Prompts.synthesize_response(
                reasoning=reasoning.reasoning_trace[:1000],
                solution=reasoning.solution,
                confidence=reasoning.confidence.score,
                graph_context=context[:1000],
            )
            return self._llm_call(query=prompt)
        except Exception as e:
            logger.error(f"Response synthesis failed: {e}")
            return reasoning.solution

    # HELPERS
    def _store_conflict(self, conflict: dict[str, Any]) -> None:
        """Store a conflict in the graph for later resolution."""
        node = Node(
            id=f"conflict_{datetime.now().timestamp()}",
            label="Conflict",
            metadata={
                "type": "conflict",
                "data": conflict,
                "timestamp": datetime.now().isoformat(),
            },
        )
        self.memory.add_node(node)

    def _extract_solution(self, text: str) -> str:
        """Extract solution from reasoning text."""
        if "Solution:" in text:
            parts = text.split("Solution:", 1)
            return parts[1].strip() if len(parts) > 1 else text
        return text

    def _estimate_confidence(self, text: str) -> float:
        """Estimate confidence from reasoning text."""
        uncertainty_markers = [
            "not sure",
            "I think",
            "maybe",
            "possibly",
            "could be",
            "might",
            "uncertain",
            "guess",
        ]
        uncertainty_count = sum(1 for m in uncertainty_markers if m in text.lower())
        confidence = max(0.3, 0.9 - (uncertainty_count * 0.1))
        return min(1.0, confidence)

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

    # LLM Helpers
    def _llm_call(
        self,
        messages: list[dict[str, str]] | None = None,
        query: str | None = None,
    ) -> str:
        """Call the LLM with either messages or a query string."""
        if messages is not None:
            msgs = messages
        elif query is not None:
            msgs = [
                {
                    "role": "system",
                    "content": "You are the Cogito Agent, a reasoning agent with symbolic memory.",
                },
                {"role": "user", "content": query},
            ]
        else:
            raise ValueError("Either 'messages' or 'query' must be provided")

        if not msgs or not isinstance(msgs, list):
            raise ValueError(f"Invalid messages: {msgs}")

        response = self.llm.chat(messages=msgs)
        self.llm_calls += 1
        return response

    # SERIALIZATION
    def save_memory(self, filepath: str) -> None:
        """Save memory graph to file."""
        with open(filepath, "w") as f:
            json.dump(self.memory.to_json(), f, indent=2)
        logger.info(f"[{self._CLSNAME}] Memory saved to {filepath}")

    def load_memory(self, filepath: str) -> None:
        """Load memory graph from file."""
        with open(filepath, "r") as f:
            data = json.load(f)
            self.memory = MemoryGraph.from_json(data)
        logger.info(f"[{self._CLSNAME}] Memory loaded from {filepath}")

    def save_state(self, filepath: str) -> None:
        """Save agent state to file."""
        with open(filepath, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)
        logger.info(f"[{self._CLSNAME}] State saved to {filepath}")

    def load_state(self, filepath: str) -> None:
        """Load agent state from file."""
        with open(filepath, "r") as f:
            data = json.load(f)
            self.state = CogitoState.from_dict(data)
        logger.info(f"[{self._CLSNAME}] State loaded from {filepath}")

    # LOGGING
    def _log_init(self) -> None:
        """Log initialization details."""
        logger.info(f"[{self._CLSNAME}] Initialized")
        logger.info(
            f"  Memory: {len(self.memory.get_nodes())} nodes, "
            f"{len(self.memory.get_edges())} edges"
        )
        logger.info(f"  Confidence threshold: {self.confidence_threshold}")
        logger.info(f"  Context format: {self.context_format}")
        logger.info(f"  Tools: {list(self.tools.keys()) if self.tools else 'None'}")
