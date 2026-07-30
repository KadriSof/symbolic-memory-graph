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
from typing import Any, Dict, List, Optional

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

# ✅ Import from core
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
    """

    def __init__(
        self,
        llm: BaseLLM,
        memory_graph: Optional[MemoryGraph] = None,
        tools: Optional[List[Tool]] = None,
        config: Optional[Dict[str, Any]] = None,
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
        (Overrides the path for a graph-aware comprehension)
        """
        try:
            result = self._llm_structured(Prompts.comprehend(query))

            # Parse entities
            entities = []
            for e in result.get("entities", []):
                entities.append(
                    Entity(
                        id=e.get("id", ""),
                        label=e.get("label", ""),
                        type=e.get("type"),
                        metadata=e.get("metadata", {}),
                        confidence=Confidence(
                            score=result.get("confidence", 0.7),
                            source="llm_comprehension",
                        ),
                    )
                )

            # Parse relations
            relations = []
            for r in result.get("relations", []):
                relations.append(
                    Relation(
                        source=r.get("source", ""),
                        target=r.get("target", ""),
                        label=r.get("label", ""),
                        direction=r.get("direction", "asymmetric"),
                        weight=r.get("weight", 1.0),
                        confidence=Confidence(
                            score=result.get("confidence", 0.7),
                            source="llm_comprehension",
                        ),
                    )
                )

            # 2. Get the path from the LLM output
            path = result.get("modus_operandi", "REACT")

            logger.info(
                f"[{self._CLSNAME}:_comprehend] Running Graph-Aware verification.."
            )
            # 3. Graph-Aware Override: Check if the query references entities in the graph
            graph_has_info = self._graph_has_info(entities, relations)
            logger.info(f"[{self._CLSNAME}:_comprehend] Graph-Aware verification done!")

            if graph_has_info and path == "REACT":
                logger.info(f"[{self._CLSNAME}] Graph overrid: REACT -> COGITO")
                path = "COGITO"
            else:
                logger.info(f"[{self._CLSNAME}] No graph info found, using REACT")
                path = "REACT"

            # Update state with context and entities
            if hasattr(self.state, "set_context"):
                self.state.set_context(result.get("active_context", ""))

            if hasattr(self.state, "add_session_entity"):
                for entity in entities:
                    self.state.add_session_entity(entity.id)

            return ComprehensionResult(
                reconstructed_query=result.get("reconstructed_query", query),
                intent=result.get("user_intent", "ask"),
                path=path,
                context=result.get("active_context", ""),
                entities=entities,
                relations=relations,
                confidence=Confidence(
                    score=result.get("confidence", 0.7), source="llm_comprehension"
                ),
            )

        except Exception as e:
            logger.error(f"Comprehension failed: {e}")
            return ComprehensionResult.fallback(query)

    def _graph_has_info(
        self, entities: List[Entity], relations: List[Relation]
    ) -> bool:
        """
        Check if the graph contains information relevant to the query.
        """
        # Simple and safe: just check if any entity exists in the graph
        for entity in entities:
            if entity and entity.id:
                if self.memory.has_node(entity.id):
                    return True

        # Check if any relation source/target exists
        for relation in relations:
            if relation:
                if relation.source and self.memory.has_node(relation.source):
                    return True
                if relation.target and self.memory.has_node(relation.target):
                    return True

        return False

    # Step 2: RETRIEVE (Graph-First)
    def _retrieve(self, entities: List[Entity]) -> str:
        """
        Retrieve graph context optimized for LLM consumption.

        Strategy:
        - Small graph (<50 nodes): Full graph representation
        - Large graph with entities: Subgraph extraction
        - Large graph without entities: Compact summary
        """
        node_count = len(self.memory.get_nodes())

        if node_count == 0:
            return "The memory graph is empty. No previous knowledge available."

        # Small graph: Full representation
        if node_count < self.max_graph_nodes:
            return GraphRepresentation.to_llm_context(
                graph=self.memory,
                format=self.context_format,
                max_nodes=self.max_graph_nodes,
                max_edges=100,
                include_metadata=True,
            )

        # Large graph with entities: Subgraph extraction
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

        # Large graph without entities: Compact summary
        return GraphRepresentation.to_llm_context(
            graph=self.memory,
            format="linearized",
            max_nodes=20,
            max_edges=30,
            include_metadata=False,
        )

    def _extract_subgraph(self, entity_ids: List[str]) -> MemoryGraph:
        """Extract a subgraph centered on the given entities."""
        nodes = set()

        # Collect entities and their neighbors
        for eid in entity_ids:
            if self.memory.has_node(eid):
                nodes.add(eid)
                # Add neighbors (depth 1)
                neighbors = self.memory.get_neighbors(eid)
                for neighbor in neighbors[: self.max_edges_per_node]:
                    nodes.add(neighbor.get_id())

        # Build subgraph
        subgraph = MemoryGraph({"type": "subgraph", "source": "retrieval"})

        # Add nodes
        for nid in nodes:
            if self.memory.has_node(nid):
                subgraph.add_node(self.memory.get_node(nid))

        # Add edges between nodes in subgraph
        for edge in self.memory.get_edges():
            if edge.get_type() == EdgeType.SYMMETRIC:
                conn = set(edge.get_connections())
                if conn.issubset(nodes):
                    subgraph.add_edge(edge)
            else:
                conn = edge.get_connections()
                if conn[0] in nodes and conn[1] in nodes:
                    subgraph.add_edge(edge)

        return subgraph

    # Step 3: CONSOLIDATE (Delta)
    def _consolidate(
        self, comprehension: ComprehensionResult, context: str
    ) -> KnowledgeDelta:
        """
        Consolidate new information into a knowledge delta.
        """
        try:
            # Prepare primitives for the LLM
            primitives = {
                "entities": [e.to_dict() for e in comprehension.entities],
                "relations": [r.to_dict() for r in comprehension.relations],
            }

            logger.info(
                f"[{self._CLSNAME}] Consolidating with {len(primitives['entities'])} entities, {len(primitives['relations'])} relations"
            )

            result = self._llm_structured(
                Prompts.consolidate(
                    context=context,
                    new_primitives=json.dumps(primitives, indent=2),
                    query=comprehension.reconstructed_query,
                )
            )

            # ✅ LOG: Log the raw result keys
            logger.info(
                f"[{self._CLSNAME}] Consolidation result keys: {list(result.keys())}"
            )

            # Parse confidence
            confidence_score = result.get("confidence", 0.7)
            confidence = Confidence(score=confidence_score, source="llm_consolidation")

            # ✅ LOG: Log what we're about to parse
            new_nodes_data = result.get("new_nodes", [])
            safe_new_nodes = []
            for n in new_nodes_data:
                if isinstance(n, dict):
                    if n.get("metadata") is None:
                        n["metadata"] = {}
                    safe_new_nodes.append(n)
                else:
                    logger.warning(f"Skipping invalid node: {n}")

            new_edges_data = result.get("new_edges", [])
            safe_new_edges = []
            for e in new_edges_data:
                if isinstance(e, dict):
                    if e.get("metadata") is None:
                        e["metadata"] = {}
                    safe_new_edges.append(e)
                else:
                    logger.warning(f"Skipping invalid edge: {e}")  # Build delta

            delta = KnowledgeDelta(
                new_nodes=[Node.from_json(n) for n in new_nodes_data],
                new_edges=[Edge.from_json(e) for e in new_edges_data],
                modified_nodes=[
                    Node.from_json(n) for n in result.get("modified_nodes", [])
                ],
                modified_edges=[
                    Edge.from_json(e) for e in result.get("modified_edges", [])
                ],
                conflicts=result.get("conflicts", []),
                gaps=result.get("gaps", []),
                confidence=confidence,
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
        """
        Reason using the consolidated graph context.
        """
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
            reasoning_text = self._llm_call(query=prompt)
            logger.info(
                f"[{self._CLSNAME}] Received reasoning response (length: {len(reasoning_text)})"
            )

            # ✅ LOG: Log first 200 chars of response
            logger.info(
                f"[{self._CLSNAME}] Reasoning response preview: {reasoning_text[:200]}..."
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
            f"[{self._CLSNAME}] Updating memory with {len(delta.new_nodes)} new nodes, {len(delta.new_edges)} new edges"
        )

        # ✅ LOG: Log the first new node to see what we're dealing with
        if delta.new_nodes:
            first_node = delta.new_nodes[0]
            logger.info(
                f"[{self._CLSNAME}] First new node: id={first_node.get_id()}, label={first_node.get_label()}, metadata={first_node.get_metadata()}"
            )

        if delta.new_edges:
            first_edge = delta.new_edges[0]
            logger.info(
                f"[{self._CLSNAME}] First new edge: id={first_edge.get_id()}, label={first_edge.get_label()}"
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
    def _store_conflict(self, conflict: Dict[str, Any]) -> None:
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

    def _call_tools(self, gaps: List[str]) -> List[Dict[str, Any]]:
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
        messages: Optional[List[Dict[str, str]]] = None,
        query: Optional[str] = None,
    ) -> str:
        """
        Call the LLM with either messages or a query string.
        """
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

    def _llm_structured(self, query: str) -> Dict[str, Any]:
        """Call the LLM and parse structured output."""
        messages = [
            {
                "role": "system",
                "content": "You are the Cogito Agent. Return valid JSON only.",
            },
            {"role": "user", "content": query},
        ]
        response = self._llm_call(messages=messages)

        # ✅ LOG: Log the raw response
        logger.info(
            f"[{self._CLSNAME}] Structured response (first 300 chars): {response[:300]}..."
        )

        if not response:
            logger.warning(f"[{self._CLSNAME}] Empty response from LLM")
            return {"raw": ""}

        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                result = json.loads(json_str)
                if isinstance(result, dict):
                    # ✅ LOG: Successfully parsed JSON
                    logger.info(
                        f"[{self._CLSNAME}] Successfully parsed JSON with keys: {list(result.keys())}"
                    )
                    return result
            logger.warning(f"[{self._CLSNAME}] No JSON object found in response")
            return {"raw": response}
        except json.JSONDecodeError as e:
            logger.error(f"[{self._CLSNAME}] JSON decode error: {e}")
            return {"raw": response}

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
            f"  Memory: {len(self.memory.get_nodes())} nodes, {len(self.memory.get_edges())} edges"
        )
        logger.info(f"  Confidence threshold: {self.confidence_threshold}")
        logger.info(f"  Context format: {self.context_format}")
        logger.info(f"  Tools: {list(self.tools.keys()) if self.tools else 'None'}")

    def _safe_json_dump(data, max_len=500):
        """Safely dump JSON data for debugging."""
        try:
            if isinstance(data, dict):
                return json.dumps(data, indent=2)[:max_len]
            return str(data)[:max_len]
        except Exception:
            return "Unable to serialize"
