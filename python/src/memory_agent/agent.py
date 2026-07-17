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
class ConsolidationResult:
    """Result of the consolidation step."""

    current_knowledge: str
    new_information: list[Primitive]
    gaps: list[str]
    updates: list[dict[str, Any]]
    confidence_adjustments: list[dict[str, Any]]


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


class CogitoAgent:
    """
    Cogito Agent - A symbolic memory-augmented agent.

    The agent thinks through the graph, not just the prompt.
    Memory is intrinsic, not optional.
    """

    def __init__(
        self,
        llm: LLMInterface,
        memory_graph: MemoryGraph | None = None,
        tools: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the Cogito Agent.

        Args:
            llm: LLM interface for reasoning and generation
            memory_graph: Optional existing memory graph (creates new if None)
            tools: Dictionary of available tools
            config: Configuration options
        """
        self.llm = llm
        self.memory = memory_graph or MemoryGraph({"type": "cogito_agent"})
        self.tools = tools or {}
        self.config = config or {}

        # State tracking
        self.conversation_history: list[dict[str, str]] = []
        self.context_window: list[str] = []
        self.confidence_threshold = self.config.get("confidence_threshold", 0.6)
        self.max_context_nodes = self.config.get("max_context_nodes", 100)
        self.max_graph_depth = self.config.get("max_graph_depth", 3)

        logger.info("[CogitoAgent] Cogito Agent initialized")
        logger.info(f"[CogitoAgent] Memory node: {len(self.memory.get_nodes())}")
        logger.info(f"[CogitoAgent] Memory edges: {len(self.memory.get_edges())}")
        logger.info(
            f"[CogitoAgent] Tools: {list(self.tools.keys()) if self.tools else 'None'}"
        )

    def process_query(self, user_query: str) -> str:
        """
        Process a user query through the Cogito loop.

        Args:
            user_query: The user's input

        Returns:
            The agent's response
        """

        logger.info(
            f"[CogitoAgent:process_query] Processing query:\n---{user_query[:100]}\n---"
        )

        # Step 1: COMPREHEND
        comprehended = self._comprehend(user_query)
        logger.info(f"[CogitoAgent] Path: {comprehended['path']}")

        if comprehended["path"] == PathType.REACT:
            return self._react_loop(user_query, comprehended)

        # Step 2: RETRIEVE
        context = self._retrieve(comprehended["primitives"])
        logger.info(f"[CogitoAgent] Retrieved: {len(context)} characters of context")

        # Step 3: CONSOLIDATE
        consolidated = self._consolidate(comprehended["primitives"], context)
        logger.info(
            f"[CogitoAgent] New info: {len(consolidated.new_information)} items"
        )
        logger.info(f"[CogitoAgent] Gaps: {len(consolidated.gaps)} items")

        # Step 4: REASON
        reasoning_result = self._reason(
            query=user_query,
            context=context,
            consolidated=consolidated,
            tool_results=None,  # Will be filled if tools are called
        )
        logger.info(
            f"[CogitoAgent] Reasoning confidence: {reasoning_result.confidence}"
        )

        # Step 5: Update
        if consolidated.updates:
            self._update_memory(consolidated)
            logger.info(
                f"[CogitoAgent] Memory updated: {len(consolidated.updates)} items"
            )

        # Step 6: RESPOND
        response = self._respond(reasoning_result, context)
        logger.info(f"[CogitoAgent] Response length: {len(response)} characters")

        # Update conversation history
        self.conversation_history.append(
            {
                "user": user_query,
                "assistant": response,
                "timestamp": datetime.now().isoformat(),
                "path": comprehended["path"].value,
            }
        )

        return response

    def _comprehend(self, query: str):
        """Comprehend the user query."""
        try:
            # 1. Reconstruct query
            reconstructed = self._llm_call(Prompts.reconstruct_query(query=query))

            # 2. Extract primitives
            primitives_json = self._llm_structured(
                Prompts.extract_primitives(query=query)
            )
            primitives = self._parse_primitives(primitives_json)

            # 3. Classify complexity
            complexity = self._llm_structured(
                Prompts.classify_complexity(
                    primitives=json.dumps(primitives), query=query
                )
            )

            # Determine path
            if complexity.get("complexity") == "complex" or primitives.get(
                "intent"
            ) in ["task"]:
                path = PathType.COGITO
            else:
                path = PathType.REACT

            return {
                "path": path,
                "primitives": primitives,
                "intent": primitives.get("intent"),
                "reconstructed": reconstructed,
                "complexity": complexity,
            }

        except Exception as e:
            logger.error(f"[CogitoAgent] Comprehension failed: {e}")
            # Fallback to simple ReAct
            return {"path": PathType.REACT, "primitives": {}}

    # Step 2: RETRIEVE
    def _retrieve(self, primitives: dict[str, Any]) -> str:
        """Retrieve relevant context from the memory graph"""
        entities = primitives.get("entities", [])
        keywords = self._extract_keywords(entities)

        # Build context
        context_parts = []

        # Search for relevant nodes
        visited = set()
        for entity in entities:
            entity_id = entity.get("id")
            if not entity_id:
                continue
            if entity_id in visited:
                continue
            visited.add(entity_id)

            # Get node and neighbors
            if self.memory.has_node(node_id=entity_id):
                node = self.memory.get_node(node_id=entity_id)
                neighbors = self.memory.get_neighbors(node_id=entity_id)

                # Add node description
                context_parts.append(
                    f"Entity: {node.get_id()} (label: {node.get_label()})"
                )
                context_parts.append(f"--Metadata: {json.dumps(node.get_metadata())}")

                # Add connections
                if neighbors:
                    neighbor_desc = ", ".join([n.get_label() for n in neighbors[:5]])
                    context_parts.append(f"--Connections: {neighbor_desc}")

                # Add edges
                edges = [e for e in self.memory.get_edges() if entity_id in e.get_id()]
                if edges:
                    edge_desc = ", ".join([e.get_label() for e in edges[:3]])
                    context_parts.append(f"--Relations: {edge_desc}")

        # Add group memberships
        group_edges = self.memory.get_group_edges()
        for edge in group_edges:
            if edge.is_group_edge():
                members = self.memory.get_group_members(group_id=edge.get_id())
                context_parts.append(
                    f"Group: {edge.get_label()} (members: {', '.join(members)}"
                )

        # Add full graph if manageable
        node_count = len(self.memory.get_nodes())  # this can be pre-computed
        if node_count < 50:
            context_parts.append("\nFull graph summary:")
            context_parts.append(f"--Total nodes: {node_count}")
            context_parts.append(f"--Total edges: {len(self.memory.get_edges())}")

        # Also add fuzzy search results
        # TODO: [PRIORITY:High] Verify whether we can implement fuzzy retrieval on the graph level.
        if keywords:
            fuzzy_results = self.fuzzy_search(keywords)
            if fuzzy_results:
                context_parts.append("\nFuzzy search results.")
                context_parts.extend(fuzzy_results)

        return (
            "\n".join(context_parts) if context_parts else "No relevant context found."
        )

    # Step 3: CONSOLIDATE
    def _consolidate(
        self, primitives: dict[str, Any], context: str
    ) -> ConsolidationResult:
        """Consolidate new information with existing knowledge"""
        try:
            # Analyze and consolidate
            result = self._llm_structured(
                prompt=Prompts.consolidate(
                    context=context,
                    new_primitives=json.dumps(primitives),
                    query=context[:500],  # Context as a query
                )
            )

            # Parse the result
            new_info = []
            for item in result.get("new_information", []):
                new_info.append(
                    Primitive(
                        primitive_type=item.get("type", "entity"),
                        data=item.get("data", {}),
                        confidence=item.get("confidence", 0.5),
                        source="llm_consolidation",
                    )
                )

            return ConsolidationResult(
                current_knowledge=result.get("knowledge_summary", ""),
                new_information=new_info,
                gaps=result.get("gaps", []),
                updates=result.get("updates", []),
                confidence_adjustments=result.get("confidence_adjustments", []),
            )

        except Exception as e:
            logger.error(f"[CogitoAgent] Consolidation failed:\n---\n{e}\n---")
            return ConsolidationResult(
                current_knowledge="",
                new_information=[],
                gaps=[],
                updates=[],
                confidence_adjustments=[],
            )

    # Step 4: REASON
    def _reason(
        self,
        query: str,
        context: str,
        consolidated: ConsolidationResult,
        tool_results: list[dict[str, Any]] | None = None,
    ) -> ReasoningResult:
        """Reason using the consolidated graph."""
        try:
            # Fill gaps using tools if needed
            tool_output = None
            if consolidated.gaps and self.tools:
                tool_output = self._call_tools(consolidated.gaps)
                tool_results = tool_output

            # Generate reasoning
            reasoning_text = self._llm_call(
                prompt=Prompts.reason(
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
            logger.error(f"[CogitoAgent] Reasoning failed:\n---\n{e}\n---")
            return ReasoningResult(
                reasoning_trace="",
                solution="I encountered an issue while reasoning. Could you rephrase your question?",
                confidence=0.3,
            )

    # Step 5: UPDATE
    def _update_memory(self, consolidated: ConsolidationResult):
        """Update the memory graph with new information"""
        for update in consolidated.updates:
            update_type = update.get("type")
            data = update.get("data", {})
            confidence = data.get("confidence", 0.5)

            # Skip low confidence updates
            if confidence < self.confidence_threshold:
                logger.debug(
                    f"[CogitoAgent] Skipping update with low confidence: {confidence}"
                )
                continue

            try:
                if update_type == "add" or update_type == "modify":
                    self.apply_update(data)
                elif update_type == "conflict":
                    self._handle_conflict(data)

            except Exception as e:
                logger.error(f"[CogitoAgent] Failed to appy update: {e}")
        pass

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
            logger.error(f"Response synthesis failed: {e}")
            return reasoning_result.solution

    # ReAct Loop (Fallback)
    # TODO: Use the memory agent project ccode
    def _react_loop(self, query: str, comprehended: dict[str, Any]) -> str:
        """Simple ReAct loop for straightforward tasks."""
        # Simple ReAct: reason → act → observe → respond
        try:
            # Think
            thought = self._llm_call(f"Think about this query: {query}")

            # Check if tools are needed
            if self.tools and "tool" in thought.lower():
                tool_result = self._call_tools([query])
                # Observe
                observation = self._llm_call(
                    f"Tool result: {tool_result}\nQuery: {query}"
                )
            else:
                observation = thought

            # Respond
            response = self._llm_call(f"Respond to: {query}\nBased on: {observation}")
            return response.strip()

        except Exception as e:
            logger.error(f"ReAct loop failed: {e}")
            return (
                "I encountered an issue processing your request. Could you rephrase it?"
            )

    # LLM Helpers
    def _llm_call(self, prompt: str) -> str:
        """Call the LLM with a prompt."""
        messages = [
            {
                "role": "system",
                "content": "You are the Cogito Agent, a reasoning agent with symbolic memory.",
            },
            {"role": "user", "content": prompt},
        ]
        return self.llm.chat(messages)

    def _llm_structured(self, prompt: str) -> dict[str, Any]:
        """Call the LLM and parse structured output."""
        messages = [
            {
                "role": "system",
                "content": "You are the Cogito Agent, a reasoning agent with symbolic memory. Return valid JSON.",
            },
            {"role": "user", "content": prompt},
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

    def _parse_primitives(self, data: dict[str, Any]) -> dict[str, Any]:
        """Parse extracted primitives"""
        return {
            "entities": data.get("entities", []),
            "relations": data.get("relations", []),
            "attributes": data.get("attributes", []),
            "intent": data.get("intent", "ask"),
        }

    # TODO: This can be improved?
    def _extract_keywords(self, entities: list[dict[str, Any]]) -> list[str]:
        """Extract keywords from entities for fuzzy search."""
        keywords = []
        for entity in entities:
            keywords.append(entity.get("id", ""))
            keywords.append(entity.get("label", ""))

        return [k for k in keywords if k and len(k) > 2]

    # TODO: We can use RapidFuzz for the fuzzy search
    def fuzzy_search(self, keywords: list[str]) -> list[str]:
        """Perform fuzzy search for nodes matching keywords."""
        results = []
        all_nodes = self.memory.get_nodes()
        for node in all_nodes:
            node_id = node.get_id()
            label = node.get_label()
            for kw in keywords:
                if kw.lower() in node_id.lower() or kw.lower() in label.lower():
                    results.append(f"--Found: {node_id} ({label})")
                    break

        return results[:10]  # truncated for now

    def _call_tools(self, gaps: list[str]) -> list[dict[str, Any]]:
        """Call tools to fill gaps."""
        results = []
        for gap in gaps:
            # Simple tool routing based on gap description (can be enhanced)
            for tool_name, tool_func in self.tools.items():
                if tool_name in gap.lower():
                    try:
                        result = tool_func(gap)
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

    # TODO: Use structured output for extraction or special markers
    def _extract_solution(self, reasoning_text: str) -> str:
        """Extract solution from reasoning text."""
        if "solutin:" in reasoning_text:
            parts = reasoning_text.split("Solution:")
            return parts[-1].strip() if len(parts) > 1 else reasoning_text
        return reasoning_text

    def _estimate_confidence(
        self, reasoning_text: str, consolidated: ConsolidationResult
    ) -> float:
        """Estrimate confidence in the reasoning"""
        # Simple heuristic to check for uncertainty markers (for now)
        uncertainty_markers = [
            "I am not sure",
            "I think",
            "maybe",
            "possibility",
            "could be",
        ]
        uncertainty_count = sum(
            1 for m in uncertainty_markers if m in reasoning_text.lower()
        )

        # Higher confidence with fewer uncertainty markers
        base_confidence = 0.9
        confidence = max(0.3, base_confidence - (uncertainty_count * 0.1))

        # Adjust based on consolidated confidence
        if consolidated.new_information:
            avg_confidence = sum(
                i.confidence for i in consolidated.new_information
            ) / len(consolidated.new_information)
            confidence = (confidence + avg_confidence) / 2

        return min(1.0, confidence)

    def apply_update(self, data: dict[str, Any]):
        """Apply a single update to the graph"""
        primitive_type = data.get("type")

        if primitive_type == "entity":
            # Add or update node
            entity_id = data.get("id", "")
            label = data.get("label", entity_id)
            metadata = data.get("metadata", {})

            if self.memory.has_node(node_id=entity_id):
                # Update existing node
                node = self.memory.get_node(node_id=entity_id)
                # Update metadata
                node.update_metadata("updated_at", datetime.now().isoformat())
                for key, value in metadata.items:
                    node.update_metadata(key, value)
                node.set_label(label=label)

            else:
                # Create new node
                node = Node(id=entity_id, label=label, metadata=metadata)
                self.memory.add_node(node=node)

        elif primitive_type == "relation":
            # Add edge
            source = data.get("source", "")
            target = data.get("target", "")
            label = data.get("label", "")
            direction = data.get("direction", "asymmetric")
            weight = data.get("weight", 0.8)
            metadata = data.get("metadata", {})

            if direction == "symmetric":
                # Use group edge for symmetric (will group edge work for bidirectional relations)
                self.memory.add_group_edge(
                    id=f"{source}_{target}",
                    label=label,
                    node_ids={source, target},
                    weight=weight,
                    metadata=metadata,
                )
            else:
                # Use symmetric edge with exactly 2 nodes
                from memory_graph import EdgeType

                conn = AsymmetricConnections((source, target))
                edge = Edge(
                    id=f"{source}_{target}",
                    label=label,
                    type=EdgeType.ASYMMETRIC,
                    connections=conn,
                    weight=weight,
                    metadata=metadata,
                )
                self.memory.add_edge(edge)

            logger.info(f"  Updated relation: {source} → {target} ({label})")

    def _handle_conflict(self, data: dict[str, Any]):
        """Handle a conflict in the graph."""
        existing = data.get("existing", {})
        new = data.get("new", {})

        # Log conflict for review
        logger.warning("Conflict detected:")
        logger.warning(f"  Existing: {existing}")
        logger.warning(f"  New: {new}")

        # Store conflict in graph metadata for later resolution
        conflict_node = Node(
            id=f"conflict_{datetime.now().timestamp()}",
            label="Conflict",
            metadata={
                "type": "conflict",
                "existing": existing,
                "new": new,
                "timestamp": datetime.now().isoformat(),
            },
        )
        self.memory.add_node(conflict_node)

    # Utilities
    def save_memory(self, filepath: str):
        """Save the memory graph to a file."""
        with open(filepath, "w") as f:
            json.dump(self.memory.to_json(), f, indent=2)
        logger.info(f"Memory saved to {filepath}")

    def load_memory(self, filepath: str):
        """Load a memory graph from a file."""
        with open(filepath, "r") as f:
            data = json.load(f)
            self.memory = MemoryGraph.from_json(data)
        logger.info(f"Memory loaded from {filepath}")
