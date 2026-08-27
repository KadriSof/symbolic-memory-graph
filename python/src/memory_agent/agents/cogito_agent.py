"""
Cogito Core Data Structures - Mecha-Class Agent Foundation

Defines the essential data structures for the Cogito Agent system.
Implements a clean, type-safe, and robust workflow orchestration pattern.
"""

from __future__ import annotations

import time

from dataclasses import dataclass, field

from typing import Any

from ..core import Prompts
from ..core import ComprehensionResult
from ..core import ComprehensionSchema

from .react_agent import ReactState, ReactAgent
from ..core import WorkingMemory, SymbolicMemory, EpisodicMemory
from ..core import WorkflowOrchestrator, CogitoStatus
from ..core import Entity

from memory_graph.core import MemoryGraph


# COGITO STATE
@dataclass
class CogitoState(ReactState):
    """
    Cogito-specific state extending ReactState.

    Manages the WorkingMemory scratchpad and holds base64-encoded
    binary snapshots of the C++ Graphs for safe, crash-free persistence.
    """

    working_memory: WorkingMemory = field(default_factory=WorkingMemory)

    symbolic_memory_b64: str = ""
    episodic_memory_b64: str = ""

    def reset(self, keep_metadata: bool = False) -> None:
        """Reset state and working memory, preserving graph snapshots if requested."""
        super().reset(keep_metadata=keep_metadata)
        self.working_memory.reset()
        if not keep_metadata:
            self.symbolic_memory_b64 = ""
            self.episodic_memory_b64 = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize state, safely handling the C++ graph via binary encoding."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "working_memory": self.working_memory.to_dict(),
                "symbolic_memory_b64": self.symbolic_memory_b64,
                "episodic_memory_b64": self.episodic_memory_b64,
            }
        )
        return base_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CogitoState":
        """Deserialize state."""
        state = super().from_dict(data)

        if not isinstance(state, CogitoState):
            raise TypeError(f"Expected {cls.__name__}, got {type(state).__name__}")

        state.working_memory = WorkingMemory.from_dict(data.get("working_memory", {}))
        state.symbolic_memory_b64 = data.get("symbolic_memory_b64", "")
        state.episodic_memory_b64 = data.get("episodic_memory_b64", "")

        return state


# COGITO AGENT
class CogitoAgent(ReactAgent):
    """COGITO: The Ultimate Mecha-Class Agent"""

    def __init__(
        self,
        llm: Any,
        tools: list | None = None,
        config: dict[str, Any] | None = None,
        state: CogitoState | None = None,
        knowledge_graph: MemoryGraph | None = None,
        episodic_memory: MemoryGraph | None = None,
    ) -> None:
        super().__init__(llm, tools, config, state)
        self.symbolic_memory = SymbolicMemory(graph=knowledge_graph)
        self.episodic_memory = EpisodicMemory(graph=episodic_memory)
        self.orchestrator = WorkflowOrchestrator(
            entry_node_id="comprehend", max_steps=50, max_history=50
        )

        self._build_workflow()

    def _initialize_state(self) -> CogitoState:
        return CogitoState()

    @property
    def state(self) -> CogitoState:
        return super().state  # type: ignore

    @classmethod
    def _state_class(cls) -> type[CogitoState]:
        return CogitoState

    # WORKFLOW DEFINITION (The Nervous System)
    def _build_workflow(self) -> None:
        orch = self.orchestrator

        # Decoupled Learning: Comprehend -> Learn -> Retrieve -> Reason
        orch.add_node(
            "comprehend",
            "Comprehend",
            "Extract goal, plan, entities, relations",
            self._comprehend_task,
        )
        orch.add_node(
            "learn",
            "Learn",
            "Consolidate new knowledge into symbolic memory",
            self._learn_task,
        )
        orch.add_node(
            "retrieve",
            "Retrieve",
            "Query knowledge and episodic memory (5-Tier)",
            self._retrieve_task,
        )
        orch.add_node(
            "reason",
            "Reason",
            "Generate reasoning using consolidated knowledge",
            self._reason_task,
        )
        orch.add_node(
            "tool_executor",
            "Tool Executor",
            "Execute external tools",
            self._tool_executor_task,
        )
        orch.add_node(
            "reply", "Reply", "Format final answer for the user", self._reply_task
        )

        # Linear progression for learning and retrieval
        orch.add_edge("comprehend", "learn")
        orch.add_edge("learn", "retrieve")
        orch.add_edge("retrieve", "reason")

        # Dynamic branching for reasoning
        orch.add_edge(
            "reason",
            "reply",
            condition=lambda wm: wm.reasoning_complete and not wm.needs_more_data,
        )
        orch.add_edge(
            "reason",
            "tool_executor",
            condition=lambda wm: wm.needs_more_data
            and bool(self.tools)
            and not wm.retrieved_knowledge.get("tool_attempted"),
        )
        orch.add_edge(
            "reason",
            "reply",
            condition=lambda wm: wm.needs_more_data
            and (not self.tools or wm.retrieved_knowledge.get("tool_attempted")),
        )

        # Cycle: Tool results feed back into reasoning
        orch.add_edge("tool_executor", "reason")

    # CORE EXECUTION OVERRIDE
    def _process_query_impl(self, query: str) -> str:
        wm = self.state.working_memory
        wm.reset()
        wm.goal = query

        self.episodic_memory.add_turn(
            role="user", content=query, metadata={"action": "query"}
        )

        ctx = {
            "agent": self,
            "symbolic_memory": self.symbolic_memory,
            "episodic_memory": self.episodic_memory,
            "llm": self.llm,
            "tools": self.tools,
        }

        start_time = time.perf_counter()
        self.orchestrator.status = CogitoStatus.COMPREHENDING

        try:
            # 1. Comprehend & Learn
            self._comprehend_task(wm, ctx)
            self._learn_task(wm, ctx)
            self._retrieve_task(wm, ctx)

            # 2. Dynamic Fast-Path Routing
            if wm.modus_operandi == "REACT":
                self.logger.info(
                    "[WORKFLOW] Routing to native ReAct loop for efficient tool use..."
                )
                self.orchestrator.status = CogitoStatus.IDLE

                # Inject retrieved knowledge into the query for the robust ReAct agent
                graph_context = wm.retrieved_knowledge.get(
                    "graph_context", "No additional context."
                )
                augmented_query = (
                    f"SYMBOLIC MEMORY CONTEXT:\n{graph_context}\n\nUSER QUERY: {query}"
                )

                # Leverage the battle-tested, built-in ReAct loop (instant tool parsing & execution)
                return super()._process_query_impl(augmented_query)

            # 3. COGITO Path: Run the orchestrator for complex, multi-step reasoning
            self.logger.info("[WORKFLOW] Entering COGITO reasoning workflow...")
            self.orchestrator.status = CogitoStatus.REASONING

            # Temporarily change entry node to 'reason' since we already did comprehend/learn/retrieve
            original_entry = self.orchestrator.entry_node_id
            self.orchestrator.entry_node_id = "reason"

            self.orchestrator.run(wm, ctx)

            # Restore original entry
            self.orchestrator.entry_node_id = original_entry

            duration = time.perf_counter() - start_time
            final_answer = (
                wm.final_answer or "I was unable to generate a complete answer."
            )
            self.orchestrator.record_execution(query, final_answer, duration)
            return final_answer

        except Exception as e:
            self.orchestrator.status = CogitoStatus.ERROR
            self.logger.error(f"[WORKFLOW] Cogito workflow failed: {e}", exc_info=True)
            if self.config.get("fallback_to_react", True):
                self.logger.info(
                    "[WORKFLOW] Falling back to standard ReAct processing..."
                )
                return super()._process_query_impl(query)
            raise
        finally:
            self.orchestrator.status = CogitoStatus.IDLE

    # PROCEDURE NODE TASKS (The Limbs)
    def _comprehend_task(self, wm: WorkingMemory, ctx: dict) -> None:
        self.logger.info("[COMPREHENDING] Comprehending query...")
        try:
            prompt = Prompts.comprehend(wm.goal)
            messages = [{"role": "user", "content": prompt}]
            schema_result = self.llm.get_structured(
                messages=messages, model=ComprehensionSchema
            )

            if schema_result is None:
                self.logger.warning(
                    "[COMPREHENDING] Comprehension returned None, using fallback"
                )
                result = ComprehensionResult.fallback(wm.goal)
            else:
                result = schema_result.to_comprehension_result()

            wm.goal = result.goal
            wm.plan = result.plan
            wm.extracted_entities = result.entities
            wm.extracted_relations = result.relations
            wm.modus_operandi = result.path

            # Graph-Aware Override: If entities exist in graph, force COGITO mode
            if result.is_react() and result.entities:
                sym_mem = ctx["symbolic_memory"]
                has_graph_info = any(
                    sym_mem.has_concept(entity.id) for entity in result.entities
                )
                if has_graph_info:
                    self.logger.info(
                        "[COMPREHENDING] Graph-Aware Override: REACT -> COGITO (Graph has relevant info)"
                    )
                    wm.modus_operandi = "COGITO"

        except Exception as e:
            self.logger.warning(
                f"[COMPREHENDING] Comprehension LLM failed," f"Error:\n----{e}\n----"
            )

            # Robust fallback
            fallback = ComprehensionResult.fallback(wm.goal)
            wm.goal = fallback.goal
            wm.plan = fallback.plan
            wm.modus_operandi = fallback.path
            wm.extracted_entities = fallback.entities
            wm.extracted_relations = fallback.relations

    def _learn_task(self, wm: WorkingMemory, ctx: dict) -> None:
        """Decoupled Learning: Update graph BEFORE answering to prevent amnesia."""
        self.logger.info("[CONSOLIDATING] Consolidating new knowledge...")
        sym_mem: SymbolicMemory = ctx["symbolic_memory"]
        added_nodes, added_edges = 0, 0

        for entity in wm.extracted_entities:
            node = entity.to_node()
            if not sym_mem.has_concept(node.get_id()):
                try:
                    sym_mem.graph.add_node(node)
                    added_nodes += 1
                except Exception as e:
                    self.logger.warning(
                        f"[CONSOLIDATING] Failed to add concept '{entity.label}',"
                        f"Error:\n----{e}\n----"
                    )

        for relation in wm.extracted_relations:
            edge = relation.to_edge()
            if not sym_mem.graph.has_edge(edge.get_id()):
                try:
                    sym_mem.graph.add_edge(edge)
                    added_edges += 1
                except Exception as e:
                    self.logger.warning(
                        f"[CONSOLIDATING] Failed to add link '{relation.source}' -> '{relation.target}',"
                        f"Error:\n----{e}\n----"
                    )

        self.logger.info(
            f"[CONSOLIDATING] Learned: +{added_nodes} nodes, +{added_edges} edges"
        )

    def _retrieve_task(self, wm: WorkingMemory, ctx: dict) -> None:
        """5-Tier Retrieval Strategy optimized for LLM token efficiency."""
        self.logger.info("[RETRIEVING] Retrieving from memory (5-Tier Strategy)...")
        sym_mem: SymbolicMemory = ctx["symbolic_memory"]

        valid_entities = [
            e for e in wm.extracted_entities if e.label and sym_mem.has_concept(e.id)
        ]
        context_parts = []
        node_count = len(sym_mem)

        if node_count == 0:
            wm.retrieved_knowledge["graph_context"] = (
                "The memory graph is empty. No previous knowledge available."
            )
            return

        # Strategy 1: Path Discovery (2+ entities)
        if len(valid_entities) >= 2:
            try:
                from memory_graph.core.traversal import find_shortest_path_with_details

                path_details = find_shortest_path_with_details(
                    sym_mem.graph,
                    valid_entities[0].id,
                    valid_entities[1].id,
                )
                if path_details and len(path_details.get("path", [])) > 1:
                    context_parts.append(
                        self._format_path_for_llm(
                            path_details, valid_entities[0], valid_entities[1]
                        )
                    )
            except Exception:
                pass

        # Strategy 2: Confidence-Filtered Subgraph (1+ entities)
        if not context_parts and valid_entities:
            try:
                from memory_graph.core.traversal import get_confidence_filtered_subgraph

                primary_entity = valid_entities[0]
                filtered_graph = get_confidence_filtered_subgraph(
                    graph=sym_mem.graph,
                    center=primary_entity.id,
                    radius=2,
                    min_confidence=0.5,
                    min_weight=0.5,
                )
                if len(filtered_graph.get_nodes()) > 1:
                    context_parts.append(
                        self._format_subgraph_for_llm(
                            filtered_graph, primary_entity.label
                        )
                    )
            except Exception:
                pass

        # Strategy 3: Community Discovery (large graphs, no specific entities)
        if not context_parts and node_count > 50:
            try:
                from memory_graph.core.traversal import find_communities

                communities = find_communities(sym_mem.graph, min_nodes=3)
                if communities:
                    context_parts.append(f"""
                        ## Graph Context
                        Found a community of {len(communities[0])} related concepts.
                        """)
            except Exception:
                pass

        # Strategy 4 & 5: Fallback to simple neighbor retrieval or linearized summary
        if not context_parts:
            if valid_entities:
                entity = valid_entities[0]
                try:
                    neighbors = sym_mem.get_neighbors(entity.id, max_depth=1)
                    neighbors_labels = [n.get_label() for n in neighbors[:5]]
                    context_parts.append(f"""
                        ## Context for '{entity.label}'
                        Related to: {', '.join(neighbors_labels) if neighbors_labels else 'None'}
                        """)
                except Exception:
                    pass
            else:
                context_parts.append("""
                    ## Graph Context
                    No specific entities matched, relying on parametric knowledge.
                    """)

        # Episodic Memory Retrieval
        epi_context = []
        for entity in valid_entities[:2]:
            try:
                epi_ctx = ctx["episodic_memory"].query(entity.label, max_results=2)
                if epi_ctx:
                    epi_context.extend(
                        [n.get_metadata().get("content", "")[:150] for n in epi_ctx]
                    )
            except Exception:
                pass

        if epi_context:
            context_parts.append("## Episodic Memory\n" + "\n".join(epi_context))

        wm.retrieved_knowledge["graph_context"] = "\n\n".join(context_parts)
        self.logger.info(
            f"[RETRIEVING] Retrieved context length: {len(wm.retrieved_knowledge['graph_context'])} chars"
        )

    def _reason_task(self, wm: WorkingMemory, _ctx: dict) -> None:
        """Step 4: Generate reasoning using consolidated knowledge and tool results."""
        self.logger.info(f"[REASONING] Reasoning (Mode: {wm.modus_operandi})...")

        # 1. Gather context for the prompt
        graph_context = wm.retrieved_knowledge.get(
            "graph_context", "No specific graph context available."
        )

        # 2. Gather tool results (if the tool_executor ran previously)
        tool_results = []
        for k, v in wm.retrieved_knowledge.items():
            if k.startswith("tool_"):
                tool_results.append(f"- {k.replace('tool_', '')}: {v}")
        tool_results_str = "\n".join(tool_results) if tool_results else None

        # 3. Gather gaps
        gaps_str = "\n".join(f"- {gap}" for gap in wm.gaps) if wm.gaps else "None"

        # 4. Build prompt using the unified template
        prompt = Prompts.reason(
            query=wm.goal,
            context=graph_context,
            gaps=gaps_str,
            tool_results=tool_results_str,
        )

        try:
            # 5. Use the robust LLM call method (handles token limits and retries)
            messages = [{"role": "user", "content": prompt}]
            reasoning = self._llm_call(messages)

            wm.reasoning = (
                str(reasoning).strip() if reasoning else "No reasoning generated."
            )

            # 6. Extract final answer
            if "Solution:" in wm.reasoning:
                wm.final_answer = wm.reasoning.split("Solution:", 1)[1].strip()
            else:
                wm.final_answer = wm.reasoning

            # 7. Detect uncertainty to trigger retrieval/tool cycles
            uncertainty_markers = [
                "not sure",
                "i need more information",
                "i don't know",
                "unclear",
                "cannot answer",
                "insufficient information",
            ]

            if any(marker in wm.reasoning.lower() for marker in uncertainty_markers):
                wm.needs_more_data = True
                wm.reasoning_complete = False
                self.logger.info(
                    "[REASONING] Reasoning indicates more data is needed. Routing to tool/retrieval..."
                )
            else:
                wm.reasoning_complete = True
                wm.needs_more_data = False
                self.logger.info("[REASONING] Reasoning complete.")

        except Exception as e:
            self.logger.error(f"[REASONING] Reasoning failed: {e}", exc_info=True)
            wm.reasoning = f"Error during reasoning: {e}"
            wm.final_answer = "I encountered an error while processing your request. Please try again."
            wm.reasoning_complete = True
            wm.needs_more_data = False

    def _tool_executor_task(self, wm: WorkingMemory, _ctx: dict) -> None:
        self.logger.info("[TOOL_EXECUTION] Executing tools...")
        targets = wm.gaps + [wm.goal] if wm.gaps else [wm.goal]
        executed_any = False

        for target in targets:
            for tool_name, tool in self.tools.items():
                if tool_name.lower() in target.lower() or any(
                    kw in target.lower()
                    for kw in ["weather", "calculate", "math", "search"]
                ):
                    try:
                        arg = (
                            target.split()[-1].strip(".,?")
                            if target.split()
                            else "unknown"
                        )
                        param_name = (
                            list(tool.parameters.keys())[0]
                            if tool.parameters
                            else "query"
                        )
                        result = self._execute_tool(tool_name, **{param_name: arg})
                        wm.retrieved_knowledge[f"tool_{tool_name}"] = result
                        executed_any = True
                        self.logger.info(
                            f"[TOOL_EXECUTION] Tool {tool_name} executed with {param_name}='{arg}': {result}"
                        )
                        break
                    except Exception as e:
                        self.logger.warning(
                            f"[TOOL_EXECUTION] Tool {tool_name} failed: {e}"
                        )

        wm.retrieved_knowledge["tool_attempted"] = True
        if not executed_any:
            self.logger.info(
                "[TOOL_EXECUTION] No applicable tools found for the current context."
            )

    def _reply_task(self, wm: WorkingMemory, ctx: dict) -> None:
        self.logger.info("[REPLYING] Generating reply...")
        final_answer = wm.final_answer or "I was unable to generate a complete answer."

        turn_id = ctx["episodic_memory"].add_turn(
            role="assistant",
            content=final_answer,
            metadata={
                "action": "reply",
                "confidence": 0.9,
                "modus_operandi": wm.modus_operandi,
            },
        )

        for entity in wm.extracted_entities:
            entity_label = entity.label
            if entity_label:
                try:
                    ctx["episodic_memory"].link_to_concept(
                        turn_id, entity_label, relation="DISCUSSES"
                    )
                except Exception:
                    pass

        self.logger.info("[REPLYING] Reply generated and memory consolidated.")

    # HELPERS & PROMPTS
    def _format_path_for_llm(
        self, path_details: dict, from_entity: Entity, to_entity: Entity
    ) -> str:
        lines = [
            "## Direct Relationship Path Found",
            f"Between: '{from_entity.label}' and '{to_entity.label}'",
            f"Length: {len(path_details.get('path', [])) - 1} hops",
            "### Nodes & Connections",
        ]
        for i, node in enumerate(path_details.get("nodes", [])):
            conf = node.get_metadata().get("confidence", 0.5)
            lines.append(f"  {i + 1}. {node.get_label()} (confidence: {conf:.2f})")
            if i < len(path_details.get("edges", [])):
                edge = path_details["edges"][i]
                lines.append(
                    f"     └─ {edge.get_label()} → (weight: {edge.get_weight():.2f})"
                )

        return "\n".join(lines)

    def _format_subgraph_for_llm(self, graph: Any, center: str) -> str:
        lines = [f"## Knowledge Subgraph centered on '{center}'"]
        for node in graph.get_nodes()[:10]:
            meta = node.get_metadata()
            lines.append(
                f"- {node.get_label()} ({meta.get('type', 'concept')}, conf: {meta.get('confidence', 1.0):.2f})"
            )
        return "\n".join(lines)

    # SERIALIZATION & DIAGNOSTICS
    def save_state(self, path: str) -> None:
        self.state.symbolic_memory_b64 = self.symbolic_memory.to_binary_b64()
        self.state.episodic_memory_b64 = self.episodic_memory.to_binary_b64()
        super().save_state(path)
        self.logger.info(f"[SAVE_STATE] Cogito state and memory graphs saved to {path}")

    def load_state(self, path: str) -> None:
        super().load_state(path)
        self.symbolic_memory = SymbolicMemory.from_binary_b64(
            self.state.symbolic_memory_b64
        )
        self.episodic_memory = EpisodicMemory.from_binary_b64(
            self.state.episodic_memory_b64
        )
        self.logger.info(
            f"[LOAD_STATE] Cogito state and memory graphs loaded from {path}"
        )

    def get_memory_stats(self) -> dict[str, Any]:
        return {
            "symbolic_memory": {
                "nodes": len(self.symbolic_memory),
                "edges": len(self.symbolic_memory.graph.get_edges()),
            },
            "episodic_memory": {"turns": len(self.episodic_memory)},
            "working_memory": self.state.working_memory.to_dict(),
            "orchestrator": self.orchestrator.get_state(),
        }

    def __repr__(self) -> str:
        return f"CogitoAgent(status={self.orchestrator.status.value}, step={self.state.working_memory.current_step}, sym_nodes={len(self.symbolic_memory)}, epi_turns={len(self.episodic_memory)}, tools={len(self.tools)})"


# USAGE EXAMPLE & FUNCTIONAL VERIFICATION
if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    try:
        from memory_agent.llm import OpenRouterLLM
        from memory_agent.tools.base import Tool
        from memory_agent.llm.base import GenerationConfig
    except ImportError:
        from ..llm import OpenRouterLLM
        from ..tools.base import Tool
        from ..llm.base import GenerationConfig

    config = GenerationConfig(max_tokens=2048, temperature=0.2)
    llm = OpenRouterLLM(model="dots-studio/dots-3-note-preview:free", config=config)

    def get_weather(city: str) -> str:
        return {
            "Paris": "Sunny, 25°C",
            "London": "Cloudy, 18°C",
            "Tokyo": "Rainy, 22°C",
        }.get(city, f"Weather data not available for {city}")

    def calculate(expression: str) -> float:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            raise ValueError("Invalid characters")
        return eval(expression)

    weather_tool = Tool(
        name="get_weather",
        description="Get current weather",
        parameters={"city": str},
        fn=get_weather,
    )
    calc_tool = Tool(
        name="calculate",
        description="Perform math",
        parameters={"expression": str},
        fn=calculate,
    )

    print("⚡ COGITO: Mobile Suit Activation Sequence...")
    agent = CogitoAgent(
        llm=llm,
        tools=[weather_tool, calc_tool],
        config={"max_steps": 10, "max_retries": 2, "debug": True},
    )

    print("\n🚀 Cogito Activated!\n" + "=" * 60)

    for i, query in enumerate(
        ["What's the weather in Tokyo?", "Tell me about the lore of the Witcher."], 1
    ):
        print(f"\n📝 Query {i}: {query}\n" + "-" * 40)
        try:
            response = agent.process_query(query)
            print(f"🤖 Response: {response}")
            if i == 1:
                stats = agent.get_memory_stats()
                print(
                    f"\n📊 Memory Stats:\n   Symbolic: {stats['symbolic_memory']['nodes']} nodes, {stats['symbolic_memory']['edges']} edges\n   Episodic: {stats['episodic_memory']['turns']} turns\n   Orchestrator Steps: {stats['orchestrator']['execution_count']}"
                )
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "=" * 60 + "\n💫 Cogito Mobile Suit Mission Complete!")

    test_path = "test_cogito_state.json"
    agent.save_state(test_path)
    agent2 = CogitoAgent(llm=llm, tools=[weather_tool, calc_tool])
    agent2.load_state(test_path)
    print(
        f"✅ State loaded successfully. Episodic turns restored: {len(agent2.episodic_memory)}"
    )
    if os.path.exists(test_path):
        os.remove(test_path)
