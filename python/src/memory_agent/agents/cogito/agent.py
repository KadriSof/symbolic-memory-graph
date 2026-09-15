# src/memory_agent/agents/cogito/agent.py

import time

from typing import Any

from ..base import BaseAgent, AgentStatus
from ..react import ReactEngine
from ...tools import Tool

from .prompts import Prompts
from .state import CogitoState, CognitivePhase
from .types import Entity, ReasoningDecision, ComprehensionResult
from .schemas import ReasoningSchema, ComprehensionSchema
from .orchestrator import WorkflowOrchestrator
from .memory import WorkingMemory, SymbolicMemory, EpisodicMemory


from memory_graph import MemoryGraph


# COGITO AGENT
class CogitoAgent(BaseAgent):
    """COGITO: The Ultimate Mecha-Class Agent"""

    def __init__(
        self,
        llm: Any,
        tools: list[Tool] | None = None,
        config: dict[str, Any] | None = None,
        state: CogitoState | None = None,
        knowledge_graph: MemoryGraph | None = None,
        episodic_memory: MemoryGraph | None = None,
    ) -> None:
        super().__init__(llm, tools, config, state)

        self.symbolic_memory = SymbolicMemory(graph=knowledge_graph)
        self.episodic_memory = EpisodicMemory(graph=episodic_memory)

        self.react_engine = ReactEngine(tools=self.tools)
        self.orchestrator = WorkflowOrchestrator(
            entry_node_id="comprehend", max_steps=50, max_history=50
        )

        self._build_workflow()

    def _initialize_state(self) -> CogitoState:
        return CogitoState()

    @property
    def state(self) -> CogitoState:
        state = super().state

        if not isinstance(state, CogitoState):
            raise TypeError("CogitoAgent requires CogitoState")

        return state

    @classmethod
    def _state_class(cls) -> type[CogitoState]:
        return CogitoState

    def _build_system_prompt(self) -> str:
        return """
        You are Cogito, a cognitive reasoning agent.

        Your cognitive process is:

        1. COMPREHEND
        Understand the user's request.

        2. RECALL
        Retrieve relevant information from memory.

        3. REASON
        Decide whether to:
        - REPLY when sufficient information exists.
        - REQUEST clarification when essential information is missing.
        - REACT when external tools or actions are useful.

        4. REACT
        Use tools when reasoning determines they are useful.
        After receiving tool observations, return to REASON.

        5. REPLY
        Produce the final answer.

        Do not request clarification when the available information is
        sufficient. Do not use tools merely because tools are available.
        """.strip()

    # WORKFLOW DEFINITION (The Nervous System)
    def _build_workflow(self) -> None:
        # Decoupled Learning: Comprehend -> Recall -> Reason
        self.orchestrator.add_node(
            "comprehend",
            "Comprehend",
            "Assimilate user input input into a structured cognitive representation",
            self._comprehension_procedure,
        )
        self.orchestrator.add_node(
            "recall",
            "Recall",
            "Retrieve relevant symbolic and episodic knowledge",
            self._recall_procedure,
        )
        self.orchestrator.add_node(
            "reason",
            "Reason",
            "Determine whether to reply, request clarification, or react",
            self._reasoning_procedure,
        )
        self.orchestrator.add_node(
            "react",
            "React",
            "Execute tools selected by the reasonong process",
            self._react_procedure,
        )
        self.orchestrator.add_node(
            "request",
            "Request",
            "Produce a clarification request and pause execution.",
            self._request_procedure,
        )

        self.orchestrator.add_node(
            "reply",
            "Reply",
            "Finalize and persist the assistant response.",
            self._replying_procedure,
        )

        # Linear cognitive trajectory
        self.orchestrator.add_edge(
            "comprehend",
            "recall",
        )

        self.orchestrator.add_edge(
            "recall",
            "reason",
        )

        # Reasoning decisions
        self.orchestrator.add_edge(
            "reason",
            "reply",
            condition=lambda wm: (wm.next_action == ReasoningDecision.REPLY),
        )

        self.orchestrator.add_edge(
            "reason",
            "request",
            condition=lambda wm: (wm.next_action == ReasoningDecision.REQUEST),
        )

        self.orchestrator.add_edge(
            "reason",
            "react",
            condition=lambda wm: (wm.next_action == ReasoningDecision.REACT),
        )

        # ReAct always returns to Reason.
        self.orchestrator.add_edge(
            "react",
            "reason",
        )

    # CORE EXECUTION OVERRIDE
    def _process_query_impl(self, query: str) -> str:
        state: CogitoState = self.state

        started = time.perf_counter()

        state.start_turn(reset_working_memory=True)
        state.working_memory.goal = query

        self.orchestrator.entry_node_id = "comprehend"

        # Context shared by workflow nodes
        context = {
            "agent": self,
            "state": state,
            "query": query,
            "react_engine": self.react_engine,
            "symbolic_memory": self.symbolic_memory,
            "episodic_memory": self.episodic_memory,
        }

        try:
            results = self.orchestrator.run(self.state.working_memory, context)

            # REQUEST
            if state.current_phase == CognitivePhase.REQUEST:
                state.awaiting_clarification = True
                state.end_turn()

                request = (
                    self.state.working_memory.clarification_request
                    or "Could you provide more informnation"
                )
                return request

            # REPLY
            if self.state.working_memory.next_action == ReasoningDecision.REPLY:
                self._consolidate_after_reply(self.state.working_memory, context)
                state.complete()
                state.end_turn()
                duration = time.perf_counter() - started
                self.orchestrator.record_execution(
                    query=query,
                    final_answer=self.state.working_memory.final_answer,
                    duration=duration,
                )

                return self.state.working_memory.final_answer

            raise RuntimeError(
                "Cogito workflow terminated without a reply or clarification request"
            )

        except Exception as exc:
            state.record_error(str(exc))
            state.fail()
            state.end_turn()
            self.logger.error(f"[CogitoAgent] Execution failed: {exc}", exc_info=True)
            return "I encountered an error while processing your request"

    # EXPLICIT CLARIFICATION RESPONSE
    def respond_to_clarification(
        self,
        response: str,
    ) -> str:
        """
        Continue a pending clarification task.

        This method is intentionally separate from process_query().

        Therefore:

            process_query("Tell me about Witcher")

        is ALWAYS a new task.

        Whereas:

            respond_to_clarification("Tokyo")

        explicitly resumes a previous REQUEST.
        """

        if not response or not response.strip():
            raise ValueError("Clarification response cannot be empty.")

        state = self.state

        if not state.awaiting_clarification:
            raise RuntimeError("Cogito is not awaiting clarification.")

        state.resume_after_clarification(response=response)

        wm = state.working_memory

        context = {
            "agent": self,
            "state": state,
            "query": wm.goal,
            "react_engine": self.react_engine,
            "symbolic_memory": self.symbolic_memory,
            "episodic_memory": self.episodic_memory,
        }

        self.orchestrator.entry_node_id = "reason"

        started = time.perf_counter()

        try:

            self.status = AgentStatus.RUNNING

            self.orchestrator.run(
                wm,
                context,
            )

            # Another clarification
            if state.current_phase == CognitivePhase.REQUEST:

                state.awaiting_clarification = True
                state.end_turn()

                return (
                    wm.clarification_request
                    or "Could you provide a little more information?"
                )

            # Final answer
            if wm.next_action == ReasoningDecision.REPLY:

                self._consolidate_after_reply(
                    wm,
                    context,
                )

                state.complete()
                state.end_turn()

                self.orchestrator.record_execution(
                    query=wm.goal,
                    final_answer=wm.final_answer,
                    duration=(time.perf_counter() - started),
                )

                return wm.final_answer

            raise RuntimeError(
                "Clarification workflow terminated " "without REPLY or REQUEST."
            )

        except Exception:

            state.fail()
            state.end_turn()

            raise

    # COMPREHENSION
    def _comprehension_procedure(self, wm: WorkingMemory, ctx: dict[str, Any]) -> None:
        self.state.current_phase = CognitivePhase.COMPREHEND
        self.logger.info("[COMPREHENDING] Comprehending query...")

        try:
            prompt = Prompts.comprehend(wm.goal)
            messages = [{"role": "user", "content": prompt}]
            result = self.llm.get_structured(
                messages=messages, model=ComprehensionSchema
            )

            if result is None:
                self.logger.warning(
                    "[COMPREHENDING] Comprehension returned None, using fallback"
                )
                fallback = ComprehensionResult.fallback(wm.goal)
                wm.goal = fallback.goal
                wm.plan = fallback.plan
                wm.extracted_entities = fallback.entities
                wm.extracted_relations = fallback.relations

                return

            comprehension = result.to_comprehension_result()

            wm.goal = comprehension.goal
            wm.plan = comprehension.plan
            wm.extracted_entities = comprehension.entities
            wm.extracted_relations = comprehension.relations

        except Exception as e:
            self.logger.warning(
                f"[COMPREHENDING] Comprehension LLM failed," f"Error:\n----{e}\n----"
            )

            # Robust fallback
            fallback = ComprehensionResult.fallback(wm.goal)
            wm.goal = fallback.goal
            wm.plan = fallback.plan
            wm.extracted_entities = fallback.entities
            wm.extracted_relations = fallback.relations

    def _recall_procedure(self, wm: WorkingMemory, ctx: dict) -> None:
        """5-Tier Retrieval Strategy optimized for LLM token efficiency."""
        self.logger.info("[RECALL] Retrieving from memory (5-Tier Strategy)...")
        self.state.current_phase = CognitivePhase.RECALL
        sym_mem: SymbolicMemory = ctx["symbolic_memory"]

        valid_entities = [
            entity
            for entity in wm.extracted_entities
            if entity.label and sym_mem.has_concept(entity.id)
        ]
        context_parts: list[str] = []
        node_count = len(sym_mem)

        if node_count == 0:
            wm.retrieved_knowledge["graph_context"] = (
                "The memory graph is empty. No previous knowledge available."
            )

        else:
            # Strategy 1: Path Discovery (2+ entities)
            if len(valid_entities) >= 2:
                try:
                    from memory_graph.core.traversal import (
                        find_shortest_path_with_details,
                    )

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
                except Exception as exc:
                    self.logger.debug(
                        f"[RECALL] Path retrieval failed:\n----\n{exc}\n----\n"
                    )

            # Strategy 2: Confidence-Filtered Subgraph (1+ entities)
            if not context_parts and valid_entities:
                try:
                    from memory_graph.core.traversal import (
                        get_confidence_filtered_subgraph,
                    )

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
                except Exception as exc:
                    self.logger.debug(
                        f"[RECALL] Subgraph retrieval failed:\n----\n{exc}\n----\n"
                    )

            # Strategy 3: Community Discovery (large graphs, no specific entities)
            if not context_parts and node_count > 50:
                try:
                    from memory_graph.core.traversal import find_communities

                    communities = find_communities(sym_mem.graph, min_nodes=3)
                    if communities:
                        context_parts.append(
                            f"## Graph Context\n"
                            f"Found a community of {len(communities[0])} related concepts."
                        )
                except Exception as exc:
                    self.logger.debug(
                        f"[RECALL] Community retrieval failed:\n----\n{exc}\n----\n"
                    )

            # Strategy 4: Fallback to simple neighbor retrieval or linearized summary
            if not context_parts and valid_entities:
                entity = valid_entities[0]
                try:
                    neighbors = sym_mem.get_neighbors(entity.id, max_depth=1)
                    labels = [n.get_label() for n in neighbors[:5]]
                    context_parts.append(
                        f"## Context for '{entity.label}'\n"
                        f"Related to: {', '.join(labels) if labels else 'None'}"
                    )
                except Exception as exc:
                    self.logger.debug(
                        f"[RECALL] neighbor retrieval failed:\n----\n{exc}\n----\n"
                    )

            # Strategy 5: No graph match
            if not context_parts:
                context_parts.append(
                    "## Graph context:\n"
                    "No specific entities matched"
                    "Use parametric knowledge where appropriate"
                )

        # Episodic memory
        episodic_parts: list[str] = []

        for entity in valid_entities[:2]:

            try:

                memories = ctx["episodic_memory"].query(
                    entity.label,
                    max_results=2,
                )

                episodic_parts.extend(
                    memory.get_metadata().get("content", "")[:150]
                    for memory in memories
                )

            except Exception as exc:

                self.logger.debug(
                    "[RECALL] Episodic retrieval failed: %s",
                    exc,
                )

        if episodic_parts:

            context_parts.append("## Episodic Memory\n" + "\n".join(episodic_parts))

        wm.retrieved_knowledge["graph_context"] = "\n\n".join(context_parts)

        self.logger.info(
            "[RECALL] Context size: %d chars",
            len(wm.retrieved_knowledge["graph_context"]),
        )

    # REASON
    def _reasoning_procedure(self, wm: WorkingMemory, _ctx: dict) -> None:
        """Step 4: Generate reasoning using consolidated knowledge and tool results."""
        self.state.current_phase = CognitivePhase.REASON
        self.logger.info("[REASONING] Determining next cognitive action.")

        # 1. Gather context for the prompt
        graph_context = wm.retrieved_knowledge.get(
            "graph_context", "No specific graph context available."
        )

        # 2. Gather tool results
        tool_results = wm.retrieved_knowledge.get("observations", [])
        tool_results_str = str(tool_results) if tool_results else None

        # 3. Gather gaps
        gaps_str = "\n".join(f"- {gap}" for gap in wm.gaps) if wm.gaps else "None"

        # 4. Build prompt using the unified template
        prompt = Prompts.reason(
            query=wm.goal,
            context=graph_context,
            gaps=gaps_str,
            tool_results=tool_results_str,
            tools=self.tools,
        )

        try:
            result = self.llm.get_structured(
                messages=[{"role": "user", "content": prompt}], model=ReasoningSchema
            )

            if result is None:
                raise RuntimeError("Reasoning structured output was None")

            reasoning = result.to_reasoning_result()
            wm.next_action = reasoning.decision
            wm.reasoning = reasoning.reasoning
            wm.final_answer = reasoning.answer
            wm.clarification_request = reasoning.clarification_request or None
            wm.gaps = reasoning.gaps
            wm.conflicts = reasoning.conflicts

            # Enforce clarification limit.
            if (
                wm.next_action == ReasoningDecision.REQUEST
                and wm.clarification_cycle_exhausted()
            ):

                self.logger.info(
                    "[REASON] Clarification limit " "reached; forcing reply."
                )

                wm.next_action = ReasoningDecision.REPLY

                if not wm.final_answer:

                    wm.final_answer = (
                        "I do not have enough "
                        "information to provide a "
                        "fully reliable answer."
                    )

        except Exception as exc:

            self.logger.error(
                "[REASON] Reasoning failed: %s",
                exc,
                exc_info=True,
            )

            # Safe fallback.
            wm.next_action = ReasoningDecision.REPLY

            wm.reasoning = f"Reasoning failure: {exc}"

            wm.final_answer = (
                "I encountered an error while " "reasoning about your request."
            )

    # REACT
    def _react_procedure(
        self,
        wm: WorkingMemory,
        ctx: dict[str, Any],
    ) -> None:

        self.state.current_phase = CognitivePhase.REACT

        self.logger.info("[REACT] Executing tool-assisted reasoning.")

        react_engine: ReactEngine = ctx["react_engine"]

        # ReactEngine should use CogitoState as its
        # structural ReAct state.
        #
        # IMPORTANT:
        # ReactEngine should NOT be allowed to turn
        # the final Cogito answer directly into REPLY.
        # It returns control to REASON.

        result = react_engine.execute(
            query=wm.goal,
            state=self.state,
            llm_caller=self._llm_call,
        )

        if result:

            wm.retrieved_knowledge["react_result"] = result

        # Force return to reasoning.
        wm.next_action = None

    # REQUEST
    def _request_procedure(
        self,
        wm: WorkingMemory,
        ctx: dict[str, Any],
    ) -> None:

        del ctx

        self.state.current_phase = CognitivePhase.REQUEST

        if wm.clarification_cycle_exhausted():

            self.logger.warning("[REQUEST] Clarification limit reached.")

            wm.next_action = ReasoningDecision.REPLY

            wm.final_answer = (
                "I do not have enough information " "to answer this reliably."
            )

            return

        request = wm.clarification_request

        if not request:

            if wm.gaps:

                request = (
                    "I need a little more information "
                    "before I can answer accurately. "
                    f"In particular: {', '.join(wm.gaps)}"
                )

            else:

                request = (
                    "Could you provide a little more "
                    "information so I can answer accurately?"
                )

        wm.clarification_request = request

        wm.register_clarification_request()

        self.state.awaiting_clarification = True

        self.logger.info(
            "[REQUEST] Clarification request #%d",
            wm.clarification_requests_count,
        )

    # REPLY
    def _replying_procedure(
        self,
        wm: WorkingMemory,
        ctx: dict[str, Any],
    ) -> None:

        self.state.current_phase = CognitivePhase.REPLY

        final_answer = (
            wm.final_answer.strip()
            if wm.final_answer
            else "I was unable to generate a complete answer."
        )

        wm.final_answer = final_answer

        self.logger.info("[REPLY] Final answer generated.")

        # Episodic persistence
        turn_id = ctx["episodic_memory"].add_turn(
            role="assistant",
            content=final_answer,
            metadata={
                "action": "reply",
                "confidence": 0.9,
            },
        )

        # Do NOT link using entity.label.
        #
        # link_to_concept() expects a concept identifier.
        #
        # Whether this works also depends on SymbolicMemory and
        # EpisodicMemory sharing the same MemoryGraph.
        #
        # We therefore deliberately leave cross-memory linking
        # disabled until that architecture is unified.

        self.logger.info(
            "[REPLY] Episodic turn persisted: %s",
            turn_id,
        )

    # POST-REPLY CONSOLIDATION
    def _consolidate_after_reply(
        self,
        wm: WorkingMemory,
        ctx: dict[str, Any],
    ) -> None:
        """
        Persist extracted knowledge after a successful reply.

        Consolidation is intentionally outside the cognitive
        workflow because it is a memory side effect, not a
        reasoning phase.
        """

        self.state.current_phase = CognitivePhase.REPLY

        sym_mem: SymbolicMemory = ctx["symbolic_memory"]

        added_nodes = 0
        added_edges = 0

        # Concepts
        for entity in wm.extracted_entities:

            if not entity.label:
                continue

            try:

                node = entity.to_node()

                if sym_mem.has_concept(node.get_id()):
                    continue

                sym_mem.graph.add_node(node)

                added_nodes += 1

            except Exception as exc:

                self.logger.warning(
                    "[CONSOLIDATE] Failed to add " "concept '%s': %s",
                    entity.label,
                    exc,
                )

        # Relations
        for relation in wm.extracted_relations:

            try:

                edge = relation.to_edge()

                if sym_mem.graph.has_edge(edge.get_id()):
                    continue

                sym_mem.graph.add_edge(edge)

                added_edges += 1

            except Exception as exc:

                self.logger.warning(
                    "[CONSOLIDATE] Failed to add " "relation '%s' -> '%s': %s",
                    relation.source,
                    relation.target,
                    exc,
                )

        self.logger.info(
            "[CONSOLIDATE] Learned: +%d nodes, +%d edges",
            added_nodes,
            added_edges,
        )

    # HELPERS
    def _format_path_for_llm(
        self,
        path_details: dict[str, Any],
        from_entity: Entity,
        to_entity: Entity,
    ) -> str:

        lines = [
            "## Direct Relationship Path Found",
            (f"Between: '{from_entity.label}' " f"and '{to_entity.label}'"),
            (f"Length: " f"{len(path_details.get('path', [])) - 1} hops"),
            "### Nodes & Connections",
        ]

        for i, node in enumerate(path_details.get("nodes", [])):

            confidence = node.get_metadata().get("confidence", 0.5)

            lines.append(
                f"  {i + 1}. " f"{node.get_label()} " f"(confidence: {confidence:.2f})"
            )

            if i < len(path_details.get("edges", [])):

                edge = path_details["edges"][i]

                lines.append(
                    f"     └─ "
                    f"{edge.get_label()} "
                    f"→ "
                    f"(weight: "
                    f"{edge.get_weight():.2f})"
                )

        return "\n".join(lines)

    def _format_subgraph_for_llm(
        self,
        graph: Any,
        center: str,
    ) -> str:

        lines = [f"## Knowledge Subgraph " f"centered on '{center}'"]

        for node in graph.get_nodes()[:10]:

            metadata = node.get_metadata()

            lines.append(
                f"- {node.get_label()} "
                f"({metadata.get('type', 'concept')}, "
                f"conf: "
                f"{metadata.get('confidence', 1.0):.2f})"
            )

        return "\n".join(lines)

    # DIAGNOSTICS
    def get_memory_stats(
        self,
    ) -> dict[str, Any]:

        return {
            "symbolic_memory": {
                "nodes": len(self.symbolic_memory),
                "edges": len(self.symbolic_memory.graph.get_edges()),
            },
            "episodic_memory": {
                "turns": len(self.episodic_memory),
            },
            "working_memory": (self.state.working_memory.to_dict()),
            "cognitive_phase": (self.state.current_phase.value),
            "awaiting_clarification": (self.state.awaiting_clarification),
            "orchestrator": (self.orchestrator.get_state()),
        }

    def __repr__(self) -> str:

        return (
            "CogitoAgent("
            f"status={self.state.status.value}, "
            f"phase={self.state.current_phase.value}, "
            f"turn={self.state.current_turn}, "
            f"sym_nodes={len(self.symbolic_memory)}, "
            f"epi_turns={len(self.episodic_memory)}, "
            f"tools={len(self.tools)}"
            ")"
        )


# USAGE EXAMPLE & FUNCTIONAL VERIFICATION
if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    try:
        from memory_agent.llm import OpenRouterLLM
        from memory_agent.llm import LlamaCppLLM
        from memory_agent.tools.base import Tool
        from memory_agent.llm.base import GenerationConfig
        from pydantic import BaseModel, Field
    except ImportError:
        from ...llm import OpenRouterLLM
        from ...tools.base import Tool
        from ...llm.base import GenerationConfig

    model_path = os.path.expanduser("~/models/granite-4.1-3b-Q4_K_M.gguf")

    config = GenerationConfig(max_tokens=2048, temperature=0.2)
    llm = OpenRouterLLM(model="dots-studio/dots-3-note-preview:free", config=config)
    llm = LlamaCppLLM(
        model_path=model_path,
        config=config,
    )

    # Define tools
    class WeatherToolArgs(BaseModel):
        city: str = Field(..., description="The city to fetch weather for")

    def get_weather(city: str) -> str:
        """Get weather for a city."""
        weather_data = {
            "Paris": "Sunny, 25°C",
            "London": "Cloudy, 18°C",
            "Tokyo": "Rainy, 22°C",
        }
        return weather_data.get(city, f"Weather data not available for {city}")

    class CalcToolArgs(BaseModel):
        expression: str = Field(..., description="Mathematical operation expression")

    def calculate(expression: str) -> float:
        """Safe mathematical calculation."""
        # Simple safe eval
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            raise ValueError("Invalid characters in expression")
        return eval(expression)

    # Create tools
    weather_tool = Tool(
        name="get_weather",
        description="Get current weather for a city. Input: {{'city': 'city_name'}}",
        args_model=WeatherToolArgs,
        fn=get_weather,
    )

    calc_tool = Tool(
        name="calculate",
        description="Perform mathematical calculations. Input: {{'expression': '2+2'}}",
        args_model=CalcToolArgs,
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
