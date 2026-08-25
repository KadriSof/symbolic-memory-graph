"""
Cogito Core Data Structures - Mecha-Class Agent Foundation

Defines the essential data structures for the Cogito Agent system.
Implements a clean, type-safe, and robust workflow orchestration pattern.
"""

from __future__ import annotations

import time
import json

from enum import Enum
from collections import deque
from datetime import datetime
from dataclasses import dataclass, field

from typing import Any, Callable, Literal

from .react_agent import ReactState
from ..core import WorkingMemory
from memory_graph.core import (
    MemoryGraph,
)


# ENUMS
class NodeStatus(str, Enum):
    """Status of a procedure node."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TransitionType(str, Enum):
    """Type of workflow edge."""

    SEQUENTIAL = "sequential"
    CONDITIONAL = "conditional"


class CogitoStatus(str, Enum):
    """Cogito Agent status."""

    IDLE = "idle"
    COMPREHENDING = "comprehending"
    RETRIEVING = "retrieving"
    CONSOLIDATING = "consolidating"
    REASONING = "reasoning"
    CLARIFYING = "clarifying"
    REPLYING = "replying"
    ERROR = "error"


# WORKFLOW GRAPH & PROCEDURES
@dataclass
class ProcedureNode:
    """
    A resusable procedure node combining static definition and runtime state.
    """

    id: str
    name: str
    description: str
    func: Callable[..., Any]
    max_retries: int = 3
    timeout: float = 30.0

    # Runtime state
    status: NodeStatus = NodeStatus.PENDING
    result: Any = None
    error: str | None = None
    retry_count: int = 0
    start_time: float | None = None
    end_time: float | None = None

    # Metrics
    call_count: int = 0
    total_time: float = 0.0
    error_count: int = 0

    def reset(self) -> None:
        """Reset runtime state for a new execution."""
        self.status = NodeStatus.PENDING
        self.result = None
        self.error = None
        self.retry_count = 0
        self.start_time = None
        self.end_time = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize node definition and metrics."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "status": self.status.value,
            "call_count": self.call_count,
            "total_time": self.total_time,
            "error_count": self.error_count,
        }


@dataclass
class TransitionEdge:
    """A directed edge in the workflow graph."""

    source: str
    target: str
    edge_type: TransitionType = TransitionType.SEQUENTIAL
    condition: Callable[[WorkingMemory], bool] | None = None

    def evaluate_condition(self, working_memory: WorkingMemory) -> bool:
        """Safely evaluate edge condition."""
        if self.condition is None:
            return True

        try:
            return self.condition(working_memory)
        except Exception:
            return False


class WorkflowGraph:
    """
    Workflow Grap.
    Manages procedure nodes and transitions with robust queue-based execution.
    """

    def __init__(
        self, name: str = "default_workflow", max_iterations: int = 100
    ) -> None:
        self.name = name
        self.nodes: dict[str, ProcedureNode] = {}
        self.edges: list[TransitionEdge] = []
        self.entry_nodes: list[str] = []
        self.max_iterations = max_iterations

    def add_node(self, node: ProcedureNode) -> ProcedureNode:
        """Register a procedure node."""
        self.nodes[node.id] = node
        if not self.entry_nodes:
            self.entry_nodes.append(node.id)

        return node

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: TransitionType = TransitionType.SEQUENTIAL,
        condition: Callable[[WorkingMemory], bool] | None = None,
    ) -> None:
        """Add a directed edge between nodes."""
        if source not in self.nodes or target not in self.nodes:
            raise ValueError(
                f"Edge reference unknown, Source: '{source}', Target: '{target}'"
            )
        self.edges.append(TransitionEdge(source, target, edge_type, condition))

    def run(
        self,
        start_node: str,
        working_memory: WorkingMemory,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the workflow using a robust, queue-based traversal."""
        context = context or {}
        completed = set()
        ready_queue = deque([start_node])
        results = {}
        iterations = 0

        # Reset all nodes for fresh execution
        for node in self.nodes.values():
            node.reset()

        while ready_queue and iterations < self.max_iterations:
            iterations += 1
            node_id = ready_queue.popleft()

            if node_id in completed:
                continue

            node = self.nodes[node.id]
            node.status = NodeStatus.RUNNING
            node.start_time = time.perf_counter()

            try:
                # Execute the prrocedure
                result = node.func(working_memory, context)

                if not working_memory.goal:
                    print(f"Node '{node_id}' cleared working_memory.goal")

                working_memory.step_times[node_id] = (
                    time.perf_counter() - node.start_time
                )

                node.result = result
                node.status = NodeStatus.COMPLETED
                node.call_count += 1
                node.total_time += time.perf_counter() - node.start_time
                completed.add(node_id)
                results[node_id] = result

                # Enqueue next nodes based on outgoing edges
                for edge in self.edges:
                    if edge.source == node_id:
                        if edge.edge_type == TransitionType.CONDITIONAL:
                            if edge.evaluate_condition(working_memory):
                                ready_queue.append(edge.target)
                        else:
                            ready_queue.append(edge.target)

            except Exception as e:
                node.error = str(e)
                node.error_count += 1
                node.retry_count += 1
                node.end_time = time.time()

                if node.retry_count < node.max_retries:
                    node.status = NodeStatus.PENDING
                    backoff = 2 ** (node.retry_count - 1)
                    time.sleep(backoff)
                    ready_queue.append(node_id)  # Retry
                else:
                    node.status = NodeStatus.FAILED
                    raise RuntimeError(
                        f"Node '{node_id}' failed after {node.max_retries} retries: {e}"
                    )
            else:
                node.end_time = time.time()

        if iterations >= self.max_iterations:
            raise RuntimeError(
                f"Workflow exceeded maximum safe iterations ({self.max_iterations})"
            )

        return results


class Orchestrator:
    """
    The Nervous System - Coordinates the workflow graph and state transitions.
    """

    def __init__(self, workflow: WorkflowGraph, max_history: int = 50):
        self.workflow = workflow
        self.state: CogitoStatus = CogitoStatus.IDLE
        self.working_memory = WorkingMemory()
        # deque to prevent unbounded memory growth
        self.execution_history: deque[dict[str, Any]] = deque(maxlen=max_history)

    def process(self, query: str, context: dict[str, Any] | None = None) -> str:
        """
        Process a query through the workflow.
        """
        context = context or {}
        self.state = CogitoStatus.COMPREHENDING
        self.working_memory.reset()
        self.working_memory.goal = query
        self.working_memory.start_time = time.time()

        try:
            if self.workflow.entry_nodes:
                start_node = self.workflow.entry_nodes[0]
            else:
                start_node = next(iter(self.workflow.nodes))

            results = self.workflow.run(
                start_node=start_node,
                working_memory=self.working_memory,
                context=context,
            )

            final_answer = (
                self.working_memory.final_answer
                or "I was unable to generate a response."
            )
            self.state = CogitoStatus.IDLE

            self.execution_history.append(
                {
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "duration": time.time() - self.working_memory.start_time,
                    "final_answer": final_answer,
                }
            )

            return final_answer

        except Exception as e:
            self.state = CogitoStatus.ERROR
            raise

    def get_state(self) -> dict[str, Any]:
        """Get current orchestrator state snapshot."""
        return {
            "state": self.state.value,
            "working_memory": self.working_memory.to_dict(),
            "node_count": len(self.workflow.nodes),
            "execution_count": len(self.execution_history),
        }


# COGITO STATE
class CogitoState(ReactState):
    """
    Cogito-specific state extending ReactState.
    Manages WorkingMemory and efficient C++ Graph persistence.
    """

    def __init__(self):
        super().__init__()

        self.working_memory: WorkingMemory = field(default_factory=WorkingMemory)
        self.symbolic_memory: MemoryGraph | None = None  # Knowledge Memory
        self.episodic_memory: MemoryGraph | None = None  # Persistence Memory

        self.graph_snapshot_b64: str = ""
        self.graph_version: int = 0

        self.current_mode: Literal["COGITO", "REACT"] = "COGITO"
        self.graph_ops: int = 0
        self.confidence_threshold: float = 0.7

    def reset(self, keep_metadata: bool = False) -> None:
        """Reset state and working memory, preserving graph if requested"""
        super().reset(keep_metadata=keep_metadata)
        self.working_memory.reset()
        if not keep_metadata:
            self.graph_snapshot_b64 = ""
            self.graph_version = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize state, safely handling the C++ graph via binary encoding."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "working_memory": self.working_memory.to_dict(),
                "current_mode": self.current_mode,
                "graph_snapshot_b64": self.graph_snapshot_b64,
                "graph_version": self.graph_version,
            }
        )
        return base_dict

    # @classmethod
    # def from_dict(cls, data: dict[str, Any]) -> "CogitoState":
    #     """Deserialize state."""
    #     state = super().from_dict(data)
    #     # ISSUE: . Cannot assign to attribute "working_memory", "graph_snapshot_b64", "graph_version"
    #     # for class "ReactState" Attribute "working_memory" is unknown [reportAttributeAccessIssue]
    #     state.working_memory = WorkingMemory.from_dict(data.get("working_memory", {}))
    #     state.graph_snapshot_b64 = data.get("graph_snapshot_b64", "")
    #     state.graph_version = data.get("graph_version", 0)
    #     # ISSUE: Diagnostics:1. Type "ReactState" is not assignable to return type "CogitoState"
    #     # "ReactState" is not assignable to "CogitoState" [reportReturnType]
    #     return state

    def save_graph_state(
        self, graph: "MemoryGraph", previous_graph: "MemoryGraph | None" = None
    ) -> None:
        """
        Efficiently save graph state using delta compression if possible.
        Falls back to full binary serialization for the first snapshot.
        """


class CogitoAgent: ...
