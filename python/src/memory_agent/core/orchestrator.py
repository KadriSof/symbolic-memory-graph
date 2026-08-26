import time
import logging

from enum import Enum
from collections import deque
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger("[WorkflowOrchestrator]")


class CogitoStatus(str, Enum):
    """Cogito Agent macro-phase status."""

    IDLE = "idle"
    COMPREHENDING = "comprehending"
    RETRIEVING = "retrieving"
    CONSOLIDATING = "consolidating"
    REASONING = "reasoning"
    CLARIFYING = "clarifying"
    REPLYING = "replying"
    ERROR = "error"


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProcedureNode:
    id: str
    name: str
    description: str
    func: Callable[[Any, dict], None]
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
    """A directed edge in the workflow graph with optional condition."""

    source: str
    target: str
    condition: Callable[[Any], bool] | None = None

    def evaluate_condition(self, working_memory: Any) -> bool:
        """Safely evaluate edge condition. Returns False on error (fail-safe)."""
        if self.condition is None:
            return True
        try:
            return self.condition(working_memory)
        except Exception as e:
            logger.warning(
                f"Condition evaluation failed for edge {self.source}->{self.target}: {e}"
            )
            return False


class WorkflowOrchestrator:
    """
    Minimalist, robust execution engine for the Cogito Agent's macro-phases.

    Features:
    - Linear/conditional traversal (no over-engineered BFS)
    - Full observability (metrics, status tracking)
    - Exponential backoff on retries
    - Bounded execution history
    - Fail-safe condition evaluation
    """

    def __init__(self, entry_node_id: str, max_steps: int = 50, max_history: int = 50):
        self.entry_node_id = entry_node_id
        self.max_steps = max_steps
        self.nodes: dict[str, ProcedureNode] = {}
        self.edges: list[TransitionEdge] = []
        self.execution_history: deque[dict[str, Any]] = deque(maxlen=max_history)
        self.status: CogitoStatus = CogitoStatus.IDLE

    def add_node(
        self,
        node_id: str,
        name: str,
        description: str,
        func: Callable,
        max_retries: int = 3,
        timeout: float = 30.0,
    ) -> None:
        """Register a procedure node."""
        if node_id in self.nodes:
            raise ValueError(f"Node '{node_id}' already exists")
        self.nodes[node_id] = ProcedureNode(
            id=node_id,
            name=name,
            description=description,
            func=func,
            max_retries=max_retries,
            timeout=timeout,
        )

    def add_edge(
        self, source: str, target: str, condition: Callable | None = None
    ) -> None:
        """Add a directed edge between nodes."""
        if source not in self.nodes or target not in self.nodes:
            raise ValueError(f"Edge references unknown node: {source} -> {target}")
        self.edges.append(
            TransitionEdge(source=source, target=target, condition=condition)
        )

    def run(self, wm: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute the workflow graph with full observability."""
        context = context or {}
        current_id = self.entry_node_id
        steps = 0
        results = {}

        # Reset all nodes for a fresh run
        for node in self.nodes.values():
            node.reset()

        while current_id:
            if steps >= self.max_steps:
                raise RuntimeError(
                    f"Orchestrator exceeded max steps ({self.max_steps})"
                )
            steps += 1

            node = self.nodes[current_id]
            node.status = NodeStatus.RUNNING
            node.start_time = time.perf_counter()

            # Execution with retry and exponential backoff
            success = False
            for attempt in range(node.max_retries + 1):
                try:
                    result = node.func(wm, context)
                    node.result = result
                    node.status = NodeStatus.COMPLETED
                    node.call_count += 1
                    node.total_time += time.perf_counter() - node.start_time
                    results[node.id] = result
                    success = True
                    break
                except Exception as e:
                    node.error = str(e)
                    node.error_count += 1
                    node.retry_count += 1
                    if attempt < node.max_retries:
                        backoff = 2**attempt  # Exponential backoff
                        logger.warning(
                            f"Node '{node.id}' failed (attempt {attempt + 1}), retrying in {backoff}s: {e}"
                        )
                        time.sleep(backoff)

            if not success:
                node.status = NodeStatus.FAILED
                node.end_time = time.perf_counter()
                raise RuntimeError(
                    f"Node '{node.id}' failed after {node.max_retries} retries: {node.error}"
                )

            node.end_time = time.perf_counter()

            # Determine next node based on edges and conditions
            next_id = None
            for edge in self.edges:
                if edge.source == current_id:
                    if edge.evaluate_condition(wm):
                        next_id = edge.target
                        break  # Take the first matching edge

            current_id = next_id

        return results

    def record_execution(self, query: str, final_answer: str, duration: float) -> None:
        """Record an execution in the bounded history."""
        self.execution_history.append(
            {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "duration": duration,
                "final_answer": final_answer,
            }
        )

    def get_node_metrics(self) -> dict[str, dict[str, Any]]:
        """Get metrics for all nodes (useful for debugging/monitoring)."""
        return {node_id: node.to_dict() for node_id, node in self.nodes.items()}

    def get_state(self) -> dict[str, Any]:
        """Get orchestrator state snapshot."""
        return {
            "status": self.status.value,
            "node_count": len(self.nodes),
            "execution_count": len(self.execution_history),
            "node_metrics": self.get_node_metrics(),
        }
