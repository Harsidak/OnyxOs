"""
OnyxOS Agent — Inference Task Descriptor (ITD)

The fundamental unit of work in OnyxOS. An Agent wraps an AI model
(or pipeline) and participates in the Lagrangian market auction to
compete for compute, memory, and power resources.

This replaces the POSIX process as the primary schedulable entity.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from onyx_kernel.config import AgentClass, AgentState, QuantLevel


@dataclass
class AgentMetrics:
    """Real-time performance metrics for an agent."""
    tokens_generated: int = 0
    inference_latency_ms: float = 0.0
    total_inferences: int = 0
    times_degraded: int = 0
    times_evicted: int = 0
    total_utility_earned: float = 0.0
    total_cost_paid: float = 0.0
    uptime_seconds: float = 0.0
    last_inference_time: float = 0.0


class AgentITD:
    """Inference Task Descriptor — The OnyxOS process equivalent.
    
    Each agent has:
    - A utility function U(q) that maps quantization level to value produced
    - A cost profile (power draw ε, memory footprint M)
    - A priority class (RT_HARD, RT_SOFT, BG)
    - Market participation logic (bid/accept/degrade/evict)
    
    The market scheduler decides which agents run based on:
        RUN(i) ⟺ ∂Uᵢ/∂rᵢ > λ_P·εᵢ + λ_M·Mᵢ(q)
    
    Agents with utility exceeding their resource cost (at current market prices)
    are admitted; others are suspended or evicted.
    """

    def __init__(
        self,
        name: str,
        agent_class: AgentClass,
        base_utility: float,
        power_cost_watts: float,
        ram_mb: float,
        gpu_tops_required: float = 0.0,
        utility_exponent: float = 0.5,
        is_rigid: bool = False,
        inference_fn: Optional[Callable] = None,
        description: str = "",
    ):
        # Identity
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.description = description
        self.agent_class = agent_class

        # Resource profile (at full precision)
        self.base_utility = base_utility
        self.power_cost_watts = power_cost_watts  # ε_i — Watts consumed during inference
        self.ram_mb = ram_mb                       # M_i — Resident memory at full precision
        self.gpu_tops_required = gpu_tops_required # G_i — GPU compute required
        self.utility_exponent = utility_exponent   # α — how sharply utility drops with degradation

        # Constraints
        self.is_rigid = is_rigid  # RT_HARD agents cannot be degraded

        # Current state
        self.state: AgentState = AgentState.REGISTERED
        self.quant_level: QuantLevel = QuantLevel.FP16
        self._previous_state: AgentState = AgentState.REGISTERED

        # Market participation
        self.current_utility: float = 0.0
        self.current_cost: float = 0.0
        self.current_surplus: float = 0.0  # utility - cost (agent's "profit")

        # Inference callback
        self._inference_fn = inference_fn

        # Metrics
        self.metrics = AgentMetrics()
        self._created_at = time.time()

        # Event log (ring buffer, last 100 events)
        self.event_log: List[Dict[str, Any]] = []
        self._max_events = 100

    # ── Utility & Cost Functions ─────────────────────────────────

    def utility(self, quant: Optional[QuantLevel] = None) -> float:
        """Compute agent utility at a given quantization level.
        
        U_i(q) = U_base * q^α
        
        Where q ∈ (0, 1] is the quantization fraction and α controls
        how steeply utility drops with degradation.
        
        For rigid agents (RT_HARD), any degradation returns -∞.
        """
        q = (quant or self.quant_level).value
        if self.is_rigid and q < 1.0:
            return float('-inf')
        return self.base_utility * (q ** self.utility_exponent)

    def memory_footprint(self, quant: Optional[QuantLevel] = None) -> float:
        """Memory footprint at given quantization level (MB).
        
        M_i(q) = M_base * q
        
        Lower quantization = proportionally less memory.
        """
        q = (quant or self.quant_level).value
        return self.ram_mb * q

    def total_cost(self, lambda_p: float, lambda_m: float, lambda_g: float = 0.0,
                   quant: Optional[QuantLevel] = None) -> float:
        """Total resource cost at current market prices.
        
        Cost_i = λ_P·ε_i + λ_M·M_i(q) + λ_G·G_i
        """
        q = quant or self.quant_level
        return (
            lambda_p * self.power_cost_watts
            + lambda_m * self.memory_footprint(q)
            + lambda_g * self.gpu_tops_required
        )

    def surplus(self, lambda_p: float, lambda_m: float, lambda_g: float = 0.0,
                quant: Optional[QuantLevel] = None) -> float:
        """Economic surplus = Utility - Cost.
        
        Positive surplus → agent should run.
        Negative surplus → agent should be suspended/evicted.
        """
        q = quant or self.quant_level
        return self.utility(q) - self.total_cost(lambda_p, lambda_m, lambda_g, q)

    def optimal_quant(self, lambda_p: float, lambda_m: float,
                      lambda_g: float = 0.0) -> QuantLevel:
        """Find the quantization level that maximizes surplus.
        
        argmax_q [U_i(q) - λ_P·ε_i - λ_M·M_i(q) - λ_G·G_i]
        
        Rigid agents always return FP16.
        """
        if self.is_rigid:
            return QuantLevel.FP16

        best_level = QuantLevel.FP16
        best_surplus = float('-inf')

        for level in QuantLevel:
            s = self.surplus(lambda_p, lambda_m, lambda_g, level)
            if s > best_surplus:
                best_surplus = s
                best_level = level

        return best_level

    # ── State Transitions ────────────────────────────────────────

    def _transition(self, new_state: AgentState, reason: str = "") -> None:
        """Perform a state transition with logging."""
        self._previous_state = self.state
        self.state = new_state
        self._log_event("state_change", {
            "from": self._previous_state.name,
            "to": new_state.name,
            "reason": reason,
        })

    def admit(self, quant: QuantLevel = QuantLevel.FP16) -> None:
        """Admit agent to run at specified quantization."""
        self.quant_level = quant
        if quant == QuantLevel.FP16:
            self._transition(AgentState.RUNNING, f"admitted at {quant.name}")
        else:
            self._transition(AgentState.DEGRADED, f"admitted degraded at {quant.name}")
            self.metrics.times_degraded += 1

    def suspend(self, reason: str = "market_decision") -> None:
        """Suspend agent (model stays in memory)."""
        self._transition(AgentState.SUSPENDED, reason)

    def evict(self, reason: str = "resource_pressure") -> None:
        """Evict agent (model unloaded from memory)."""
        self._transition(AgentState.EVICTED, reason)
        self.metrics.times_evicted += 1

    def degrade(self, new_quant: QuantLevel, reason: str = "price_spike") -> None:
        """Degrade agent to lower quantization level."""
        if self.is_rigid:
            raise ValueError(f"Cannot degrade rigid agent {self.name}")
        old_quant = self.quant_level
        self.quant_level = new_quant
        self._transition(AgentState.DEGRADED,
                         f"{old_quant.name} → {new_quant.name}: {reason}")
        self.metrics.times_degraded += 1

    # ── Inference ────────────────────────────────────────────────

    def run_inference(self, input_data: Any = None) -> Optional[Any]:
        """Execute one inference step.
        
        Returns the inference result, or None if agent is not running.
        """
        if self.state not in (AgentState.RUNNING, AgentState.DEGRADED):
            return None

        start = time.time()
        result = None

        if self._inference_fn is not None:
            result = self._inference_fn(input_data, self.quant_level)

        latency = (time.time() - start) * 1000  # ms
        self.metrics.inference_latency_ms = latency
        self.metrics.total_inferences += 1
        self.metrics.last_inference_time = time.time()

        return result

    # ── Logging ──────────────────────────────────────────────────

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event to the ring buffer."""
        event = {
            "timestamp": time.time(),
            "agent": self.name,
            "type": event_type,
            **data,
        }
        self.event_log.append(event)
        if len(self.event_log) > self._max_events:
            self.event_log.pop(0)

    # ── Display ──────────────────────────────────────────────────

    def status_line(self) -> str:
        """One-line status for HUD display."""
        state_icons = {
            AgentState.RUNNING: "🟢",
            AgentState.DEGRADED: "🟡",
            AgentState.SUSPENDED: "🔴",
            AgentState.EVICTED: "⚫",
            AgentState.READY: "🔵",
            AgentState.REGISTERED: "⚪",
        }
        icon = state_icons.get(self.state, "❓")
        return (
            f"{icon} {self.name:<20s} "
            f"[{self.agent_class.name:>7s}] "
            f"Q={self.quant_level.name:<4s} "
            f"U={self.current_utility:>6.1f} "
            f"C={self.current_cost:>6.1f} "
            f"S={self.current_surplus:>+6.1f} "
            f"RAM={self.memory_footprint():>6.0f}MB"
        )

    def __repr__(self) -> str:
        return (
            f"AgentITD(name={self.name!r}, class={self.agent_class.name}, "
            f"state={self.state.name}, quant={self.quant_level.name})"
        )
