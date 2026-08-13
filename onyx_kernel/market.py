"""
OnyxOS Market Scheduler — Lagrangian Auction Engine

The heart of OnyxOS. Replaces traditional OS schedulers (CFS, EEVDF)
with a market-based resource allocation mechanism.

Each tick:
  1. Thermal governor publishes power/memory/GPU budgets
  2. Agents compute their optimal quantization given current prices
  3. Auction admits agents whose utility exceeds their resource cost
  4. Dual prices update via gradient ascent on constraint violations

The key insight: agents *voluntarily* degrade themselves when resources
are scarce (prices are high), rather than being OOM-killed or starved.

Mathematical foundation:
  max  Σᵢ Uᵢ(qᵢ) · xᵢ
  s.t. Σᵢ εᵢ · xᵢ ≤ P_budget        (power constraint)
       Σᵢ Mᵢ(qᵢ) · xᵢ ≤ M_budget    (memory constraint)
       Σᵢ Gᵢ · xᵢ ≤ G_budget         (GPU constraint)
       xᵢ ∈ {0,1}, qᵢ ∈ QuantLevels

  Lagrangian relaxation with dual variables λ_P, λ_M, λ_G
  solved via dual ascent (subgradient method).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from onyx_kernel.config import (
    AgentClass,
    AgentState,
    MarketConfig,
    QuantLevel,
    SystemConfig,
    DEFAULT_CONFIG,
)
from onyx_kernel.agent import AgentITD


@dataclass
class AuctionResult:
    """Result of a single market auction tick."""
    tick_id: int
    timestamp: float

    # Dual prices after this tick
    lambda_power: float
    lambda_memory: float
    lambda_gpu: float

    # Resource usage
    total_power_used: float
    total_memory_used: float
    total_gpu_used: float

    # Budgets (constraints)
    power_budget: float
    memory_budget: float
    gpu_budget: float

    # Agent decisions
    admitted: List[str] = field(default_factory=list)      # agent names
    suspended: List[str] = field(default_factory=list)
    degraded: List[str] = field(default_factory=list)
    evicted: List[str] = field(default_factory=list)

    # Convergence
    converged: bool = False
    rounds: int = 0
    total_utility: float = 0.0
    total_cost: float = 0.0

    @property
    def power_utilization(self) -> float:
        return self.total_power_used / max(self.power_budget, 0.01)

    @property
    def memory_utilization(self) -> float:
        return self.total_memory_used / max(self.memory_budget, 0.01)


class LagrangianMarket:
    """Lagrangian dual-ascent market for agent resource allocation.
    
    This is the core scheduler of OnyxOS. It replaces priority-based
    scheduling with an economic mechanism where:
    
    - Resources have PRICES (λ_P for power, λ_M for memory, λ_G for GPU)
    - Agents have UTILITY functions that decrease with degradation
    - An agent RUNS if its utility exceeds its cost at current prices
    - Prices RISE when demand exceeds supply (dual ascent)
    - Prices FALL when supply exceeds demand
    
    This naturally produces:
    - Graceful degradation (agents self-degrade to save money)
    - Fair allocation (high-utility agents outbid low-utility ones)
    - Thermal compliance (power price rises with temperature)
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or DEFAULT_CONFIG
        mc = self.config.market

        # Dual variables (Lagrange multipliers = resource prices)
        self.lambda_power: float = mc.LAMBDA_POWER_INIT
        self.lambda_memory: float = mc.LAMBDA_MEMORY_INIT
        self.lambda_gpu: float = mc.LAMBDA_GPU_INIT

        # Price history (for HUD sparklines)
        self.price_history: List[Dict[str, float]] = []
        self._max_history = 300  # 5 minutes at 1Hz

        # Tick counter
        self.tick_count: int = 0

        # Event log
        self.events: List[Dict[str, Any]] = []
        self._max_events = 500

    def run_auction(
        self,
        agents: List[AgentITD],
        power_budget: float,
        memory_budget: float,
        gpu_budget: float,
    ) -> AuctionResult:
        """Run one tick of the Lagrangian market auction.
        
        This is the main scheduling function, called once per tick.
        
        Steps:
          1. Each agent finds its optimal quantization at current prices
          2. Admission decision: run if surplus > 0 (utility > cost)
          3. Enforce hard constraints (RT_HARD agents always run)
          4. Dual ascent: update prices based on constraint violations
          5. Return auction result with all decisions
        
        Args:
            agents: List of agents participating in the auction
            power_budget: Maximum power draw (Watts), from thermal governor
            memory_budget: Available memory (MB)
            gpu_budget: Available GPU compute (TOPS)
        
        Returns:
            AuctionResult with all decisions and updated prices
        """
        self.tick_count += 1
        mc = self.config.market

        result = AuctionResult(
            tick_id=self.tick_count,
            timestamp=time.time(),
            lambda_power=self.lambda_power,
            lambda_memory=self.lambda_memory,
            lambda_gpu=self.lambda_gpu,
            total_power_used=0.0,
            total_memory_used=0.0,
            total_gpu_used=0.0,
            power_budget=power_budget,
            memory_budget=memory_budget,
            gpu_budget=gpu_budget,
        )

        # ── Step 1: Optimal quantization for each agent ──────────
        for agent in agents:
            optimal_q = agent.optimal_quant(
                self.lambda_power, self.lambda_memory, self.lambda_gpu
            )
            agent.quant_level = optimal_q

        # ── Step 2: Admission decisions ──────────────────────────
        # Sort by priority class first, then by surplus (descending)
        class_priority = {
            AgentClass.RT_HARD: 0,
            AgentClass.RT_SOFT: 1,
            AgentClass.BG: 2,
        }

        sorted_agents = sorted(
            agents,
            key=lambda a: (
                class_priority.get(a.agent_class, 99),
                -a.surplus(self.lambda_power, self.lambda_memory, self.lambda_gpu),
            ),
        )

        total_power = 0.0
        total_memory = 0.0
        total_gpu = 0.0

        for agent in sorted_agents:
            utility = agent.utility()
            cost = agent.total_cost(
                self.lambda_power, self.lambda_memory, self.lambda_gpu
            )
            surplus = utility - cost

            agent.current_utility = utility
            agent.current_cost = cost
            agent.current_surplus = surplus

            # ── Step 3: RT_HARD always runs ──────────────────────
            if agent.agent_class == AgentClass.RT_HARD:
                agent.admit(QuantLevel.FP16)
                total_power += agent.power_cost_watts
                total_memory += agent.memory_footprint()
                total_gpu += agent.gpu_tops_required
                result.admitted.append(agent.name)
                agent.metrics.total_utility_earned += utility
                agent.metrics.total_cost_paid += cost
                self._log_event("ADMIT", agent, "RT_HARD guaranteed")
                continue

            # ── Admission: surplus must be positive ──────────────
            if surplus > 0:
                # Check if admitting would exceed hard budgets
                would_power = total_power + agent.power_cost_watts
                would_memory = total_memory + agent.memory_footprint()
                would_gpu = total_gpu + agent.gpu_tops_required

                # Allow slight oversubscription (prices will correct)
                if would_power <= power_budget * 1.1 and \
                   would_memory <= memory_budget * 1.05:
                    if agent.quant_level != QuantLevel.FP16:
                        agent.admit(agent.quant_level)
                        result.degraded.append(agent.name)
                        self._log_event("DEGRADE", agent,
                                        f"admitted at {agent.quant_level.name}")
                    else:
                        agent.admit(QuantLevel.FP16)
                        result.admitted.append(agent.name)
                        self._log_event("ADMIT", agent,
                                        f"surplus={surplus:.2f}")

                    total_power += agent.power_cost_watts
                    total_memory += agent.memory_footprint()
                    total_gpu += agent.gpu_tops_required
                    agent.metrics.total_utility_earned += utility
                    agent.metrics.total_cost_paid += cost
                else:
                    # Over budget — try degrading further
                    degraded = self._try_degrade_to_fit(
                        agent, power_budget - total_power,
                        memory_budget - total_memory
                    )
                    if degraded:
                        total_power += agent.power_cost_watts
                        total_memory += agent.memory_footprint()
                        total_gpu += agent.gpu_tops_required
                        result.degraded.append(agent.name)
                    else:
                        agent.suspend("over_budget")
                        result.suspended.append(agent.name)
                        self._log_event("SUSPEND", agent, "over budget")
            else:
                # Negative surplus — not worth running at current prices
                if agent.state in (AgentState.RUNNING, AgentState.DEGRADED):
                    agent.suspend("negative_surplus")
                    result.suspended.append(agent.name)
                    self._log_event("SUSPEND", agent,
                                    f"surplus={surplus:.2f}")
                elif agent.state == AgentState.SUSPENDED:
                    result.suspended.append(agent.name)
                else:
                    agent.evict("negative_surplus")
                    result.evicted.append(agent.name)
                    self._log_event("EVICT", agent, f"surplus={surplus:.2f}")

        # ── Step 4: Dual ascent — update prices ──────────────────
        result.total_power_used = total_power
        result.total_memory_used = total_memory
        result.total_gpu_used = total_gpu

        # Subgradient: price += η * (demand - supply)
        power_violation = total_power - power_budget
        memory_violation = total_memory - memory_budget
        gpu_violation = total_gpu - gpu_budget

        # Update with dampening (exponential moving average)
        new_lambda_p = self.lambda_power + mc.ETA_POWER * power_violation
        new_lambda_m = self.lambda_memory + mc.ETA_MEMORY * memory_violation
        new_lambda_g = self.lambda_gpu + mc.ETA_GPU * gpu_violation

        # Apply dampening
        self.lambda_power = (
            mc.PRICE_DAMPENING * self.lambda_power
            + (1 - mc.PRICE_DAMPENING) * new_lambda_p
        )
        self.lambda_memory = (
            mc.PRICE_DAMPENING * self.lambda_memory
            + (1 - mc.PRICE_DAMPENING) * new_lambda_m
        )
        self.lambda_gpu = (
            mc.PRICE_DAMPENING * self.lambda_gpu
            + (1 - mc.PRICE_DAMPENING) * new_lambda_g
        )

        # Clamp prices to bounds
        self.lambda_power = max(mc.LAMBDA_MIN,
                                min(mc.LAMBDA_MAX, self.lambda_power))
        self.lambda_memory = max(mc.LAMBDA_MIN,
                                 min(mc.LAMBDA_MAX, self.lambda_memory))
        self.lambda_gpu = max(mc.LAMBDA_MIN,
                              min(mc.LAMBDA_MAX, self.lambda_gpu))

        # Update result with final prices
        result.lambda_power = self.lambda_power
        result.lambda_memory = self.lambda_memory
        result.lambda_gpu = self.lambda_gpu
        result.total_utility = sum(
            a.current_utility for a in agents
            if a.state in (AgentState.RUNNING, AgentState.DEGRADED)
        )
        result.total_cost = sum(
            a.current_cost for a in agents
            if a.state in (AgentState.RUNNING, AgentState.DEGRADED)
        )

        # Record price history
        self.price_history.append({
            "tick": self.tick_count,
            "lambda_P": self.lambda_power,
            "lambda_M": self.lambda_memory,
            "lambda_G": self.lambda_gpu,
            "power_util": result.power_utilization,
            "memory_util": result.memory_utilization,
        })
        if len(self.price_history) > self._max_history:
            self.price_history.pop(0)

        return result

    def _try_degrade_to_fit(
        self, agent: AgentITD,
        power_headroom: float, memory_headroom: float
    ) -> bool:
        """Try progressively lower quantization levels to fit the budget."""
        if agent.is_rigid:
            return False

        for level in [QuantLevel.INT8, QuantLevel.INT4, QuantLevel.INT2]:
            mem = agent.memory_footprint(level)
            if agent.power_cost_watts <= power_headroom and mem <= memory_headroom:
                surplus = agent.surplus(
                    self.lambda_power, self.lambda_memory, self.lambda_gpu, level
                )
                if surplus > 0:
                    agent.admit(level)
                    self._log_event("DEGRADE", agent,
                                    f"degraded to {level.name} to fit budget")
                    return True
        return False

    def _log_event(self, action: str, agent: AgentITD, reason: str) -> None:
        """Log a market event."""
        event = {
            "tick": self.tick_count,
            "timestamp": time.time(),
            "action": action,
            "agent": agent.name,
            "class": agent.agent_class.name,
            "quant": agent.quant_level.name,
            "utility": agent.current_utility,
            "cost": agent.current_cost,
            "surplus": agent.current_surplus,
            "reason": reason,
        }
        self.events.append(event)
        if len(self.events) > self._max_events:
            self.events.pop(0)

    def get_price_sparkline(self, key: str = "lambda_P", width: int = 40) -> str:
        """Generate a sparkline string for price history.
        
        Uses Unicode block characters for a compact visualization.
        """
        if not self.price_history:
            return "▁" * width

        values = [h[key] for h in self.price_history[-width:]]
        if not values:
            return "▁" * width

        min_val = min(values)
        max_val = max(values)
        value_range = max_val - min_val if max_val != min_val else 1.0

        blocks = "▁▂▃▄▅▆▇█"
        sparkline = ""
        for v in values:
            normalized = (v - min_val) / value_range
            idx = int(normalized * (len(blocks) - 1))
            sparkline += blocks[idx]

        return sparkline.ljust(width, "▁")

    def status_summary(self) -> str:
        """One-line market status for display."""
        return (
            f"Tick {self.tick_count:>4d} | "
            f"λ_P={self.lambda_power:>6.3f} "
            f"λ_M={self.lambda_memory:>6.3f} "
            f"λ_G={self.lambda_gpu:>6.3f}"
        )
