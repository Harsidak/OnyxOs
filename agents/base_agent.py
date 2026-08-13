"""
OnyxOS Base Agent — Abstract Base for All OnyxOS Agents

Every agent in OnyxOS extends this base class, which provides:
  - Registration with the agent registry
  - Market participation (utility/cost reporting)
  - Lifecycle hooks (on_admit, on_suspend, on_evict, on_degrade)
  - Inference request interface
  - Heartbeat loop
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from onyx_kernel.config import AgentClass, AgentState, QuantLevel
from onyx_kernel.agent import AgentITD


class BaseAgent(ABC):
    """Abstract base class for all OnyxOS agents.
    
    Subclasses must implement:
      - setup() — Initialize models/resources
      - process() — Run one inference step
      - teardown() — Cleanup resources
    
    Optional hooks:
      - on_admit() — Called when admitted to run
      - on_suspend() — Called when suspended
      - on_evict() — Called when evicted
      - on_degrade(old_quant, new_quant) — Called when degraded
    """

    def __init__(
        self,
        name: str,
        agent_class: AgentClass,
        base_utility: float,
        power_cost_watts: float,
        ram_mb: float,
        gpu_tops_required: float = 0.0,
        is_rigid: bool = False,
        description: str = "",
    ):
        self.itd = AgentITD(
            name=name,
            agent_class=agent_class,
            base_utility=base_utility,
            power_cost_watts=power_cost_watts,
            ram_mb=ram_mb,
            gpu_tops_required=gpu_tops_required,
            is_rigid=is_rigid,
            inference_fn=self._inference_wrapper,
            description=description,
        )
        self._running = False
        self._setup_complete = False

    @property
    def name(self) -> str:
        return self.itd.name

    @property
    def state(self) -> AgentState:
        return self.itd.state

    # ── Abstract Methods (must implement) ────────────────────────

    @abstractmethod
    def setup(self) -> None:
        """Initialize the agent (load models, open devices, etc).
        
        Called once when the agent is first readied.
        """
        ...

    @abstractmethod
    def process(self, input_data: Any = None,
                quant_level: QuantLevel = QuantLevel.FP16) -> Any:
        """Run one inference/processing step.
        
        Args:
            input_data: Input to process (camera frame, audio, text, etc)
            quant_level: Current quantization level (agent should adapt)
        
        Returns:
            Processing result
        """
        ...

    @abstractmethod
    def teardown(self) -> None:
        """Cleanup resources (unload models, close devices, etc).
        
        Called when the agent is evicted or the system shuts down.
        """
        ...

    # ── Lifecycle Hooks (optional overrides) ─────────────────────

    def on_admit(self, quant_level: QuantLevel) -> None:
        """Called when the agent is admitted to run."""
        pass

    def on_suspend(self, reason: str) -> None:
        """Called when the agent is suspended."""
        pass

    def on_evict(self, reason: str) -> None:
        """Called when the agent is evicted."""
        self.teardown()

    def on_degrade(self, old_quant: QuantLevel, new_quant: QuantLevel) -> None:
        """Called when the agent is degraded to a lower quantization."""
        pass

    # ── Internal ─────────────────────────────────────────────────

    def _inference_wrapper(self, input_data: Any,
                           quant_level: QuantLevel) -> Any:
        """Wrapper that calls the agent's process() method."""
        return self.process(input_data, quant_level)

    def initialize(self) -> None:
        """Full initialization — setup + mark ready."""
        if not self._setup_complete:
            self.setup()
            self._setup_complete = True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, state={self.state.name})"
