"""
OnyxOS Loop Agent — Background Reasoning & Memory Consolidation (BG)

The lowest-priority agent. Performs background tasks:
  - Long-horizon reasoning (multi-step planning)
  - Memory consolidation (summarize recent interactions)
  - Context pre-computation (anticipate user needs)
  - Knowledge graph updates

First to be degraded/evicted when resources are scarce.
Represents the "System 2" slow thinking mode.

For demo: simulates background reasoning with progress updates
"""

from __future__ import annotations

import random
import time
from typing import Any

from onyx_kernel.config import AgentClass, QuantLevel
from agents.base_agent import BaseAgent


REASONING_TASKS = [
    {
        "task": "Analyzing meeting patterns for schedule optimization",
        "steps": 8,
        "insight": "You tend to have 40% more productive meetings before noon",
    },
    {
        "task": "Consolidating today's conversations into memory",
        "steps": 5,
        "insight": "3 action items extracted from 12 interactions",
    },
    {
        "task": "Pre-computing context for tomorrow's meetings",
        "steps": 6,
        "insight": "Prepared briefings for 2 upcoming meetings",
    },
    {
        "task": "Updating personal knowledge graph",
        "steps": 10,
        "insight": "Added 7 new connections between concepts",
    },
    {
        "task": "Analyzing health patterns from sensor data",
        "steps": 7,
        "insight": "Heart rate variability suggests elevated stress this week",
    },
]


class LoopAgent(BaseAgent):
    """Background reasoning agent — slow thinking.
    
    Priority: BG (first to degrade/evict when resources are scarce)
    
    This is the "expendable" agent — the market will sacrifice it
    first when thermal pressure or memory pressure rises. It
    gracefully accepts this and resumes when conditions improve.
    """

    def __init__(self):
        super().__init__(
            name="Loop_Reason",
            agent_class=AgentClass.BG,
            base_utility=30.0,
            power_cost_watts=8.0,
            ram_mb=1536.0,
            gpu_tops_required=10.0,
            is_rigid=False,
            description="Background reasoning, memory consolidation, anticipation",
        )
        self._current_task: dict = {}
        self._current_step: int = 0
        self._tasks_completed: int = 0
        self._insights: list = []

    def setup(self) -> None:
        """Initialize background reasoning state."""
        self._pick_new_task()

    def process(self, input_data: Any = None,
                quant_level: QuantLevel = QuantLevel.FP16) -> dict:
        """Execute one step of background reasoning.
        
        At lower quant levels, processing is simpler but still progresses.
        """
        if not self._current_task:
            self._pick_new_task()

        # Simulate processing time (background = slower is fine)
        latency_map = {
            QuantLevel.FP16: (0.3, 0.8),
            QuantLevel.INT8: (0.2, 0.5),
            QuantLevel.INT4: (0.1, 0.3),
            QuantLevel.INT2: (0.05, 0.15),
        }
        lo, hi = latency_map.get(quant_level, (0.2, 0.5))
        time.sleep(random.uniform(lo, hi))

        self._current_step += 1
        total_steps = self._current_task.get("steps", 5)
        progress = min(1.0, self._current_step / total_steps)

        result = {
            "task": self._current_task["task"],
            "step": self._current_step,
            "total_steps": total_steps,
            "progress": round(progress, 2),
            "quant_level": quant_level.name,
            "insight": None,
        }

        # Task complete
        if self._current_step >= total_steps:
            result["insight"] = self._current_task["insight"]
            self._insights.append(self._current_task["insight"])
            self._tasks_completed += 1
            self._pick_new_task()

        return result

    def _pick_new_task(self) -> None:
        """Select a new background reasoning task."""
        self._current_task = random.choice(REASONING_TASKS)
        self._current_step = 0

    def teardown(self) -> None:
        """Save reasoning state."""
        pass

    def on_suspend(self, reason: str) -> None:
        """Gracefully accept suspension — we'll resume later."""
        pass

    def on_evict(self, reason: str) -> None:
        """Accept eviction — save current progress."""
        self.teardown()
