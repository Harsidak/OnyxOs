"""
OnyxOS Voice Agent — Voice NLU + LLM Inference (RT_SOFT)

Handles the voice interaction pipeline:
  Wake word → Speech-to-Text → NLU → LLM Inference → TTS

Can be degraded (smaller model, simpler responses) but should
not be evicted during an active conversation.

For demo: simulates voice interaction with canned responses
"""

from __future__ import annotations

import random
import time
from typing import Any, Optional

from onyx_kernel.config import AgentClass, QuantLevel
from agents.base_agent import BaseAgent


DEMO_RESPONSES = {
    QuantLevel.FP16: [
        "I've analyzed the document. Here are three key insights: "
        "First, the revenue grew 23% year-over-year. Second, the APAC region "
        "showed the strongest performance. Third, operating margins improved by 2.1 points.",
        "Based on your calendar, you have a meeting with the design team in 45 minutes. "
        "I've prepared a summary of yesterday's discussion points.",
        "The weather forecast shows clear skies until 3 PM, then a 60% chance of rain. "
        "I'd recommend heading out for your walk within the next hour.",
    ],
    QuantLevel.INT8: [
        "Revenue up 23% YoY. APAC strongest. Margins improved 2.1 points.",
        "Design team meeting in 45 min. Summary ready.",
        "Clear until 3 PM, then rain likely. Walk soon.",
    ],
    QuantLevel.INT4: [
        "Revenue: +23%. Margins: +2.1pt.",
        "Meeting 45 min. Summary ready.",
        "Rain after 3 PM.",
    ],
    QuantLevel.INT2: [
        "Revenue up.",
        "Meeting soon.",
        "Rain later.",
    ],
}


class VoiceAgent(BaseAgent):
    """Voice interaction agent — NLU + LLM inference.
    
    Priority: RT_SOFT (can be degraded, reluctant to evict)
    When degraded, produces shorter/simpler responses.
    """

    def __init__(self):
        super().__init__(
            name="Voice_NLU",
            agent_class=AgentClass.RT_SOFT,
            base_utility=80.0,
            power_cost_watts=8.0,
            ram_mb=1536.0,
            gpu_tops_required=12.0,
            is_rigid=False,
            description="Voice-first interaction: wake word → NLU → LLM → TTS",
        )
        self._in_conversation: bool = False
        self._last_response: str = ""
        self._response_count: int = 0

    def setup(self) -> None:
        """Load voice pipeline models."""
        # Production: load Whisper + LLM + TTS models
        # Demo: no-op
        pass

    def process(self, input_data: Any = None,
                quant_level: QuantLevel = QuantLevel.FP16) -> dict:
        """Process voice input and generate response.
        
        Degradation behavior:
          FP16 → Full, detailed responses
          INT8 → Concise responses
          INT4 → Minimal responses
          INT2 → One-word responses
        """
        # Simulate inference latency based on quant level
        latency_map = {
            QuantLevel.FP16: (0.5, 1.2),
            QuantLevel.INT8: (0.3, 0.7),
            QuantLevel.INT4: (0.1, 0.3),
            QuantLevel.INT2: (0.05, 0.1),
        }
        lo, hi = latency_map.get(quant_level, (0.3, 0.7))
        time.sleep(random.uniform(lo, hi))

        # Select response based on quant level
        responses = DEMO_RESPONSES.get(quant_level, DEMO_RESPONSES[QuantLevel.INT8])
        self._last_response = random.choice(responses)
        self._response_count += 1

        return {
            "response": self._last_response,
            "quant_level": quant_level.name,
            "tokens": len(self._last_response.split()),
            "response_id": self._response_count,
        }

    def teardown(self) -> None:
        """Unload voice models."""
        self._in_conversation = False

    def on_degrade(self, old_quant: QuantLevel, new_quant: QuantLevel) -> None:
        """When degraded, switch to simpler response generation."""
        pass

    def on_suspend(self, reason: str) -> None:
        """If suspended during conversation, mark conversation as interrupted."""
        if self._in_conversation:
            self._in_conversation = False
