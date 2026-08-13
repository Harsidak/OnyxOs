"""
OnyxOS Emotion Agent — Real-Time Emotion Detection (RT_HARD)

This is a RIGID agent — it cannot be degraded or evicted.
It represents the safety-critical emotion detection pipeline
that must always run on AR glasses to detect user distress.

In production: camera frame → face detection → emotion classification
For demo: simulated emotion cycling with realistic timing
"""

from __future__ import annotations

import random
import time
from typing import Any

from onyx_kernel.config import AgentClass, QuantLevel
from agents.base_agent import BaseAgent


EMOTIONS = [
    ("neutral", 0.85),
    ("happy", 0.72),
    ("focused", 0.91),
    ("surprised", 0.68),
    ("stressed", 0.77),
    ("calm", 0.89),
    ("confused", 0.63),
    ("excited", 0.74),
]


class EmotionAgent(BaseAgent):
    """Real-time emotion detection from camera input.
    
    Priority: RT_HARD (cannot be degraded or evicted)
    This agent is rigid — the market MUST always allocate
    resources for it, regardless of price pressure.
    """

    def __init__(self):
        super().__init__(
            name="Emotion_RT",
            agent_class=AgentClass.RT_HARD,
            base_utility=100.0,
            power_cost_watts=5.0,
            ram_mb=1024.0,
            gpu_tops_required=8.0,
            is_rigid=True,
            description="Real-time emotion detection from camera feed",
        )
        self._current_emotion: str = "neutral"
        self._confidence: float = 0.0
        self._frame_count: int = 0

    def setup(self) -> None:
        """Load emotion detection model."""
        # In production: load ONNX/TensorRT emotion model
        # For demo: no-op, we simulate
        self._current_emotion = "neutral"
        self._confidence = 0.85

    def process(self, input_data: Any = None,
                quant_level: QuantLevel = QuantLevel.FP16) -> dict:
        """Process one camera frame for emotion detection.
        
        Returns:
            dict with 'emotion', 'confidence', 'frame_id'
        """
        # Simulate inference with realistic timing
        time.sleep(random.uniform(0.01, 0.03))  # 10-30ms inference

        # Cycle through emotions with some randomness
        self._frame_count += 1
        if self._frame_count % 30 == 0:  # Change emotion every ~30 frames
            emotion, base_conf = random.choice(EMOTIONS)
            self._current_emotion = emotion
            self._confidence = base_conf + random.uniform(-0.05, 0.05)

        # Add slight confidence jitter per frame
        jittered_conf = self._confidence + random.uniform(-0.02, 0.02)
        jittered_conf = max(0.0, min(1.0, jittered_conf))

        return {
            "emotion": self._current_emotion,
            "confidence": round(jittered_conf, 3),
            "frame_id": self._frame_count,
        }

    def teardown(self) -> None:
        """Unload emotion model."""
        pass

    def on_admit(self, quant_level: QuantLevel) -> None:
        """RT_HARD — always admitted at FP16."""
        pass
