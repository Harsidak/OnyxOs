"""
OnyxOS World Model Agent — Spatial Awareness & Scene Understanding (RT_SOFT)

Processes camera input to build a spatial understanding of the environment:
  - Object detection and tracking
  - Depth estimation
  - Scene graph construction
  - Navigation cues for AR overlay

Can be degraded to lower-resolution processing when resources are scarce.

For demo: simulates scene detection with rotating objects
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List

from onyx_kernel.config import AgentClass, QuantLevel
from agents.base_agent import BaseAgent


SCENE_OBJECTS = [
    {"label": "desk", "confidence": 0.94, "distance_m": 0.8},
    {"label": "monitor", "confidence": 0.91, "distance_m": 1.2},
    {"label": "coffee_cup", "confidence": 0.87, "distance_m": 0.5},
    {"label": "keyboard", "confidence": 0.93, "distance_m": 0.7},
    {"label": "person", "confidence": 0.89, "distance_m": 2.5},
    {"label": "door", "confidence": 0.96, "distance_m": 4.0},
    {"label": "window", "confidence": 0.92, "distance_m": 3.2},
    {"label": "plant", "confidence": 0.85, "distance_m": 1.8},
    {"label": "book", "confidence": 0.78, "distance_m": 0.6},
    {"label": "phone", "confidence": 0.90, "distance_m": 0.4},
]


class WorldModelAgent(BaseAgent):
    """Spatial awareness agent — camera → scene understanding.
    
    Priority: RT_SOFT (can be degraded to lower resolution)
    
    Degradation behavior:
      FP16 → Full scene graph with 10+ objects, depth estimation
      INT8 → Top-5 objects, approximate depth
      INT4 → Top-3 objects, no depth
      INT2 → Single dominant object only
    """

    def __init__(self):
        super().__init__(
            name="World_Model",
            agent_class=AgentClass.RT_SOFT,
            base_utility=70.0,
            power_cost_watts=15.0,
            ram_mb=2048.0,
            gpu_tops_required=20.0,
            is_rigid=False,
            description="Spatial awareness: object detection, depth, scene graph",
        )
        self._frame_count: int = 0
        self._scene_objects: List[Dict] = []

    def setup(self) -> None:
        """Load vision models."""
        # Production: load YOLO/DETR + MiDaS + scene graph model
        # Demo: pre-populate with simulated objects
        self._scene_objects = SCENE_OBJECTS.copy()

    def process(self, input_data: Any = None,
                quant_level: QuantLevel = QuantLevel.FP16) -> dict:
        """Process one camera frame for scene understanding.
        
        The number of detected objects decreases with quantization.
        """
        # Simulate inference latency
        latency_map = {
            QuantLevel.FP16: (0.05, 0.10),
            QuantLevel.INT8: (0.03, 0.06),
            QuantLevel.INT4: (0.02, 0.04),
            QuantLevel.INT2: (0.01, 0.02),
        }
        lo, hi = latency_map.get(quant_level, (0.03, 0.06))
        time.sleep(random.uniform(lo, hi))

        self._frame_count += 1

        # Number of objects detected depends on quant level
        max_objects = {
            QuantLevel.FP16: 10,
            QuantLevel.INT8: 5,
            QuantLevel.INT4: 3,
            QuantLevel.INT2: 1,
        }
        n_objects = max_objects.get(quant_level, 5)

        # Add some variance to detections
        detected = []
        for obj in self._scene_objects[:n_objects]:
            detected.append({
                "label": obj["label"],
                "confidence": round(
                    obj["confidence"] + random.uniform(-0.03, 0.03), 3
                ),
                "distance_m": round(
                    obj["distance_m"] + random.uniform(-0.1, 0.1), 2
                ) if quant_level in (QuantLevel.FP16, QuantLevel.INT8) else None,
            })

        return {
            "objects": detected,
            "object_count": len(detected),
            "frame_id": self._frame_count,
            "quant_level": quant_level.name,
            "has_depth": quant_level in (QuantLevel.FP16, QuantLevel.INT8),
        }

    def teardown(self) -> None:
        """Unload vision models."""
        self._scene_objects = []

    def on_degrade(self, old_quant: QuantLevel, new_quant: QuantLevel) -> None:
        """Switching to lower-resolution detection."""
        pass
