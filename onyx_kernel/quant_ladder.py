"""
OnyxOS Quantization Ladder — Model Precision Manager

Agents can trade precision for resource savings. The quant ladder
provides the mapping between quantization levels and their
utility/memory/compute characteristics.

When memory prices (λ_M) spike, agents voluntarily step down the
ladder to reduce their footprint. This is the key mechanism that
prevents OOM kills — agents degrade gracefully instead of crashing.

Levels:
  FP16  → Full precision, best quality    (1.0x resources)
  INT8  → 8-bit quantized, good quality   (0.5x resources)
  INT4  → 4-bit quantized, acceptable     (0.25x resources)
  INT2  → 2-bit quantized, degraded       (0.125x resources)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from onyx_kernel.config import QuantLevel


@dataclass(frozen=True)
class QuantProfile:
    """Characteristics of a quantization level for a model."""
    level: QuantLevel
    memory_fraction: float    # Fraction of FP16 memory used
    compute_fraction: float   # Fraction of FP16 compute used
    quality_fraction: float   # Expected quality relative to FP16
    latency_multiplier: float # Expected latency relative to FP16
    description: str


# Default quantization profiles
DEFAULT_PROFILES: Dict[QuantLevel, QuantProfile] = {
    QuantLevel.FP16: QuantProfile(
        level=QuantLevel.FP16,
        memory_fraction=1.0,
        compute_fraction=1.0,
        quality_fraction=1.0,
        latency_multiplier=1.0,
        description="Full precision — maximum quality",
    ),
    QuantLevel.INT8: QuantProfile(
        level=QuantLevel.INT8,
        memory_fraction=0.5,
        compute_fraction=0.6,
        quality_fraction=0.95,
        latency_multiplier=0.7,
        description="8-bit integer — near-lossless, 2x memory savings",
    ),
    QuantLevel.INT4: QuantProfile(
        level=QuantLevel.INT4,
        memory_fraction=0.25,
        compute_fraction=0.35,
        quality_fraction=0.85,
        latency_multiplier=0.5,
        description="4-bit integer — good quality, 4x memory savings",
    ),
    QuantLevel.INT2: QuantProfile(
        level=QuantLevel.INT2,
        memory_fraction=0.125,
        compute_fraction=0.2,
        quality_fraction=0.6,
        latency_multiplier=0.3,
        description="2-bit integer — degraded quality, 8x memory savings",
    ),
}


class QuantLadder:
    """Manages the quantization ladder for an agent.
    
    The ladder defines which quantization levels an agent supports
    and provides methods to find the optimal level given current
    market prices.
    """

    def __init__(
        self,
        supported_levels: Optional[List[QuantLevel]] = None,
        profiles: Optional[Dict[QuantLevel, QuantProfile]] = None,
    ):
        """Initialize the quantization ladder.
        
        Args:
            supported_levels: Which levels this agent supports.
                Defaults to all levels.
            profiles: Custom quantization profiles.
                Defaults to DEFAULT_PROFILES.
        """
        self.profiles = profiles or DEFAULT_PROFILES.copy()

        if supported_levels:
            self.supported_levels = sorted(
                supported_levels, key=lambda l: l.value, reverse=True
            )
        else:
            self.supported_levels = sorted(
                list(QuantLevel), key=lambda l: l.value, reverse=True
            )

    def get_profile(self, level: QuantLevel) -> QuantProfile:
        """Get the profile for a quantization level."""
        return self.profiles[level]

    def memory_at(self, base_ram_mb: float, level: QuantLevel) -> float:
        """Compute memory footprint at a given quantization level.
        
        Args:
            base_ram_mb: Full-precision (FP16) memory in MB
            level: Target quantization level
        
        Returns:
            Memory in MB at the target level
        """
        profile = self.profiles[level]
        return base_ram_mb * profile.memory_fraction

    def utility_at(
        self,
        base_utility: float,
        level: QuantLevel,
        exponent: float = 0.5,
    ) -> float:
        """Compute utility at a given quantization level.
        
        U(q) = U_base · quality_fraction^exponent
        
        Args:
            base_utility: Full-precision utility
            level: Target quantization level
            exponent: How sharply utility drops (lower = more tolerant)
        
        Returns:
            Utility at the target level
        """
        profile = self.profiles[level]
        return base_utility * (profile.quality_fraction ** exponent)

    def find_optimal_level(
        self,
        base_utility: float,
        base_ram_mb: float,
        power_cost: float,
        lambda_p: float,
        lambda_m: float,
        lambda_g: float = 0.0,
        gpu_required: float = 0.0,
        utility_exponent: float = 0.5,
        is_rigid: bool = False,
    ) -> Tuple[QuantLevel, float]:
        """Find the quantization level that maximizes surplus.
        
        surplus(q) = U(q) - λ_P·ε - λ_M·M(q) - λ_G·G
        
        Args:
            base_utility: Full-precision utility
            base_ram_mb: Full-precision memory (MB)
            power_cost: Power cost (Watts) — same at all quant levels
            lambda_p: Current power price
            lambda_m: Current memory price
            lambda_g: Current GPU price
            gpu_required: GPU compute required (TOPS)
            utility_exponent: How sharply utility drops
            is_rigid: If True, only FP16 is valid
        
        Returns:
            Tuple of (best_level, best_surplus)
        """
        if is_rigid:
            surplus = (
                self.utility_at(base_utility, QuantLevel.FP16, utility_exponent)
                - lambda_p * power_cost
                - lambda_m * self.memory_at(base_ram_mb, QuantLevel.FP16)
                - lambda_g * gpu_required
            )
            return QuantLevel.FP16, surplus

        best_level = QuantLevel.FP16
        best_surplus = float('-inf')

        for level in self.supported_levels:
            utility = self.utility_at(base_utility, level, utility_exponent)
            memory = self.memory_at(base_ram_mb, level)
            cost = lambda_p * power_cost + lambda_m * memory + lambda_g * gpu_required
            surplus = utility - cost

            if surplus > best_surplus:
                best_surplus = surplus
                best_level = level

        return best_level, best_surplus

    def step_down(self, current: QuantLevel) -> Optional[QuantLevel]:
        """Get the next lower quantization level.
        
        Returns None if already at the lowest level.
        """
        levels = self.supported_levels
        for i, level in enumerate(levels):
            if level == current and i + 1 < len(levels):
                return levels[i + 1]
        return None

    def step_up(self, current: QuantLevel) -> Optional[QuantLevel]:
        """Get the next higher quantization level.
        
        Returns None if already at the highest level.
        """
        levels = self.supported_levels
        for i, level in enumerate(levels):
            if level == current and i > 0:
                return levels[i - 1]
        return None

    def ladder_display(self, current: QuantLevel, base_ram_mb: float) -> str:
        """Generate a visual ladder display for HUD."""
        lines = []
        for level in self.supported_levels:
            profile = self.profiles[level]
            mem = self.memory_at(base_ram_mb, level)
            marker = "→" if level == current else " "
            lines.append(
                f"  {marker} {level.name:<4s} | "
                f"Quality: {profile.quality_fraction * 100:>5.1f}% | "
                f"RAM: {mem:>6.0f}MB | "
                f"Speed: {1/profile.latency_multiplier:>4.1f}x"
            )
        return "\n".join(lines)
