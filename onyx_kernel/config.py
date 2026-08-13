"""
OnyxOS Kernel — Hardware Constants & System Configuration

All physical parameters for the NVIDIA Jetson Orin Nano Super
and thermal model for AR glasses form factor.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict


class AgentClass(Enum):
    """Agent priority classifications."""
    RT_HARD = auto()   # Real-time hard — cannot be degraded or evicted (e.g., safety)
    RT_SOFT = auto()   # Real-time soft — can be degraded but not easily evicted
    BG = auto()        # Background — first to degrade/evict when resources scarce


class AgentState(Enum):
    """Agent lifecycle states."""
    REGISTERED = auto()  # Known to registry but not loaded
    READY = auto()       # Model loaded, waiting for admission
    RUNNING = auto()     # Actively executing inference
    DEGRADED = auto()    # Running at lower quantization
    SUSPENDED = auto()   # Temporarily paused, model still in memory
    EVICTED = auto()     # Model unloaded from memory


class QuantLevel(Enum):
    """Quantization levels for the quantization ladder.
    
    Each level trades precision for memory/compute savings.
    Value represents the fraction of full-precision resources used.
    """
    FP16 = 1.0     # Full precision — best quality, most expensive
    INT8 = 0.5     # 8-bit integer — good quality, half memory
    INT4 = 0.25    # 4-bit integer — acceptable quality, quarter memory
    INT2 = 0.125   # 2-bit integer — degraded quality, minimal memory


@dataclass(frozen=True)
class ThermalConfig:
    """Thermal model parameters for AR glasses form factor.
    
    Based on skin-contact thermal limits (IEC 62368-1) and
    simplified RC thermal network model.
    """
    T_AMBIENT: float = 25.0       # Ambient temperature (°C)
    T_SKIN_LIMIT: float = 43.0    # Skin-contact hard limit (°C) — IEC 62368-1
    T_THROTTLE: float = 41.0      # Start throttling (°C) — 2°C safety margin
    T_EMERGENCY: float = 44.0     # Emergency shutdown (°C)
    R_THERMAL: float = 1.5        # Thermal resistance (°C/W) — package to skin
    C_THERMAL: float = 10.0       # Thermal capacitance (J/°C) — thermal mass
    SAFETY_FACTOR: float = 0.8    # α — safety margin for power budget (0-1)
    TICK_INTERVAL: float = 1.0    # H — scheduler tick interval (seconds)

    @property
    def P_SUSTAINABLE(self) -> float:
        """Steady-state sustainable power dissipation (Watts).
        P_sus = (T_limit - T_amb) / R_th
        """
        return (self.T_SKIN_LIMIT - self.T_AMBIENT) / self.R_THERMAL


@dataclass(frozen=True)
class HardwareConfig:
    """NVIDIA Jetson Orin Nano Super hardware specifications."""
    # Memory
    RAM_TOTAL_MB: int = 8192          # 8GB LPDDR5 unified memory
    RAM_RESERVED_MB: int = 1024       # Reserved for Linux + display
    RAM_AVAILABLE_MB: int = 7168      # Available for agents (8192 - 1024)
    RAM_BANDWIDTH_GBPS: float = 102.0 # Memory bandwidth

    # Compute
    GPU_CUDA_CORES: int = 1024        # Ampere CUDA cores
    GPU_TENSOR_CORES: int = 32        # Tensor cores
    GPU_TOPS: float = 67.0            # INT8 TOPS
    CPU_CORES: int = 6                # Arm Cortex-A78AE

    # Power
    POWER_MIN_W: float = 7.0          # Minimum power mode
    POWER_MAX_W: float = 25.0         # MAXN SUPER mode
    POWER_DEFAULT_W: float = 15.0     # Default power budget

    # Sensors
    THERMAL_ZONES: int = 8            # Number of thermal zones
    HAS_INA3221: bool = True          # Power monitor present


@dataclass(frozen=True)
class MarketConfig:
    """Lagrangian dual-ascent market parameters.
    
    These control the convergence speed and stability of the
    resource market auction.
    """
    # Initial dual prices (Lagrange multipliers)
    LAMBDA_POWER_INIT: float = 1.0    # Initial price of 1 Watt
    LAMBDA_MEMORY_INIT: float = 1.0   # Initial price of 1 MB
    LAMBDA_GPU_INIT: float = 1.0      # Initial price of 1 TOPS

    # Learning rates for dual ascent (η)
    ETA_POWER: float = 0.1            # Power price learning rate
    ETA_MEMORY: float = 0.05          # Memory price learning rate
    ETA_GPU: float = 0.08             # GPU price learning rate

    # Price bounds (prevent runaway)
    LAMBDA_MIN: float = 0.01          # Minimum price floor
    LAMBDA_MAX: float = 100.0         # Maximum price ceiling

    # Convergence
    PRICE_DAMPENING: float = 0.9      # Exponential moving average factor
    CONVERGENCE_EPSILON: float = 0.01 # Price change threshold for convergence
    MAX_AUCTION_ROUNDS: int = 50      # Max iterations per tick


@dataclass
class SystemConfig:
    """Top-level OnyxOS configuration."""
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    market: MarketConfig = field(default_factory=MarketConfig)

    # Scheduler
    tick_rate_hz: float = 1.0          # Market auction frequency
    enable_thermal_governor: bool = True
    enable_quant_ladder: bool = True

    # Logging
    log_auction_events: bool = True
    log_thermal_events: bool = True
    log_agent_state_changes: bool = True

    # Demo mode
    demo_mode: bool = False            # Use mock data instead of real hardware
    demo_speed_multiplier: float = 1.0 # Speed up simulation for demos


# Singleton default configuration
DEFAULT_CONFIG = SystemConfig()
