"""
OnyxOS Thermal Governor — Physics-Aware Power Budget Controller

Reads thermal data (real sysfs on Jetson, mock on dev machines) and
computes a dynamic power budget that the market scheduler uses as
its power constraint.

The thermal model is an RC network:
  T(t+Δ) = T(t) + [P_total - (T(t) - T_amb) / R_th] / C_th · Δt

The power budget is:
  P_budget = P_sustainable + α · C_th · (T_limit - T_current) / H

Where:
  P_sustainable = (T_limit - T_ambient) / R_thermal
  α = safety factor (0.8)
  H = tick interval (1s)

When temperature exceeds T_throttle, the governor reduces the budget
aggressively. At T_emergency, it triggers an emergency shutdown.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional

from onyx_kernel.config import HardwareConfig, ThermalConfig, SystemConfig, DEFAULT_CONFIG


class ThermalState(Enum):
    """Current thermal state of the system."""
    NOMINAL = auto()     # T < T_throttle — running normally
    THROTTLED = auto()   # T_throttle ≤ T < T_emergency — reducing power
    EMERGENCY = auto()   # T ≥ T_emergency — emergency shutdown
    UNKNOWN = auto()     # Cannot read thermal sensors


@dataclass
class ThermalSnapshot:
    """A point-in-time capture of all thermal data."""
    timestamp: float
    cpu_temp: float
    gpu_temp: float
    soc_temp: float       # Highest of all zones (worst case)
    ambient_temp: float
    state: ThermalState
    power_budget_watts: float
    power_sustainable_watts: float
    thermal_headroom: float  # T_limit - T_current
    power_draw_watts: float  # Actual power draw (if available)


class ThermalGovernor:
    """Physics-aware power budget controller.
    
    On a real Jetson, reads from /sys/devices/virtual/thermal/.
    On dev machines, simulates thermal behavior based on reported power draw.
    """

    # Jetson thermal zone paths
    JETSON_THERMAL_ZONES = {
        "cpu": "/sys/devices/virtual/thermal/thermal_zone0/temp",
        "gpu": "/sys/devices/virtual/thermal/thermal_zone1/temp",
        "cv": "/sys/devices/virtual/thermal/thermal_zone2/temp",
        "soc0": "/sys/devices/virtual/thermal/thermal_zone3/temp",
        "soc1": "/sys/devices/virtual/thermal/thermal_zone4/temp",
        "soc2": "/sys/devices/virtual/thermal/thermal_zone5/temp",
        "tj": "/sys/devices/virtual/thermal/thermal_zone6/temp",
        "pmic": "/sys/devices/virtual/thermal/thermal_zone7/temp",
    }

    # Jetson INA3221 power monitor paths
    JETSON_POWER_PATHS = {
        "total": "/sys/bus/i2c/drivers/ina3221/7-0040/hwmon/hwmon3/in1_input",
        "gpu": "/sys/bus/i2c/drivers/ina3221/7-0040/hwmon/hwmon3/in2_input",
        "cpu": "/sys/bus/i2c/drivers/ina3221/7-0040/hwmon/hwmon3/in3_input",
    }

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.tc = self.config.thermal
        self.hc = self.config.hardware

        # Detect if running on Jetson
        self.is_jetson = self._detect_jetson()

        # Simulation state (used when not on Jetson)
        self._sim_temp: float = 30.0  # Start at a warm-ish temperature
        self._sim_power_draw: float = 0.0

        # History for HUD
        self.history: List[ThermalSnapshot] = []
        self._max_history = 300

        # Current state
        self.current_state = ThermalState.NOMINAL
        self.current_temp: float = self.tc.T_AMBIENT
        self.current_budget: float = self.tc.P_SUSTAINABLE

    def _detect_jetson(self) -> bool:
        """Detect if we're running on a Jetson device."""
        thermal_path = Path("/sys/devices/virtual/thermal/thermal_zone0/temp")
        return thermal_path.exists()

    def _read_sysfs_temp(self, path: str) -> Optional[float]:
        """Read temperature from sysfs (millidegrees Celsius → Celsius)."""
        try:
            with open(path, 'r') as f:
                return int(f.read().strip()) / 1000.0
        except (FileNotFoundError, PermissionError, ValueError):
            return None

    def _read_sysfs_power(self, path: str) -> Optional[float]:
        """Read power from sysfs (milliwatts → Watts)."""
        try:
            with open(path, 'r') as f:
                return int(f.read().strip()) / 1000.0
        except (FileNotFoundError, PermissionError, ValueError):
            return None

    def _read_jetson_thermals(self) -> Dict[str, float]:
        """Read all thermal zones from a Jetson device."""
        temps = {}
        for name, path in self.JETSON_THERMAL_ZONES.items():
            temp = self._read_sysfs_temp(path)
            if temp is not None:
                temps[name] = temp
        return temps

    def _simulate_thermal(self, power_draw: float, dt: float = 1.0) -> float:
        """Simulate thermal behavior using RC model.
        
        T(t+Δ) = T(t) + [P - (T - T_amb)/R_th] / C_th · Δt
        """
        cooling = (self._sim_temp - self.tc.T_AMBIENT) / self.tc.R_THERMAL
        delta_t = (power_draw - cooling) / self.tc.C_THERMAL * dt
        self._sim_temp += delta_t

        # Clamp to physical bounds
        self._sim_temp = max(self.tc.T_AMBIENT - 5.0,
                             min(self.tc.T_EMERGENCY + 10.0, self._sim_temp))
        return self._sim_temp

    def compute_power_budget(self, current_temp: Optional[float] = None) -> float:
        """Compute the dynamic power budget based on current temperature.
        
        P_budget = P_sus + α · C_th · (T_limit - T_current) / H
        
        Returns the maximum watts that agents are allowed to consume.
        """
        if current_temp is None:
            current_temp = self.current_temp

        headroom = self.tc.T_SKIN_LIMIT - current_temp
        p_sus = self.tc.P_SUSTAINABLE

        if headroom <= 0:
            # Over the limit — emergency, allow only minimum
            return self.hc.POWER_MIN_W

        if current_temp >= self.tc.T_THROTTLE:
            # Throttle zone — reduce budget proportionally
            throttle_range = self.tc.T_SKIN_LIMIT - self.tc.T_THROTTLE
            throttle_factor = headroom / throttle_range if throttle_range > 0 else 0
            return p_sus * throttle_factor

        # Normal operation — budget includes thermal headroom
        budget = p_sus + (
            self.tc.SAFETY_FACTOR
            * self.tc.C_THERMAL
            * headroom
            / self.tc.TICK_INTERVAL
        )

        # Clamp to hardware limits
        return max(self.hc.POWER_MIN_W, min(self.hc.POWER_MAX_W, budget))

    def tick(self, actual_power_draw: float = 0.0) -> ThermalSnapshot:
        """Run one tick of the thermal governor.
        
        Args:
            actual_power_draw: The actual power consumed by agents this tick (Watts)
        
        Returns:
            ThermalSnapshot with current state and updated power budget
        """
        # Read current temperature
        if self.is_jetson:
            temps = self._read_jetson_thermals()
            cpu_temp = temps.get("cpu", self.tc.T_AMBIENT)
            gpu_temp = temps.get("gpu", self.tc.T_AMBIENT)
            soc_temp = max(temps.values()) if temps else self.tc.T_AMBIENT

            # Read actual power draw from INA3221
            real_power = self._read_sysfs_power(
                self.JETSON_POWER_PATHS.get("total", "")
            )
            if real_power is not None:
                actual_power_draw = real_power
        else:
            # Simulation mode
            self._sim_power_draw = actual_power_draw
            soc_temp = self._simulate_thermal(actual_power_draw)
            cpu_temp = soc_temp - 2.0  # CPU slightly cooler
            gpu_temp = soc_temp + 1.0  # GPU slightly hotter

        self.current_temp = soc_temp

        # Determine thermal state
        if soc_temp >= self.tc.T_EMERGENCY:
            self.current_state = ThermalState.EMERGENCY
        elif soc_temp >= self.tc.T_THROTTLE:
            self.current_state = ThermalState.THROTTLED
        else:
            self.current_state = ThermalState.NOMINAL

        # Compute power budget
        self.current_budget = self.compute_power_budget(soc_temp)

        # Create snapshot
        snapshot = ThermalSnapshot(
            timestamp=time.time(),
            cpu_temp=cpu_temp,
            gpu_temp=gpu_temp,
            soc_temp=soc_temp,
            ambient_temp=self.tc.T_AMBIENT,
            state=self.current_state,
            power_budget_watts=self.current_budget,
            power_sustainable_watts=self.tc.P_SUSTAINABLE,
            thermal_headroom=self.tc.T_SKIN_LIMIT - soc_temp,
            power_draw_watts=actual_power_draw,
        )

        # Store history
        self.history.append(snapshot)
        if len(self.history) > self._max_history:
            self.history.pop(0)

        return snapshot

    def get_temp_bar(self, width: int = 20) -> str:
        """Generate a colored temperature bar for HUD display."""
        if self.current_temp <= self.tc.T_AMBIENT:
            fill = 0
        else:
            temp_range = self.tc.T_EMERGENCY - self.tc.T_AMBIENT
            fill = (self.current_temp - self.tc.T_AMBIENT) / temp_range
        fill = max(0.0, min(1.0, fill))

        filled = int(fill * width)
        bar = "█" * filled + "░" * (width - filled)

        return f"[{bar}] {self.current_temp:.1f}°C"

    def status_summary(self) -> str:
        """One-line thermal status for display."""
        state_icons = {
            ThermalState.NOMINAL: "🟢",
            ThermalState.THROTTLED: "🟡",
            ThermalState.EMERGENCY: "🔴",
            ThermalState.UNKNOWN: "⚪",
        }
        icon = state_icons.get(self.current_state, "❓")
        return (
            f"{icon} {self.current_temp:.1f}°C | "
            f"Budget: {self.current_budget:.1f}W / {self.hc.POWER_MAX_W:.0f}W | "
            f"Headroom: {self.tc.T_SKIN_LIMIT - self.current_temp:.1f}°C"
        )
