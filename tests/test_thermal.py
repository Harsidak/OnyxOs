"""Tests for the OnyxOS Thermal Governor."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from onyx_kernel.config import SystemConfig, ThermalConfig
from onyx_kernel.thermal_governor import ThermalGovernor, ThermalState


class TestThermalGovernor:
    """Test thermal governor behavior."""

    def test_nominal_state_at_low_temp(self):
        """System should be NOMINAL when temperature is well below throttle."""
        gov = ThermalGovernor()
        gov._sim_temp = 30.0  # Well below throttle (41°C)

        snap = gov.tick(actual_power_draw=10.0)
        assert snap.state == ThermalState.NOMINAL

    def test_throttled_state_near_limit(self):
        """System should be THROTTLED when near skin limit."""
        gov = ThermalGovernor()
        gov._sim_temp = 42.0  # Above throttle (41°C), below limit (43°C)

        snap = gov.tick(actual_power_draw=10.0)
        assert snap.state == ThermalState.THROTTLED

    def test_emergency_state_over_limit(self):
        """System should be EMERGENCY when over emergency temp."""
        gov = ThermalGovernor()
        gov._sim_temp = 45.0  # Above emergency (44°C)

        snap = gov.tick(actual_power_draw=10.0)
        assert snap.state == ThermalState.EMERGENCY

    def test_power_budget_decreases_with_temperature(self):
        """Higher temperature should result in lower power budget."""
        gov = ThermalGovernor()

        # Use temps that won't both clamp to POWER_MAX_W
        budget_warm = gov.compute_power_budget(38.0)
        budget_hot = gov.compute_power_budget(42.0)
        budget_very_hot = gov.compute_power_budget(42.8)

        assert budget_warm > budget_hot > budget_very_hot

    def test_sustainable_power_calculation(self):
        """P_sus = (T_limit - T_amb) / R_th."""
        config = SystemConfig()
        tc = config.thermal
        expected = (tc.T_SKIN_LIMIT - tc.T_AMBIENT) / tc.R_THERMAL

        assert abs(tc.P_SUSTAINABLE - expected) < 0.01

    def test_budget_never_below_minimum(self):
        """Power budget should never go below hardware minimum."""
        gov = ThermalGovernor()

        budget = gov.compute_power_budget(50.0)  # Way over limit
        assert budget >= gov.hc.POWER_MIN_W

    def test_budget_never_above_maximum(self):
        """Power budget should never exceed hardware maximum."""
        gov = ThermalGovernor()

        budget = gov.compute_power_budget(20.0)  # Very cool
        assert budget <= gov.hc.POWER_MAX_W

    def test_simulation_heats_up_under_load(self):
        """Temperature should increase when power draw is high."""
        gov = ThermalGovernor()
        gov._sim_temp = 30.0

        initial_temp = gov._sim_temp
        for _ in range(20):
            gov.tick(actual_power_draw=20.0)  # High power

        assert gov.current_temp > initial_temp

    def test_simulation_cools_down_with_no_load(self):
        """Temperature should decrease when power draw is zero."""
        gov = ThermalGovernor()
        gov._sim_temp = 40.0  # Start hot

        initial_temp = gov._sim_temp
        for _ in range(20):
            gov.tick(actual_power_draw=0.0)  # No power

        assert gov.current_temp < initial_temp

    def test_history_bounded(self):
        """Temperature history should not grow unbounded."""
        gov = ThermalGovernor()

        for _ in range(500):
            gov.tick(actual_power_draw=10.0)

        assert len(gov.history) <= gov._max_history

    def test_temp_bar_output(self):
        """Temperature bar should be a valid string."""
        gov = ThermalGovernor()
        gov._sim_temp = 35.0
        gov.tick(actual_power_draw=10.0)

        bar = gov.get_temp_bar(width=20)
        assert "°C" in bar
        assert len(bar) > 20  # bar + temp text

    def test_status_summary(self):
        """Status summary should contain key info."""
        gov = ThermalGovernor()
        gov.tick(actual_power_draw=10.0)

        summary = gov.status_summary()
        assert "°C" in summary
        assert "W" in summary
