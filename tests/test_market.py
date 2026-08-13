"""Tests for the OnyxOS Lagrangian Market Scheduler."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from onyx_kernel.config import (
    AgentClass, AgentState, QuantLevel, SystemConfig, MarketConfig,
)
from onyx_kernel.agent import AgentITD
from onyx_kernel.market import LagrangianMarket


def make_test_agents():
    """Create a standard set of test agents."""
    return [
        AgentITD(
            name="Emotion_RT",
            agent_class=AgentClass.RT_HARD,
            base_utility=100.0,
            power_cost_watts=5.0,
            ram_mb=1024.0,
            is_rigid=True,
        ),
        AgentITD(
            name="Voice_NLU",
            agent_class=AgentClass.RT_SOFT,
            base_utility=80.0,
            power_cost_watts=8.0,
            ram_mb=1536.0,
        ),
        AgentITD(
            name="World_Model",
            agent_class=AgentClass.RT_SOFT,
            base_utility=70.0,
            power_cost_watts=15.0,
            ram_mb=2048.0,
        ),
        AgentITD(
            name="Loop_Agent",
            agent_class=AgentClass.BG,
            base_utility=30.0,
            power_cost_watts=8.0,
            ram_mb=1536.0,
        ),
    ]


class TestMarketBasics:
    """Test basic market auction behavior."""

    def test_rt_hard_always_admitted(self):
        """RT_HARD agents must always be admitted regardless of prices."""
        market = LagrangianMarket()
        agents = make_test_agents()

        # Set extremely high prices to make everything unprofitable
        market.lambda_power = 50.0
        market.lambda_memory = 50.0

        result = market.run_auction(
            agents=agents,
            power_budget=100.0,
            memory_budget=8192.0,
            gpu_budget=67.0,
        )

        # Emotion_RT (RT_HARD) must always be admitted
        assert "Emotion_RT" in result.admitted
        emotion = next(a for a in agents if a.name == "Emotion_RT")
        assert emotion.state in (AgentState.RUNNING, AgentState.DEGRADED)

    def test_bg_agents_evicted_under_pressure(self):
        """BG agents should be evicted when budget is very tight."""
        config = SystemConfig()
        market = LagrangianMarket(config)
        agents = make_test_agents()

        # Very tight power budget — only enough for RT_HARD
        result = market.run_auction(
            agents=agents,
            power_budget=6.0,  # Only 6W — barely enough for Emotion_RT (5W)
            memory_budget=2048.0,  # Tight memory too
            gpu_budget=67.0,
        )

        # Loop_Agent (BG) should not be running
        loop = next(a for a in agents if a.name == "Loop_Agent")
        assert loop.state in (AgentState.SUSPENDED, AgentState.EVICTED)

    def test_prices_rise_when_over_budget(self):
        """Dual prices should increase when demand exceeds supply."""
        market = LagrangianMarket()
        agents = make_test_agents()

        initial_lambda_p = market.lambda_power

        # Run multiple ticks with tight budget so demand > supply
        # (dampening factor means single tick may not show net increase)
        for _ in range(10):
            market.run_auction(
                agents=agents,
                power_budget=10.0,  # Total demand ~36W, budget 10W
                memory_budget=4096.0,
                gpu_budget=67.0,
            )

        # Power price should have increased over multiple ticks
        assert market.lambda_power > initial_lambda_p

    def test_prices_fall_when_under_budget(self):
        """Dual prices should decrease when supply exceeds demand."""
        market = LagrangianMarket()
        agents = make_test_agents()

        # Start with high prices
        market.lambda_power = 10.0
        market.lambda_memory = 10.0

        # Run with very generous budget
        market.run_auction(
            agents=agents,
            power_budget=200.0,
            memory_budget=16384.0,
            gpu_budget=200.0,
        )

        # Prices should have decreased (or stayed clamped at min)
        assert market.lambda_power <= 10.0

    def test_agents_self_degrade(self):
        """Non-rigid agents should degrade when memory price is high."""
        market = LagrangianMarket()
        agents = make_test_agents()

        # Set high memory price to encourage degradation
        market.lambda_memory = 5.0

        result = market.run_auction(
            agents=agents,
            power_budget=100.0,
            memory_budget=3000.0,  # Tight memory
            gpu_budget=67.0,
        )

        # Check if any non-rigid agent degraded
        voice = next(a for a in agents if a.name == "Voice_NLU")
        world = next(a for a in agents if a.name == "World_Model")

        # At least one should have degraded or the market should have
        # found an optimal quantization below FP16
        non_rigid_quants = [voice.quant_level, world.quant_level]
        # With high memory prices, at least some agent should degrade
        assert any(q != QuantLevel.FP16 for q in non_rigid_quants) or \
               len(result.suspended) > 0

    def test_rigid_agent_never_degrades(self):
        """Rigid (RT_HARD) agents must never be degraded."""
        market = LagrangianMarket()
        agents = make_test_agents()

        # Extreme prices
        market.lambda_memory = 100.0

        market.run_auction(
            agents=agents,
            power_budget=100.0,
            memory_budget=8192.0,
            gpu_budget=67.0,
        )

        emotion = next(a for a in agents if a.name == "Emotion_RT")
        assert emotion.quant_level == QuantLevel.FP16


class TestMarketConvergence:
    """Test that the market converges to stable allocations."""

    def test_market_stabilizes_over_ticks(self):
        """After many ticks with constant conditions, prices should stabilize."""
        market = LagrangianMarket()
        agents = make_test_agents()

        prices = []
        for _ in range(50):
            market.run_auction(
                agents=agents,
                power_budget=30.0,
                memory_budget=6144.0,
                gpu_budget=67.0,
            )
            prices.append(market.lambda_power)

        # Price variance in last 10 ticks should be much less than first 10
        early_variance = max(prices[:10]) - min(prices[:10])
        late_variance = max(prices[-10:]) - min(prices[-10:])

        # Late variance should be smaller (market converging)
        # Allow for some oscillation
        assert late_variance <= early_variance + 0.5

    def test_total_utility_maximized(self):
        """Higher-utility agents should be preferred over lower-utility ones."""
        market = LagrangianMarket()
        agents = make_test_agents()

        # Tight budget — can't run everything
        result = market.run_auction(
            agents=agents,
            power_budget=15.0,
            memory_budget=3000.0,
            gpu_budget=67.0,
        )

        # Emotion_RT (U=100) should always run
        assert "Emotion_RT" in result.admitted

        # If any agent is suspended, it should be the lowest-utility one (Loop_Agent, U=30)
        if result.suspended:
            assert "Loop_Agent" in result.suspended or "Loop_Agent" in result.evicted


class TestMarketSparklines:
    """Test HUD-related market features."""

    def test_sparkline_generation(self):
        """Sparkline should generate valid unicode string."""
        market = LagrangianMarket()
        agents = make_test_agents()

        # Run a few ticks to build history
        for _ in range(5):
            market.run_auction(
                agents=agents,
                power_budget=30.0,
                memory_budget=6144.0,
                gpu_budget=67.0,
            )

        sparkline = market.get_price_sparkline("lambda_P", width=20)
        assert len(sparkline) == 20
        assert all(c in "▁▂▃▄▅▆▇█" for c in sparkline)

    def test_price_history_bounded(self):
        """Price history should not grow unbounded."""
        market = LagrangianMarket()
        agents = make_test_agents()

        for _ in range(500):
            market.run_auction(
                agents=agents,
                power_budget=30.0,
                memory_budget=6144.0,
                gpu_budget=67.0,
            )

        assert len(market.price_history) <= market._max_history
