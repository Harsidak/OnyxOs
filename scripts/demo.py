#!/usr/bin/env python3
"""
OnyxOS Demo Launcher — One-command demo for YC

Usage:
    python scripts/demo.py           # Launch full TUI dashboard
    python scripts/demo.py --headless # Run without TUI (print to stdout)
    python scripts/demo.py --ticks 50 # Run for 50 ticks then exit
"""

import sys
import os
import argparse
from pathlib import Path

# Fix Unicode output on Windows (cp1252 → UTF-8)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="OnyxOS Demo — AI-Native Operating System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/demo.py              # Full TUI dashboard
  python scripts/demo.py --headless   # Headless mode (stdout)
  python scripts/demo.py --ticks 30   # Run 30 ticks then exit
        """,
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without TUI, print status to stdout"
    )
    parser.add_argument(
        "--ticks", type=int, default=0,
        help="Number of ticks to run (0 = infinite)"
    )
    parser.add_argument(
        "--tick-rate", type=float, default=1.0,
        help="Ticks per second (default: 1.0)"
    )

    args = parser.parse_args()

    if args.headless:
        run_headless(args.ticks, args.tick_rate)
    else:
        # Launch the full TUI dashboard
        from onyx_hud.dashboard import main as dashboard_main
        dashboard_main()


def run_headless(max_ticks: int = 0, tick_rate: float = 1.0):
    """Run OnyxOS in headless mode — print to stdout."""
    import time
    from onyx_kernel.config import SystemConfig, AgentState
    from onyx_kernel.market import LagrangianMarket
    from onyx_kernel.thermal_governor import ThermalGovernor
    from onyx_kernel.agent_registry import AgentRegistry
    from agents.emotion_agent import EmotionAgent
    from agents.voice_agent import VoiceAgent
    from agents.world_model_agent import WorldModelAgent
    from agents.loop_agent import LoopAgent

    config = SystemConfig(demo_mode=True)
    market = LagrangianMarket(config)
    thermal = ThermalGovernor(config)
    registry = AgentRegistry(config)

    # Setup agents
    agents = [EmotionAgent(), VoiceAgent(), WorldModelAgent(), LoopAgent()]
    for agent in agents:
        agent.initialize()
        registry.register(agent.itd)
        registry.ready(agent.itd.id)

    print("◆ OnyxOS v0.1.0 — Headless Mode")
    print(f"  Agents: {len(agents)} registered")
    print(f"  Tick rate: {tick_rate} Hz")
    print(f"  Max ticks: {'∞' if max_ticks == 0 else max_ticks}")
    print()

    tick = 0
    try:
        while True:
            tick += 1
            if max_ticks > 0 and tick > max_ticks:
                break

            # Compute actual power draw
            actual_power = sum(
                a.itd.power_cost_watts
                for a in agents
                if a.itd.state in (AgentState.RUNNING, AgentState.DEGRADED)
            )
            snap = thermal.tick(actual_power)

            # Run auction
            result = market.run_auction(
                agents=[a.itd for a in agents],
                power_budget=snap.power_budget_watts,
                memory_budget=config.hardware.RAM_AVAILABLE_MB,
                gpu_budget=config.hardware.GPU_TOPS,
            )

            # Run inference
            for agent in agents:
                if agent.itd.state in (AgentState.RUNNING, AgentState.DEGRADED):
                    agent.itd.run_inference()

            # Print status
            print(
                f"T{tick:>4d} | "
                f"{snap.soc_temp:.1f}°C | "
                f"P:{result.total_power_used:.0f}/{snap.power_budget_watts:.0f}W | "
                f"M:{result.total_memory_used:.0f}/{config.hardware.RAM_AVAILABLE_MB}MB | "
                f"λ_P={market.lambda_power:.3f} "
                f"λ_M={market.lambda_memory:.3f} | "
                f"Running: {', '.join(result.admitted + result.degraded) or 'none'} | "
                f"{'Suspended: ' + ', '.join(result.suspended) if result.suspended else ''}"
            )

            time.sleep(1.0 / tick_rate)

    except KeyboardInterrupt:
        print("\n◆ OnyxOS shutdown.")

    print(f"\nCompleted {tick} ticks.")


if __name__ == "__main__":
    main()
