"""
OnyxOS Dashboard — Real-Time Market Visualization HUD

A stunning terminal UI built with Rich + Textual that shows:
  🌡️ Thermal gauge with live temperature
  💰 Market prices with sparkline history  
  🤖 Agent status board (state, utility, cost, quant level)
  ⚡ Power budget vs actual draw
  📊 Live auction event log
  💬 Agent inference output stream

This is the "wow" factor for the YC demo — a real-time view into
the market-based agent scheduling happening inside OnyxOS.
"""

from __future__ import annotations

import sys
import time
import asyncio
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich.progress_bar import ProgressBar
from rich import box

# Add parent to path for imports
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from onyx_kernel.config import (
    AgentClass, AgentState, QuantLevel, SystemConfig, DEFAULT_CONFIG,
)
from onyx_kernel.agent import AgentITD
from onyx_kernel.market import LagrangianMarket, AuctionResult
from onyx_kernel.thermal_governor import ThermalGovernor, ThermalState
from onyx_kernel.quant_ladder import QuantLadder
from onyx_kernel.agent_registry import AgentRegistry
from agents.emotion_agent import EmotionAgent
from agents.voice_agent import VoiceAgent
from agents.world_model_agent import WorldModelAgent
from agents.loop_agent import LoopAgent


# ── Color Palette ────────────────────────────────────────────────
COLORS = {
    "bg": "#0a0a0f",
    "panel": "#12121a",
    "accent": "#7c3aed",       # Purple
    "accent_light": "#a78bfa",
    "green": "#10b981",
    "yellow": "#f59e0b",
    "red": "#ef4444",
    "orange": "#f97316",
    "blue": "#3b82f6",
    "cyan": "#06b6d4",
    "text": "#e2e8f0",
    "text_dim": "#64748b",
    "gold": "#fbbf24",
}

STATE_STYLES = {
    AgentState.RUNNING: ("bold green", "🟢"),
    AgentState.DEGRADED: ("bold yellow", "🟡"),
    AgentState.SUSPENDED: ("bold red", "🔴"),
    AgentState.EVICTED: ("dim", "⚫"),
    AgentState.READY: ("bold blue", "🔵"),
    AgentState.REGISTERED: ("dim", "⚪"),
}

THERMAL_STYLES = {
    ThermalState.NOMINAL: ("bold green", "NOMINAL"),
    ThermalState.THROTTLED: ("bold yellow", "THROTTLED"),
    ThermalState.EMERGENCY: ("bold red", "EMERGENCY"),
    ThermalState.UNKNOWN: ("dim", "UNKNOWN"),
}


class OnyxDashboard:
    """The OnyxOS real-time dashboard.
    
    Orchestrates the entire demo: initializes agents, runs the
    market scheduler, and renders the TUI.
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.console = Console()

        # Core systems
        self.market = LagrangianMarket(self.config)
        self.thermal = ThermalGovernor(self.config)
        self.registry = AgentRegistry(self.config)
        self.quant_ladder = QuantLadder()

        # Agents
        self.agents: List[Any] = []
        self._inference_outputs: List[Dict] = []
        self._max_outputs = 15

        # Demo state
        self._tick_count = 0
        self._last_result: Optional[AuctionResult] = None
        self._start_time = time.time()

    def setup_agents(self) -> None:
        """Initialize and register all built-in agents."""
        agent_classes = [
            EmotionAgent,
            VoiceAgent,
            WorldModelAgent,
            LoopAgent,
        ]

        for AgentCls in agent_classes:
            agent = AgentCls()
            agent.initialize()
            self.registry.register(agent.itd)
            self.registry.ready(agent.itd.id)
            self.agents.append(agent)

    def tick(self) -> AuctionResult:
        """Run one tick of the OnyxOS scheduler."""
        self._tick_count += 1

        # 1. Thermal governor computes power budget
        actual_power = sum(
            a.itd.power_cost_watts
            for a in self.agents
            if a.itd.state in (AgentState.RUNNING, AgentState.DEGRADED)
        )
        thermal_snap = self.thermal.tick(actual_power)

        # 2. Market auction
        agent_itds = [a.itd for a in self.agents]
        result = self.market.run_auction(
            agents=agent_itds,
            power_budget=thermal_snap.power_budget_watts,
            memory_budget=self.config.hardware.RAM_AVAILABLE_MB,
            gpu_budget=self.config.hardware.GPU_TOPS,
        )
        self._last_result = result

        # 3. Run inference on admitted agents
        for agent in self.agents:
            if agent.itd.state in (AgentState.RUNNING, AgentState.DEGRADED):
                output = agent.itd.run_inference()
                if output:
                    self._inference_outputs.append({
                        "tick": self._tick_count,
                        "agent": agent.name,
                        "quant": agent.itd.quant_level.name,
                        "output": output,
                    })
                    if len(self._inference_outputs) > self._max_outputs:
                        self._inference_outputs.pop(0)

        return result

    # ── Panel Renderers ──────────────────────────────────────────

    def _render_header(self) -> Panel:
        """Render the OnyxOS header."""
        uptime = time.time() - self._start_time
        minutes = int(uptime) // 60
        seconds = int(uptime) % 60

        header_text = Text()
        header_text.append("◆ ", style="bold #7c3aed")
        header_text.append("OnyxOS", style="bold white")
        header_text.append(" v0.1.0", style="dim")
        header_text.append("  │  ", style="dim")
        header_text.append("AI-Native Operating System", style="#a78bfa")
        header_text.append("  │  ", style="dim")
        header_text.append(f"Tick {self._tick_count}", style="bold cyan")
        header_text.append("  │  ", style="dim")
        header_text.append(f"Uptime {minutes:02d}:{seconds:02d}", style="dim")
        header_text.append("  │  ", style="dim")

        mode = "SIMULATION" if not self.thermal.is_jetson else "JETSON LIVE"
        mode_style = "bold yellow" if not self.thermal.is_jetson else "bold green"
        header_text.append(f"[{mode}]", style=mode_style)

        return Panel(
            Align.center(header_text),
            style="#12121a",
            box=box.HEAVY,
            border_style="#7c3aed",
        )

    def _render_thermal(self) -> Panel:
        """Render the thermal gauge panel."""
        tc = self.thermal
        snap = tc.history[-1] if tc.history else None

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(width=14)
        table.add_column(width=40)

        # Temperature bar
        temp = tc.current_temp
        t_min = tc.tc.T_AMBIENT
        t_max = tc.tc.T_EMERGENCY
        fill = max(0, min(1, (temp - t_min) / (t_max - t_min)))
        bar_width = 30
        filled = int(fill * bar_width)

        if temp >= tc.tc.T_THROTTLE:
            bar_color = "red"
        elif temp >= tc.tc.T_THROTTLE - 3:
            bar_color = "yellow"
        else:
            bar_color = "green"

        bar = f"[{bar_color}]{'█' * filled}[/][dim]{'░' * (bar_width - filled)}[/]"
        table.add_row(
            Text("🌡️ SoC Temp", style="bold"),
            Text.from_markup(f"{bar} {temp:.1f}°C")
        )

        # Thermal state
        state_style, state_text = THERMAL_STYLES.get(
            tc.current_state, ("dim", "UNKNOWN")
        )
        table.add_row(
            Text("   State", style="dim"),
            Text(state_text, style=state_style)
        )

        # Power budget
        budget = tc.current_budget
        max_w = tc.hc.POWER_MAX_W
        budget_fill = max(0, min(1, budget / max_w))
        b_filled = int(budget_fill * bar_width)
        table.add_row(
            Text("⚡ Power Budget", style="bold"),
            Text.from_markup(
                f"[cyan]{'█' * b_filled}[/][dim]{'░' * (bar_width - b_filled)}[/] "
                f"{budget:.1f}W / {max_w:.0f}W"
            )
        )

        # Actual power draw
        if snap:
            draw = snap.power_draw_watts
            draw_fill = max(0, min(1, draw / max_w))
            d_filled = int(draw_fill * bar_width)
            draw_color = "red" if draw > budget else "green"
            table.add_row(
                Text("   Draw", style="dim"),
                Text.from_markup(
                    f"[{draw_color}]{'█' * d_filled}[/]"
                    f"[dim]{'░' * (bar_width - d_filled)}[/] "
                    f"{draw:.1f}W"
                )
            )

        # Headroom
        headroom = tc.tc.T_SKIN_LIMIT - temp
        hr_style = "green" if headroom > 5 else "yellow" if headroom > 2 else "red"
        table.add_row(
            Text("   Headroom", style="dim"),
            Text(f"{headroom:.1f}°C to skin limit", style=hr_style)
        )

        return Panel(
            table,
            title="[bold #f97316]THERMAL GOVERNOR[/]",
            border_style="#f97316",
            box=box.ROUNDED,
        )

    def _render_market(self) -> Panel:
        """Render the market prices panel with sparklines."""
        m = self.market

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(width=12)
        table.add_column(width=8)
        table.add_column(width=30)

        # Power price
        spark_p = m.get_price_sparkline("lambda_P", 25)
        table.add_row(
            Text("λ_Power", style="bold #ef4444"),
            Text(f"{m.lambda_power:.4f}", style="bold"),
            Text(spark_p, style="#ef4444"),
        )

        # Memory price
        spark_m = m.get_price_sparkline("lambda_M", 25)
        table.add_row(
            Text("λ_Memory", style="bold #3b82f6"),
            Text(f"{m.lambda_memory:.4f}", style="bold"),
            Text(spark_m, style="#3b82f6"),
        )

        # GPU price
        spark_g = m.get_price_sparkline("lambda_G", 25)
        table.add_row(
            Text("λ_GPU", style="bold #10b981"),
            Text(f"{m.lambda_gpu:.4f}", style="bold"),
            Text(spark_g, style="#10b981"),
        )

        # Utilization
        if self._last_result:
            r = self._last_result
            table.add_row(Text(""), Text(""), Text(""))
            table.add_row(
                Text("Power Util", style="dim"),
                Text(f"{r.power_utilization * 100:.0f}%",
                     style="bold" if r.power_utilization > 0.8 else ""),
                Text(f"({r.total_power_used:.1f}W / {r.power_budget:.1f}W)",
                     style="dim"),
            )
            table.add_row(
                Text("Memory Util", style="dim"),
                Text(f"{r.memory_utilization * 100:.0f}%",
                     style="bold" if r.memory_utilization > 0.8 else ""),
                Text(f"({r.total_memory_used:.0f}MB / {r.memory_budget:.0f}MB)",
                     style="dim"),
            )

        return Panel(
            table,
            title="[bold #7c3aed]MARKET PRICES[/]",
            border_style="#7c3aed",
            box=box.ROUNDED,
        )

    def _render_agents(self) -> Panel:
        """Render the agent status board."""
        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold #a78bfa",
            border_style="dim",
            padding=(0, 1),
        )
        table.add_column("", width=2)
        table.add_column("Agent", width=14)
        table.add_column("Class", width=8)
        table.add_column("State", width=10)
        table.add_column("Quant", width=5)
        table.add_column("Utility", width=8, justify="right")
        table.add_column("Cost", width=8, justify="right")
        table.add_column("Surplus", width=9, justify="right")
        table.add_column("RAM", width=8, justify="right")

        for agent_wrapper in self.agents:
            a = agent_wrapper.itd
            style, icon = STATE_STYLES.get(a.state, ("dim", "❓"))

            surplus_style = "green" if a.current_surplus > 0 else "red"
            class_style = {
                AgentClass.RT_HARD: "bold red",
                AgentClass.RT_SOFT: "bold yellow",
                AgentClass.BG: "dim",
            }.get(a.agent_class, "")

            table.add_row(
                icon,
                Text(a.name, style="bold"),
                Text(a.agent_class.name, style=class_style),
                Text(a.state.name, style=style),
                Text(a.quant_level.name, style="cyan"),
                Text(f"{a.current_utility:.1f}", style=""),
                Text(f"{a.current_cost:.1f}", style=""),
                Text(f"{a.current_surplus:+.1f}", style=surplus_style),
                Text(f"{a.memory_footprint():.0f}MB", style="dim"),
            )

        return Panel(
            table,
            title="[bold #06b6d4]AGENT STATUS BOARD[/]",
            border_style="#06b6d4",
            box=box.ROUNDED,
        )

    def _render_auction_log(self) -> Panel:
        """Render recent auction events."""
        events = self.market.events[-10:]

        lines = []
        for evt in reversed(events):
            action = evt["action"]
            action_style = {
                "ADMIT": "[green]ADMIT [/]",
                "SUSPEND": "[red]SUSPND[/]",
                "EVICT": "[bold red]EVICT [/]",
                "DEGRADE": "[yellow]DGRADE[/]",
            }.get(action, f"[dim]{action:<6s}[/]")

            line = Text.from_markup(
                f"[dim]T{evt['tick']:>3d}[/] {action_style} "
                f"[bold]{evt['agent']:<14s}[/] "
                f"[dim]{evt['reason']}[/]"
            )
            lines.append(line)

        if not lines:
            lines.append(Text("Waiting for first auction...", style="dim italic"))

        content = Text("\n").join(lines)

        return Panel(
            content,
            title="[bold #fbbf24]AUCTION LOG[/]",
            border_style="#fbbf24",
            box=box.ROUNDED,
        )

    def _render_inference(self) -> Panel:
        """Render live agent inference outputs."""
        outputs = self._inference_outputs[-8:]

        lines = []
        for out in reversed(outputs):
            agent_name = out["agent"]
            quant = out["quant"]
            data = out["output"]

            if agent_name == "Emotion_RT":
                emotion = data.get("emotion", "?")
                conf = data.get("confidence", 0)
                line = Text.from_markup(
                    f"[bold red]EMO[/] [dim]Q={quant}[/] "
                    f"→ {emotion} ({conf:.0%})"
                )
            elif agent_name == "Voice_NLU":
                resp = data.get("response", "")[:60]
                tokens = data.get("tokens", 0)
                line = Text.from_markup(
                    f"[bold yellow]VOX[/] [dim]Q={quant}[/] "
                    f"→ {resp}{'...' if len(data.get('response', '')) > 60 else ''}"
                )
            elif agent_name == "World_Model":
                n_obj = data.get("object_count", 0)
                depth = "✓" if data.get("has_depth") else "✗"
                objs = [o["label"] for o in data.get("objects", [])[:3]]
                line = Text.from_markup(
                    f"[bold cyan]WLD[/] [dim]Q={quant}[/] "
                    f"→ {n_obj} objects [dim](depth:{depth})[/] "
                    f"[dim]{', '.join(objs)}[/]"
                )
            elif agent_name == "Loop_Reason":
                progress = data.get("progress", 0)
                task = data.get("task", "")[:40]
                insight = data.get("insight")
                if insight:
                    line = Text.from_markup(
                        f"[bold green]BGD[/] [dim]Q={quant}[/] "
                        f"→ ✨ {insight[:55]}"
                    )
                else:
                    bar_w = 10
                    filled = int(progress * bar_w)
                    bar = "█" * filled + "░" * (bar_w - filled)
                    line = Text.from_markup(
                        f"[bold green]BGD[/] [dim]Q={quant}[/] "
                        f"→ [{bar}] {task}"
                    )
            else:
                line = Text.from_markup(
                    f"[dim]{agent_name}[/] → {str(data)[:50]}"
                )

            lines.append(line)

        if not lines:
            lines.append(Text("No inference outputs yet...", style="dim italic"))

        content = Text("\n").join(lines)

        return Panel(
            content,
            title="[bold #10b981]INFERENCE STREAM[/]",
            border_style="#10b981",
            box=box.ROUNDED,
        )

    def render(self) -> Layout:
        """Render the full dashboard layout."""
        layout = Layout()

        # Top: Header
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        layout["header"].update(self._render_header())

        # Body: 2 columns
        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1),
        )

        # Left column: Thermal + Market
        layout["left"].split_column(
            Layout(name="thermal", size=10),
            Layout(name="market", size=13),
            Layout(name="auction_log"),
        )

        layout["thermal"].update(self._render_thermal())
        layout["market"].update(self._render_market())
        layout["auction_log"].update(self._render_auction_log())

        # Right column: Agents + Inference
        layout["right"].split_column(
            Layout(name="agents", size=11),
            Layout(name="inference"),
        )

        layout["agents"].update(self._render_agents())
        layout["inference"].update(self._render_inference())

        # Footer
        footer_text = Text()
        footer_text.append(" ◆ OnyxOS ", style="bold white on #7c3aed")
        footer_text.append("  Lagrangian Market Scheduler  ", style="dim")
        footer_text.append("│", style="dim")
        footer_text.append("  Jetson Orin Nano Super ", style="#a78bfa")
        footer_text.append("│", style="dim")
        footer_text.append("  8GB LPDDR5  ", style="dim")
        footer_text.append("│", style="dim")
        footer_text.append("  67 TOPS  ", style="dim")
        footer_text.append("│", style="dim")
        footer_text.append("  Press Ctrl+C to exit", style="dim italic")

        layout["footer"].update(
            Panel(
                Align.center(footer_text),
                style="#12121a",
                box=box.HEAVY,
                border_style="dim",
            )
        )

        return layout


def main():
    """Launch the OnyxOS dashboard demo."""
    console = Console()

    # Banner
    console.print()
    console.print(
        "  [bold #7c3aed]◆[/] [bold white]OnyxOS[/] "
        "[dim]v0.1.0[/]  —  "
        "[#a78bfa]AI-Native Operating System[/]",
    )
    console.print(
        "  [dim]Initializing market scheduler...[/]",
    )
    console.print()

    # Initialize
    config = SystemConfig(demo_mode=True)
    dashboard = OnyxDashboard(config)
    dashboard.setup_agents()

    console.print(
        f"  [green]✓[/] {len(dashboard.agents)} agents registered",
    )
    console.print(
        "  [green]✓[/] Thermal governor online (simulation mode)",
    )
    console.print(
        "  [green]✓[/] Lagrangian market initialized",
    )
    console.print(
        "  [dim]Starting live dashboard...[/]",
    )
    console.print()

    time.sleep(1)

    # Run live dashboard
    try:
        with Live(
            dashboard.render(),
            console=console,
            refresh_per_second=2,
            screen=True,
        ) as live:
            while True:
                dashboard.tick()
                live.update(dashboard.render())
                time.sleep(0.5)
    except KeyboardInterrupt:
        console.print()
        console.print(
            "  [bold #7c3aed]◆[/] [bold]OnyxOS[/] "
            "[dim]shutdown complete.[/]"
        )
        console.print()

        # Print final stats
        total_ticks = dashboard._tick_count
        total_utility = sum(
            a.itd.metrics.total_utility_earned for a in dashboard.agents
        )
        total_evictions = sum(
            a.itd.metrics.times_evicted for a in dashboard.agents
        )
        total_degradations = sum(
            a.itd.metrics.times_degraded for a in dashboard.agents
        )

        console.print(f"  Ticks: {total_ticks}")
        console.print(f"  Total utility earned: {total_utility:.1f}")
        console.print(f"  Total degradations: {total_degradations}")
        console.print(f"  Total evictions: {total_evictions}")
        console.print()


if __name__ == "__main__":
    main()
