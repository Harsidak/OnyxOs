# ◆ OnyxOS — AI-Native Operating System

<p align="center">
  <strong>The world's first operating system where AI agents compete for resources through a real-time market auction.</strong>
</p>

<p align="center">
  <em>No scheduler — a market. No processes — agents. No OOM kills — graceful degradation.</em>
</p>

---

## What is OnyxOS?

OnyxOS is a stripped-down Linux-based operating system purpose-built for AI inference on edge devices (AR glasses, wearables). Instead of traditional process scheduling, it uses a **Lagrangian market-based auction** where AI agents bid for compute, memory, and power using economic prices.

### Key Innovations

| Traditional OS | OnyxOS |
|---|---|
| CFS/EEVDF scheduler | Lagrangian market auction |
| Priority queues | Economic utility functions |
| OOM killer | Voluntary quantization degradation |
| systemd (PID 1) | `onyx-init` (Rust, <500ms boot) |
| cgroups (manual) | Auto cgroups per agent class |
| `/var/log` | Structured event stream |

### How It Works

```
1. Thermal Governor reads SoC temperature
2. Computes dynamic power budget: P = P_sus + α·C·(T_limit - T)/H
3. Market Auction runs with dual variables λ_P, λ_M, λ_G
4. Agents bid: RUN(i) ⟺ U_i(q) > λ_P·ε_i + λ_M·M_i(q)
5. Winners get resources, losers self-degrade or suspend
6. Prices update via dual ascent: λ += η·(demand - supply)
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### Run the Demo

```bash
# Install dependencies
pip install rich textual numpy psutil

# Launch the full TUI dashboard
python scripts/demo.py

# Or run headless
python scripts/demo.py --headless --ticks 30
```

### What You'll See

A real-time terminal dashboard showing:
- 🌡️ **Thermal gauge** — live temperature with color gradient
- 💰 **Market prices** — λ_P, λ_M, λ_G with sparkline history
- 🤖 **Agent board** — each agent's state, utility, cost, quant level
- ⚡ **Power budget** — budget vs actual draw
- 📊 **Auction log** — real-time bid/win/evict events
- 💬 **Inference stream** — live output from running agents

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│  LAYER 3: AGENTS                                 │
│  Emotion(RT_HARD) Voice(RT_SOFT) World BG        │
├──────────────────────────────────────────────────┤
│  LAYER 2: ONYXD (Central Daemon)                 │
│  Market Auction │ Thermal Governor │ Inference    │
├──────────────────────────────────────────────────┤
│  LAYER 1: ONYX-INIT (Rust PID 1)                │
│  Mount FS │ cgroup v2 │ Launch OnyxD             │
├──────────────────────────────────────────────────┤
│  LAYER 0: STRIPPED LINUX KERNEL                  │
│  L4T 5.15 │ NVIDIA GPU │ onyx_sched.ko          │
├──────────────────────────────────────────────────┤
│  NVIDIA Jetson Orin Nano Super                   │
│  8GB │ 67 TOPS │ 1024 CUDA │ 7-25W              │
└──────────────────────────────────────────────────┘
```

---

## Project Structure

```
OnyxOS/
├── onyx_kernel/          # Python market scheduler (P0)
│   ├── config.py         # Hardware constants & enums
│   ├── agent.py          # Agent ITD (Inference Task Descriptor)
│   ├── market.py         # Lagrangian auction engine
│   ├── thermal_governor.py
│   ├── quant_ladder.py
│   └── agent_registry.py
├── onyx_hud/             # Terminal UI dashboard
│   └── dashboard.py      # Rich/Textual TUI
├── agents/               # Built-in AI agents
│   ├── base_agent.py
│   ├── emotion_agent.py  # RT_HARD — emotion detection
│   ├── voice_agent.py    # RT_SOFT — voice NLU + LLM
│   ├── world_model_agent.py  # RT_SOFT — spatial awareness
│   └── loop_agent.py     # BG — background reasoning
├── onyx-init/            # Custom PID 1 (Rust)
├── onyx-sched/           # Kernel scheduler module (C)
├── onyxd/                # Central daemon (Rust)
├── tests/
├── scripts/
│   └── demo.py           # One-command demo
└── docs/
```

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Target Hardware

**NVIDIA Jetson Orin Nano Super**
- 8GB LPDDR5 unified memory (102 GB/s)
- 1024 CUDA cores + 32 Tensor cores
- 67 TOPS (INT8)
- 6-core Arm Cortex-A78AE
- 7-25W power envelope
- $249

---

## License

MIT

---

<p align="center">
  <strong>◆ OnyxOS</strong> — Where AI agents trade resources in a market, not fight over a scheduler.
</p>
