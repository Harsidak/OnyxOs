"""
OnyxOS Agent Registry — Agent Lifecycle & Communication Manager

Manages the lifecycle of all agents in the system:
  REGISTERED → READY → RUNNING ⇄ DEGRADED → SUSPENDED → EVICTED

Also provides:
  - Agent discovery (scan manifests from /etc/onyx/agents/)
  - Inter-agent message bus (simple pub/sub)
  - Agent health monitoring (heartbeat + watchdog)
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set

from onyx_kernel.config import AgentClass, AgentState, SystemConfig, DEFAULT_CONFIG
from onyx_kernel.agent import AgentITD


class AgentRegistry:
    """Central registry for all OnyxOS agents.
    
    This is the "process table" of OnyxOS. Every agent must register
    here before it can participate in market auctions.
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or DEFAULT_CONFIG

        # Agent storage
        self._agents: Dict[str, AgentITD] = {}  # id → agent
        self._agents_by_name: Dict[str, AgentITD] = {}  # name → agent

        # Message bus
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)

        # Health monitoring
        self._heartbeats: Dict[str, float] = {}  # agent_id → last heartbeat
        self._watchdog_timeout: float = 10.0  # seconds

        # Event log
        self.events: List[Dict[str, Any]] = []
        self._max_events = 200

    # ── Agent Lifecycle ──────────────────────────────────────────

    def register(self, agent: AgentITD) -> None:
        """Register a new agent with the system.
        
        After registration, the agent is in REGISTERED state and
        must be readied before it can be admitted to run.
        """
        if agent.id in self._agents:
            raise ValueError(f"Agent {agent.id} already registered")
        if agent.name in self._agents_by_name:
            raise ValueError(f"Agent name '{agent.name}' already taken")

        self._agents[agent.id] = agent
        self._agents_by_name[agent.name] = agent
        self._heartbeats[agent.id] = time.time()

        agent._transition(AgentState.REGISTERED, "registered with system")
        self._log_event("REGISTER", agent)
        self._publish("agent.registered", {"agent": agent.name, "id": agent.id})

    def ready(self, agent_id: str) -> None:
        """Mark an agent as ready (model loaded, waiting for admission)."""
        agent = self._get_agent(agent_id)
        agent._transition(AgentState.READY, "model loaded")
        self._log_event("READY", agent)
        self._publish("agent.ready", {"agent": agent.name})

    def unregister(self, agent_id: str) -> None:
        """Remove an agent from the system."""
        agent = self._get_agent(agent_id)
        agent._transition(AgentState.EVICTED, "unregistered")
        self._log_event("UNREGISTER", agent)
        self._publish("agent.unregistered", {"agent": agent.name})

        del self._agents[agent_id]
        del self._agents_by_name[agent.name]
        self._heartbeats.pop(agent_id, None)

    def heartbeat(self, agent_id: str) -> None:
        """Record a heartbeat from an agent."""
        self._heartbeats[agent_id] = time.time()

    # ── Queries ──────────────────────────────────────────────────

    def get_agent(self, name: str) -> Optional[AgentITD]:
        """Get an agent by name."""
        return self._agents_by_name.get(name)

    def get_agent_by_id(self, agent_id: str) -> Optional[AgentITD]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def all_agents(self) -> List[AgentITD]:
        """Get all registered agents."""
        return list(self._agents.values())

    def active_agents(self) -> List[AgentITD]:
        """Get all agents that can participate in auctions (READY+)."""
        return [
            a for a in self._agents.values()
            if a.state not in (AgentState.REGISTERED, AgentState.EVICTED)
        ]

    def running_agents(self) -> List[AgentITD]:
        """Get all currently running agents."""
        return [
            a for a in self._agents.values()
            if a.state in (AgentState.RUNNING, AgentState.DEGRADED)
        ]

    def by_class(self, agent_class: AgentClass) -> List[AgentITD]:
        """Get all agents of a specific class."""
        return [
            a for a in self._agents.values()
            if a.agent_class == agent_class
        ]

    @property
    def count(self) -> int:
        return len(self._agents)

    @property
    def running_count(self) -> int:
        return len(self.running_agents())

    # ── Message Bus ──────────────────────────────────────────────

    def subscribe(self, topic: str, callback: Callable) -> None:
        """Subscribe to a message topic.
        
        Topics follow dot-notation:
          - agent.registered
          - agent.ready
          - agent.admitted
          - agent.suspended
          - agent.evicted
          - market.tick
          - thermal.update
          - system.shutdown
        """
        self._subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callable) -> None:
        """Unsubscribe from a topic."""
        if topic in self._subscribers:
            self._subscribers[topic] = [
                cb for cb in self._subscribers[topic] if cb != callback
            ]

    def _publish(self, topic: str, data: Dict[str, Any]) -> None:
        """Publish a message to all subscribers of a topic."""
        for callback in self._subscribers.get(topic, []):
            try:
                callback(topic, data)
            except Exception:
                pass  # Don't let subscriber errors crash the registry

    # ── Health Monitoring ────────────────────────────────────────

    def check_health(self) -> List[str]:
        """Check for agents that have missed their heartbeat.
        
        Returns list of agent names that are unresponsive.
        """
        now = time.time()
        unresponsive = []

        for agent_id, last_beat in self._heartbeats.items():
            if now - last_beat > self._watchdog_timeout:
                agent = self._agents.get(agent_id)
                if agent and agent.state in (AgentState.RUNNING, AgentState.DEGRADED):
                    unresponsive.append(agent.name)
                    self._log_event("WATCHDOG", agent, "missed heartbeat")

        return unresponsive

    # ── Stats ────────────────────────────────────────────────────

    def total_memory_usage(self) -> float:
        """Total memory used by all running agents (MB)."""
        return sum(
            a.memory_footprint()
            for a in self.running_agents()
        )

    def total_power_usage(self) -> float:
        """Total power used by all running agents (Watts)."""
        return sum(
            a.power_cost_watts
            for a in self.running_agents()
        )

    def total_utility(self) -> float:
        """Total utility produced by all running agents."""
        return sum(
            a.current_utility
            for a in self.running_agents()
        )

    def status_summary(self) -> str:
        """One-line registry status for display."""
        total = self.count
        running = self.running_count
        suspended = len([
            a for a in self._agents.values()
            if a.state == AgentState.SUSPENDED
        ])
        return (
            f"Agents: {running}/{total} running, "
            f"{suspended} suspended | "
            f"RAM: {self.total_memory_usage():.0f}MB | "
            f"Power: {self.total_power_usage():.1f}W"
        )

    # ── Internal ─────────────────────────────────────────────────

    def _get_agent(self, agent_id: str) -> AgentITD:
        """Get an agent by ID, raising if not found."""
        agent = self._agents.get(agent_id)
        if agent is None:
            raise KeyError(f"Agent {agent_id} not found")
        return agent

    def _log_event(self, action: str, agent: AgentITD,
                   detail: str = "") -> None:
        """Log a registry event."""
        event = {
            "timestamp": time.time(),
            "action": action,
            "agent": agent.name,
            "agent_id": agent.id,
            "state": agent.state.name,
            "class": agent.agent_class.name,
            "detail": detail,
        }
        self.events.append(event)
        if len(self.events) > self._max_events:
            self.events.pop(0)
