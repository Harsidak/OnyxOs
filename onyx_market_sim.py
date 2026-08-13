import math
import random
import time

# --- System Constants & Physical Limits (Section 2 & 3) ---
T_AMB = 22.0        # Ambient temperature (Celsius)
T_LIMIT = 43.0      # Skin-contact hard limit
R_TH = 1.5          # Thermal resistance
C_TH = 10.0         # Thermal capacitance
M_RAM = 4096.0      # Total RAM in MB (4GB Edge device)
P_SUS = (T_LIMIT - T_AMB) / R_TH  # Steady state sustainable power

# --- The Dual Prices (Section 4) ---
# These are the global market prices for Power and Memory
lambda_P = 1.0  # Price of 1 Watt
lambda_M = 1.0  # Price of 1 MB

class AgentITD:
    """Inference Task Descriptor (ITD) - Replaces POSIX processes"""
    def __init__(self, name, a_class, base_utility, base_power_cost, base_ram, is_rigid=False):
        self.name = name
        self.a_class = a_class  # 'RT_HARD', 'RT_SOFT', 'BG'
        self.U_base = base_utility
        self.eps = base_power_cost # eps_i (Joules per token/frame)
        self.ram = base_ram        # M_i (Resident memory)
        self.is_rigid = is_rigid   # RT_HARD cannot be degraded
        
        # State
        self.running = False
        self.quantization = 1.0 # 1.0 = INT8/INT4 (L0), 0.5 = L2 degraded

    def utility(self, q):
        # Utility drops as quantization (precision) drops
        if self.is_rigid and q < 1.0: return -9999
        return self.U_base * (q ** 0.5)
        
    def mem_footprint(self, q):
        return self.ram * q

def simulate_tick(tick_id, agents, current_temp):
    global lambda_P, lambda_M
    
    print(f"\n--- TICK {tick_id} | Temp: {current_temp:.1f}°C | λ_P: {lambda_P:.3f} | λ_M: {lambda_M:.3f} ---")
    
    # 1. Thermal Governor: Calculate Power Budget (Section 2)
    headroom = C_TH * (T_LIMIT - current_temp)
    P_budget = P_SUS + (0.8 * headroom / 1.0) # α=0.8 safety margin, H=1s
    print(f"Governor published Power Budget: {P_budget:.1f} W")
    
    # 2. Memory Manager: Quantization Ladder (Section 3)
    total_ram_used = 0
    for agent in agents:
        if agent.is_rigid:
            agent.quantization = 1.0
        else:
            # Agent solves: argmax_q [U_i(q) - λ_M * M_i(q)]
            # Simple simulation: if λ_M is high, degrade to 50% RAM
            if agent.utility(1.0) - (lambda_M * agent.mem_footprint(1.0)) < \
               agent.utility(0.5) - (lambda_M * agent.mem_footprint(0.5)):
                agent.quantization = 0.5
                print(f"[{agent.name}] Voluntarily degraded to Q-level L2 to save RAM")
            else:
                agent.quantization = 1.0
        
        total_ram_used += agent.mem_footprint(agent.quantization)

    # 3. Multi-Agent Router: Lagrangian Auction (Section 4)
    total_power_used = 0
    for agent in agents:
        # RUN(i) ⟺ ∂Uᵢ/∂rᵢ > λ_P·εᵢ (Simplified to total Utility > total Cost)
        utility_val = agent.utility(agent.quantization)
        cost_val = lambda_P * agent.eps
        
        if utility_val > cost_val:
            agent.running = True
            total_power_used += agent.eps
            print(f"[{agent.name}] RUNNING (Utility {utility_val:.1f} > Cost {cost_val:.1f})")
        else:
            agent.running = False
            print(f"[{agent.name}] SUSPENDED (Utility {utility_val:.1f} < Cost {cost_val:.1f})")

    # 4. Market Update (Dual Ascent)
    eta_P = 0.1 # Learning rate for Power Price
    eta_M = 0.05 # Learning rate for Memory Price
    
    lambda_P = max(0.1, lambda_P + eta_P * (total_power_used - P_budget))
    lambda_M = max(0.1, lambda_M + eta_M * (total_ram_used - M_RAM))
    
    # 5. Physics Update (Heat generation)
    # T(t+Δ) = T(t) + (Power - Cooling) / Capacitance
    current_temp = current_temp + (total_power_used - ((current_temp - T_AMB)/R_TH)) / C_TH
    
    return current_temp

# --- Initialization ---
agents = [
    AgentITD("Emotion_RT", "RT_HARD", base_utility=100, base_power_cost=5.0, base_ram=1024, is_rigid=True),
    AgentITD("World_Model", "RT_SOFT", base_utility=80, base_power_cost=15.0, base_ram=2048),
    AgentITD("Loop_Agent_1", "BG", base_utility=30, base_power_cost=8.0, base_ram=1536),
    AgentITD("Loop_Agent_2", "BG", base_utility=20, base_power_cost=8.0, base_ram=1024)
]

sim_temp = 35.0 # Start warm

# Run the simulation loop
for tick in range(1, 10):
    sim_temp = simulate_tick(tick, agents, sim_temp)
    time.sleep(0.5)