# ==============================================================================
# Title:       Multiple Access Sizing for UAV-Enabled Multiuser Networks
#
# Author:      Andreas Manitsas
# Email:       amanitsb@ece.auth.gr
# 
# Course:      UAV13 Advanced Topics in Wireless Communications
# Program:     MSc Aerial Autonomous Systems
# Institution: Aristotle University of Thessaloniki, Faculty of Polytechnics
# Term:        2025-2026
#
# Description: This script implements the sizing of UAV fleets for multiple access in UAV-enabled multiuser communications.
#
# Disclaimer:  AI assistance may have been used during development
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. SYSTEM PARAMETERS
# ==========================================
BETA_0 = 1.42e-4
W = 10e6
P_U_DBM = -10.0
N0_DBM_HZ = -169.0
RHO = 0.005 # User density
G0 = 2.2846
TOTAL_AREA = 1e6 # 1 square kilometer (1,000,000 m^2)

P_u = (10 ** (P_U_DBM / 10.0)) / 1000.0
N0 = (10 ** (N0_DBM_HZ / 10.0)) / 1000.0

# ==========================================
# 2. UPLINK MAC RATE FUNCTION
# ==========================================
def rate_mac_max(rho):
    """Calculates the maximum Uplink MAC rate at optimal Theta (1.3195 rad)"""
    theta_opt = 1.3195 
    eta = (P_u * BETA_0 * G0 * rho * np.pi) / (N0 * W)
    t_tan = np.tan(theta_opt)**2
    t_cos = np.cos(theta_opt)**2
    
    term1 = (1.0 / (t_cos * t_tan)) * np.log2(1.0 + (eta * np.sin(theta_opt)**2) / (theta_opt**2))
    term2 = (1.0 / t_tan) * np.log2(1.0 + (eta * t_tan) / (theta_opt**2))
    term3 = (eta / theta_opt**2) * np.log2(1.0 + (theta_opt**2 * t_tan) / (theta_opt**2 + eta * t_tan))
    
    return term1 - term2 + term3

# ==========================================
# 3. FLEET SIZING LOGIC
# ==========================================
R_target_range = np.linspace(0.1, 2.0, 20) # Target rates in bps/Hz

def compute_required_uavs(density):
    required_uavs = []
    max_single_uav_rate = rate_mac_max(density)
    total_users = density * TOTAL_AREA
    
    for r_tar in R_target_range:
        total_capacity_needed = total_users * r_tar
        num_nodes = np.ceil(total_capacity_needed / max_single_uav_rate)
        required_uavs.append(int(num_nodes))
    return required_uavs

# Compute for baseline and 20% increased density
uavs_baseline = compute_required_uavs(RHO)
uavs_congested = compute_required_uavs(RHO * 1.2)

# ==========================================
# 4. PLOTTING
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(R_target_range, uavs_baseline, 'b-o', label=r'Baseline Density ($\rho = 0.005$)')
plt.plot(R_target_range, uavs_congested, 'r-s', label=r'Congested Density ($1.2\rho = 0.006$)')

plt.title('Question 5: Minimum UAV Fleet Allocation')
plt.xlabel(r'Target Rate per User $R_{target}$ (bps/Hz)')
plt.ylabel('Minimum Required UAVs ($N_{min}$)')
plt.grid(True, linestyle='--')
plt.legend()

plt.savefig('q5_sizing.png', dpi=300, bbox_inches='tight')
plt.show()