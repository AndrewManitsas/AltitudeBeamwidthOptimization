# ==============================================================================
# Title:       Joint Altitude and Beamwidth Optimization in UAV-Enabled Multiuser Networks
#
# Author:      Andreas Manitsas
# Email:       amanitsb@ece.auth.gr
# 
# Course:      UAV13 Advanced Topics in Wireless Communications
# Program:     MSc Aerial Autonomous Systems
# Institution: Aristotle University of Thessaloniki, Faculty of Polytechnics
# Term:        2025-2026
#
# Description: This script implements the joint altitude and beamwidth optimization for UAV-enabled multiuser communications.
#
# Disclaimer:  AI assistance may have been used during development
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. SYSTEM PARAMETERS (From Section VI)
# ==========================================
BETA_0 = 1.42e-4       # Channel power gain at reference distance d0=1m
W = 10e6               # Total bandwidth (10 MHz)
P_D_DBM = 10.0         # Downlink transmit power (10 dBm)
P_U_DBM = -10.0        # Uplink transmit power (-10 dBm)
N0_DBM_HZ = -169.0     # Noise power spectrum density (-169 dBm/Hz)
RHO_DEFAULT = 0.005    # Default GT density (GTs/m^2)

# Convert dBm to Watts
P_d = (10 ** (P_D_DBM / 10.0)) / 1000.0
P_u = (10 ** (P_U_DBM / 10.0)) / 1000.0
N0 = (10 ** (N0_DBM_HZ / 10.0)) / 1000.0

# Antenna constant G0 approx 2.2846
G0 = (30000 / (2**2)) * ((np.pi / 180)**2)

# Fundamental scaling variables used in equations
a = (P_d * G0 * BETA_0) / (N0 * W)

# ==========================================
# 2. ANALYTICAL RATE FUNCTIONS
# ==========================================
def rate_multicast(H, theta, rho=RHO_DEFAULT):
    """Equation (6): Multicast Sum Rate"""
    # K_s coefficient
    ks_coef = (3 * np.sqrt(3) / 2) * rho * (H**2) * (np.tan(theta)**2)
    snr_edge = (a * (np.cos(theta)**2)) / ((theta**2) * (H**2))
    # Return rate scaled to Mbps/Hz to match Fig 2
    return (ks_coef * np.log2(1.0 + snr_edge)) / 1e6

def rate_broadcast(H, theta):
    """Equation (8): Broadcast Sum Rate"""
    term1 = (1.0 / (np.sin(theta)**2)) * np.log2(1.0 + (a * np.cos(theta)**2) / (theta**2 * H**2))
    term2 = (1.0 / (np.tan(theta)**2)) * np.log2(1.0 + a / (theta**2 * H**2))
    
    num = (theta**2 * H**2) + a * np.cos(theta)**2
    den = (theta**2 * H**2 * np.cos(theta)**2) + a * np.cos(theta)**2
    term3 = (a / (theta**2 * H**2 * np.tan(theta)**2)) * np.log2(num / den)
    
    return term1 - term2 + term3

def rate_mac(theta, rho):
    """Equation (10): Uplink MAC Sum Rate"""
    eta = (P_u * BETA_0 * G0 * rho * np.pi) / (N0 * W)
    t_tan = np.tan(theta)**2
    t_cos = np.cos(theta)**2
    
    term1 = (1.0 / (t_cos * t_tan)) * np.log2(1.0 + (eta * np.sin(theta)**2) / (theta**2))
    term2 = (1.0 / t_tan) * np.log2(1.0 + (eta * t_tan) / (theta**2))
    term3 = (eta / theta**2) * np.log2(1.0 + (theta**2 * t_tan) / (theta**2 + eta * t_tan))
    
    return term1 - term2 + term3

# ==========================================
# 3. PLOTTING FIGURES
# ==========================================
# Define ranges (avoiding 0 to prevent division by zero)
theta_range = np.linspace(0.01, np.pi/2 - 0.01, 200)
H_range = np.linspace(100, 4500, 200)

plt.figure(figsize=(12, 10))

# --- Fig 2: Multicast Rate vs Theta ---
plt.subplot(2, 2, 1)
for h_val in [500, 1500, 2500, 3500, 4500]:
    rates = [rate_multicast(h_val, t) for t in theta_range]
    plt.plot(theta_range, rates, label=f'H = {h_val}m')
plt.title(r'Fig 2: $\overline{R}_{MC}$ vs $\Theta$')
plt.xlabel(r'$\Theta$ (rad)')
plt.ylabel(r'$\overline{R}_{MC}$ (Mbps/Hz)')
plt.grid(True, linestyle='--')
plt.legend()

# --- Fig 3(a): Broadcast Rate vs H ---
plt.subplot(2, 2, 2)
fixed_theta = np.pi / 10
rbc_vs_H = [rate_broadcast(h, fixed_theta) for h in H_range]
plt.plot(H_range, rbc_vs_H, 'b-s', markevery=20)
plt.title(r'Fig 3(a): $\overline{R}_{BC}$ vs $H$ ($\Theta = \pi/10$)')
plt.xlabel('H (m)')
plt.ylabel(r'$\overline{R}_{BC}$ (bps/Hz)')
plt.grid(True, linestyle='--')

# --- Fig 3(b): Broadcast Rate vs Theta ---
plt.subplot(2, 2, 3)
fixed_H = 500
rbc_vs_theta = [rate_broadcast(fixed_H, t) for t in theta_range]
plt.plot(theta_range, rbc_vs_theta, 'b-s', markevery=15)
plt.title(r'Fig 3(b): $\overline{R}_{BC}$ vs $\Theta$ ($H = 500$m)')
plt.xlabel(r'$\Theta$ (rad)')
plt.ylabel(r'$\overline{R}_{BC}$ (bps/Hz)')
plt.grid(True, linestyle='--')

# --- Fig 4: MAC Rate vs Theta ---
plt.subplot(2, 2, 4)
for rho_val, marker in zip([0.001, 0.005, 0.010], ['s', '*', '+']):
    rmac_vals = [rate_mac(t, rho_val) for t in theta_range]
    plt.plot(theta_range, rmac_vals, label=rf'$\rho = {rho_val}$', marker=marker, markevery=15)
plt.axvline(x=1.3195, color='k', linestyle='--', label=r'Optimal $\Theta^* \approx 1.3195$')
plt.title(r'Fig 4: $\overline{R}_{MAC}$ vs $\Theta$')
plt.xlabel(r'$\Theta$ (rad)')
plt.ylabel(r'$\overline{R}_{MAC}$ (bps/Hz)')
plt.grid(True, linestyle='--')
plt.legend()

plt.tight_layout()
plt.savefig('Figure_1.png', dpi=300, bbox_inches='tight')
plt.show()