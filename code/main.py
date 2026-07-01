import numpy as np
import matplotlib.pyplot as plt

# --- SYSTEM CONSTANTS (Section VI Parameters) ---
BETA_0 = 1.42e-4      # Channel gain at reference d0=1m
W = 10e6              # Total bandwidth (10 MHz)
P_D_DBM = 10.0        # Downlink power (10 dBm)
P_U_DBM = -10.0       # Uplink power (-10 dBm)
N0_DBM_HZ = -169.0    # Noise floor (-169 dBm/Hz)
RHO = 0.005           # Baseline user density (GTs/m^2)
G0 = 2.2846           # Idealized antenna constant

# Conversion to Linear Units
P_d = 10**(P_D_DBM / 10.0) / 1000.0
P_u = 10**(P_U_DBM / 10.0) / 1000.0
N0 = 10**(N0_DBM_HZ / 10.0) / 1000.0

# Fundamental Scaling Constants
a = (P_d * G0 * BETA_0) / (N0 * W)
eta = (P_u * BETA_0 * G0 * RHO * np.pi) / (N0 * W)

# --- ANALYTICAL FORMULATIONS ---
def calculate_rmc(H, theta):
    # Hexagonal deployment multiplier
    k_s_coef = 1.5 * np.sqrt(3) * RHO * (H**2) * (np.tan(theta)**2)
    snr_edge = (a * (np.cos(theta)**2)) / ((theta**2) * (H**2))
    return k_s_coef * np.log2(1.0 + snr_edge)

def calculate_rbc(H, theta):
    term1 = (1.0 / (np.sin(theta)**2)) * np.log2(1.0 + (a * np.cos(theta)**2)/(theta**2 * H**2))
    term2 = (1.0 / (np.tan(theta)**2)) * np.log2(1.0 + a / (theta**2 * H**2))
    num_log3 = (theta**2 * H**2) + a * np.cos(theta)**2
    den_log3 = (theta**2 * H**2 * np.cos(theta)**2) + a * np.cos(theta)**2
    term3 = (a / (theta**2 * H**2 * np.tan(theta)**2)) * np.log2(num_log3 / den_log3)
    return term1 - term2 + term3

def calculate_rmac(theta, current_rho=RHO):
    local_eta = (P_u * BETA_0 * G0 * current_rho * np.pi) / (N0 * W)
    t_tan = np.tan(theta)**2
    t_cos = np.cos(theta)**2
    term1 = (1.0 / (t_cos * t_tan)) * np.log2(1.0 + (local_eta * np.sin(theta)**2)/(theta**2))
    term2 = (1.0 / t_tan) * np.log2(1.0 + (local_eta * t_tan)/(theta**2))
    term3 = (local_eta / theta**2) * np.log2(1.0 + (theta**2 * t_tan)/(theta**2 + local_eta * t_tan))
    return term1 - term2 + term3

# --- GENERATE REPRODUCTION PLOTS (Question 4) ---
theta_range = np.linspace(0.01, np.pi/2 - 0.01, 200)
H_range = np.linspace(100, 4500, 200)

plt.figure(figsize=(14, 10))

# 1. Downlink Multicast Reproduction
plt.subplot(2, 2, 1)
for h_val in [500, 1500, 2500, 3500, 4500]:
    rates = [calculate_rmc(h_val, t) / 1e6 for t in theta_range]  # Mbps/Hz scaling
    plt.plot(theta_range, rates, label=f'H = {h_val}m')
plt.title(r'Fig 2: Multicast Rate vs $\Theta$')
plt.xlabel(r'$\Theta$ (rad)')
plt.ylabel(r'$\overline{R}_{MC}$ (Mbps/Hz)')
plt.grid(True); plt.legend()

# 2. Downlink Broadcast vs Altitude
plt.subplot(2, 2, 2)
rbc_vs_H = [calculate_rbc(h, np.pi/10) for h in H_range]
plt.plot(H_range, rbc_vs_H, 'b-')
plt.title(r'Fig 3a: Broadcast Rate vs Altitude ($\Theta=\pi/10$)')
plt.xlabel('Altitude H (m)')
plt.ylabel(r'$\overline{R}_{BC}$ (bps/Hz)')
plt.grid(True)

# 3. Downlink Broadcast vs Beamwidth
plt.subplot(2, 2, 3)
rbc_vs_theta = [calculate_rbc(500, t) for t in theta_range]
plt.plot(theta_range, rbc_vs_theta, 'r-')
plt.title(r'Fig 3b: Broadcast Rate vs $\Theta$ (H=500m)')
plt.xlabel(r'$\Theta$ (rad)')
plt.ylabel(r'$\overline{R}_{BC}$ (bps/Hz)')
plt.grid(True)

# 4. Uplink Multiple Access Reproduction
plt.subplot(2, 2, 4)
for rho_val in [0.001, 0.005, 0.010]:
    rmac_vals = [calculate_rmac(t, rho_val) for t in theta_range]
    plt.plot(theta_range, rmac_vals, label=f'$\\rho$ = {rho_val}')
plt.title(r'Fig 4: Uplink MAC Rate vs $\Theta$')
plt.xlabel(r'$\Theta$ (rad)')
plt.ylabel(r'$\overline{R}_{MAC}$ (bps/Hz)')
plt.grid(True); plt.legend()

plt.tight_layout()
plt.show()

# --- SIZING OPTIMIZATION (Question 5) ---
# Scenario Parameters
TOTAL_AREA = 1e6      # Region dimension: 1,000,000 m^2
R_target_range = np.linspace(0.1, 2.0, 10)  # Target rates in bps/Hz

def compute_required_uavs(density):
    required_uavs = []
    # Uplink model evaluation context
    best_theta = 1.3195  # Globally optimized uplink beamwidth
    max_area_sum_rate = calculate_rmac(best_theta, density)
    users_in_area = density * TOTAL_AREA
    
    for r_tar in R_target_range:
        # Minimum fleet allocation sizing logic
        total_capacity_needed = users_in_area * r_tar
        num_nodes = np.ceil(total_capacity_needed / max_area_sum_rate)
        required_uavs.append(int(num_nodes))
    return required_uavs

uavs_baseline = compute_required_uavs(RHO)
uavs_congested = compute_required_uavs(RHO * 1.2)

plt.figure(figsize=(8, 5))
plt.plot(R_target_range, uavs_baseline, 'o-', label='Baseline Density ($\\rho$)')
plt.plot(R_target_range, uavs_congested, 's--', label='Congested Density ($1.2\\rho$)')
plt.title('Question 5: Minimum UAV Fleet Allocation Profile')
plt.xlabel('Target Rate Threshold $R_{target}$ (bps/Hz)')
plt.ylabel('Minimum Required Coordinated UAVs ($N_{min}$)')
plt.grid(True); plt.legend()
plt.show()