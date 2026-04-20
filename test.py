
import numpy as np
from scipy.stats import norm, expon
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.stats import multivariate_normal as mvn

# time related
from time import time
from tqdm import tqdm

# data
import pickle

# copy
from copy import deepcopy
import os

# Import filter classes
from MGF_improved import extended_kalman_filter as EKF
from MGF_improved import particle_filter as PF
from MGF_improved import multi_model_particle_filter as MMPF
from MGF_improved import multiple_filters as MF
from MGF_improved import fun_ind, fun_linear
from MGF_improved import plot_belief

# Import simulation class
from sim_improved import simulation
import argparse
# --- ARGUMENT PARSING START ---
parser = argparse.ArgumentParser(description='Run Drone Simulation Scenarios')
parser.add_argument('--R', type=float, default=3.0, help='Room height')
parser.add_argument('--L', type=int, default=20, help='Simulation duration')
parser.add_argument('--sig_u', type=float, default=0.03, help='Std of acceleration noise')
parser.add_argument('--sig_v', type=float, default=0.01, help='Std of observation noise')
parser.add_argument('--output_prefix', type=str, default='default', help='Prefix for log files')
args = parser.parse_args()

# Map arguments to your simulation variables
Ts = 0.02
R = args.R
L = args.L
sig_u = args.sig_u
sig_v = args.sig_v
# --- ARGUMENT PARSING END ---

n_scenarios = 20
scenarios = {f"SCENARIO_{i+1}": i for i in range(n_scenarios)}

print("="*80)
print(f"GENERATING SIMULATION DATA FOR {n_scenarios} SCENARIOS")
print("="*80)

# Store all scenarios
all_data = {}

for idx, (scenario_name, variant_seed) in enumerate(scenarios.items()):
    print(f"\n{scenario_name} Scenario:")
    data = simulation(Ts=Ts, R=R, sig_u=sig_u, sig_v=sig_v, L=L)
    data.sparse_obstacles_map_variant(1000 + idx)  # Distinct sparse layout per scenario
    np.random.seed(idx)  # Different realization (drone path, measurements)
    data.run_drone()
    data.run_measure()
    
    # Calculate obstacle coverage
    obstacle_coverage = np.sum((data.au > 0) | (data.ad > 0)) / data.N * 100
    print(f"  Obstacle coverage: {obstacle_coverage:.1f}%")
    print(f"  Number of time steps: {data.N}")
    
    all_data[scenario_name] = data

# Create a unique subdirectory for this specific configuration
plot_dir = os.path.join('plots', args.output_prefix)
os.makedirs(plot_dir, exist_ok=True)

for scenario_name, scenario_data in all_data.items():
    scenario_data.plot(scenario_name)
    # Move the file from default location to our prefixed folder
    if os.path.exists(f'plots/{scenario_name}.png'):
        os.rename(f'plots/{scenario_name}.png', f'{plot_dir}/{scenario_name}.png')
    print(f"  Saved plot: {plot_dir}/{scenario_name}.png")

# Use first scenario for KDE/filter setup
first_scenario_key = list(scenarios.keys())[0]
data = all_data[first_scenario_key]
print(f"\n{'='*80}")
print(f"Using {first_scenario_key} for filter setup (all scenarios run in main loop)")
print(f"{'='*80}\n")

print(f"Number of time steps: {data.N}")
print(f"State dimensions: {data.x.shape}")
print(f"Measurement dimensions: {data.y.shape}")

# ## 2. Define Obstacle Transition Models
# 
# We use 4 models for each obstacle (ceiling and floor):
# 1. **No obstacle**: height = 0
# 2. **Uniform**: height ~ U(0, R)
# 3. **Exponential**: height ~ Exp(μ)
# 4. **KDE**: Kernel Density Estimate from training data


# Collect obstacle height data for KDE
h_ceiling = data.au  # Ceiling obstacle heights
h_floor = data.ad    # Floor obstacle heights

# Model 1: No obstacle
def no_obstacle(x):
    return np.zeros_like(x)

# Model 2: Uniform distribution
def uniform_obstacle(x, a=0, b=1.5):
    return np.random.uniform(a, b, size=x.shape)

# Model 3: Exponential distribution
def exponential_obstacle(x, mu=0.5):
    return np.random.exponential(scale=mu, size=x.shape)

# Model 4: Kernel Density Estimate (FIXED to return proper shape)
class KDE_obstacle:
    def __init__(self, data, bandwidth=0.1):
        self.data = data[data > 0] if np.any(data > 0) else np.array([0.01])
        self.bandwidth = bandwidth
        
    def __call__(self, x):
        M = x.shape[0]
        # Sample from kernel centers
        centers = np.random.choice(self.data, size=M)
        # Add kernel noise
        result = np.maximum(0, centers + np.random.randn(M) * self.bandwidth)
        # IMPORTANT: Return 2D array with shape (M, 1)
        return result.reshape(-1, 1)

kde_ceiling = KDE_obstacle(h_ceiling, bandwidth=0.1)
kde_floor = KDE_obstacle(h_floor, bandwidth=0.1)

print(f"KDE ceiling: {len(kde_ceiling.data)} samples")
print(f"KDE floor: {len(kde_floor.data)} samples")

# ## 3. Helper Classes for Particle Filters


# Observation function class for obstacle particle filters
class ObservationFunction:
    """Observation function for particle filter that returns likelihoods"""
    def __init__(self, R, sigma_v, obs_idx=0):
        """
        Args:
            R: Room height
            sigma_v: Observation noise std
            obs_idx: 0 for ceiling (y1), 1 for floor (y2)
        """
        self.R = R
        self.sigma_v = sigma_v
        self.obs_idx = obs_idx
        self.y = None  # Will be set by particle filter
        self.x_pre = np.zeros(5)  # Full state [x1, x2, x3, x4, x5]
        
    def __call__(self, particles):
        """Compute observation likelihoods for each particle
        
        For ceiling (obs_idx=0): y1 = R - x4 - x1 + noise
        For floor (obs_idx=1): y2 = x1 - x5 + noise
        """
        x1 = self.x_pre[0]  # Drone height from public state
        y_obs = float(self.y[self.obs_idx] if self.y.ndim > 0 else self.y)
        # Vectorized: all particles at once (much faster)
        if self.obs_idx == 0:  # Ceiling: y1 = R - x4 - x1
            y_pred = self.R - particles[:, 0] - x1
        else:  # Floor: y2 = x1 - x5
            y_pred = x1 - particles[:, 0]
        likelihoods = norm.pdf(y_obs, loc=y_pred, scale=self.sigma_v)
        return likelihoods + 1e-300  # Avoid zeros

# Transition function wrapper for obstacles
class TransitionFunction:
    """Wraps transition models to work with particle arrays"""
    def __init__(self, model_fn, noise_std=0.01):
        self.model_fn = model_fn
        self.noise_std = noise_std
        
    def __call__(self, particles):
        # Apply model transition
        new_particles = self.model_fn(particles)
        # Add small process noise
        new_particles = new_particles + np.random.randn(*particles.shape) * self.noise_std
        # Ensure non-negative (obstacles are heights)
        new_particles = np.maximum(new_particles, 0)
        return new_particles

print("Helper classes defined")

# ## 4. Setup Filters for RB-MPF
# 
# ### Group [1]: Extended Kalman Filter for Drone (Rao-Blackwellization!)


# Drone transition model (linear)
F = np.array([[1, Ts, 0.5*Ts**2],
              [0, 1,  Ts],
              [0, 0,  1]])
tr_drone = fun_linear(F)

# Process noise covariance
Q = np.diag([0, 0, sig_u**2])

# Observation model: operates on full state [x1, x2, x3, x4, x5]
# y1 = R - x4 - x1 + noise
# y2 = x1 - x5 + noise
ob_drone_full = fun_linear(
    A=np.array([[-1, 0, 0, -1, 0],   # R - x4 - x1
                [ 1, 0, 0,  0, -1]]), # x1 - x5
    B=np.array([R, 0])
)

# Extract only drone states [0, 1, 2]
ob_drone = fun_ind(ob_drone_full, ind=[0, 1, 2])

# Measurement noise covariance
R_obs = np.eye(2) * sig_v**2

# Create Extended Kalman Filter for Group [1]
filter_drone_kf = EKF(
    mu=np.array([1.5, 0, 0]),  # Initial: height=1.5m, velocity=0, accel=0
    S=np.eye(3) * 0.1,          # Initial covariance
    tr=tr_drone,
    ob=ob_drone,
    St=Q,
    So=R_obs
)

print("Group [1] EKF created for drone states (Rao-Blackwellized)")

# ### Group [2] & [3]: Pairwise obstacle MMPFs (built inside `build_pairwise_mpf` below)


# Number of particles per model (paper: M=1000)
M = 1000
n_iter = 2
rho = 0.6
noise_std = 0.01

# Drone transition/observation for Original MPF (PF for drone)
class DroneTransition:
    def __init__(self, Ts, sig_u):
        self.Ts = Ts
        self.sig_u = sig_u
        self.F = np.array([[1, Ts, 0.5*Ts**2], [0, 1, Ts], [0, 0, 1]])
    def __call__(self, particles):
        new_particles = (self.F @ particles.T).T
        new_particles[:, 2] += np.random.randn(particles.shape[0]) * self.sig_u
        return new_particles

class DroneObservation:
    def __init__(self, R, sig_v):
        self.R, self.sig_v = R, sig_v
        self.y = None
        self.x_pre = np.zeros(5)
    def __call__(self, particles):
        x4, x5 = self.x_pre[3], self.x_pre[4]
        y1_pred = self.R - x4 - particles[:, 0]
        y2_pred = particles[:, 0] - x5
        lik1 = norm.pdf(self.y[0], loc=y1_pred, scale=self.sig_v)
        lik2 = norm.pdf(self.y[1], loc=y2_pred, scale=self.sig_v)
        return lik1 * lik2 + 1e-300

def build_pairwise_mpf(config, seed=None, use_free_energy=False, lambda_reg=0.5):
    """Build standard MPF (no RB) with selectable model switching method."""
    if seed is not None:
        np.random.seed(seed)
    
    M = 1000
    # In build_pairwise_mpf:
    if use_free_energy:
        rho = 0.6  # FE handles temporal consistency via entropy regularization
        lambda_reg = 0.1 # Tune this: 0.1-0.5 usually works
    else:
        rho = 0.6  # Likelihood needs forgetting to prevent dithering
        lambda_reg = 0.0
        
    # --- DRONE FILTER: Always PF (No Rao-Blackwellization) ---
    filter_drone = PF(
        dx=3, dy=2, 
        x0=np.random.randn(M, 3) * 0.1 + np.array([1.5, 0, 0]),
        tr=DroneTransition(Ts, sig_u), 
        ob=DroneObservation(R, sig_v)
    )
    
    # --- OBSTACLE MMPFs with method selection ---
    def make_mmpf(obs_idx):
        trs, x0s = [], []
        
        # Model 0: No obstacle
        trs.append(TransitionFunction(no_obstacle, noise_std=0.01))
        x0s.append(np.zeros((M, 1)))
        
        # Model 1: Config-specific (F2/F3/F4)
        if config == "F2":
            trs.append(TransitionFunction(lambda x: uniform_obstacle(x, a=0, b=R), noise_std=0.01))
            x0s.append(np.random.uniform(0, R, (M, 1)))
        elif config == "F3":
            trs.append(TransitionFunction(lambda x: exponential_obstacle(x, mu=0.5), noise_std=0.01))
            x0s.append(np.random.exponential(0.5, (M, 1)))
        elif config == "F4":
            kde = kde_ceiling if obs_idx == 0 else kde_floor
            trs.append(TransitionFunction(kde, noise_std=0.01))
            x0s.append(np.maximum(kde(np.zeros((M, 1))), 0.0).reshape(M, 1))
        
        pfs = [PF(dx=1, dy=1, x0=x0s[i], tr=trs[i], 
                  ob=ObservationFunction(R, sig_v, obs_idx=obs_idx)) for i in range(2)]
        
        # NEW: Assign model complexity IDs and priors
        # Model 0 (no obstacle) is simplest - gets STRONG preference
        pfs[0].model_id = 0
        pfs[0].model_prior = 2.0  # MUCH stronger positive prior (was 0.5)
        
        # Model 1 (obstacle present) is more complex
        pfs[1].model_id = 1
        pfs[1].model_prior = -1.5  # MUCH stronger negative prior (was -0.3)
        
        # Create MMPF with selected method
        mmpf = MMPF(dx=1, dy=1, use_free_energy=use_free_energy, lambda_reg=lambda_reg)
        mmpf.rho = rho
        for pf in pfs:
            mmpf.add_filter(pf)
        return mmpf
    
    mmpf_ceiling = make_mmpf(0)
    mmpf_floor = make_mmpf(1)
    
    # Coordinator (always MF, no RB)
    mf = MF(n_iter=2)
    mf.add_filter(mmpf_ceiling, [3])
    mf.add_filter(mmpf_floor, [4])
    mf.add_filter(filter_drone, [0, 1, 2])  # Standard PF, not EKF
    mf.set_rho(rho)
    
    return mf

configs = ["F2", "F3", "F4"]
variants = [
    ("Likelihood-Based", False, 0.0),   # Standard: rho=0.6 (inside build_pairwise_mpf logic)
    ("Free Energy (Aggressive)", True, 2.5)  # AGGRESSIVE: Lambda=2.5 for very strong regularization
]

# COMPREHENSIVE TESTING: All scenarios x All configs x All variants
all_results = {}
all_estimates = {}

for scenario_name in scenarios.keys():
    print(f"\n{'='*80}")
    print(f"TESTING SCENARIO: {scenario_name}")
    print(f"{'='*80}")
    
    # Use the appropriate scenario data
    scenario_data = all_data[scenario_name]
    
    all_results[scenario_name] = {}
    all_estimates[scenario_name] = {}
    
    for config in configs:
        all_results[scenario_name][config] = {}
        
        for name, use_fe, lam in variants:
            mf = build_pairwise_mpf(
                config, 
                seed=None, 
                use_free_energy=use_fe, 
                lambda_reg=lam
            )
            
            x_hat = np.zeros((scenario_data.N, 5))
            for i in range(scenario_data.N):
                x_hat[i] = mf.estimate(scenario_data.y[i])
            
            if config == "F2":
                all_estimates[scenario_name][name] = x_hat.copy()
                
            # Calculate RMSEs
            rmse_drone = np.sqrt(np.mean((x_hat[:, 0] - scenario_data.x[:, 0])**2))
            rmse_ceiling = np.sqrt(np.mean((x_hat[:, 3] - scenario_data.au)**2))
            rmse_floor = np.sqrt(np.mean((x_hat[:, 4] - scenario_data.ad)**2))
            rmse_overall = np.sqrt(np.mean((x_hat[:, [0,3,4]] - scenario_data.x[:, [0,3,4]])**2))
            
            all_results[scenario_name][config][name] = {
                "drone": rmse_drone,
                "ceiling": rmse_ceiling,
                "floor": rmse_floor,
                "overall": rmse_overall
            }
            print(f"  {scenario_name}/{config}/{name}: Overall RMSE = {rmse_overall:.4f} m")

# Display comprehensive comparison
print("\n" + "="*80)
print("COMPREHENSIVE COMPARISON ACROSS ALL SCENARIOS")
print("="*80)

for scenario_name in scenarios.keys():
    print(f"\n{'*'*80}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'*'*80}")
    
    for config in configs:
        r = all_results[scenario_name][config]
        print(f"\n{config}:")
        for name in ["Likelihood-Based", "Free Energy (Aggressive)"]:
            v = r[name]
            print(f"  {name:<30}: Overall={v['overall']:.4f}m  Drone={v['drone']:.4f}m  Ceil={v['ceiling']:.4f}m  Floor={v['floor']:.4f}m")
        
        if "Free Energy (Aggressive)" in r and "Likelihood-Based" in r:
            fe, lb = r["Free Energy (Aggressive)"], r["Likelihood-Based"]
            imp = (1 - fe['overall']/lb['overall'])*100
            print(f"  → Improvement: {imp:+.2f}%")

# Summary table
print("\n" + "="*80)
print("SUMMARY: Average Improvement by Scenario")
print("="*80)
print(f"{'Scenario':<15} {'Avg Improvement':<20} {'Best Config':<15}")
print("-"*80)

for scenario_name in scenarios.keys():
    improvements = []
    best_config = None
    best_imp = -999
    
    for config in configs:
        r = all_results[scenario_name][config]
        fe, lb = r["Free Energy (Aggressive)"], r["Likelihood-Based"]
        imp = (1 - fe['overall']/lb['overall'])*100
        improvements.append(imp)
        if imp > best_imp:
            best_imp = imp
            best_config = config
    
    avg_imp = np.mean(improvements)
    print(f"{scenario_name:<15} {avg_imp:+.2f}%              {best_config:<15}")

print("="*80)

# --- True vs estimated state plots (F2, Likelihood vs Free Energy) ---
for scenario_name in scenarios.keys():
    scenario_data = all_data[scenario_name]
    T = scenario_data.T
    x_lik = all_estimates[scenario_name]["Likelihood-Based"]
    x_fe = all_estimates[scenario_name]["Free Energy (Aggressive)"]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # # Drone height
    # axes[0].plot(T, scenario_data.x[:, 0], 'k-', label='True', linewidth=2)
    # axes[0].plot(T, x_lik[:, 0], 'b--', label='Likelihood-Based', alpha=0.8)
    # axes[0].plot(T, x_fe[:, 0], 'r-.', label='Free Energy', alpha=0.8)
    # axes[0].set_ylabel('Height (m)')
    # axes[0].set_title('Drone height')
    # axes[0].legend(loc='upper right')
    # axes[0].grid(True, alpha=0.3)
    
    # Ceiling obstacle
    axes[0].plot(T, scenario_data.au, 'k-', label='True', linewidth=2)
    axes[0].plot(T, x_lik[:, 3], 'b--', label='Likelihood-Based', alpha=0.8)
    axes[0].plot(T, x_fe[:, 3], 'r-.', label='Free Energy', alpha=0.8)
    axes[0].set_ylabel('Height (m)')
    axes[0].set_title('Ceiling obstacle')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Floor obstacle
    axes[1].plot(T, scenario_data.ad, 'k-', label='True', linewidth=2)
    axes[1].plot(T, x_lik[:, 4], 'b--', label='Likelihood-Based', alpha=0.8)
    axes[1].plot(T, x_fe[:, 4], 'r-.', label='Free Energy', alpha=0.8)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Height (m)')
    axes[1].set_title('Floor obstacle')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'True vs estimated states — {scenario_name} (F2)', fontsize=12)
    plt.tight_layout()
    # Updated save path
    plt.savefig(f'{plot_dir}/true_vs_estimates_{scenario_name}.png', dpi=150)
    plt.close()
    print(f"  Saved plot: {plot_dir}/true_vs_estimates_{scenario_name}.png")

# Drone transition for particle filter
class DroneTransition:
    def __init__(self, Ts, sig_u):
        self.Ts = Ts
        self.sig_u = sig_u
        self.F = np.array([[1, Ts, 0.5*Ts**2],
                           [0, 1,  Ts],
                           [0, 0,  1]])
    
    def __call__(self, particles):
        M = particles.shape[0]
        # Apply dynamics
        new_particles = (self.F @ particles.T).T
        # Add process noise (only to acceleration)
        new_particles[:, 2] += np.random.randn(M) * self.sig_u
        return new_particles

# Drone observation for particle filter
class DroneObservation:
    def __init__(self, R, sig_v):
        self.R = R
        self.sig_v = sig_v
        self.y = None
        self.x_pre = np.zeros(5)  # [x1, x2, x3, x4, x5]
        
    def __call__(self, particles):
        """particles shape: (M, 3) containing [x1, x2, x3]"""
        M = particles.shape[0]
        
        # Get obstacle heights from public state
        x4 = self.x_pre[3]  # Ceiling obstacle
        x5 = self.x_pre[4]  # Floor obstacle
        
        # Predicted measurements for each particle
        # y1 = R - x4 - x1
        y1_pred = self.R - x4 - particles[:, 0]
        # y2 = x1 - x5
        y2_pred = particles[:, 0] - x5
        
        # Likelihood for both measurements
        lik1 = norm.pdf(self.y[0], loc=y1_pred, scale=self.sig_v)
        lik2 = norm.pdf(self.y[1], loc=y2_pred, scale=self.sig_v)
        
        return lik1 * lik2 + 1e-300

# Create Particle Filter for drone (Original MPF)
filter_drone_pf = PF(
    dx=3,
    dy=2,
    x0=np.random.randn(M, 3) * 0.1 + np.array([1.5, 0, 0]),  # Init around [1.5, 0, 0]
    tr=DroneTransition(Ts, sig_u),
    ob=DroneObservation(R, sig_v)
)

print("Group [1] PF created for drone states (Original MPF baseline)")

# ### Create Obstacle Filters for Original MPF


# Create ceiling obstacle filters (same as RB-MPF)
pf_ceiling_models_orig = []

pf_ceiling_models_orig.append(PF(
    dx=1, dy=1, x0=np.zeros((M, 1)),
    tr=TransitionFunction(no_obstacle, noise_std=0.01),
    ob=ObservationFunction(R, sig_v, obs_idx=0)
))

pf_ceiling_models_orig.append(PF(
    dx=1, dy=1, x0=np.random.uniform(0, R, (M, 1)),
    tr=TransitionFunction(lambda x: uniform_obstacle(x, a=0, b=R), noise_std=0.01),
    ob=ObservationFunction(R, sig_v, obs_idx=0)
))

pf_ceiling_models_orig.append(PF(
    dx=1, dy=1, x0=np.random.exponential(0.5, (M, 1)),
    tr=TransitionFunction(lambda x: exponential_obstacle(x, mu=0.5), noise_std=0.01),
    ob=ObservationFunction(R, sig_v, obs_idx=0)
))

pf_ceiling_models_orig.append(PF(
    dx=1, dy=1, x0=kde_ceiling(np.zeros((M, 1))),
    tr=TransitionFunction(kde_ceiling, noise_std=0.01),
    ob=ObservationFunction(R, sig_v, obs_idx=0)
))

mmpf_ceiling_orig = MMPF(dx=1, dy=1)
for pf in pf_ceiling_models_orig:
    mmpf_ceiling_orig.add_filter(pf)
mmpf_ceiling_orig.rho = 0.6

# Create floor obstacle filters
pf_floor_models_orig = []

pf_floor_models_orig.append(PF(
    dx=1, dy=1, x0=np.zeros((M, 1)),
    tr=TransitionFunction(no_obstacle, noise_std=0.01),
    ob=ObservationFunction(R, sig_v, obs_idx=1)
))

pf_floor_models_orig.append(PF(
    dx=1, dy=1, x0=np.random.uniform(0, R, (M, 1)),
    tr=TransitionFunction(lambda x: uniform_obstacle(x, a=0, b=R), noise_std=0.01),
    ob=ObservationFunction(R, sig_v, obs_idx=1)
))

pf_floor_models_orig.append(PF(
    dx=1, dy=1, x0=np.random.exponential(0.5, (M, 1)),
    tr=TransitionFunction(lambda x: exponential_obstacle(x, mu=0.5), noise_std=0.01),
    ob=ObservationFunction(R, sig_v, obs_idx=1)
))

pf_floor_models_orig.append(PF(
    dx=1, dy=1, x0=kde_floor(np.zeros((M, 1))),
    tr=TransitionFunction(kde_floor, noise_std=0.01),
    ob=ObservationFunction(R, sig_v, obs_idx=1)
))

mmpf_floor_orig = MMPF(dx=1, dy=1)
for pf in pf_floor_models_orig:
    mmpf_floor_orig.add_filter(pf)
mmpf_floor_orig.rho = 0.6

print("Obstacle filters created for Original MPF")

# ### Create Original MPF Coordinator


# Create Original MPF
mpf_original = MF(n_iter=2)

# Add Group [1]: Drone states [x1, x2, x3] - Particle Filter!
mpf_original.add_filter(filter_drone_pf, index=[0, 1, 2])

# Add Group [2]: Ceiling obstacle [x4]
mpf_original.add_filter(mmpf_ceiling_orig, index=[3])

# Add Group [3]: Floor obstacle [x5]
mpf_original.add_filter(mmpf_floor_orig, index=[4])

# Set forgetting parameter
mpf_original.set_rho(0.6)

print("Original MPF coordinator created (baseline)")