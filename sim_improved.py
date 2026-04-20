#%% Data generation process
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class simulation:
    def __init__(self,
                Ts, # sampling interval
                R, # height of room
                sig_u, # std of noise on acceleration
                sig_v, # std of noise on observation
                N=None, # number of samples
                L=None # Time duration length of simulation
                ):
        # Setting parameters
        self.Ts = Ts
        self.R = R
        self.sig_u = sig_u
        self.sig_v = sig_v
        
        assert (N is not None) ^ (L is not None), "Provide and only provide one of N or L"
        if N is not None:
            self.N = N
        if L is not None:
            self.N = int(L / Ts)
        
        # place holder of the physical objects
        self.T = np.arange(self.N) * self.Ts # time slots
        self.au = np.zeros(self.N) # up
        self.ad = np.zeros(self.N) # down
        
    def select_section(self, R1, R2):
        """
        return 1: ind:
            an array same size as self.T
            where the percent of time between R1 and R2 are makred as true
        return 2: t:
            an array contains a time starting from T1
            the length of this array is the number of True in return 1
        """
        T1 = R1 * np.max(self.T)
        T2 = R2 * np.max(self.T)
        
        ind = np.logical_and(self.T>=T1, self.T<T2)
        N = np.sum(ind)
        t = np.arange(N) / N * (R2 - R1) * 25
        return ind, t
    
    def sparse_obstacles_map(self):
        """
        SPARSE: Only 15-20% of time has obstacles
        Free Energy should excel here by sticking to "no obstacle" model
        """
        # Ceiling obstacle: short burst
        ind, t = self.select_section(0.2, 0.25)
        self.au[ind] = 0.4
        
        # Floor obstacle: brief appearance
        ind, t = self.select_section(0.5, 0.53)
        self.ad[ind] = 0.5
        
        # Ceiling obstacle: quick ramp
        ind, t = self.select_section(0.75, 0.78)
        self.au[ind] = t / np.max(t) * 0.6
        
        # Floor obstacle: tiny bump
        ind, t = self.select_section(0.88, 0.90)
        self.ad[ind] = 0.3
        
        # Total obstacle time: ~11% (very sparse!)

    def sparse_obstacles_map_variant(self, seed):
        """
        SPARSE but distinct: different obstacle placements per seed.
        Each call with a different seed yields a different sparse layout
        (~10-18% obstacle time).
        """
        rng = np.random.default_rng(seed)
        self.au = np.zeros(self.N)
        self.ad = np.zeros(self.N)
        # 2-3 ceiling segments, 2-3 floor segments; each segment 2-4% of time
        n_ceiling = rng.integers(2, 4)
        n_floor = rng.integers(2, 4)
        for _ in range(n_ceiling):
            start = rng.uniform(0.0, 0.88)
            width = rng.uniform(0.02, 0.05)
            R1, R2 = start, min(1.0, start + width)
            ind, t = self.select_section(R1, R2)
            if np.sum(ind) > 0:
                t_max = np.max(t) if np.max(t) > 0 else 1.0
                shape = rng.choice(["flat", "ramp", "bump"])
                if shape == "flat":
                    self.au[ind] = rng.uniform(0.25, 0.6)
                elif shape == "ramp":
                    self.au[ind] = (t / t_max) * rng.uniform(0.3, 0.6)
                else:
                    self.au[ind] = 0.3 + 0.2 * np.sin(np.pi * t / t_max)
        for _ in range(n_floor):
            start = rng.uniform(0.0, 0.88)
            width = rng.uniform(0.02, 0.05)
            R1, R2 = start, min(1.0, start + width)
            ind, t = self.select_section(R1, R2)
            if np.sum(ind) > 0:
                t_max = np.max(t) if np.max(t) > 0 else 1.0
                shape = rng.choice(["flat", "ramp", "bump"])
                if shape == "flat":
                    self.ad[ind] = rng.uniform(0.25, 0.6)
                elif shape == "ramp":
                    self.ad[ind] = (t / t_max) * rng.uniform(0.3, 0.6)
                else:
                    self.ad[ind] = 0.3 + 0.2 * np.sin(np.pi * t / t_max)
        
    def moderate_obstacles_map(self):
        """
        MODERATE: 30-40% of time has obstacles
        This is where Free Energy's advantage is less clear
        """
        # Ceiling: flat section
        ind, t = self.select_section(0.1, 0.18)
        self.au[ind] = 0.4
        
        # Floor: rising slope
        ind, t = self.select_section(0.25, 0.32)
        self.ad[ind] = t / np.max(t) * 0.8
        
        # Ceiling: falling slope
        ind, t = self.select_section(0.45, 0.52)
        self.au[ind] = (1 - t / np.max(t)) * 0.7
        
        # Floor: wavy section
        ind, t = self.select_section(0.65, 0.75)
        self.ad[ind] = 0.4 + 0.2 * np.sin(2 * np.pi * t / np.max(t))
        
        # Both: complex section
        ind, t = self.select_section(0.85, 0.92)
        self.au[ind] = 0.3
        self.ad[ind] = 0.35
        
        # Total obstacle time: ~36%
        
    def dense_obstacles_map(self):
        """
        DENSE: 60-70% of time has obstacles
        Likelihood-based might do better here
        """
        # Ceiling: long flat
        ind, t = self.select_section(0.05, 0.2)
        self.au[ind] = 0.5
        
        # Floor: long flat
        ind, t = self.select_section(0.15, 0.3)
        self.ad[ind] = 0.4
        
        # Ceiling: complex waveform
        ind, t = self.select_section(0.35, 0.55)
        self.au[ind] = 0.6 + 0.3 * np.sin(3 * np.pi * t / np.max(t))
        
        # Floor: rising then falling
        ind, t = self.select_section(0.5, 0.65)
        tt = t / np.max(t)
        self.ad[ind] = 0.8 * (1 - 4 * (tt - 0.5)**2)  # Parabola
        
        # Both: irregular section
        ind, t = self.select_section(0.7, 0.85)
        self.au[ind] = 0.3 + 0.2 * np.sin(5 * t / np.max(t))
        ind, t = self.select_section(0.72, 0.88)
        self.ad[ind] = 0.4 + 0.15 * np.sin(7 * t / np.max(t))
        
        # Total obstacle time: ~68%
    
    def default_map(self):
        """Original map - kept for backwards compatibility"""
        # one side flat
        ind, t = self.select_section(0.1, 0.15)
        self.ad[ind] = 0.3
        ind, t = self.select_section(0.15, 0.2)
        self.au[ind] = 0.3
        # one side raising 
        ind, t = self.select_section(0.25, 0.3)
        self.ad[ind] = t / np.max(t)
        ind, t = self.select_section(0.3, 0.35)
        self.au[ind] = t / np.max(t)
        # one side falling
        ind, t = self.select_section(0.4, 0.45)
        self.ad[ind] = 1 - t / np.max(t)
        ind, t = self.select_section(0.45, 0.5)
        self.au[ind] = 1 - t / np.max(t)
        # One side irregular
        ind, t = self.select_section(0.55, 0.65)
        self.ad[ind] = 0.6 + 0.4 * np.sin(3 * np.pi * t / np.max(t))
        ind, t = self.select_section(0.65, 0.75)
        self.au[ind] = 0.6 + 0.4 * np.sin(3 * np.pi * t / np.max(t))
        # Both side irregular
        ind, t = self.select_section(0.8, 0.9)
        self.ad[ind] = 0.3 + 0.1 * np.sin(10 * t / np.max(t)) + 0.3 * np.sin(2 * t / np.max(t))
        ind, t = self.select_section(0.8, 0.9)
        self.au[ind] = 0.1 -0.1 * np.sin(5 + 10 * t / np.max(t)) + 0.8 * np.sin(2 * t / np.max(t))
            
    
    def run_drone(self):
        while True:
            x = 1.5 # initial height of drone
            x_1 = 0 # initial speed of the drone

            x_list = []
            for i in range(self.N):
                x_2 = np.random.randn() * self.sig_u
                x_1 = x_1 + self.Ts * x_2
                x = x + self.Ts * x_1 + 0.5 * self.Ts ** 2 * x_2
                x_list.append([x, x_1, x_2, self.au[i], self.ad[i]])
                
                if self.R - self.au[i] - x < 0.3 or x - self.ad[i] < 0.3:
                    break
            else:
                break
            continue
        self.x = np.array(x_list)
        return 
    
    def run_measure(self):
        y_list = []
        for i in range(self.N):
            x = self.x[i, 0]
            yu = self.R - self.au[i] - x + np.random.randn() * self.sig_v
            yd = x - self.ad[i] + np.random.randn() * self.sig_v
            y_list.append([yu, yd])
        self.y = np.array(y_list)
        return
    
    def plot(self, scenario_name = "default", output_dir = "plots"):
        # Calculate obstacle coverage
        obstacle_time = np.sum((self.au > 0) | (self.ad > 0)) / self.N * 100
        
        # plot ground truth
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.T, self.R - self.au, label='Ceiling')
        plt.plot(self.T, self.ad, label='Floor')
        plt.plot(self.T, self.x[:, 0], label='Drone', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Height (m)')
        plt.title(f'Ground Truth (Obstacle Coverage: {obstacle_time:.1f}%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # plot measurements
        plt.subplot(1, 2, 2)
        plt.plot(self.T, self.x[:, 0], label='True Drone Height', linewidth=2)
        plt.plot(self.T, self.x[:, 0] + self.y[:, 0], label='Drone + Ceiling Measurement', alpha=0.5)
        plt.plot(self.T, self.x[:, 0] - self.y[:, 1], label='Drone - Floor Measurement', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Height (m)')
        plt.title('Noisy Measurements')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'{output_dir}/{scenario_name}.png')
        plt.close()
        
        
if __name__ == "__main__":
    print("Testing SPARSE obstacles (Free Energy should excel):")
    data = simulation(Ts=0.02, R=3, sig_u=0.03, sig_v=0.03, L=10)
    data.sparse_obstacles_map()
    data.run_drone()
    data.run_measure()
    data.plot("sparse")
    
    print("\nTesting MODERATE obstacles:")
    data = simulation(Ts=0.02, R=3, sig_u=0.03, sig_v=0.03, L=10)
    data.moderate_obstacles_map()
    data.run_drone()
    data.run_measure()
    data.plot("moderate")
    
    print("\nTesting DENSE obstacles (Likelihood might compete):")
    data = simulation(Ts=0.02, R=3, sig_u=0.03, sig_v=0.03, L=10)
    data.dense_obstacles_map()
    data.run_drone()
    data.run_measure()
    data.plot("dense")