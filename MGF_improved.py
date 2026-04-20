import numpy as np
from numpy.linalg import inv
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod

from time import time


def colize(x):
    return x.flatten().reshape(x.size,-1) 
def rowize(x):
    return x.flatten().reshape(-1, x.size) 

def print_arrays(obj):
    for attr_name in dir(obj):
        attr = getattr(obj, attr_name)
        if type(attr) == type(np.zeros(1)):
            print(attr_name, attr.shape)
    print('\n')

def plot_belief(t, x, sig):
    plt.plot(t, x)
    
    plt.fill_between(t, x + 1.96 * sig, x - 1.96 * sig,
        color='blue',       
        alpha=0.2)          
    
def moments(x, w):
    mu = w @ x
    S = (mu - x).transpose() @ (w[:, np.newaxis] * (mu - x))
    return mu, S



class my_function(ABC):
    def __init__(self, din, dout):
        self.din = din
        self.dout = dout
        
    def __call__(self, x):
        x = x.flatten()
        assert(x.shape[0] == self.din)
        y = self.call_(x).flatten()
        assert(y.shape[0] == self.dout)
        return y
    
    @abstractmethod
    def call_(self, x):
        """
        in:
            din vector
        out:
            dout vector
        """
        pass
    
    def jacobian(self, x):
        x = x.flatten()
        assert(x.shape[0] == self.din)
        J = self.jacobian_(x)
        assert(J.shape == (self.dout, self.din))
        return J
    
    @abstractmethod
    def jacobian_(self, x):
        """
        in:
            din vector
        out:
            dout by din matrix
        """
        pass
    
class fun_linear(my_function):
    def __init__(self, A, B=None):
        """
        dimensions:
            A: dout by din matrix
            B: dout verctor
        """
        super().__init__(din=A.shape[1], dout=A.shape[0])
        self.A = A
        if B is None:
            self.B = np.zeros(self.dout).flatten()
        else: 
            assert(B.flatten().shape[0] == self.dout)
            self.B = B.flatten()
            
    def call_(self, x):
        """
        in:
            din N vector
        out:
            dout M vector
        """
        
        return self.A @ x.flatten() + self.B
    
    def jacobian_(self, x):
        """
        in:
            din vector
        out:
            dout by din matrix
        """
        return self.A
    
class fun_ind(my_function):
    def __init__(self, fun, ind):
        assert(isinstance(fun, my_function))
        super().__init__(din=len(ind), dout=fun.dout)
        self.fun = fun
        self.x_pre = np.zeros(fun.din)
        self.ind = ind
        
    def set_x_pre(self, x_pre):
        """
        x_pre: vector, 
            dimension will match the input formate of fun
        """
        assert(x_pre.flatten().shape[0] == fun.din)
        self.x_pre = x_pre.flatten()
        
        
    def fill(self, x_in):
        x_in = x_in.flatten()
        assert(x_in.shape[0] == self.din)
        x = self.x_pre + 0
        x[self.ind] = x_in
        return x
    
    def call_(self, x):
        return self.fun(self.fill(x)).flatten()
    
    def jacobian_(self, x):
        J = self.fun.jacobian(self.fill(x))
        return J[:, self.ind]


class filtering(ABC):
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy
        
    @abstractmethod
    def predict(self):
        pass
    
    @abstractmethod
    def set_observation_pre(self, x_pre):
        pass
        
    @abstractmethod
    def estimate(self):
        pass

    def set_rho(self, rho):
        self.rho = rho
        
class gau_filtering(filtering):
    """
    Bayesian filtering method that uses Gaussian distribution as belief
    """
    def __init__(self, dx, dy, mu=None, S=None):
        super().__init__(dx, dy)
        
        if mu is None:
            self.mu = np.zeros(dx)
        else:
            assert(mu.flatten().shape[0] == self.dx)
            self.mu = mu.flatten()
            
        if S is None:
            self.S = np.eye(dx)
        else:
            assert(S.shape == (dx, dx))
            self.S = S
            
        self.mu_pr = self.mu + 0
        self.S_pr = self.S + 0
    
class extended_kalman_filter(gau_filtering):
    def __init__(self, mu, S, tr, ob, St, So):
        """
        Define a kalman filter that has a SSM as below:
        x = tr(x) + N(0, St)
        y = ob(x) + N(0, So)
        in:
            mu: Dx vector
            S: Dx * Dx Matrix
            tr(): callable
                in: Dx vector
                out: Dx vector
            tr.jacobian():
                in: Dx vector
                out Dx * Dx Matrix
            ob(): callable
                in: Dx vector
                out: Dy vector
            ob.jacobian():
                in: Dx vector
                out: Dy * Dx Matrix
            St: Dx * Dx Matirx
            So: Dy * Dy Matrix
        """
        super().__init__(dx=ob.din, dy=ob.dout, mu=mu, S=S)
        self.St, self.So = St, So
        self.tr, self.ob = tr, ob
        
    def predict(self):
        
        self.A = self.tr.jacobian(self.mu)
        
        self.mu_pr = self.tr(self.mu)
        self.S_pr = self.A @ self.S @ self.A.transpose() + self.St
        
    def set_observation_pre(self, x_pre):
        assert(isinstance(self.ob, fun_ind))
        assert(x_pre.flatten().shape[0] == self.ob.fun.din)
        self.ob.x_pre = x_pre.flatten()
        
    def estimate(self, y):
        y = y.flatten()
        
        self.C = self.ob.jacobian(self.mu_pr)
        
        self.mu_i = y - self.ob(self.mu_pr)
        self.S_i = self.C @ self.S_pr @ self.C.transpose() + self.So
        
        self.K = self.S_pr @ self.C.transpose() @ inv(self.S_i)
        
        self.mu_po = (colize(self.mu_pr) + self.K @ colize(self.mu_i)).flatten()
        self.S_po = (np.eye(self.mu.size) - self.K @ self.C) @ self.S_pr
        
        self.model_log_likelihood = mvn.logpdf(
            y,
            mean=self.ob(self.mu_pr), 
            cov=self.S_i)
        
        self.mu, self.S = self.mu_po, self.S_po
        return self.mu_po


        
class particle_filter(filtering):      
    def __init__(self, dx=1, dy=1, x0=None, M=200, tr=None, ob=None):
        if x0 is not None:
            M, dx = x0.shape
            self.x = x0
        else:
            self.x = np.random.randn(M, dx)
        super().__init__(dx, dy)
        self.free_energy = 0.0
        self.lambda_reg = 1.0  # Fixed lambda as per paper formulation
        self.mu = self.x.mean(axis=0)
        self.mu_pr = self.mu + 0
        self.tr, self.ob = tr, ob
        self.M = M
        
        
        self.model_id = 0  
        self.model_prior = 0.0  
        
    def predict(self):
        """
        fit self.x in to self.mu and self.S which might be given
        but here is not necessary
        """
        
        self.x_pr = self.tr(self.x)
        
        self.mu_pr, self.S_pr = moments(self.x_pr, np.ones(self.M) / self.M)
        
    def set_observation_pre(self, x_pre):
        self.ob.x_pre = x_pre.flatten()
        
    def estimate(self, y):
        y = y.flatten()
        self.ob.y = y + 0
        
        # 1. Compute Likelihoods p(y|x)
        raw_lik = self.ob(self.x_pr)
        
        # Store log likelihoods for Free Energy calculation before normalization
        # F_term1 = - < log p(y|x) >
        # Add epsilon to raw_lik to prevent log(0)
        self.log_likelihoods = np.log(raw_lik + 1e-300)
        
        # 2. Normalize Weights (Importance Sampling)
        sum_lik = np.sum(raw_lik)
        
        if sum_lik < 1e-300:
            # Handle numerical collapse: if all particles have 0 likelihood
            # Reset to uniform weights to allow the filter to recover/survive
            self.w = np.ones(self.M) / self.M
            self.model_log_likelihood = -1e300 # Extremely low likelihood
        else:
            self.w = raw_lik / sum_lik
            # Standard marginal likelihood
            self.model_log_likelihood = np.log(sum_lik / self.M + 1e-300)
        
        # 3. Calculate Bayesian Free Energy
        # Formula: F = - <log p(y|x)>_w + lambda * H(w)
        
        # Term 1: Accuracy (Expected Negative Log-Likelihood)
        expected_nll = -np.sum(self.w * self.log_likelihoods)
        
        # Term 2: Complexity (Shannon Entropy) H(w) = - sum(w * log w)
        # Note: We penalize entropy, so we ADD lambda * Entropy
        entropy = -np.sum(self.w * np.log(self.w + 1e-300))
        
        self.free_energy = expected_nll + self.lambda_reg * entropy
        
        # 4. Resample / State Estimation
        self.mu_po, self.S_po = moments(self.x_pr, self.w)
        
        # Explicitly enforce sum to 1.0 for np.random.choice to avoid ValueError
        self.w /= np.sum(self.w)
        
        try:
            ind = np.random.choice(self.M, size=self.M, p=self.w)
        except ValueError:
            # Fallback if floating point issues persist
            self.w = np.ones(self.M) / self.M
            ind = np.random.choice(self.M, size=self.M, p=self.w)
            
        self.x = self.x_pr[ind, :]
        
        return self.mu
    
class multi_model_particle_filter(gau_filtering):
    def __init__(self, dx, dy, use_free_energy=False, lambda_reg=1.0):
        super().__init__(dx, dy)
        
        self.rho = 0
        self.filter_list = []
        self.N = 0
        
        self.use_free_energy = use_free_energy
        self.lambda_reg = lambda_reg
        
    def add_filter(self, filter):
        """
        add one particle filter
        in:
            filter: particle_filter
        """
        self.filter_list.append(filter)
        self.N += 1
        self.model_belief = np.ones(self.N) / self.N
        self.model_log_belief = np.zeros(self.N) / self.N
        self.mu_pr = filter.mu_pr + 0
        return
    
    def predict(self):
        """Required by ABC. Delegates to child filters."""
        for i in range(self.N):
            self.filter_list[i].predict()
        
        self.mu_pr = np.zeros_like(self.filter_list[0].mu_pr)
        for i in range(self.N):
            self.mu_pr += self.filter_list[i].mu_pr * self.model_belief[i]
    
    def set_observation_pre(self, x_pre):
        """Required by ABC. Delegates to child filters."""
        for i in range(self.N):
            self.filter_list[i].set_observation_pre(x_pre)
            
    def estimate(self, y):
        y = y.flatten() + 0
        
        # 1. Update individual filters
        for i in range(self.N):
            # Pass the fixed lambda to the child filters
            self.filter_list[i].lambda_reg = self.lambda_reg
            self.filter_list[i].estimate(y)
            
        # 2. Update Model Probabilities (Model Switching)
        if self.use_free_energy:
            # --- Free Energy Logic (Softmin) ---
            # pi(m) = exp(-F_m) / sum(exp(-F_k))
            
            # Collect Free Energies
            Es = np.array([self.filter_list[i].free_energy for i in range(self.N)])
            
            # Softmin Update
            # Subtract min(Es) for numerical stability to prevent underflow
            exp_neg_E = np.exp(-(Es - np.min(Es)))
            self.model_belief = exp_neg_E / np.sum(exp_neg_E)
            
            # Sync log belief for consistency
            self.model_log_belief = np.log(self.model_belief + 1e-300)
            
        else:
            # --- Standard Bayesian Logic (Likelihood-based) ---
            # Posterior ~ Likelihood * Prior
            
            # Collect Log Likelihoods
            LLs = np.array([self.filter_list[i].model_log_likelihood for i in range(self.N)])
            
            # Update Log Belief
            # log p(m|y) \propto log p(y|m) + log p(m)
            self.model_log_belief = LLs + np.log(self.model_belief + 1e-300)
            
            # Normalize
            self.model_log_belief -= np.max(self.model_log_belief)
            self.model_belief = np.exp(self.model_log_belief)
            self.model_belief /= np.sum(self.model_belief)

        # 3. Compute Global Estimate (Bayesian Model Averaging)
        self.mu_po = np.zeros_like(self.filter_list[0].mu_po)
        for i in range(self.N):
            self.mu_po += self.filter_list[i].mu_po * self.model_belief[i]
            
        # Compute Global Covariance
        self.S_po = np.zeros_like(self.filter_list[0].S_po)
        for i in range(self.N):
            diff = colize(self.mu_po - self.filter_list[i].mu_po)
            self.S_po += (self.filter_list[i].S_po + diff @ diff.T) * self.model_belief[i]
        
        # Sync back to filters (optional, depending on architecture)
        for i in range(self.N):
            self.filter_list[i].mu = self.mu_po
            self.filter_list[i].S = self.S_po
    
        return self.mu_po.flatten()

class multiple_filters:
    def __init__(self, n_iter=5):
        
        self.N = 0
        
        self.filter_list = []
        
        self.group = []
        
        self.n_iter = n_iter
        
    def add_filter(self, filter, index):
        """ 
        filter: can be particle_filter or multi_model_particle_filter
            need to have
                filter.estimate()
                filter.prediction
        index: list of int: the index of states
        """
        self.filter_list.append(filter)
        self.group.append(index)
        self.N += 1
        
    def predict(self):
        pass
        
    def estimate(self, y):
        
        for i in range(self.N):
            self.filter_list[i].predict()
            
        
        self.mu_pr = np.zeros(sum([len(index) for index in self.group]))
        for i in range(self.N):
            for j in range(len(self.group[i])):
                self.mu_pr[self.group[i][j]] = self.filter_list[i].mu_pr[j]
        
        
        for t in range(self.n_iter):
            for i in range(len(self.filter_list)):
                
                if hasattr(self.filter_list[i], 'ob') and hasattr(self.filter_list[i].ob, 'fun'):
                    expected_dim = self.filter_list[i].ob.fun.din
                    if self.mu_pr.shape[0] > expected_dim:
                        
                        self.filter_list[i].set_observation_pre(self.mu_pr[:expected_dim])
                    else:
                        self.filter_list[i].set_observation_pre(self.mu_pr)
                else:
                    
                    try:
                        self.filter_list[i].set_observation_pre(self.mu_pr)
                    except:
                        pass
                
            
            for i in range(self.N):
                x_hat_filter = self.filter_list[i].estimate(y)
                for j in range(len(self.group[i])):
                    self.mu_pr[self.group[i][j]] = x_hat_filter[j]
                    
            self.x_hat = self.mu_pr
            
        return self.x_hat
        
    def set_rho(self, rho):
        for filter in self.filter_list:
            try:
                filter.rho = rho
            except:
                pass

class RaoBlackwellized_PF(filtering):
    """
    Rao-Blackwellized Particle Filter for the Indoor UAV Problem.
    - Particles sample the Obstacles (Ceiling Height, Floor Height) and Model States.
    - Each particle has its own Kalman Filter for the Drone State (Height, Velocity, Accel).
    """
    def __init__(self, M=100, R=3.0, obstacle_transition=None):
        
        super().__init__(dx=5, dy=2)
        
        
        self.dx_drone = 3  
        self.dx_obs = 2    
        self.dy = 2        
        self.M = M         
        self.R = R         
        
        
        self.obstacle_transition = obstacle_transition

        self.x_obs = np.abs(np.random.randn(M, self.dx_obs) * 0.1)
        
        
        self.w = np.ones(M) / M

        
        
        self.mu_drone = np.zeros((M, self.dx_drone))
        self.mu_drone[:, 0] = 1.5  
        self.P_drone = np.zeros((M, self.dx_drone, self.dx_drone))
        for i in range(M):
            self.P_drone[i] = np.eye(self.dx_drone) * 0.1

        
        Ts = 0.02 
        self.F = np.array([[1, Ts, 0.5*Ts**2],
                           [0, 1,  Ts],
                           [0, 0,  1]])
        self.Q = np.diag([0, 0, 0.001]) 

        
        
        
        self.H = np.array([[-1, 0, 0], 
                           [ 1, 0, 0]])
        self.R_cov = np.eye(2) * 0.001 

    def predict(self):
        
        
        
        if self.obstacle_transition is not None:
            
            
            
            if isinstance(self.obstacle_transition, dict):
                
                if 'up' in self.obstacle_transition:
                    self.x_obs[:, 0] = self.obstacle_transition['up'](self.x_obs[:, 0:1]).flatten()
                if 'down' in self.obstacle_transition:
                    self.x_obs[:, 1] = self.obstacle_transition['down'](self.x_obs[:, 1:2]).flatten()
            else:
                
                self.x_obs = self.obstacle_transition(self.x_obs)
        else:
            
            
            jump_prob = 0.05
            
            
            noise_static = np.random.randn(self.M, self.dx_obs) * 0.01
            
            noise_jump = np.random.randn(self.M, self.dx_obs) * 0.5
            
            mask = np.random.rand(self.M, self.dx_obs) < jump_prob
            
            self.x_obs = self.x_obs + noise_static * (~mask) + noise_jump * mask
            
            self.x_obs = np.maximum(self.x_obs, 0)

        
        
        
        self.mu_drone = (self.F @ self.mu_drone.T).T 
        
        
        
        for i in range(self.M):
            self.P_drone[i] = self.F @ self.P_drone[i] @ self.F.T + self.Q

    def set_observation_pre(self, x_pre):
        pass 
    
    def set_rho(self, rho):
        
        
        pass

    def estimate(self, y):
        
        y = y.flatten()
        
        
        log_w = np.zeros(self.M)
        
        for i in range(self.M):

            
            obs_offset = np.array([self.R - self.x_obs[i, 0], -self.x_obs[i, 1]])
            z_innov = y - (self.H @ self.mu_drone[i] + obs_offset)
            
            
            S = self.H @ self.P_drone[i] @ self.H.T + self.R_cov
            
            
            
            log_w[i] = mvn.logpdf(z_innov, mean=np.zeros(2), cov=S)
            
            
            K = self.P_drone[i] @ self.H.T @ inv(S)
            self.mu_drone[i] = self.mu_drone[i] + K @ z_innov
            self.P_drone[i] = (np.eye(3) - K @ self.H) @ self.P_drone[i]

        
        log_w = log_w - np.max(log_w)
        self.w = self.w * np.exp(log_w)
        self.w += 1.e-300 
        self.w /= np.sum(self.w)
        
        
        
        mu_drone_est = np.average(self.mu_drone, weights=self.w, axis=0)
        
        x_obs_est = np.average(self.x_obs, weights=self.w, axis=0)
        
        
        self.x_hat = np.hstack([mu_drone_est, x_obs_est])
        
        
        N_eff = 1.0 / np.sum(self.w**2)
        if N_eff < self.M / 2:
            indices = np.random.choice(self.M, size=self.M, p=self.w)
            self.x_obs = self.x_obs[indices]
            self.mu_drone = self.mu_drone[indices]
            self.P_drone = self.P_drone[indices]
            self.w = np.ones(self.M) / self.M
            
        return self.x_hat