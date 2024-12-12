import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy import special

class H_pot():
    k_b = 1.0 #Can be chosen arb. since it is not used really #1.380649e-23 
    T0 = 1.0#273.15
    def __init__(self, T, v_func, v_func_args):
        self.v_func = v_func
        self.v_func_args = v_func_args
        self.T = T
        self.p_dist = None
        self.p_func = None

    def __call__(self, xs):
        return self.v_func(xs, **self.v_func_args)

    def boltzmann_factor(self, energy):
        return np.exp(-(energy)/(self.k_b*self.T))

    def boltzmann_diff_factor(self, energy, energy_prime):
        return np.exp(-(energy_prime-energy)/(self.k_b*self.T))

    def get_pdist(self, method, **kwargs):
        if self.p_dist is None:
            if method == "montecarlo_sampling":
                self.p_dist = self.montecarlo_sampling(**kwargs)
            if method == "metro_montecarlo_sampling":
                self.p_dist = self.metro_montecarlo_sampling(**kwargs)
            return self.p_dist
        else:
            return self.p_dist
        
    def montecarlo_sampling(self, Y, x_min, x_max, N, **kwargs):
        p_dist = []
        while len(p_dist) < N+1:
            p = np.random.rand(1)
            x = np.random.rand(1)*(x_max-x_min) + x_min
            if p <= Y(x, **kwargs):
                p_dist.append(x)
        return np.array(p_dist)
    
    def direct_integration(self, func, x_interval):
        exp_val = quad(func, *x_interval)[0]
        return exp_val

    def metro_montecarlo_sampling(self, proposal_func, x0, N, terminate_sampling=(False, 0.0), **kwargs):
        accept_prop = lambda x_prime, x: self.boltzmann_diff_factor(self.v_func(x, **self.v_func_args),self.v_func(x_prime, **self.v_func_args))#self.boltzmann_factor(self.v_func(x_prime, **self.v_func_args))/self.boltzmann_factor(self.v_func(x, **self.v_func_args))
        xs = []
        x = x0
        while len(xs) < N:
            p = np.random.rand(1)
            x_prime = proposal_func(x, **kwargs)
            acc_prob = accept_prop(x=x, x_prime=x_prime)
            if p <= min(1.0, acc_prob):
                xs.append(x_prime)
                x = x_prime
            else:
                xs.append(x)
            if terminate_sampling[0] is True:
                if self(x) <= terminate_sampling[1]:
                    break
            else:
                pass
        return np.array(xs)
    
    def plot(self, ax, plot_range=[-4.0, 4.0], colors={"v_avg":"C3","v_func":"k"}):
        xs = np.linspace(plot_range[0], plot_range[1], 400)
        ax.plot(xs, self.v_func(xs, **self.v_func_args),ls="--", color=colors["v_func"], label=r"$V(x)$")
        v_avg = self.get_v_avg()
        ax.plot(xs, np.linspace(v_avg, v_avg, len(xs)), color=colors["v_avg"], label=r"$\left\langle V(x) \right\rangle$ $=$"+f" {round(v_avg,3)}")
    
    def get_expectation_value(self, operator, **kwargs):
        raise Exception("Should be implemented by subclass")

    def get_v_avg(self):
        v_func = lambda x: self.v_func(x, **self.v_func_args)
        return self.get_expectation_value(v_func)

    def get_v_avg_squared(self):
        v_func = lambda x: self.v_func(x, **self.v_func_args)**2
        return self.get_expectation_value(v_func)

    def get_cv_fluct(self):
        return 1.0/(self.T**2)*(self.get_v_avg_squared() - self.get_v_avg()**2)

class H_pot_direct_integration(H_pot):
    def __init__(self, T, v_func, v_func_args, p_func, p_func_args, integration_lims=(-4.0,4.0)):
        super().__init__(T, v_func, v_func_args)
        self.p_func = p_func
        self.p_func_args = p_func_args
        self.integration_lims = integration_lims
    
    def get_expectation_value(self, operator):
        func = lambda x: operator(x)*self.p_func(x, **self.p_func_args)
        return self.direct_integration(func=func, x_interval=self.integration_lims)

    def plot(self, ax, plot_range=[-4, 4], colors={ "v_avg": "C3","v_func": "k", "p_dist":"C0"}):
        super().plot(ax, plot_range, colors)
        xs = np.linspace(plot_range[0], plot_range[1], 400)
        ax.fill_between(xs, self.p_func(xs, **self.p_func_args), color=colors["p_dist"], label=r"$P(x)$", alpha=0.6, hatch="x", edgecolor="k")

    
    def step_temperature(self, T_step):
        self.T+=T_step
        if "T" in self.v_func_args:
            self.v_func_args["T"]+=T_step
        if "T" in self.p_func_args:
            self.p_func_args["T"]+=T_step

    def set_temperature(self, new_T):
        self.T = new_T
        if "T" in self.v_func_args:
            self.v_func_args["T"]=new_T
        if "T" in self.p_func_args:
            self.p_func_args["T"]=new_T

    def get_cv(self, method="analytical", T_step=1e-3):
        if method == "analytical":
            cv = self.get_cv_fluct()
        if method == "finite_difference":
            v_avg_before_step = self.get_v_avg()
            self.step_temperature(T_step=T_step)
            v_avg_after_step = self.get_v_avg()
            self.step_temperature(T_step=-T_step)
            cv = (v_avg_after_step - v_avg_before_step)/T_step
        return cv

class H_plain_MonteCarlo(H_pot):
    def __init__(self, T, 
                 v_func,
                 v_func_args,
                 Y,
                 Y_args,
                 N=2000,
                 sampling_range=(-4.0,4.0)
                 ):
        super().__init__(T, v_func, v_func_args)
        self.N = N
        self.Y = Y
        self.Y_args = Y_args
        self.sampling_range=sampling_range

    def get_pdist(self):
        super().get_pdist("montecarlo_sampling")
    
    def get_v_avg(self):
        if self.p_dist is None:
            self.get_pdist()
        return super().get_v_avg()
    
    def get_v_avg_squared(self):
        if self.p_dist is None:
            self.get_pdist()
        return super().get_v_avg_squared()

    def montecarlo_sampling(self):
        x_min, x_max = self.sampling_range
        return super().montecarlo_sampling(self.Y, x_min, x_max, self.N, **self.Y_args)

    def plot(self, ax, bin_size=0.2, plot_range=[-4, 4], colors={ "v_avg": "C3","v_func": "k" }):
        super().plot(ax, plot_range, colors)
        bin_edges = np.arange(plot_range[0], plot_range[1]+bin_size, bin_size)
        if self.p_dist is None:
            self.get_pdist()
        dist, bin_edges = np.histogram(self.p_dist, bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) /2
        max_dist = np.max(dist)
        ax.bar(bin_centers, dist/max_dist*1.2, width=0.8*bin_size, alpha=0.6, edgecolor="k")

    def get_expectation_value(self, operator):
        if self.p_dist is None:
            self.p_dist = self.get_pdist()
        return np.mean(operator(self.p_dist))


class H_Metro_MonteCarlo(H_pot):
    def __init__(self, T, v_func, v_func_args,
                 x0, 
                 proposal_func, 
                 proposal_func_args, 
                 N):
        super().__init__(T, v_func, v_func_args)
        self.N = N
        self.proposal_func = proposal_func
        self.proposal_func_args = proposal_func_args
        if type(x0) is float:
            self.x0 = np.array([x0])
    
    def get_pdist(self):
        return super().get_pdist("metro_montecarlo_sampling")
    
    def get_v_avg(self):
        if self.p_dist is None:
            self.get_pdist()
        return super().get_v_avg()
    
    def get_v_avg_squared(self):
        if self.p_dist is None:
            self.get_pdist()
        return super().get_v_avg_squared()
    
    def metro_montecarlo_sampling(self, terminate_sampling=(False, 0)):
        return super().metro_montecarlo_sampling(self.proposal_func, self.x0, self.N, terminate_sampling, **self.proposal_func_args)
    
    def get_expectation_value(self, operator):
        if self.p_dist is None:
            self.p_dist = self.get_pdist()
        return np.mean(operator(self.p_dist))
    
    def plot(self, ax, bin_size=0.2, plot_range=[-4, 4], colors={ "v_avg": "C3","v_func": "k" }):
        super().plot(ax, plot_range, colors)
        bin_edges = np.arange(plot_range[0], plot_range[1]+bin_size, bin_size)
        if self.p_dist is None:
            self.get_pdist()
        dist, bin_edges = np.histogram(self.p_dist, bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) /2
        max_dist = np.max(dist)
        ax.bar(bin_centers, dist/max_dist*1.2, width=0.8*bin_size, alpha=0.6, edgecolor="k")

class H_pot_ref(H_pot):
    def __init__(self, H_pot_ref,  T, v_func, v_func_args):
        super().__init__(T, v_func, v_func_args)
        self.H_ref = H_pot_ref
        self.p_dist = H_pot_ref.p_dist#H_pot_ref.get_pdist()

    def delta_v(self, x):
        return self.v_func(x, **self.v_func_args) - self.H_ref.v_func(x, **self.H_ref.v_func_args)

    def get_expectation_value(self, operator):
        f1 = lambda x: np.exp(-self.delta_v(x=x)/(self.k_b*self.T))
        t1 = np.mean(f1(self.p_dist))
        f2 = lambda x: f1(x=x)*operator(x)
        t2 = np.mean(f2(self.p_dist))
        return t2/t1
    

class MD_Simulator():
    kb = 1.0
    T0 = 1.0
    def __init__(self, temp, v_func, acc_func=None) -> None:
        self.temp = temp
        self.v_func = v_func
        if type(acc_func) == None:
            raise Exception("No acceleration function has been provided")
        else:
            self.acc_func = acc_func
    
    def verlet_integrate(self, x, v, time_step):
        x_new = x + v*time_step + 1.0/2.0*self.acc_func(x)*time_step**2
        v_new = v + 1.0/2.0*(self.acc_func(x) + self.acc_func(x_new))*time_step
        return x_new, v_new
    
    def euler_integrate(self, x, v, time_step):
        x_new = x + v*time_step
        v_new = v + self.acc_func(x)*time_step
        return x_new, v_new

    def run_N2_integration(self, N_steps, x_init, v_init, t_init=0.0,time_step=0.01, method="verlet_integrate"):
        result = np.zeros(shape=(N_steps, 3))
        t = t_init
        for i in range(N_steps):
            if method == "verlet_integrate":
                x_init, v_init = self.verlet_integrate(x=x_init, v=v_init, time_step=time_step)
            if method == "euler_integrate":
                x_init, v_init = self.euler_integrate(x=x_init, v=v_init, time_step=time_step)
            result[i] = np.array([t, x_init, v_init])
            t+=time_step
        return result

    def run_metropolis_montecarlo(self, N_points, x_init, proposal_func):
        result = np.zeros(N_points)
        acc_prob = lambda x, x_prime: np.exp((self.v_func(x)-self.v_func(x_prime))/self.temp)
        x = x_init
        i=0
        while i < N_points:
            p = np.random.rand(1)
            x_prime = proposal_func(x)
            if p < acc_prob(x=x, x_prime=x_prime):
                x = x_prime
            else:
                pass
            result[i] = x
            i+=1
        return result

    def run_md_simulation(self, N_steps, x_init, v_init, t_init=0.0, time_step=0.01, method="verlet_integrate", integrate_steps=50):
        result = np.zeros(shape=(N_steps,3))
        t = t_init
        for i in range(N_steps):
            v_init = np.random.randn(1)[0]*np.sqrt(self.kb*self.temp/1.0) #Have to divide by the mass here, but it has been set to 1.
            after_step = self.run_N2_integration(N_steps=integrate_steps, x_init=x_init, v_init=v_init, t_init=t, time_step=time_step, method=method)
            x_init = after_step[-1][1]
            t = after_step[-1][0]
            result[i] = after_step[-1]
            #print(after_step[-1])
        return result

    def get_cv(self, x_distribution):
        return (np.mean(self.v_func(x_distribution)**2) - np.mean(self.v_func(x_distribution))**2)*self.kb/self.temp**2

    def get_v_avg(self, x_distribution):
        return np.mean(self.v_func(x_distribution))

    def plot(self, ax, x_distribution, bin_size=0.5, x_lims=[-2,2]):
        #ax.set_xlim(x_lims)
        xs_plot = np.linspace(x_lims[0], x_lims[1], 1000)
        ax.plot(xs_plot, self.v_func(xs_plot), c='k')
        bin_edges = np.arange(x_lims[0], x_lims[1]+bin_size, bin_size)
        dist, bin_edges = np.histogram(x_distribution, bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) /2
        max_dist = np.max(dist)
        ax.bar(bin_centers, dist/max_dist*1.2, width=0.8*bin_size, alpha=0.6, edgecolor="k")
        ax.set_title(r"$V_{avg}=$"+f"{self.get_v_avg(x_distribution=x_distribution).round(3)}"+"   "+r"$C_{v}=$"+f"{self.get_cv(x_distribution=x_distribution).round(3)}")
