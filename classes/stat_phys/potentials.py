import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy import special


class P_dist_handler():
    k_b = 1.0 #Can be chosen arb. since it is not used really #1.380649e-23 
    T0 = 1.0#273.15
    def __init__(self, x_min, x_max, N=2000, p_func=None, Y=None, proposal_func=None) -> None:
        self.p_func = p_func
        self.Y = Y
        self.proposal_func = proposal_func
        self.x_max = x_max
        self.x_min = x_min
        self.p_dist = None
        self.N = N

    def get_expectation_value(self, operator, operator_args=None, prop_args=None):
        if self.p_func == None:
            func = lambda x: operator(x, *operator_args)
            if type(self.p_dist) == type(None):
                self.p_dist = self.get_p_distribution(args=prop_args)
                return np.mean(func(self.p_dist))
            else:
                return np.mean(func(self.p_dist))
        else:
            func = lambda x: operator(x, *operator_args)*self.p_func(x, *prop_args[1::])
            return self.direct_integration(func=func)
    
    def direct_integration(self, func):
        exp_val = quad(func, self.x_min, self.x_max)[0]
        return exp_val

    def metro_montecarlo(self, args):
        x = args[3]
        accept_prop = lambda x_prime, x: np.exp((args[0](x, *args[1::])-args[0](x_prime, *args[1::]))/args[1])
        xs = []
        if args[-1] != None:
            while len(xs) < self.N+1:
                p = np.random.rand(1)
                x_prime = self.proposal_func(x)
                acc_prob = accept_prop(x=x, x_prime=x_prime)
                if acc_prob >= 1.0:
                    if p <= 1.0:
                        xs.append(x_prime)
                        x = x_prime
                        #pass
                    else:
                        pass
                else: 
                    if p <= acc_prob:
                        xs.append(x_prime)
                        x = x_prime
                    else:
                        pass
            return np.array(xs)
    
    def montecarlo(self, args):
        p_dist = []
        xs = np.linspace(self.x_min, self.x_max, self.N)
        for x in xs:
            p = np.random.rand(1)
            if p <= self.Y(x, args[0], args[1], args[2]):
                p_dist.append(x)
        return np.array(p_dist)

    def get_p_distribution(self, args):
        if self.p_func == None:
            if self.proposal_func == None:
                print("We did plain-montecarlo")
                return self.montecarlo(args=args[1::])
            else:
                print("We did metro-montecarlo")
                return self.metro_montecarlo(args=args)    
        else:
            print("We return the analytical probability distribution")
            return self.p_func

class H_pot():
    k_b = 1.0 #Can be chosen arb. since it is not used really #1.380649e-23 
    T0 = 1.0#273.15
    def __init__(self,
            T, 
            k=1.0,
            x_0=0.0, 
            p_handler=None,
            v_func = None,
            ) -> None:
        
        self.T = T
        self.beta = 1.0/(self.k_b*T)
        self.k = k
        self.x_0 = x_0
        if v_func == None:
            self.v_func = lambda x, T, k, x_0: 1.0/2.0*self.k*(x-self.x_0)**2
        else:
            self.v_func = v_func
        self.v_avg = None
        self.v_avg_squared = None
        self.cv = None
        self.H_ref = None
        self.use_H_ref = False
        self.p_handler = p_handler

    @property
    def get_v_avg(self):
        if self.H_ref == None or self.use_H_ref == False:
            if self.p_handler == None:
                raise Exception("No p_handler has been assigned yet, so probabilities cannot be computed")
            else:
                if self.v_avg == None:
                    self.v_avg = self.p_handler.get_expectation_value(operator=self.v_func, operator_args=(self.T, self.k, self.x_0), prop_args=(self.v_func, self.T, self.k, self.x_0))
                    return self.v_avg
                else:
                    return self.v_avg
        else:
            if self.v_avg == None:
                delta_v = lambda x: self.v_func(x, self.T, self.k, self.x_0) - self.H_ref.v_func(x, self.H_ref.T, self.H_ref.k, self.H_ref.x_0)
                exp_factor = lambda x, T, k, x_0: np.exp(-1.0/T*delta_v(x))
                v_new = lambda x, T, k, x_0: self.v_func(x, self.T, self.k, self.x_0)*exp_factor(x=x,T=T, k=k, x_0=x_0)
                args = (self.T, self.k, self.x_0)
                t1 = self.H_ref.p_handler.get_expectation_value(operator=exp_factor, operator_args=args, prop_args=args)
                t2 = self.H_ref.p_handler.get_expectation_value(operator=v_new, operator_args=args, prop_args=args)
                self.v_avg = t2/t1
                return self.v_avg
            else:
                return self.v_avg
    @property
    def get_v_avg_squared(self):
        if self.p_handler == None:
            raise Exception("No p_handler has been assigned yet, so probabilities cannot be computed")
        else:
            if self.v_avg_squared == None:
                v_op_squared = lambda x, T, k, x_0: self.v_func(x, T=T, k=k, x_0=x_0)**2
                self.v_avg_squared = self.p_handler.get_expectation_value(v_op_squared, operator_args=(self.T, self.k, self.x_0), prop_args=(self.v_func, self.T, self.k, self.x_0))
                return self.v_avg_squared
            else:
                return self.v_avg_squared
    
    def clear_p_handler(self, new_p_handler=None):
        self.cv = None
        self.v_avg = None
        self.v_avg_squared = None
        self.p_handler = new_p_handler

    def get_cv(self, method="analytical", T_step=1e-5):
        if method == "analytical":
            if self.v_avg == None:
                self.get_v_avg
            if self.v_avg_squared == None:
                self.get_v_avg_squared
            self.cv = 1.0/(self.T**2)*(self.v_avg_squared - self.v_avg**2)
            return self.cv

        if method == "finite_difference":
            v_e_step = self.p_handler.get_expectation_value(operator=self.v_func, operator_args=(self.T+T_step, self.k, self.x_0), prop_args=(self.v_func, self.T+T_step, self.k, self.x_0))
            if self.v_avg == None:
                self.get_v_avg
            self.cv = (v_e_step - self.v_avg)/T_step
            return self.cv
        
    def use_H_pot_ref(self, H_ref):
        self.H_ref = H_ref
        self.use_H_ref = True
        self.clear_p_handler()

    def find_V_min(self, proposal_func, N_step, E_crit):
        x = self.x_0
        accept_prop = lambda x_prime, x: np.exp((self.v_func(x, self.T, self.k, self.x_0)-self.v_func(x_prime, self.T, self.k, self.x_0))/self.T)
        xs = []
        while len(xs) < N_step+1:
            p = np.random.rand(1)
            x_prime = proposal_func(x)
            if self.v_func(x_prime, self.T, self.k, self.x_0) < E_crit:
                    xs.append(x_prime)
                    break
            acc_prob = accept_prop(x=x, x_prime=x_prime)
            if acc_prob >= 1.0:
                if p <= 1.0:
                    xs.append(x_prime)
                    x = x_prime
                    #pass
                else:
                    pass
            else: 
                if p <= acc_prob:
                    xs.append(x_prime)
                    x = x_prime
                else:
                    pass
        return np.array(xs)

    def plot(self, ax, plot_range=[-4.0, 4.0], color="C0", bin_size=0.5):
        xs = np.linspace(plot_range[0], plot_range[1], 400)
        ax.plot(xs, self.v_func(xs,self.T, self.k, self.x_0), color=color)
        if self.v_avg == None:
            print("v_average has not been calculated yet, and cannot be displayed")
        else:
            ax.plot(xs, np.linspace(self.v_avg, self.v_avg, len(xs)), color=color)
        if self.p_handler.p_func == None:

            xs, bins = np.histogram((self.p_handler.p_dist), abs(int((max(self.p_handler.p_dist)-min(self.p_handler.p_dist))/bin_size)))
            xs = xs/(len(self.p_handler.p_dist)/10.0)
            ax.stairs(xs, bins, color=color, alpha=0.4, fill=True)
            for x, bin in zip(xs,bins):
                ax.plot([bin,bin], [0,x], c="k", lw=0.8)
            ax.stairs(xs, bins, color="k")
        else:
            ax.plot(xs, self.p_handler.p_func(xs, self.T, self.k, self.x_0), color="red")




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
