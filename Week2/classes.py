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
    def __init__(self, x_min, x_max, N=2000, p_func=None, Y=None) -> None:
        self.p_func = p_func
        self.Y = Y
        self.x_max = x_max
        self.x_min = x_min
        self.p_dist = None
        self.N = N

    def get_expectation_value(self, operator, args=None):
        if self.p_func == None:
            func = lambda x: operator(x, *args)
            if type(self.p_dist) == type(None):
                self.p_dist = self.get_p_distribution(args=args)
                return np.mean(func(self.p_dist))
            else:
                return np.mean(func(self.p_dist))
        else:
            func = lambda x: operator(x, *args)*self.p_func(x, *args)
            return self.direct_integration(func=func)
    
    def direct_integration(self, func):
        exp_val = quad(func, self.x_min, self.x_max)[0]
        return exp_val

    def metro_montecarlo(self):
        pass
    
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
            if self.Y == None:
                return self.metro_montecarlo(args=args)
            else:
                return self.montecarlo(args=args)    
        else:
            return self.p_func

class H_pot():
    k_b = 1.0 #Can be chosen arb. since it is not used really #1.380649e-23 
    T0 = 1.0#273.15
    def __init__(self,
            T, 
            k=1.0,
            x_0=0.0, 
            p_handler=None,
            ) -> None:
        
        self.T = T
        self.beta = 1.0/(self.k_b*T)
        self.k = k
        self.x_0 = x_0
        self.v_func = lambda x, T, k, x_0: 1.0/2.0*self.k*(x-self.x_0)**2
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
                    self.v_avg = self.p_handler.get_expectation_value(operator=self.v_func, args=(self.T, self.k, self.x_0))
                    return self.v_avg
                else:
                    return self.v_avg
        else:
            if self.v_avg == None:
                delta_v = lambda x: self.v_func(x, self.T, self.k, self.x_0) - self.H_ref.v_func(x, self.H_ref.T, self.H_ref.k, self.H_ref.x_0)
                exp_factor = lambda x, T, k, x_0: np.exp(-1.0/T*delta_v(x))
                v_new = lambda x, T, k, x_0: self.v_func(x, self.T, self.k, self.x_0)*exp_factor(x=x,T=T, k=k, x_0=x_0)
                t1 = self.H_ref.p_handler.get_expectation_value(operator=exp_factor, args=(self.T, self.k, self.x_0))
                t2 = self.H_ref.p_handler.get_expectation_value(operator=v_new, args=(self.T, self.k, self.x_0))
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
                self.v_avg_squared = self.p_handler.get_expectation_value(v_op_squared, args=(self.T, self.k, self.x_0))
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
            v_e_step = self.p_handler.get_expectation_value(operator=self.v_func, args=(self.T+T_step, self.k, self.x_0))
            if self.v_avg == None:
                self.get_v_avg
            self.cv = (v_e_step - self.v_avg)/T_step
            return self.cv
        
    def use_H_pot_ref(self, H_ref):
        self.H_ref = H_ref
        self.use_H_ref = True
        self.clear_p_handler()


    def plot(self, ax, plot_range=[-4.0, 4.0], color="C0"):
        xs = np.linspace(plot_range[0], plot_range[1], 400)
        ax.plot(xs, self.v_func(xs,self.T, self.k, self.x_0), color=color)
        if self.v_avg == None:
            print("v_average has not been calculated yet, and cannot be displayed")
        else:
            ax.plot(xs, np.linspace(self.v_avg, self.v_avg, len(xs)), color=color)
        if self.p_handler.p_func == None:
            xs, bins = np.histogram((self.p_handler.p_dist), 30)
            xs = xs/(len(self.p_handler.p_dist)/10.0)
            ax.stairs(xs, bins, color=color, alpha=0.4, fill=True)
            for x, bin in zip(xs,bins):
                ax.plot([bin,bin], [0,x], c="k", lw=0.8)
            ax.stairs(xs, bins, color="k")
        else:
            ax.plot(xs, self.p_handler.p_func(xs, self.T, self.k, self.x_0), color="red")