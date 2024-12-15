import numpy as np
from classes.calculators.calculator import Calculator, AutoDiff_Calculator
import copy

class LennardJones(Calculator):
    def __init__(self, eps0=5.0, sigma=2**(-1/6)):
        self.eps0 = eps0
        self.sigma = sigma
        super().__init__()

    def __deepcopy__(self, memo):
        new_object = LennardJones(eps0=self.eps0*1.0, sigma=self.sigma*1.0)
        memo[id(self)] = new_object
        return new_object

    def _dV_dr(self, r):
        t1 = 6.0*(self.sigma**6)/(r**7)
        t2 = 12.0*(self.sigma**12)/(r**13)        
        return 4.0*self.eps0*(t1 - t2)
    
    def _V(self,r):
        return 4.0*self.eps0*((self.sigma/r)**12-(self.sigma/r)**6)


class LJGauss(LennardJones):
    def __init__(self, pbc=False, pbc_handler=None, 
                 gauss_sigma2=0.02,
                 gauss_eps=1.0,
                 r0=1.0,
                 lennard_eps=1.0, 
                 lennard_sigma=1.0
                 ):
        self.gauss_sigma2 = gauss_sigma2
        self.gauss_eps = gauss_eps
        self.r0 = r0
        super().__init__(pbc, pbc_handler, lennard_eps, lennard_sigma)
    
    def _V(self, r):
        lennard_term = super()._V(r)
        gauss_term = self.gauss_eps*np.exp(-(r-self.r0)**2/(2.0*self.gauss_sigma2))
        return lennard_term - gauss_term
    
    def _dV_dr(self, r):
        lennard_term = super()._dV_dr(r)
        gauss_term = self.gauss_eps/self.gauss_sigma2*np.exp(-(r-self.r0)**2/(2.0*self.gauss_sigma2))*(r-self.r0)
        return lennard_term + gauss_term

class LennardJones_AutoDiff(AutoDiff_Calculator):
    def __init__(self, pbc=False, pbc_handler=None, eps0=5.0, sigma=2**(-1/6)):
        self.eps0 = eps0
        self.sigma = sigma
        super().__init__(pbc, pbc_handler)

    def _V(self,r):
        return 4.0*self.eps0*((self.sigma/r)**12-(self.sigma/r)**6)