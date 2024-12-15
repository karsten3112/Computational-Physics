import numpy as np
from scipy.spatial.distance import pdist
import torch
import copy

class Calculator():
    def __init__(self):
        pass

    def __call__(self, r):
        return self._V(r=r)
    
    def _V(self, r):
        print("NO potential_function has been assigned yet")

    def _dV_dr(self, r):
        print("No derivative of the potential_function has been assigned yet")

    def forces(self, distances):
        return self._dV_dr(r=distances)
        
    def energy(self, distances):
        return np.sum(self._V(distances))

class AutoDiff_Calculator(Calculator):
    def __init__(self, pbc=False, pbc_handler=None):
        super().__init__(pbc, pbc_handler)

    def _dV_dr(self, r):
        r = torch.tensor(r, requires_grad=True)
        dv_dr = torch.autograd.grad(self._V(r),r,torch.ones_like(r), create_graph=True)[0]
        with torch.no_grad():
            return dv_dr.numpy()