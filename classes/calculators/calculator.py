import numpy as np
from scipy.spatial.distance import pdist
import torch

class Calculator():
    def __init__(self, pbc=False, pbc_handler=None):
        self.pbc = pbc
        self.pbc_handler = pbc_handler

    def __call__(self, r):
        return self._V(r=r)
    
    def _V(self, r):
        print("NO potential_function has been assigned yet")

    def _dV_dr(self, r):
        print("No derivative of the potential_function has been assigned yet")

    def forces(self, pos):
        if self.pbc == True:
            diff = self.pbc_handler.get_periodic_dist_vector(atom_pos=pos)
        else:
            diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
        r = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(r, np.inf)
        force_magnitude = self._dV_dr(r)
        forces = np.sum(force_magnitude[..., np.newaxis] * diff / \
                        r[..., np.newaxis], axis=1)
        return forces
    
    def stress_tensor(self,pos,step_size=1e-3):
        if self.pbc == True:
            stress_comps = np.zeros(shape=(2,2))
            l01,l02 = (self.pbc_handler.v1, self.pbc_handler.v2)
            strains = step_size*np.array([[1.0/np.linalg.norm(l01), 0.0], [0.0, 1.0/np.linalg.norm(l01)]])
            E0 = self.energy(pos=pos)
            pos0=pos*1.0
            for i, strain in enumerate(strains):
                scaled_pos = self.pbc_handler.scale_cell_and_coords(atom_pos=pos0, scale_x=strain[0], scale_y=strain[1])
                E_step = self.energy(pos=scaled_pos)
                stress_comps[i] = - (E_step - E0)/strain[i]
                self.pbc_handler.update_params(new_unit_cell=(l01, l02))
            return stress_comps.reshape(2,2)
        else:
            raise Exception("PBC is not set to on, so no unit-cell is given and no volume can be computed")

    def energy(self, pos):
        if self.pbc == True:
            dists = self.pbc_handler.get_periodic_dist(atom_pos=pos)
            return np.sum(self._V(dists))
        if self.pbc == False:
            return np.sum(self._V(pdist(pos)))

class AutoDiff_Calculator(Calculator):
    def __init__(self, pbc=False, pbc_handler=None):
        super().__init__(pbc, pbc_handler)

    def _dV_dr(self, r):
        r = torch.tensor(r, requires_grad=True)
        dv_dr = torch.autograd.grad(self._V(r),r,torch.ones_like(r), create_graph=True)[0]
        with torch.no_grad():
            return dv_dr.numpy()