from scipy.spatial.distance import pdist
import numpy as np


class LennardJones():
    def __init__(self,eps0=5,sigma=2**(-1/6)):
        self.eps0 = eps0
        self.sigma = sigma
    
    def _V(self,r):
        return 4.0*self.eps0*((self.sigma/r)**12-(self.sigma/r)**6)
    
    def _dV_dr(self, r):
        t1 = 6.0*(self.sigma**6)/(r**7)
        t2 = 12.0*(self.sigma**12)/(r**13)        
        return 4.0*self.eps0*(t1 - t2)

    def forces(self, pos, pbc, pbc_handler):
        if pbc == True:
            diff = pbc_handler.get_periodic_dist_vector(atom_pos=pos)
        else:
            diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
        r = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(r, np.inf)
        force_magnitude = self._dV_dr(r)
        forces = np.sum(force_magnitude[..., np.newaxis] * diff / \
                        r[..., np.newaxis], axis=1)
        return forces

    def stress_tensor(self, pos, pbc, pbc_handler, step_size=1e-3):
        if pbc == True:
            stress_comps = np.zeros(shape=(2,2))
            l01,l02 = (pbc_handler.v1, pbc_handler.v2)
            strains = step_size*np.array([[1.0/np.linalg.norm(l01), 0.0], [0.0, 1.0/np.linalg.norm(l01)]])
            E0 = self.energy(pos=pos, pbc=pbc, pbc_handler=pbc_handler)
            pos0=pos*1.0
            for i, strain in enumerate(strains):
                scaled_pos = pbc_handler.scale_cell_and_coords(atom_pos=pos0, scale_x=strain[0], scale_y=strain[1])
                E_step = self.energy(pos=scaled_pos, pbc=pbc, pbc_handler=pbc_handler)
                stress_comps[i] = - (E_step - E0)/strain[i]
                pbc_handler.update_params(new_unit_cell=(l01, l02))
            return stress_comps.reshape(2,2)
        else:
            raise Exception("PBC is not set to on, so no unit-cell is given and no volume can be computed")

    def energy(self, pos, pbc, pbc_handler):
        if pbc == True:
            dists = pbc_handler.get_periodic_dist(atom_pos=pos)
            return np.sum(self._V(dists))
        if pbc == False:
            return np.sum(self._V(pdist(pos)))