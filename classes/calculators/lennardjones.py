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
        diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
        r = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(r, np.inf)
        force_magnitude = self._dV_dr(r)
        forces = np.sum(force_magnitude[..., np.newaxis] * diff / \
                        r[..., np.newaxis], axis=1)
        return forces

    def energy(self, pos, pbc, pbc_handler):
        if pbc == True:
            dists = pbc_handler.get_periodic_dist(atom_pos=pos)
            return np.sum(self._V(dists))
        if pbc == False:
            return np.sum(self._V(pdist(pos)))