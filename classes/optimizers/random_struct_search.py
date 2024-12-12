from classes.atoms import Atom, Atom_Collection
from classes.optimizers.optimizer import Optimizer1, Optimizer
from scipy.optimize import fmin
import numpy as np
import copy


class Line_searcher1(Optimizer1):
    def __init__(self, atom_col):
        super().__init__(atom_col)
        self.best_atom_col = None

    def convergence_check(self, forces, force_crit=0.05):
        force_mags = np.linalg.norm(forces, axis=1)
        if abs(np.max(force_mags)) <= force_crit:
            return True
        else:
            return False
        
    def E_new(self, alpha, forces_unit, current_pos):
        pos_step = alpha*forces_unit
        E = self.atom_col.calculator.energy(current_pos+pos_step*self.atom_col.frozens_bin[:,None])
        return E
    
    def run(self, N_max=5000, fmax=0.05, track=False):
        converged = False
        i = 0
        current_forces = self.get_forces()*1.0
        current_position = self.get_atom_positions()*1.0
        while not(converged) and i < N_max:
            forces_unit = current_forces/np.linalg.norm(current_forces, axis=1)[:,None]
            alpha_opt = fmin(self.E_new, 0.1, args=(forces_unit,current_position), disp=False)
            step_pos = alpha_opt*forces_unit
            self.move_atoms(step_pos)
            current_forces = self.get_forces()*1.0
            current_position = self.get_atom_positions()*1.0
            if track == True:
                self.log_atom_col()
            converged = self.convergence_check(current_forces, force_crit=fmax)
            i+=1
        self.best_atom_col = copy.deepcopy(self.atom_col)
