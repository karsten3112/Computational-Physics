from classes.optimizers.random_struct_search import Line_searcher
from classes.optimizers.metropol import Metropol_new
import numpy as np
from scipy.optimize import fmin

def gauss_dist(shape, sigma=0.2):
    M, N = shape
    return np.random.randn(M,N)*sigma

class Bassin_Hopper():
    def __init__(self, atom_col, temp) -> None:
        self.temp = temp
        self.calculator = atom_col.calculator
        self.init_pos = atom_col.get_positions()
        self.init_force = atom_col.get_forces()
        self.pbc = atom_col.pbc
        self.pbc_handler = atom_col.pbc_handler
        self.frozens = (atom_col.get_frozens() == False).astype(int)
        self.best_E = atom_col.get_potential_energy()
        self.best_pos = self.init_pos*1.0
    
    def convergence_check(self, forces, force_crit=0.05):
        force_mags = np.linalg.norm(forces, axis=1)
        if abs(np.max(force_mags)) <= force_crit:
            return True
        else:
            return False

    def E_new(self, alpha, positions, forces_unit):
        if self.pbc == True:
            new_pos = self.pbc_handler.restrict_positions(positions + alpha*forces_unit*self.frozens[:,None])
        else:
            new_pos = positions + alpha*forces_unit*self.frozens[:,None]
        return self.calculator.energy(new_pos, self.pbc, self.pbc_handler)

    def line_search(self, positions, fmax=0.05, N_max=400):
        converged = False
        i = 0
        current_pos = positions*1.0
        current_forces = self.calculator.forces(current_pos, self.pbc, self.pbc_handler)
        while not(converged) and i < N_max:
            forces_unit = current_forces/np.linalg.norm(current_forces, axis=1)[:,None]
            alpha_opt = fmin(self.E_new, 0.1, args=(current_pos, forces_unit), disp=False)
            if self.pbc == True:
                current_pos = self.pbc_handler.restrict_positions(current_pos+alpha_opt*forces_unit*self.frozens[:,None])
            else:
                current_pos = current_pos+alpha_opt*forces_unit*self.frozens[:,None]

            current_forces = self.calculator.forces(current_pos, self.pbc, self.pbc_handler)
            converged = self.convergence_check(current_forces, force_crit=fmax)
            i+=1
        return current_pos

    def run(self, N_max=500, fmax=0.05, track=False, proposal_func=gauss_dist, prop_args=(0.2,)):
        i = 0
        current_pos = self.init_pos*1.0
        E_currently = self.calculator.energy(current_pos, self.pbc, self.pbc_handler)
        poses = [current_pos]
        energies = [E_currently]
        for i in range(N_max):
            new_pos = current_pos + proposal_func(current_pos.shape, *prop_args)*self.frozens[:,None]
            optimized_pos = self.line_search(positions=new_pos, fmax=fmax)
            E_new = self.calculator.energy(optimized_pos, self.pbc, self.pbc_handler)
            p = np.random.rand()
            acc_prob = np.exp(-(E_new-E_currently)/self.temp)
            if p < acc_prob:
                pos_currently = optimized_pos*1.0
                E_currently = E_new
            else:
                pass
            
            if E_new < self.best_E:
                self.best_E = E_new
                self.best_pos = optimized_pos*1.0
            
            if track == True:
                poses.append(pos_currently)
                energies.append(E_currently)
            else:
                poses = [self.best_pos]
                energies = [self.best_E]
        return np.array(poses), np.array(energies)
