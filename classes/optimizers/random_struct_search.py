from classes.atoms import Atom, Atom_Collection
from scipy.optimize import fmin
import numpy as np
import copy

class Random_Searcher():
    def __init__(self, atom_col, step_size) -> None:
        atom_list = [copy.deepcopy(atom) for atom in atom_col]
        self.atom_col = Atom_Collection(atomlist=atom_list)
        self.atom_col.calculator = atom_col.calculator
        self.step_size = step_size
        self.best_energy = 0.0

    def convergence_check(self, forces, force_crit=1e-2):
        #is_converged = True
        force_mags = np.linalg.norm(forces, axis=1)
        #print(np.max(force_mags))
        if abs(np.max(force_mags)) <= force_crit:
            return True
        else:
            return False
    
    def grad_descent(self, atom_col, gamma=1e-4, N_max=5000):
        forces = atom_col.get_forces()
        is_converged = self.convergence_check(forces=forces)
        i = 0
        while is_converged == False and i < N_max:
            atom_col.move_atoms(gamma*forces)
            forces = atom_col.get_forces()
            is_converged = self.convergence_check(forces=forces)
            i+=1
        #return atom_col
            
    def E_new(self, alpha, atom_col, force_unit):
        copied_atoms = copy.deepcopy(atom_col)
        new_pos = alpha*force_unit
        copied_atoms.move_atoms(new_pos)
        return copied_atoms.get_potential_energy()
    
    def line_search(self, atom_col, N_max=5000, fmax=0.05, track=False):
        is_converged = False
        i = 0
        opt_list = [copy.deepcopy(atom_col)]
        while is_converged == False and i < N_max:
            forces = atom_col.get_forces()
            forces_unit = forces/np.linalg.norm(forces, axis=1)[:,None]
            alpha_opt = fmin(self.E_new, 0.1, args=(atom_col, forces_unit), disp=False)
            atom_col.move_atoms(alpha_opt*forces_unit)
            opt_list.append(copy.deepcopy(atom_col))
            is_converged = self.convergence_check(atom_col.get_forces(), force_crit=fmax)
            i+=1
        if track == True:
            return opt_list
        else:
            return opt_list[-1]


    def run(self, N_max=200, E_limit=-4.5, track=False, method="grad_descent"):
        i = 0
        init_col = copy.deepcopy(self.atom_col)
        while i < N_max and self.best_energy > E_limit:
            proposal_col = copy.deepcopy(init_col)
            if i == 0:
                if method == "grad_descent":
                    self.grad_descent(atom_col=proposal_col)
                if method == "line_search":
                    self.line_search(atom_col=proposal_col)
                self.best_energy = proposal_col.get_potential_energy()
            else:
                proposal_col.rattle_atoms(delta=self.step_size, rattle_steps=3)
                if method == "grad_descent":
                    self.grad_descent(atom_col=proposal_col)
                if method == "line_search":
                    self.line_search(atom_col=proposal_col)
                new_energy = proposal_col.get_potential_energy()
                #print(new_energy)
                if new_energy < self.best_energy:
                    self.best_energy = new_energy
                    self.atom_col = proposal_col
            i+=1
        return self.atom_col