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
        force_mags = np.linalg.norm(forces, axis=1)
        if abs(np.max(force_mags)) <= force_crit:
            return True
        else:
            return False
    
    def grad_descent(self, atom_col, gamma=1e-4, N_max=5000):
        positions = atom_col.get_positions()
        forces = atom_col.calculator.forces(positions)
        frozens = atom_col.get_frozens()
        poses_for_update = (frozens == False).astype(int)
        is_converged = self.convergence_check(forces=forces)
        i = 0
        while is_converged == False and i < N_max:
            positions+=gamma*forces*poses_for_update[:,None]
            forces = atom_col.calculator.forces(positions)
            is_converged = self.convergence_check(forces=forces)
            i+=1
        atom_col.set_positions(positions)
        #return positions
            
    
    def E_new(self, alpha, calculator, positions, poses_for_update, forces_unit):
        new_pos = positions + alpha*forces_unit*poses_for_update[:,None]
        return calculator.energy(new_pos)

    def line_search(self, atom_col, N_max=5000, fmax=0.05, track=False):
        is_converged = False
        i = 0
        calculator = atom_col.calculator
        positions = atom_col.get_positions()
        frozens = atom_col.get_frozens()
        poses_for_update = (frozens == False).astype(int)
        poses_tot = [positions]
        #print(positions)
        while is_converged == False and i < N_max:
            forces = calculator.forces(positions)
            forces_unit = forces/np.linalg.norm(forces, axis=1)[:,None]
            alpha_opt = fmin(self.E_new, 0.1, args=(calculator, positions, poses_for_update, forces_unit), disp=False)
            positions = (positions+alpha_opt*forces_unit*poses_for_update[:,None])*1.0 #Makes a copy of the earlier position by multiplying 1.0
            is_converged = self.convergence_check(calculator.forces(positions), force_crit=fmax)
            i+=1
            if track == True:
                poses_tot.append(positions)
            else:
                poses_tot = [positions]
        opt_cols = [copy.deepcopy(atom_col) for i in range(len(poses_tot))]
        for opt_col, pos in zip(opt_cols, poses_tot):
            #print(pos)
            opt_col.set_positions(pos)
        return opt_cols

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