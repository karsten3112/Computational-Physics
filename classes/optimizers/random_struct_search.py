from classes.atoms import Atom, Atom_Collection
import numpy as np
import copy

class Random_Searcher():
    def __init__(self, atom_col, step_size) -> None:
        atom_list = [copy.deepcopy(atom) for atom in atom_col]
        self.atom_col = Atom_Collection(atomlist=atom_list)
        self.atom_col.calculator = atom_col.calculator
        self.step_size = step_size
        self.best_energy = 0.0

    def convergence_check(self, forces, force_crit=1e-3):
        #is_converged = True
        force_mags = np.linalg.norm(forces, axis=1)
        #print(np.max(force_mags))
        if np.max(force_mags) <= force_crit:
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
            
    def sample_E(self, atom, force, atom_col):
        alpha_samples = np.arange(1e-5, 1.0+1e-5, 1e-2)
        Es = np.zeros(len(alpha_samples))
        force_mag = np.linalg.norm(force)
        for i, alpha in enumerate(alpha_samples):
            atom.move(alpha*(force/force_mag))
            Es[i] = atom_col.get_potential_energy()
        index = np.argmin(Es)
        return alpha_samples[index]*force/force_mag

    
    def line_search(self, atom_col, N_max=5000, track=False):
        is_converged = False
        i = 0
        while is_converged == False and i < N_max:
            steps = np.zeros(shape=(len(atom_col),2))
            forces = atom_col.get_forces()
            j=0
            for k, force in enumerate(forces):
                atom_col_copy = copy.deepcopy(atom_col)
                if atom_col_copy[k].frozen == True:
                    pass
                else:
                    steps[j] =  self.sample_E(atom=atom_col_copy[k],atom_col=atom_col_copy, force=force)
                j+=1
            #print(steps)
            atom_col.move_atoms(steps)
            is_converged = self.convergence_check(atom_col.get_forces())
            i+=1



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