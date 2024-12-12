from classes.optimizers.optimizer import Optimizer1
from classes.optimizers.metropol import Metropol_new1
from classes.optimizers.random_struct_search import Line_searcher1
import numpy as np
import copy

def gauss_dist(N_atoms, sigma=0.2):
    poses = np.random.randn(N_atoms, 2)*sigma
    if N_atoms == 1:
        return poses[0]
    else:
        return poses


class Bassin_Hopper(Optimizer1):
    def __init__(self, atom_col, T, N_lin_search=2000, N_metropol=1000, start_quench=5000):
        super().__init__(atom_col)
        self.T = T
        self.best_atom_col = self.atom_col
        self.N_lin_search = N_lin_search
        self.N_metropol = N_metropol
        self.start_quench = start_quench

    def run(self, N_max=500, fmax=0.05, E_limit=-300, track=False, proposal_func=gauss_dist, prop_args=(0.2,), metro_method="all_atoms"):
        i = 0
        E_current = self.get_potential_energy()
        E_best = E_current
        #print(N_max)
        while i < N_max:
            metropol = Metropol_new1(atom_col=self.atom_col, T=self.T, proposal_func=proposal_func, prop_args=prop_args)
            if metro_method == "all_atoms":
                metropol.run_all_atoms(N_max=self.N_metropol, E_limit=E_limit, track=track, start_quench=self.start_quench)
            if metro_method == "single_atom":
                metropol.run_each_atom(N_max=self.N_metropol, E_limit=E_limit, track=track, start_quench=self.start_quench)

            self.atom_col = copy.deepcopy(metropol.atom_col)

            line_searcher = Line_searcher1(atom_col=self.atom_col)
            line_searcher.run(N_max=self.N_lin_search, fmax=fmax, track=track)
            
            self.atom_col = copy.deepcopy(line_searcher.best_atom_col)
            
            if track == True:
                self.logged_atom_cols+=metropol.logged_atom_cols
                self.logged_atom_cols+=line_searcher.logged_atom_cols
            
            E_current = self.get_potential_energy()
            if E_current < E_best:
                E_best = E_current
                self.best_atom_col = self.atom_col
            i+=1