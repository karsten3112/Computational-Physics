import numpy as np
import copy
from classes.atoms import Atom, Atom_Collection

class MD_Simulator():
    kb = 1.0
    def __init__(self) -> None:
        pass

    def verlet_integrate(self, atom_col, time_step=0.01):
        vs = atom_col.get_velocities()
        #xs = atom_col.get_positions()
        acc = atom_col.get_forces()/atom_col.get_masses()[:,None]
        #print(acc[-1])
        x_new = vs*time_step + 1.0/2.0*acc*time_step**2
        atom_col.move_atoms(x_new)

        acc_new = atom_col.get_forces()/atom_col.get_masses()[:,None]
        #print(acc_new[-1])
        v_new = 1.0/2.0*(acc + acc_new)*time_step
        atom_col.boost_velocities(v_new)
        return atom_col
    
    def euler_integrate(self, atom_col, time_step):
        vs = atom_col.get_velocities()
        acc = atom_col.get_forces()/atom_col.get_masses()[:,None]
        v_new = acc*time_step
        x_new = vs*time_step
        atom_col.move_atoms(x_new)
        atom_col.boost_velocities(v_new)
        return atom_col

    def run_N2_integration(self, atom_col, N_steps, t_init=0.0,time_step=0.01, track=True, method="verlet_integrate"):
        t = t_init
        result = []
        for i in range(N_steps):
            atom_col_copy = copy.deepcopy(atom_col)
            #print(atom_col_copy)
            if method == "verlet_integrate":
                atom_col = self.verlet_integrate(atom_col=atom_col_copy, time_step=time_step)
            if method == "euler_integrate":
                atom_col = self.euler_integrate(atom_col=atom_col, time_step=time_step)
            if track == True:
                result.append(atom_col_copy)
            t+=time_step
        if track == True:
            return result
        else:
            return [atom_col]


    def run_MD_simulation(self, atom_col, temp, t_init=0.0, N_steps=1000, time_step=0.01, track=True, method="verlet_integrate", integrate_steps=50):
        t = t_init
        atom_cols_result = []
        for i in range(N_steps):
            atom_col_copy = copy.deepcopy(atom_col)
            M, N = atom_col_copy.velocities.shape
            v_init = np.random.randn(M,N)*np.sqrt(self.kb*temp/atom_col_copy.get_masses()[:,None])
            atom_col_copy.set_velocities(v_init)
            atom_cols = self.run_N2_integration(atom_col=atom_col_copy, N_steps=integrate_steps, t_init=t, time_step=time_step, track=track, method=method)
            atom_col = atom_cols[-1]
            if track == True:
                atom_cols_result+=atom_cols
        if track == True:
            return atom_cols_result
        else:
            return [atom_col]

