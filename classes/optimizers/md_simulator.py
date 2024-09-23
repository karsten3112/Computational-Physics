import numpy as np
import copy
from classes.atoms import Atom, Atom_Collection

class MD_Simulator():
    def __init__(self, atom_col) -> None:
        atom_col


    def verlet_integrate(self, atom_col, time_step=0.01):
        vs = atom_col.get_velocities()
        xs = atom_col.get_positions()
        
        acc = atom_col.get_forces()/atom_col.get_masses()[:,None]
        x_new = xs + vs*time_step + 1.0/2.0*acc*time_step**2
        atom_col.set_positions(x_new)

        acc_new = atom_col.get_forces()/atom_col.get_masses()[:,None]
        v_new = vs + 1.0/2.0*(acc + acc_new)*time_step
        atom_col.set_velocities(v_new)
        return atom_col
    
    def euler_integrate(self, x, v, time_step):
        x_new = x + v*time_step
        v_new = v + self.acc_func(x)*time_step
        return x_new, v_new

    def run_N2_integration(self, atom_col, N_steps, t_init=0.0,time_step=0.01, track=True, method="verlet_integrate"):
        t = t_init
        result = []
        for i in range(N_steps):
            atom_col_copy = copy.deepcopy(atom_col)
            if method == "verlet_integrate":
                atom_col = self.verlet_integrate(atom_col=atom_col_copy, time_step=time_step)
            if track == True:
                result.append(atom_col_copy)
            t+=time_step
        if track == True:
            return result
        else:
            return [atom_col]


    def run_MD_simulation(self):
        pass

