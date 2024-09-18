import numpy as np
import copy
from classes.atoms import Atom, Atom_Collection

class Metropol():
    kb = 1.0
    def __init__(self, atom_col, T, step_size, proposal_func=None) -> None:
        atom_list = [copy.deepcopy(atom) for atom in atom_col]
        self.atom_col = Atom_Collection(atomlist=atom_list)
        self.atom_col.calculator = atom_col.calculator
        self.T = T
        self.step_size=step_size
        if proposal_func == None:
            pass #CODE SOMETHING HERE
        else:
            pass #CODE SOMETHING HERE
    
    def run(self, N_max=3000, E_limit=-4.5, start_quench=2000, track=False):
        i = 0
        E_new = 100
        optimized_cols = []
        E_init = self.atom_col.get_potential_energy()
        best_col = copy.deepcopy(self.atom_col)
        while i < N_max and E_new >= E_limit:
            proposal_col = copy.deepcopy(self.atom_col)
            E_init = proposal_col.get_potential_energy()
            for atom in proposal_col:
                if atom.frozen == True:
                    pass
                else:
                    p = np.random.rand(1)
                    #v = np.random.rand(1)[0]*2.0*np.pi
                    v = np.random.rand(2)*2.0 - 1.0
                    #delr = np.array([self.step_size*np.cos(v), self.step_size*np.sin(v)])
                    #delr = np.array([self.step_size*np.cos(v), self.step_size*np.sin(v)])
                    atom.move(v)
                    E_new = proposal_col.get_potential_energy()
                    acc_prob = np.exp(-(E_new-E_init)/self.T)
                    if i > start_quench:
                        if E_new < E_init:
                            self.atom_col = proposal_col
                            break
                        else:
                            pass 
                    else:
                        if p < acc_prob: #or E_new < E_init:
                            self.atom_col = proposal_col
                            break
            
            if track == True:
                optimized_cols.append(copy.deepcopy(self.atom_col))
            
            if best_col.get_potential_energy() > self.atom_col.get_potential_energy():
                best_col = self.atom_col
                #print(best_col.get_potential_energy())
            
            i+=1
        
        if track == True:
            return optimized_cols
        else:
            return best_col
            #print(i)
        