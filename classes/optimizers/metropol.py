import numpy as np
import copy
#from classes.atoms import Atom, Atom_Collection, PBC_handler
from classes.optimizers.optimizer import Optimizer1

def random_dir_step(N, step_size):
    vs = np.random.rand(N)*2.0*np.pi
    res_vec = np.zeros(shape=(N,2))
    res_vec[:,0]+=step_size*np.cos(vs)
    res_vec[:,1]+=step_size*np.sin(vs)
    return res_vec

def uniform_step(N, step_range):
    poses = np.random.rand(N,2)*step_range - 1.0/2.0*step_range
    if N == 1:
        return poses[0]
    else:
        return poses


class Metropol_new1(Optimizer1):
    kb = 1.0
    def __init__(self, atom_col, T, proposal_func=random_dir_step, prop_args=(0.1,)):
        super().__init__(atom_col)
        self.T = T
        self.proposal_func = proposal_func
        self.prop_args = prop_args
        self.best_E = None
        self.best_atom_col = None
        self.last_atom_col = None


    def run_each_atom(self, N_max=5000, E_limit=None, start_quench=5000, track=True):
        i = 0
        E_currently = self.get_potential_energy()
        prop_func = lambda *prop_args: self.proposal_func(1, *prop_args)
        pos_currently = self.get_atom_positions()*1.0
        frozens = self.atom_col.frozens
        if self.best_E is None:
            self.best_E = E_currently

        while i < N_max and E_currently >= E_limit:
            new_accept = False
            for _ in range(self.atom_col.N_atoms):
                index = np.random.randint(0, self.atom_col.N_atoms, 1) #I dont think this loop is quite fair? Should break after each atom has been moved regardless?
                if frozens[index] == True:
                    pass
                else:
                    proposal_pos = prop_func(*self.prop_args)
                    self.move_single_atom(index=index, position=proposal_pos)
                    E_new = self.get_potential_energy()
                    acc_prob = np.exp(-(E_new-E_currently)/self.T)
                    p = np.random.rand(1)
                    if i > start_quench:
                        if E_new < E_currently:
                            pos_currently = self.get_atom_positions()*1.0
                            E_currently = E_new
                            new_accept = True
                        else:
                            self.set_atom_positions(pos_currently)
                    else:
                        if p < acc_prob:
                            pos_currently = self.get_atom_positions()*1.0
                            E_currently = E_new
                            new_accept = True
                        else:
                            self.set_atom_positions(pos_currently)
                    if new_accept == True:
                        if E_currently < self.best_E:
                            self.best_E = E_currently
                            self.best_pos = pos_currently*1.0
                            self.best_atom_col = copy.deepcopy(self.atom_col) 
                    if track == True:
                        self.log_atom_col()
                    break
            i+=1
        self.last_atom_col = copy.deepcopy(self.atom_col)

    def run_all_atoms(self, N_max=5000, E_limit=None, start_quench=5000, track=True):
        i = 0
        E_currently = self.get_potential_energy()
        prop_func = lambda *prop_args: self.proposal_func(self.atom_col.N_atoms, *prop_args)
        pos_currently = self.get_atom_positions()*1.0
        if self.best_E is None:
            self.best_E = E_currently
        
        while i < N_max and E_currently >= E_limit:
            proposal_pos = prop_func(*self.prop_args)
            self.move_atoms(proposal_pos)
            #print(self.atom_col.positions)
            E_new = self.get_potential_energy()
            acc_prob = np.exp(-(E_new-E_currently)/self.T)
            p = np.random.rand(1)
            
            if i > start_quench:
                if E_new < E_currently:
                    pos_currently = self.get_atom_positions()*1.0
                    E_currently = E_new
                else:
                    #pass
                    self.set_atom_positions(pos_currently)
            else:
                if p < acc_prob:
                    pos_currently = self.get_atom_positions()*1.0
                    E_currently = E_new
                else:
                    #pass
                    self.set_atom_positions(pos_currently)

            if E_currently < self.best_E:
                self.best_E = E_currently
                self.best_pos = pos_currently*1.0
                self.best_atom_col = copy.deepcopy(self.atom_col)
           #print(self.best_E)
            if track == True:
                self.log_atom_col()
            i+=1
        self.last_atom_col = copy.deepcopy(self.atom_col)

class Metropol_new():
    kb = 1.0
    def __init__(self, atom_col, T) -> None:
        self.N = atom_col.size
        self.pbc_handler = atom_col.pbc_handler
        self.calculator = atom_col.calculator
        self.init_pos = atom_col.get_positions()
        self.pbc = atom_col.pbc
        self.T = T
        self.frozens = (atom_col.get_frozens() == False).astype(int)
        self.best_E = 0.0
        self.best_pos = None

    def run(self, N_max, E_limit, start_quench, proposal_func=random_dir_step, track=False, prop_args=(0.1,)):
        i = 0
        E_currently = self.calculator.energy(self.init_pos, self.pbc, self.pbc_handler)
        self.best_E = E_currently
        pos_currently = self.init_pos*1.0
        self.best_pos = pos_currently*1.0
        poses = []
        energies = []
        prop_func = lambda *prop_args: proposal_func(self.N, *prop_args)
        while i < N_max and E_currently >= E_limit:
            if self.pbc == True:
                proposal_pos = self.pbc_handler.restrict_positions(pos_currently + prop_func(*prop_args)*self.frozens[:,None])
            else:
                proposal_pos = pos_currently + prop_func(*prop_args)*self.frozens[:,None]

            E_new = self.calculator.energy(proposal_pos, self.pbc, self.pbc_handler)
            p = np.random.rand(1)
            acc_prob = np.exp(-(E_new-E_currently)/self.T)
            
            if i > start_quench:
                if E_new < E_currently:
                    pos_currently = proposal_pos
                    E_currently = E_new
                else:
                    pass
            else:
                if p < acc_prob:
                    pos_currently = proposal_pos
                    E_currently = E_new
                else:
                    pass
        
            if E_currently < self.best_E:
                self.best_E = E_currently
                self.best_pos = pos_currently*1.0
            
            if track == True:
                poses.append(pos_currently)
                energies.append(E_currently)
            else:
                poses = [self.best_pos]
                energies = [self.best_E]
            i+=1
        return np.array(poses), np.array(energies)
    


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
        