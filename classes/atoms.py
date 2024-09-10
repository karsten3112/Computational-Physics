import numpy as np
import copy

class Atom():
    def __init__(self, position=np.zeros(2), color="C0", size=50.0, frozen=False) -> None:
        self.pos = position
        self.color = color
        self.plot_elem = None
        self.size = size
        self.frozen = frozen

    def __deepcopy__(self, memo):
        new_object = Atom()
        new_object.pos = copy.copy(self.pos)
        new_object.color = copy.copy(self.color)
        new_object.size = copy.copy(self.size)
        new_object.frozen = copy.copy(self.frozen)
        new_object.plot_elem = None
        memo[id(self)] = new_object
        return new_object

    def plot(self, ax):
        if self.plot_elem == None:
            self.plot_elem = ax.plot(self.pos[0], self.pos[1], "o", color=self.color, ms=self.size, markeredgecolor="k")[0]
        else:
            self.plot_elem.set_data(self.pos[0], self.pos[1])
        return self.plot_elem
    
    def move(self, r):
        if self.frozen == True:
            pass
        else:
            self.pos+=r


class Atom_Collection():
    def __init__(self, atomlist) -> None:
        self.atoms = atomlist
        self.size = len(atomlist)
        self.positions = self.get_positions()
        self.calculator = None
        self.N = 0

    def __len__(self):
        return len(self.atoms)

    def __iter__(self):
        self.N = 0
        return self

    def __next__(self):
        if self.N >= len(self.atoms):
            raise StopIteration
        else:
            self.N+=1
            return self.atoms[self.N-1]

    def __getitem__(self, key):
        return self.atoms[key]
    
    #def __deepcopy__(self, memo):

    def set_sizes(self, new_sizes):
        for atom, new_size in zip(self, new_sizes):
            atom.size = new_size

    def plot(self, ax):
        return [atom.plot(ax=ax) for atom in self.atoms]
    
    def get_positions(self):
        return np.array([atom.pos for atom in self.atoms])
    
    def move_atoms(self, new_pos):
        for atom, pos in zip(self.atoms, new_pos):
            atom.move(pos)

    def move_atom(self, index, pos):
        self.atoms[index].pos+=pos
        self.positions = self.get_positions()

    def set_atom_pos(self, index, pos):
        self.atoms[index].pos=pos
        self.positions = self.get_positions()

    def rattle_atoms(self, delta=0.1, rattle_steps=1):
        for i in range(rattle_steps):
            vs = np.random.rand(len(self.atoms))*2.0*np.pi
            for atom, v in zip(self.atoms, vs):
                atom.move(delta*np.array([np.cos(v), np.sin(v)]))
        self.positions = self.get_positions()

    def freeze_atoms(self, indices):
        for index in indices:
            self.atoms[index].frozen = True
    
    def unfreeze_atoms(self, indices):
        for index in indices:
            self.atoms[index].frozen = False

    def get_forces(self):
        if self.calculator == None:
            raise Exception("No calculator has been assigned yet, therefore no force estimate can be given")
        else:
            return self.calculator.forces(self)

    def get_potential_energy(self):
        if self.calculator == None:
            raise Exception("No calculator has been assigned yet, therefore no energy estimate can be given")
        else:
            return self.calculator.energy(self)
