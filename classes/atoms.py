import numpy as np
from scipy.spatial.distance import pdist
import copy

class Atom():
    def __init__(self, position=np.zeros(2), velocity=np.zeros(2), mass=1.0, color="C0", size=50.0, frozen=False) -> None:
        self.pos = position
        self.color = color
        self.plot_elem = None
        self.size = size
        self.frozen = frozen
        self.velocity = velocity
        self.mass = mass

    def __deepcopy__(self, memo):
        new_object = Atom()
        new_object.pos = copy.copy(self.pos)
        new_object.color = copy.copy(self.color)
        new_object.size = copy.copy(self.size)
        new_object.frozen = copy.copy(self.frozen)
        new_object.velocity = copy.copy(self.velocity)
        new_object.mass = copy.copy(self.mass)
        new_object.plot_elem = None
        memo[id(self)] = new_object
        return new_object

    def plot(self, ax):
        if self.plot_elem == None:
            self.plot_elem = ax.plot(self.pos[0], self.pos[1], "o", color=self.color, ms=self.size, markeredgecolor="k")[0]
        else:
            self.plot_elem.set_data(self.pos[0], self.pos[1])
        return self.plot_elem
    
    def set_velocity(self, v):
        if self.frozen == True:
            pass
        else:
            self.velocity = v

    def boost_velocity(self, v):
        if self.frozen == True:
            pass
        else:
            self.velocity+=v
    
    def set_position(self, r):
        if self.frozen == True:
            pass
        else:
            self.pos = r
    def move(self, r):
        if self.frozen == True:
            pass
        else:
            self.pos+=r


class Atom_Collection():
    def __init__(self, atomlist) -> None:
        self.atoms = atomlist
        self.size = len(atomlist)
        self.velocities = self.get_velocities()
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
    

    def set_sizes(self, new_sizes):
        for atom, new_size in zip(self, new_sizes):
            atom.size = new_size

    def reset_plot_axes(self, new_axes=None):
        if new_axes == None:
            for atom in self.atoms:
                atom.plot_elem = None
        else:
            for atom in self.atoms:
                atom.plot_elem = new_axes

    def plot(self, ax):
        return [atom.plot(ax=ax) for atom in self.atoms]
    
    def get_positions(self):
        return np.array([atom.pos for atom in self.atoms])

    def get_velocities(self):
        return np.array([atom.velocity for atom in self.atoms])

    def get_kinetic_energy(self):
        velocities = self.get_velocities()
        masses = np.array([atom.mass for atom in self.atoms])
        return 1.0/2.0*np.sum(np.dot(masses,velocities**2))

    def move_atoms(self, new_pos):
        for atom, pos in zip(self.atoms, new_pos):
            atom.move(pos)

    def move_atom(self, index, pos):
        self.atoms[index].pos+=pos
        self.positions = self.get_positions()

    def set_atom_pos(self, index, pos):
        self.atoms[index].pos=pos
        self.positions = self.get_positions()

    def set_positions(self, pos):
        for atom, p in zip(self.atoms, pos):
            atom.set_position(p)
        self.positions = self.get_positions()

    def set_velocities(self, vels):
        for atom, vel in zip(self.atoms, vels):
            atom.set_velocity(vel)
        self.velocities = self.get_velocities()
    
    def boost_velocities(self, vels):
        for atom, vel in zip(self.atoms, vels):
            atom.boost_velocity(vel)
        self.velocities = self.get_velocities()

    def get_masses(self):
        return np.array([atom.mass for atom in self.atoms])
    
    def set_atom_vel(self, index, vel):
        self.atoms[index].set_velocity(vel)
        self.velocities = self.get_velocities()

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

    def set_colors(self, colors):
        for atom, color in zip(self.atoms, colors):
            atom.color=color

    def get_distances(self):
        pos = self.get_positions()
        return pdist(pos)

    def get_potential_energy(self):
        if self.calculator == None:
            raise Exception("No calculator has been assigned yet, therefore no energy estimate can be given")
        else:
            return self.calculator.energy(self)
