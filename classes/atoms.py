import os
import pickle
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
        self.masses = self.get_masses()
        self.frozens = self.get_frozens()
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
    
    def get_frozens(self):
        return np.array([atom.frozen for atom in self.atoms])

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
        self.frozens=self.get_frozens()
    
    def unfreeze_atoms(self, indices):
        for index in indices:
            self.atoms[index].frozen = False
        self.frozens=self.get_frozens()

    def get_forces(self):
        if self.calculator == None:
            raise Exception("No calculator has been assigned yet, therefore no force estimate can be given")
        else:
            return self.calculator.forces(self.positions)

    def set_colors(self, colors):
        for atom, color in zip(self.atoms, colors):
            atom.color=color

    def get_distances(self):
        return pdist(self.positions)

    def get_potential_energy(self):
        if self.calculator == None:
            raise Exception("No calculator has been assigned yet, therefore no energy estimate can be given")
        else:
            return self.calculator.energy(self.positions)

class Atom_File_handler_new():
    def __init__(self) -> None:
        pass
    
    def save_atom_collections(self, atom_cols, filename):
        try:
            file_obj = open(filename, "xb")
        except:
            file_obj = open(filename, "ab")
        pickle.dump(atom_cols, file=file_obj)

    def load_atom_collections(self, filename):
        try:
            file_obj = open(filename, "rb")
        except:
            raise Exception(f"The specified file: {filename} : Does not exist")
        atom_cols = pickle.load(file_obj)
        return atom_cols

class Atom_File_handler():
    save_attributes = {"position": "\t\t\t\t\t\t\t\t\t",
                       "velocity": "\t\t\t\t\t\t\t\t\t",
                       "frozen_status": "\t\t",
                       "mass": "\t\t"}
    
    load_attributes = {"position": [0,1],
                       "velocity": [2,3],
                       "frozen_status": [4],
                       "mass": [5]}

    def __init__(self) -> None:
        pass
    
    def save_atom_collections(self, atom_cols, filename, save_attributes={}):
        if len(save_attributes) == 0:
            save_attributes = self.save_attributes
        else:
            pass
        try:
            file_obj = open(filename, "x")
            header = self.create_header_line(save_attributes=save_attributes)
            file_obj.write(header)
        except:
            file_obj = open(filename, "a")
        for atom_col in atom_cols:
            for atom in atom_col:
                string_write = ""
                for save_attribute in save_attributes:
                    add_string = self.write_string(atom=atom, save_attribute=save_attribute)
                    string_write+=add_string
                string_write+="\n"
                file_obj.write(string_write)
            file_obj.write("\n")

    def create_header_line(self, save_attributes):
        header = ""
        for save_attribute in save_attributes:
            header+=save_attribute+save_attributes[save_attribute]
        return header+"\n"
    
    def write_string(self, atom, save_attribute):
        if save_attribute == "position":
            string = ""
            for coord in atom.pos:
                string+=f"{coord}\t"
            return string
        
        if save_attribute == "velocity":
            string = ""
            for coord in atom.velocity:
                string+=f"{coord}\t"
            return string
        if save_attribute == "frozen_status":
            return f"{atom.frozen}\t"
        if save_attribute == "mass":
            return f"{atom.mass}\t"

    def set_atom_params(self, string_splitted, atom, load_attributes):
        for load_attribute in load_attributes:
            if load_attribute == "position":
                pos = np.zeros(2)
                for i, index in enumerate(load_attributes[load_attribute]):
                    pos[i] = float(string_splitted[index])
                atom.pos = pos
            if load_attribute == "velocity":
                vel = np.zeros(2)
                for i, index in enumerate(load_attributes[load_attribute]):
                    vel[i] = float(string_splitted[index])
                atom.velocity = vel
            if load_attribute == "mass":
                mass = 0.0
                for i, index in enumerate(load_attributes[load_attribute]):
                    mass = float(string_splitted[index])
                atom.mass = mass
            if load_attribute == "frozen_status":
                status = False
                for i, index in enumerate(load_attributes[load_attribute]):
                    status = bool(string_splitted[index])
                atom.frozen = status

    def load_atom_collections(self, filename, load_attributes={}):
        if len(load_attributes) == 0:
            load_attributes = self.load_attributes
        else:
            pass

        file_obj = open(filename, "r")
        atom_list = []
        atom_cols = []
        for i, line in enumerate(file_obj):
            if i == 0:
                pass
            else:
                splitted_string = line.split("\t")
                if splitted_string[0] == "\n":
                    atom_col = Atom_Collection(atomlist=atom_list)
                    atom_cols.append(atom_col)
                    atom_list = []
                else:
                    atom = Atom()
                    self.set_atom_params(string_splitted=splitted_string, atom=atom, load_attributes=load_attributes)
                    atom_list.append(atom)
        return atom_cols
            