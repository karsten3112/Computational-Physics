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
    def __init__(self, atomlist, unit_cell=None, pbc=False) -> None:
        self.atoms = atomlist
        self.size = len(atomlist)
        self.pbc = pbc
        self.unit_cell = unit_cell
        self.masses = self.get_masses()
        self.frozens = self.get_frozens()
        self.velocities = self.get_velocities()
        self.calculator = None
        self.N = 0

        if pbc == True:
            self.pbc_handler = PBC_handler(unit_cell_vectors=unit_cell)
            restricted_positions = self.pbc_handler.restrict_positions(np.array([atom.pos for atom in atomlist]))
            self.set_positions(pos=restricted_positions)
        if pbc == False:
            self.positions = self.get_positions()
            self.pbc_handler = None

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

    def plot(self, ax, plot_cell=True):
        if plot_cell == True:
            try:
                l1,l2 = self.unit_cell
                l_end = l1+l2
                for l in [l1, l2]:
                    ax.plot([0.0, l[0]], [0.0, l[1]], "--", c='k')
                    ax.plot([l[0], l_end[0]],[l[1], l_end[1]], "--", c='k')
            except:
                raise Exception("No proper unit_cell has been supplied and cannot be plotted")
        return [atom.plot(ax=ax) for atom in self.atoms]
    
    def get_frozens(self):
        return np.array([atom.frozen for atom in self.atoms])

    def get_positions(self):
        positions = np.array([atom.pos for atom in self.atoms])
        #if self.pbc == True:
        #    return "hej"
        #else:
        return positions

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
        if self.pbc == True:
            return self.pbc_handler.get_periodic_dist(atom_pos=self.positions)
        else:
            return pdist(self.positions)
        
    def get_potential_energy(self):
        if self.calculator == None:
            raise Exception("No calculator has been assigned yet, therefore no energy estimate can be given")
        else:
            return self.calculator.energy(self.positions, self.pbc, self.pbc_handler)



class PBC_handler():
    def __init__(self, unit_cell_vectors):
        self.v1, self.v2 = unit_cell_vectors
        print("NOTE VECTORS SUPPLIED ALSO HAVE TO BE ORDERED, so (v1 --> [L,0]) and (v2 --> [0,L])")
        if np.dot(self.v1, self.v2).astype(int) != 0:
            raise Exception("unit_cell vectors are not orthogonal, and the algorithm will therefore not work")
        self.d1 = np.linalg.norm(self.v1)
        self.d2 = np.linalg.norm(self.v2)

    def restrict_positions(self, atom_pos):
        new_poses = atom_pos*1.0
        #print(new_poses)
        for i, d in enumerate([self.d1, self.d2]):
            res_coord_big = (atom_pos[:,i] > d).astype(int)*(-d)
            res_coord_less = (atom_pos[:,i] <= 0.0).astype(int)*(d) #PERHAPS ADD SUCH THAT THE UNITCELL CAN HAVE ARBITRARY ORIGIN
            new_poses[:,i]+=res_coord_big+res_coord_less
        return new_poses

    def get_periodic_dist(self, atom_pos):
        dists_res = []
        for i, d in enumerate([self.d1, self.d2]):
            dists = pdist(atom_pos[:,i].reshape(-1,1))
            res_coord_big = (dists > d/2.0).astype(int)*(-d)
            res_coord_less = (dists < -d/2.0).astype(int)*(d)
            dists_res.append(dists+res_coord_big+res_coord_less)
        #print(np.array(dists_res))
        return np.linalg.norm(np.array(dists_res).T, axis=1)#np.array(dists_res).reshape(len(dists_res[0]), 2)#np.linalg.norm(np.array(dists_res).reshape(len(dists_res[0]), 2), axis=-1)
    

class Atom_File_handler():
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