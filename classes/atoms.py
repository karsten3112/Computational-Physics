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
    
    def set_restricted_position(self, r):
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
            self.set_positions(pos=np.array([atom.pos for atom in atomlist]))
            self.volume = self.get_volume()
        if pbc == False:
            self.positions = self.get_positions()
            self.pbc_handler = None
            self.volume = None

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

    def plot(self, ax, plot_cell=False):
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
    
    def plot_cells(self, ax, displacement_vectors="auto", size=50):
        self.set_sizes(new_sizes=[size for i in range(len(self))])
        l1, l2 = self.unit_cell
        if displacement_vectors == "auto":
            for i in [-2.0, -1.0, 0.0, 1.0, 2.0]:
                for j in [-2.0,-1.0, 0.0, 1.0, 2.0]:
                    if i == 0.0 and j == 0.0:
                        self.plot(ax=ax, plot_cell=True)
                    else:
                        displacement = l1*i + l2*j
                        pos_plot = self.positions + displacement
                        ax.plot(pos_plot[:,0], pos_plot[:,1],'o', c="C1", ms=size, alpha=0.5, markeredgecolor="k")
        else:    
            for i, disp_vector in enumerate(displacement_vectors):
                if i == 0:
                    self.plot(ax=ax, plot_cell=True)
                else:
                    ax.plot(self.positions[:,0]+disp_vector[0], self.positions[:,1]+disp_vector[1],'o', c="C1", ms=size, alpha=0.5, markeredgecolor="k")

    def get_frozens(self):
        return np.array([atom.frozen for atom in self.atoms])

    def get_positions(self):
        return np.array([atom.pos for atom in self.atoms])

    def get_velocities(self):
        return np.array([atom.velocity for atom in self.atoms])

    def get_kinetic_energy(self): #Not sure this is right
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
        if self.pbc == True:
            restricted_pos = self.pbc_handler.restrict_positions(atom_pos=pos)
            for atom, p in zip(self.atoms, restricted_pos):
                atom.set_restricted_position(p)
        else:
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
    
    def set_colors(self, colors):
        for atom, color in zip(self.atoms, colors):
            atom.color=color
    
    def scale_volume(self, scale_x=1.0, scale_y=1.0):
        if self.pbc == True:
            scaled_positions = self.pbc_handler.scale_cell_and_coords(self.positions, scale_x=scale_x, scale_y=scale_y)
            self.set_positions(scaled_positions)
            self.unit_cell = (self.pbc_handler.v1, self.pbc_handler.v2)
            self.get_volume()
    
    def get_volume(self):
        if self.pbc == True:
            vol = self.pbc_handler.get_volume()
            self.volume = vol
            return vol

    def get_forces(self):
        if self.calculator == None:
            raise Exception("No calculator has been assigned yet, therefore no force estimate can be given")
        else:
            return self.calculator.forces(self.positions)

    def get_distances(self):
        if self.pbc == True:
            return self.pbc_handler.get_periodic_dist(atom_pos=self.positions)
        else:
            return pdist(self.positions)
        
    def get_potential_energy(self):
        if self.calculator == None:
            raise Exception("No calculator has been assigned yet, therefore no energy estimate can be given")
        else:
            return self.calculator.energy(self.positions)

    def get_stress_tensor(self, step_size=1e-3):
        if self.calculator == None:
            raise Exception("No calculator has been assigned yet, therefore no stress estimate can be given")
        else:
            return self.calculator.stress_tensor(self.positions, step_size)

    def get_pressure(self, step_size=1e-3):
        if self.calculator == None:
            raise Exception("No calculator has been assigned yet, therefore no pressure estimate can be given")
        else:
            stress_tensor = self.get_stress_tensor(step_size=step_size)
            return np.mean(stress_tensor.diagonal())/self.volume

def create_atom_col_copies(atom_col, positions, velocities):
    atom_cols = []
    for pos, vel in zip(positions, velocities):
        atom_col_copy = copy.deepcopy(atom_col)
        atom_col_copy.set_positions(pos)
        atom_col_copy.set_velocities(vel)
        atom_cols.append(atom_col_copy)
    return atom_cols

def custom_metric(u, v):
    return v-u


class PBC_handler():
    def __init__(self, unit_cell_vectors) -> None:
        self.v1, self.v2 = unit_cell_vectors
        self.v1_unit = self.get_unit_vector(self.v1)
        self.v2_unit = self.get_unit_vector(self.v2)
        self.volume = self.get_volume()
        self.trans_matrix, self.inv_trans_matrix = self.get_trans_matrices(v1=self.v1_unit, v2=self.v2_unit)
        self.v1_trans, self.v2_trans = self.project_data(self.v1), self.project_data(self.v2)
        self.d1_trans, self.d2_trans = np.linalg.norm(self.v1_trans), np.linalg.norm(self.v2_trans)

    def get_unit_vector(self, v):
        return v/np.linalg.norm(v)
    
    def get_trans_matrices(self, v1, v2):
        inv_mat = np.array([v1, v2]).T
        mat = np.linalg.inv(inv_mat)
        return mat, inv_mat

    def get_volume(self):
        return np.linalg.norm(np.cross(self.v1, self.v2))

    def project_data(self, xy_pos):
        return np.dot(self.trans_matrix, xy_pos.T).T
    
    def project_back(self, proj_pos):
        return np.dot(self.inv_trans_matrix, proj_pos.T).T

    def restrict_positions(self, atom_pos):
        projected_pos = self.project_data(xy_pos=atom_pos)
        for i, d in enumerate([self.d1_trans, self.d2_trans]):
            res_coord_big = (projected_pos[:,i] > d).astype(int)*(-d)
            res_coord_less = (projected_pos[:,i] <= 0.0).astype(int)*(d) #PERHAPS ADD SUCH THAT THE UNITCELL CAN HAVE ARBITRARY ORIGIN
            projected_pos[:,i]+=res_coord_big+res_coord_less
        return self.project_back(projected_pos)
    
    def get_periodic_dist(self, atom_pos):
        x_diffs = pdist(atom_pos[:,0].reshape(-1,1), metric=custom_metric)
        y_diffs = pdist(atom_pos[:,1].reshape(-1,1), metric=custom_metric)
        projed_data = self.project_data(xy_pos=np.array([x_diffs, y_diffs]).T)
        for i, d in enumerate([self.d1_trans, self.d2_trans]):
            res_coord_big = (projed_data[:,i] > d/2.0).astype(int)*(-d)
            res_coord_less = (projed_data[:,i] < -d/2.0).astype(int)*(d)
            projed_data[:,i]+=res_coord_big+res_coord_less
        
        diffs = self.project_back(projed_data)
        return np.linalg.norm(diffs, axis=1)
    
    def get_periodic_dist_vector(self, atom_pos):
        diff = atom_pos[np.newaxis, :, :] - atom_pos[:, np.newaxis, :]
        projed_data = self.project_data(diff.reshape(len(atom_pos)**2, 2))
        for i, d in enumerate([self.d1_trans, self.d2_trans]):
            res_coord_big = (projed_data[:,i] > d/2.0).astype(int)*(-d)
            res_coord_less = (projed_data[:,i] < -d/2.0).astype(int)*(d)
            projed_data[:,i]+=res_coord_big+res_coord_less
        diffs_final = self.project_back(proj_pos=projed_data)
        return diffs_final.reshape(len(atom_pos), len(atom_pos), 2)

    def update_params(self, new_unit_cell):
        self.v1, self.v2 = new_unit_cell
        self.v1_unit = self.get_unit_vector(self.v1)
        self.v2_unit = self.get_unit_vector(self.v2)
        self.volume = self.get_volume()
        self.trans_matrix, self.inv_trans_matrix = self.get_trans_matrices(v1=self.v1_unit, v2=self.v2_unit)
        self.v1_trans, self.v2_trans = self.project_data(self.v1), self.project_data(self.v2)
        self.d1_trans, self.d2_trans = np.linalg.norm(self.v1_trans), np.linalg.norm(self.v2_trans)

    def scale_cell_and_coords(self, atom_pos, scale_x=1.0, scale_y=1.0):
        scale_mat = np.array([[1.0+scale_x, 0.0], [0.0, 1.0+scale_y]])
        self.v1 = np.dot(scale_mat, self.v1.T)
        self.v2 = np.dot(scale_mat, self.v2.T)
        self.update_params(new_unit_cell=(self.v1, self.v2))
        scaled_pos = np.dot(scale_mat, atom_pos.T)
        return scaled_pos.T



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