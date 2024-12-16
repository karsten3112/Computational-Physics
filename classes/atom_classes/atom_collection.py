import numpy as np
from scipy.spatial.distance import pdist
from classes.atom_classes.pbc_handler import PBC_handler
from classes.atom_classes.atom import Atom
import copy

class Atom_Collection():
    def __init__(self, atomlist, unit_cell=None, pbc=False) -> None:
        self.N_atoms = len(atomlist)
        self.pbc = pbc
        self.unit_cell = unit_cell
        self.masses = np.array([atom.mass for atom in atomlist])
        self.frozens = np.array([atom.frozen for atom in atomlist])
        self.frozens_bin = (self.frozens == False).astype(int)
        self.velocities = np.array([atom.velocity for atom in atomlist])
        self.positions = np.array([atom.pos for atom in atomlist])
        self.label = None
        self.plot_axes = None
        self.plot_elems = {}
        
        self.calculator = None
       
        if pbc == True:
            self.pbc_handler = PBC_handler(unit_cell_vectors=unit_cell)
            self.set_atom_positions(pos=self.positions*1.0)
            self.volume = self.get_volume()
        if pbc == False:
            self.positions = self.get_positions()
            self.pbc_handler = None
            self.volume = None

    def __deepcopy__(self, memo):
        atomlist = []
        for pos, vel, frozen, mass in zip(self.positions*1.0, self.velocities*1.0, copy.deepcopy(self.frozens), self.masses*1.0):
            atomlist.append(Atom(position=pos, velocity=vel, frozen=frozen, mass=mass))
        new_object = Atom_Collection(atomlist=atomlist, unit_cell=self.unit_cell, pbc=self.pbc)
        new_object.calculator = copy.deepcopy(self.calculator)
        memo[id(self)] = new_object
        return new_object

    def reset_plot_axes(self, new_axes=None):
        self.plot_elems = {}
        if new_axes is None:
            self.plot_axes = None
        else:
            self.plot_axes = new_axes

    def get_frozens(self):
        return self.frozens

    def get_positions(self):
        return self.positions

    def get_velocities(self):
        return self.velocities
    
    def get_masses(self):
        return self.masses

    def get_kinetic_energy(self): #Not sure this is right
        return 1.0/2.0*np.sum(np.dot(self.masses,self.velocities**2))

    def set_atom_positions(self, pos):
        intermed_pos = self.positions*(self.frozens_bin == 0).astype(int)[:,None]
        intermed_pos += pos*self.frozens_bin[:,None]
        if self.pbc == True:
            restricted_pos = self.pbc_handler.restrict_positions(atom_pos=intermed_pos)
            self.positions = restricted_pos
        else:
            self.positions = intermed_pos

    def move_atoms(self, new_pos):
        pos = self.positions+new_pos*self.frozens_bin[:,None]
        self.set_atom_positions(pos=pos)

    def move_atom(self, index, pos):
        if self.pbc == True:
            restricted_pos = self.pbc_handler.restrict_positions(atom_pos=self.positions[index]+pos*self.frozens_bin[index])
            self.positions[index] = restricted_pos*1.0
        else:
            self.positions[index] += pos*self.frozens_bin[index]

    def set_atom_pos(self, index, pos):
        if self.pbc == True:
            restricted_pos = self.pbc_handler.restrict_positions(atom_pos=self.positions[index]*self.frozens_bin[index])
            self.positions[index]= restricted_pos*1.0
        else:
            self.positions[index]=pos*self.frozens_bin[index]

    def set_velocities(self, vels):
        self.velocities = vels*self.frozens_bin[:,None]
    
    def boost_velocities(self, vels):
        self.velocities+=vels*self.frozens_bin[:,None]
    
    def set_atom_vel(self, index, vel):
        self.velocities[index] = vel

    def rattle_atoms(self, delta=0.1, rattle_steps=1):
        for i in range(rattle_steps):
            vs = np.random.rand(self.N_atoms)*2.0*np.pi
            self.move_atoms(new_pos=delta*np.array([np.cos(vs), np.sin(vs)]).T)

    def freeze_atoms(self, indices):
        for index in indices:
            self.frozens[index] = True
        self.frozens_bin = (self.frozens == False).astype(int)
    
    def unfreeze_atoms(self, indices):
        for index in indices:
            self.frozens[index] = False
        self.frozens_bin = (self.frozens == False).astype(int)
    
    def scale_volume(self, scale_x=1.0, scale_y=1.0):
        if self.pbc == True:
            scaled_positions = self.pbc_handler.scale_cell_and_coords(self.positions*1.0, scale_x=scale_x, scale_y=scale_y)
            self.set_atom_positions(scaled_positions)
            self.unit_cell = (self.pbc_handler.v1*1.0, self.pbc_handler.v2*1.0)
            self.get_volume()
    
    def get_volume(self):
        if self.pbc == True:
            vol = self.pbc_handler.get_volume()
            self.volume = vol
            return vol

    def get_distance_vector(self):
        if self.pbc == True:
            return self.pbc_handler.get_periodic_dist_vector(atom_pos=self.positions)
        else:
            return  self.positions[np.newaxis, :, :] - self.positions[:, np.newaxis, :]
    
    def get_forces(self):
        if self.calculator == None:
            raise Exception("No calculator has been assigned yet, therefore no force estimate can be given")
        else:
            diff = self.get_distance_vector()
            r = np.sqrt(np.sum(diff**2, axis=-1))
            np.fill_diagonal(r, np.inf)
            force_magnitude = self.calculator.forces(r)
            forces = np.sum(force_magnitude[..., np.newaxis] * diff / \
                    r[..., np.newaxis], axis=1)
            return forces

    def get_stress_tensor(self, step_size=1e-3):
        if self.calculator == None:
            raise Exception("No calculator has been assigned yet, therefore no stress estimate can be given")
        else:
            stress_comps = np.zeros(shape=(2,2))
            l01, l02 = self.unit_cell
            strains = step_size*np.array([[1.0/np.linalg.norm(l01), 0.0], [0.0, 1.0/np.linalg.norm(l02)]])
            E0 = self.calculator.energy(self.get_distances())
            for i, strain in enumerate(strains):
                self.scale_volume(scale_x=strain[0], scale_y=strain[1])
                E_step = self.calculator.energy(self.get_distances())
                stress_comps[i] = - (E_step - E0)/strain[i]
                self.scale_volume(scale_x=-strain[0], scale_y=-strain[1])
            return stress_comps.reshape(2,2)

    def get_distances(self):
        if self.pbc == True:
            return self.pbc_handler.get_periodic_dist(atom_pos=self.positions)
        else:
            return pdist(self.positions)
       
    def get_potential_energy(self):
        if self.calculator == None:
            raise Exception("No calculator has been assigned yet, therefore no energy estimate can be given")
        else:
            return self.calculator.energy(self.get_distances())

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