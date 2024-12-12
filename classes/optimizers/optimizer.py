import copy
from classes.atoms import Atom, Atom_Collection


class Optimizer1():
    def __init__(self, atom_col):
        self.atom_col = copy.deepcopy(atom_col)
        self.atom_col.calculator = atom_col.calculator
        self.logged_atom_cols = []
    
    def move_single_atom(self, index, position):
        self.atom_col.move_atom(index, position)

    def move_atoms(self, positions):
        self.atom_col.move_atoms(positions)

    def get_volume(self):
        return self.atom_col.get_volume()

    def get_forces(self):
        return self.atom_col.get_forces()

    def get_potential_energy(self):
        return self.atom_col.get_potential_energy()

    def get_stress_tensor(self, step_size):
        return self.atom_col.get_stress_tensor(step_size)
    
    def get_atom_velocities(self):
        return self.atom_col.velocities
    
    def set_atom_velocities(self, velocities):
        self.atom_col.set_velocities(velocities)

    def get_atom_positions(self):
        return self.atom_col.positions
    
    def set_atom_positions(self, positions):
        self.atom_col.set_atom_positions(positions)
    
    def get_atom_masses(self):
        return self.atom_col.masses

    def scale_volume(self, scale_x, scale_y):
        self.atom_col.scale_volume(scale_x=scale_x, scale_y=scale_y)
    
    def reset_pbc_handler(self, unit_cell):
        self.pbc_handler.update_params(new_unit_cell=unit_cell)

    def get_acceleration(self):
        return self.get_forces()/self.get_masses()[:,None]

    def restrict_atom_positions(self, positions):
        return self.pbc_handler.restrict_positions(positions)
    
    def log_atom_col(self):
        self.logged_atom_cols.append(copy.deepcopy(self.atom_col))


class Optimizer():
    def __init__(self, atom_col) -> None:
        self.calculator = atom_col.calculator
        self.masses = atom_col.get_masses()
        self.frozens = (atom_col.get_frozens() == False).astype(int)
        self.init_pos = atom_col.get_positions()
        self.init_velocities = atom_col.get_velocities()
        self.pbc_handler = atom_col.pbc_handler
        self.pbc = atom_col.pbc
        if self.pbc == True:
            self.init_volume = atom_col.get_volume()
            self.init_stress = atom_col.get_stress_tensor()
            self.init_unit_cell = atom_col.unit_cell
        else:
            self.init_volume = None
            self.init_stress = None
        pass
    
    def get_volume(self):
        return self.pbc_handler.get_volume()

    def get_forces(self, positions):
        return self.calculator.forces(positions)

    def get_energy(self, positions):
        return self.calculator.energy(positions)

    def get_stress_tensor(self, positions, step_size):
        return self.calculator.stress_tensor(positions, step_size)
    
    def scale_cell(self, scale_x, scale_y):
        self.pbc_handler.scale_cell(scale_x=scale_x,scale_y=scale_y)

    def scale_velocities(self, velocities, scale_x, scale_y):
        return self.pbc_handler.scale_coords(coords=velocities, scale_x=scale_x,scale_y=scale_y)

    def scale_pos(self, positions, scale_x, scale_y):
        return self.pbc_handler.scale_coords(coords=positions, scale_x=scale_x,scale_y=scale_y)

    def scale_all_cell_params(self, positions, velocities, scale_x, scale_y):
        self.scale_cell(scale_x=scale_x, scale_y=scale_y)
        scaled_pos = self.scale_pos(positions=positions, scale_x=scale_x, scale_y=scale_y)
        scaled_vel = self.scale_velocities(velocities=velocities, scale_x=scale_x, scale_y=scale_y)
        return scaled_pos, scaled_vel

    def scale_cell_pos(self, positions, scale_x, scale_y):
        self.scale_cell(scale_x=scale_x, scale_y=scale_y)
        return self.scale_pos(positions=positions, scale_x=scale_x, scale_y=scale_y)

    def move_atom_positions(self, positions, step_position):
        if self.pbc == True:
            return self.pbc_handler.restrict_positions(positions + step_position*self.frozens[:,None])
        else:
            return positions + step_position*self.frozens[:,None]

    def add_velocities(self, velocities, step_velocity):
        return velocities + step_velocity*self.frozens[:,None]
    
    def reset_pbc_handler(self, unit_cell):
        self.pbc_handler.update_params(new_unit_cell=unit_cell)

    def get_acceleration(self, positions):
        return self.get_forces(positions=positions)/self.masses[:,None]

    def restrict_positions(self, positions):
        return self.pbc_handler.restrict_positions(positions)

    def construct_atom_cols(self):
        pass