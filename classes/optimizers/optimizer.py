import copy

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
    
    def scale_cell_and_coords(self, positions, scale_x, scale_y):
        return self.pbc_handler.scale_cell_and_coords(atom_pos=positions, scale_x=scale_x,scale_y=scale_y)

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