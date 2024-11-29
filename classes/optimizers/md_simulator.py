import numpy as np
import copy
from classes.optimizers.optimizer import Optimizer

class MD_Simulator(Optimizer):
    kb = 1.0
    def __init__(self, atom_col, temp, gamma=1e-5, step_size=1e-3, time_step=0.01, integrate_steps=50, method="verlet_integration") -> None:
        self.method = method
        self.integrate_steps=integrate_steps
        self.time_step = time_step
        self.temp = temp
        self.gamma = gamma
        self.step_size = step_size
        super().__init__(atom_col)

    def verlet_integrate(self, positions, velocities, time_step):
        acc = self.get_acceleration(positions)
        pos_step = velocities*time_step + 1.0/2.0*acc*time_step**2
        new_poses = self.move_atom_positions(positions=positions, step_position=pos_step)
        acc_new = self.get_acceleration(new_poses)
        step_velocities = 1.0/2.0*(acc + acc_new)*time_step
        new_vels = self.add_velocities(velocities=velocities, step_velocity=step_velocities)
        return new_poses, new_vels
    
    def euler_integrate(self, positions, velocities, time_step):
        return None

    def N2_integration_with_stress(self, current_pos, current_vel):
        result_poses = []
        result_vels = []
        result_stresses = []
        result_vols = []
        result_Es = []
        for i in range(self.integrate_steps):
            if self.method == "verlet_integration":
                current_pos, current_vel = self.verlet_integrate(positions=current_pos, velocities=current_vel, time_step=self.time_step)
            if self.method == "euler_integration":
                pass
            result_poses.append(current_pos)
            result_vels.append(current_vel)
            result_stresses.append(self.get_stress_tensor(positions=current_pos, step_size=self.step_size))
            result_vols.append(self.get_volume())
            result_Es.append(self.get_energy(current_pos))#+self.kinetic_energy(current_vel))
        
        return result_poses, result_vels, result_stresses, result_vols, result_Es
    
    def N2_integration_without_stress(self, current_pos, current_vel):
        result_poses = [current_pos]
        result_vels = [current_vel]
        for i in range(self.integrate_steps):
            if self.method == "verlet_integration":
                current_pos, current_vel = self.verlet_integrate(positions=current_pos, velocities=current_vel, time_step=self.time_step)
            if self.method == "euler_integration":
                pass
            result_poses.append(current_pos)
            result_vels.append(current_vel)
        return np.array(result_poses), np.array(result_vels)

    def run_MD_simulation(self):
        print("NOT IMPLEMENTED YET")

    def thermostat(self):
        M, N = self.init_pos.shape
        current_vel = np.random.randn(M,N)*np.sqrt(self.kb*self.temp/self.masses[:,None])
        return current_vel
    
    def barostat(self, current_pos, target_stress=np.zeros(shape=(2,2))):
        max_scale = 1e-3
        min_scale = -1e-3
        current_stress = self.get_stress_tensor(positions=current_pos, step_size=self.step_size)
        current_scale = (current_stress-target_stress)*self.gamma #Not sure this is quite right, i should add the previous scale factor here?
        for i in range(len(current_scale)):
            if current_scale[i,i] > max_scale:
                current_scale[i,i] = max_scale
            if current_scale[i,i] < min_scale:
                current_scale[i,i] = min_scale
        current_pos = self.scale_cell_pos(positions=current_pos, scale_x=current_scale[0,0],scale_y=current_scale[1,1]) #remember that barostat scales the cell of PBC_handler and needs to be readjusted if needed
        return current_pos

    def kinetic_energy(self, velocities):
        return 1.0/2.0*np.sum(np.dot(self.masses,velocities**2))

class MDT_Simulator(MD_Simulator):
    def __init__(self, atom_col, temp, time_step=0.01, integrate_steps=50, method="verlet_integration") -> None:
        super().__init__(atom_col, temp, 0.0, 0.0, time_step, integrate_steps, method) #gamma and step_size are not relevant for temperature only.

    def run_MD_simulation(self, N_steps=1000):
        current_pos = self.init_pos*1.0
        current_vel = self.init_velocities*1.0
        result_poses = []
        result_vels = []
        for i in range(N_steps):
            n2_poses, n2_vels = self.N2_integration_without_stress(init_pos=current_pos, init_vel=current_vel)
            
            current_pos = n2_poses[-1]
            current_vel = self.thermostat()
            
            result_poses.append(n2_poses)
            result_vels.append(n2_vels)
       
        return result_poses, result_vels

class MDTP_Simulator(MD_Simulator):
    def __init__(self, atom_col, temp, gamma=0.00001, step_size=0.001, time_step=0.01, integrate_steps=50, method="verlet_integration") -> None:
        super().__init__(atom_col, temp, gamma, step_size, time_step, integrate_steps, method)

    def run_MD_simulation(self, N_steps=1000, reset_unit_cell=True):
        current_pos = self.init_pos*1.0
        current_vel = self.init_velocities*1.0
        
        result_poses = []
        result_vels = []
        result_stresses = []
        result_vols = []
        result_Es = []
        for i in range(N_steps):
            n2_poses, n2_vels, n2_stresses, n2_vols, n2_Es = self.N2_integration_with_stress(current_pos=current_pos, current_vel=current_vel)
            current_pos = n2_poses[-1]
            result_Es.append(n2_Es)
            current_vel = self.thermostat()
            current_pos = self.restrict_positions(self.barostat(current_pos=current_pos))
            
            result_poses.append(n2_poses)
            result_vels.append(n2_vels)

            result_vols.append(n2_vols)
            result_stresses.append(n2_stresses)
        if reset_unit_cell == True:
            self.pbc_handler.update_params(new_unit_cell=self.init_unit_cell)
        return np.array(result_Es), np.array(result_poses), np.array(result_vels), np.array(result_stresses), np.array(result_vols)
