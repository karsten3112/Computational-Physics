from classes.optimizers.optimizer import Optimizer1 as Optimizer
import numpy as np
import copy



class MD_simulator(Optimizer):
    kb = 1.0
    def __init__(self, atom_col, temp, log_level="only_md_steps", integration_method="verlet_integration"):
        super().__init__(atom_col)
        self.log_level = log_level
        self.temp = temp
        self.integration_method = integration_method

    def perform_verlet_step(self, time_step=1e-3):
        masses = self.get_atom_masses()
        acc = self.get_forces()/masses[:,None]
        pos_step = self.get_atom_velocities()*time_step + 1.0/2.0*acc*time_step**2
        self.set_atom_positions(self.get_atom_positions()+pos_step)
        acc_new = self.get_forces()/masses[:,None]
        step_velocities = 1.0/2.0*(acc + acc_new)*time_step
        #print(self.get_atom_velocities()+step_velocities)
        self.set_atom_velocities((self.get_atom_velocities()+step_velocities)*1.0)
    
    def perform_euler_step(self, time_step=1e-3):
        masses = self.get_atom_masses()
        acc = self.get_forces()/masses[:,None]
        pos_step = self.get_atom_velocities()*time_step
        self.set_atom_positions(self.get_atom_positions()+pos_step)
        step_velocities = acc*time_step
        self.set_atom_velocities(self.get_atom_velocities()+step_velocities)

    def N2_integration(self, time_step=1e-3, integrate_steps=50):
        if self.log_level == "all_steps":
            if self.integration_method == "verlet_integration":
                for _ in range(integrate_steps):
                    self.perform_verlet_step(time_step=time_step)
                    self.log_atom_col()
            elif self.integration_method == "euler_integration":
                for _ in range(integrate_steps):
                    self.perform_euler_step(time_step=time_step)
                    self.log_atom_col()
            else:
                raise Exception(f"Specified integration method: {self.integration_method} - has not been implemented yet")
        else:
            if self.integration_method == "verlet_integration":
                for _ in range(integrate_steps):
                    self.perform_verlet_step(time_step=time_step)

            elif self.integration_method == "euler_integration":
                for _ in range(integrate_steps):
                    self.perform_euler_step(time_step=time_step)
            else:
                raise Exception(f"Specified integration method: {self.integration_method} - has not been implemented yet")

    

    def barostat(self, target_stress=np.zeros(shape=(2,2)), step_size=1e-5, gamma=1e-3, max_scale=5e-4):
        current_stress = self.get_stress_tensor(step_size=step_size)
        current_scale = (current_stress-target_stress)*gamma
        for i in range(len(current_scale)):
            if current_scale[i,i] > max_scale:
                current_scale[i,i] = max_scale
            if current_scale[i,i] < -max_scale:
                current_scale[i,i] = -max_scale
        self.scale_volume(scale_x=current_scale[0,0], scale_y=current_scale[1,1])

    def thermostat(self):
        M, N = self.get_atom_velocities().shape
        current_vel = np.random.randn(M,N)*np.sqrt(self.kb*self.temp/self.get_atom_masses()[:,None])
        self.set_atom_velocities(current_vel)

    def run_MDsimulation(self):
        raise Exception("NOT IMPLEMENTED YET")


class MDT_simulator(MD_simulator):
    def __init__(self, atom_col, temp, log_level="only_md_steps", integration_method="verlet_integration"):
        super().__init__(atom_col, temp, log_level, integration_method)
    
    def run_MDsimulation(self, N_steps=200, time_step=1e-3):
        if self.log_level == "only_end_result":
            for _ in range(N_steps):
                self.N2_integration(time_step=time_step)
                self.thermostat()
            self.log_atom_col()
        else:
            for _ in range(N_steps):
                self.N2_integration(time_step=time_step)
                self.thermostat()
                self.log_atom_col()

class MDTP_simulator(MD_simulator):
    def __init__(self, atom_col, temp, log_level="only_md_steps", integration_method="verlet_integration"):
        super().__init__(atom_col, temp, log_level, integration_method)

    def run_MDsimulation(self, N_steps=200, target_stress=np.zeros(shape=(2,2)), time_step=1e-3, gamma=1e-4, step_size=1e-5, max_scale=5e-4):
        if self.log_level == "only_end_result":
            for _ in range(N_steps):
                self.N2_integration(time_step=time_step)
                self.thermostat()
                self.barostat(target_stress=target_stress,step_size=step_size, gamma=gamma, max_scale=max_scale)
            self.log_atom_col()
        else:
            for _ in range(N_steps):
                self.N2_integration(time_step=time_step)
                self.thermostat()
                self.barostat(target_stress=target_stress,step_size=step_size, gamma=gamma, max_scale=max_scale)
                self.log_atom_col()

