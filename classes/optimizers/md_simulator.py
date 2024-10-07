import numpy as np
import copy
from classes.atoms import Atom, Atom_Collection

class MD_Simulator():
    kb = 1.0
    def __init__(self) -> None:
        pass

    def verlet_integrate(self, atom_col, time_step=0.01):
        vs = atom_col.get_velocities()
        acc = atom_col.get_forces()/atom_col.get_masses()[:,None]
        x_new = vs*time_step + 1.0/2.0*acc*time_step**2
        atom_col.move_atoms(x_new)

        acc_new = atom_col.get_forces()/atom_col.get_masses()[:,None]
        v_new = 1.0/2.0*(acc + acc_new)*time_step
        atom_col.boost_velocities(v_new)
        return atom_col
    
    def euler_integrate(self, atom_col, time_step):
        vs = atom_col.get_velocities()
        acc = atom_col.get_forces()/atom_col.get_masses()[:,None]
        v_new = acc*time_step
        x_new = vs*time_step
        atom_col.move_atoms(x_new)
        atom_col.boost_velocities(v_new)
        return atom_col

    def run_N2_integration(self, atom_col, N_steps, t_init=0.0,time_step=0.01, track=True, method="verlet_integrate"):
        t = t_init
        result = []
        for i in range(N_steps):
            atom_col_copy = copy.deepcopy(atom_col)
            #print(atom_col_copy)
            if method == "verlet_integrate":
                atom_col = self.verlet_integrate(atom_col=atom_col_copy, time_step=time_step)
            if method == "euler_integrate":
                atom_col = self.euler_integrate(atom_col=atom_col, time_step=time_step)
            if track == True:
                result.append(atom_col_copy)
            t+=time_step
        if track == True:
            return result
        else:
            return [atom_col]


    def run_MD_simulation(self, atom_col, temp, t_init=0.0, N_steps=1000, time_step=0.01, track=True, method="verlet_integrate", integrate_steps=50):
        t = t_init
        atom_cols_result = []
        for i in range(N_steps):
            atom_col_copy = copy.deepcopy(atom_col)
            M, N = atom_col_copy.velocities.shape
            v_init = np.random.randn(M,N)*np.sqrt(self.kb*temp/atom_col_copy.get_masses()[:,None])
            atom_col_copy.set_velocities(v_init)
            atom_cols = self.run_N2_integration(atom_col=atom_col_copy, N_steps=integrate_steps, t_init=t, time_step=time_step, track=track, method=method)
            atom_col = atom_cols[-1]
            if track == True:
                atom_cols_result+=atom_cols
        if track == True:
            return atom_cols_result
        else:
            return [atom_col]


class MD_simulator_new1():
    kb = 1.0
    def __init__(self, atom_col, temp) -> None:
        self.calculator = atom_col.calculator
        self.masses = atom_col.get_masses()
        self.frozens = (atom_col.get_frozens() == False).astype(int)
        self.init_pos = atom_col.get_positions()
        self.init_velocities = atom_col.get_velocities()
        self.pbc_handler = atom_col.pbc_handler
        self.pbc = atom_col.pbc
        self.temp = temp

    def verlet_integrate(self, positions, velocities, time_step):
        acc = self.calculator.forces(positions)/self.masses[:,None]
        if self.pbc == True:
            new_poses = self.pbc_handler.restrict_positions(positions + (velocities*time_step + 1.0/2.0*acc*time_step**2)*self.frozens[:,None])
        else:
            new_poses = positions + (velocities*time_step + 1.0/2.0*acc*time_step**2)*self.frozens[:,None]
        
        acc_new = self.calculator.forces(new_poses)/self.masses[:,None]
        new_vels = velocities + (1.0/2.0*(acc + acc_new)*time_step)*self.frozens[:,None]
    
        return new_poses, new_vels

    def euler_integrate(self, positions, velocities, time_step):
        pass

    def run_N2_integration(self, N_steps, time_step=0.01, t_init=0.0, track=True, method="verlet_integration"):
        current_pos = self.init_pos*1.0
        current_vel = self.init_velocities*1.0
        time = t_init
        result_poses = [current_pos]
        result_vels = [current_vel]
        ts = [time]
        for i in range(N_steps):
            if method == "verlet_integration":
                current_pos, current_vel = self.verlet_integrate(positions=current_pos, velocities=current_vel, time_step=time_step)
            if method == "euler_integration":
                pass
            time+=time_step
            if track == True:
                result_poses.append(current_pos)
                result_vels.append(current_vel)
                ts.append(time)
            else:
                result_poses = [current_pos]
                result_vels = [current_vel]
                ts = [time]
        return np.array(ts), np.array(result_poses), np.array(result_vels)

    def run_MD_simulation(self, t_init=0.0, N_steps=1000, time_step=0.01, track_steps=True, method="verlet_integration", integrate_steps=50):
        current_pos = self.init_pos*1.0
        time = t_init
        result_poses = [current_pos]
        result_vels = [self.init_velocities]
        ts = [time]
        for i in range(N_steps):
            current_vel = np.random.randn(len(current_pos),2)*np.sqrt(self.kb*self.temp/self.masses[:,None])
            for j in range(integrate_steps):
                if method == "verlet_integration":
                    current_pos, current_vel = self.verlet_integrate(positions=current_pos, velocities=current_vel, time_step=time_step)
                if method == "euler_integration":
                    pass
                time+=time_step
                if track_steps == True:
                    result_poses.append(current_pos)
                    result_vels.append(current_vel)
                    ts.append(time)
                else:
                    result_poses = [current_pos]
                    result_vels = [current_vel]
                    ts = [time]
        return np.array(ts), np.array(result_poses), np.array(result_vels)
    
    def convert_to_atom_cols(self, atom_col_for_copy, positions, velocities):
        pass
            
class MD_simulator_new():
    kb = 1.0
    def __init__(self, atom_col) -> None:
        self.atom_col = atom_col
        self.calculator = atom_col.calculator
        self.masses = atom_col.get_masses()
        self.frozens = atom_col.get_frozens()
        self.atoms_for_move = (self.frozens == False).astype(int)
        self.init_pos = atom_col.get_positions()
        self.init_velocities = atom_col.get_velocities()

    def verlet_integrate(self, positions, velocities, time_step=0.01):
        acc = self.calculator.forces(positions)/self.masses[:,None]
        new_poses = positions + (velocities*time_step + 1.0/2.0*acc*time_step**2)*self.atoms_for_move[:,None]
        acc_new = self.calculator.forces(new_poses)/self.masses[:,None]
        new_vels = velocities + (1.0/2.0*(acc + acc_new)*time_step)*self.atoms_for_move[:,None]
        return new_poses, new_vels
    
    def euler_integrate(self):
        pass
    
    def run_N2_integration(self, N_steps, pos_init=None, vels_init=None, t_init=0.0,time_step=0.01, track=True, method="verlet_integrate", return_type="pos_and_vels"):
        if type(pos_init) == None:
            positions = self.init_pos*1.0
        else:
            positions = pos_init
        if type(vels_init) == None:
            velocities = self.init_velocities*1.0
        else:
            velocities = vels_init
        
        t = t_init
        final_poses = []
        final_vels = []
        
        for i in range(N_steps):
            if method == "verlet_integrate":
                positions, velocities = self.verlet_integrate(positions=positions, velocities=velocities, time_step=time_step)
            if method == "euler_integrate":
                pass
            t+=time_step
            if track==True:
                final_poses.append(positions)
                final_vels.append(velocities)
            else:
                final_poses = [positions]
                final_vels = [velocities]
        
        if return_type == "pos_and_vels":
            return final_poses, final_vels
        if return_type == "atom_cols":
            atom_col_res = []
            for pos, vel in zip(final_poses, final_vels):
                atom_col = copy.deepcopy(self.atom_col)
                atom_col.set_positions(pos)
                atom_col.set_velocities(vel)
                atom_col_res.append(atom_col)
            return atom_col_res
        
    def run_MD_simulation(self, temp, pos_init=None, vels_init=None, t_init=0.0, N_steps=1000, time_step=0.01, track_steps=True, track_integration=True, method="verlet_integrate", integrate_steps=50, return_type="atom_cols"):
        try:
            if pos_init == None:
                positions = self.init_pos*1.0 #- np.sum(self.init_pos*self.masses[:,None], axis=0)/np.sum(self.masses)
        except:
            positions = pos_init
        try:
            if vels_init == None:
                velocities = self.init_velocities*1.0
        except:
            velocities = vels_init

        M, N = self.init_velocities.shape
        t = t_init
        velocities_res_run = []
        positions_res_run = []
        #print(velocities, positions)
        for i in range(N_steps):
            #R = np.sum(self.init_pos*self.masses[:,None], axis=0)/np.sum(self.masses)
            #positions-=R
            velocities = np.random.randn(M,N)*np.sqrt(self.kb*temp/self.masses[:,None])
            #velocities-=np.sum(velocities*self.masses[:,None], axis=0)/np.sum(self.masses)
            pos_from_int, vel_from_int = self.run_N2_integration(N_steps=integrate_steps, pos_init=positions, vels_init=velocities, t_init=t_init, time_step=time_step, track=track_integration, method=method, return_type="pos_and_vels")
            #print(pos_from_int)
            if track_steps == True:
                velocities_res_run.append(vel_from_int)
                positions_res_run.append(pos_from_int)
            if track_steps == False:
                velocities_res=vel_from_int
                positions_res=pos_from_int
            t+=time_step
            positions = pos_from_int[-1]
        if return_type == "pos_and_vels":
            return positions_res, velocities_res
        #print(positions_res[1])
        if return_type == "atom_cols":
            atom_col_res = []
            for pos_run, vel_run in zip(positions_res_run, velocities_res_run):
                for poses, velocities in zip(pos_run, vel_run):
                    atom_col = copy.deepcopy(self.atom_col)
                    atom_col.set_positions(poses)
                    atom_col.set_velocities(velocities)
                    atom_col_res.append(atom_col)
            return atom_col_res