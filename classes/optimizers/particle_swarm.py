import numpy as np
import copy
from classes.optimizers.optimizer import Optimizer1
from classes.atoms import Atom, Atom_Collection

class Particle_swarm():
    def __init__(self, rattled_atom_cols, c1=1.5, c2=2.0, w=0.8, gamma=0.1):
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.particles = [Particle(atom_col=rattled_col, gamma=gamma) for rattled_col in rattled_atom_cols]
        self.global_best_pos = None
        self.global_best_energy = None
        self.global_best_particle = None
    
    def find_global_best(self):
        energies = [particle.best_energy for particle in self.particles]
        index = np.argmin(energies)
        if self.global_best_energy is None:
            self.global_best_pos = self.particles[index].best_pos
            self.global_best_energy = self.particles[index].best_energy
            self.global_best_particle = self.particles[index]
    
        elif energies[index] < self.global_best_energy:
            self.global_best_pos = self.particles[index].best_pos
            self.global_best_energy = self.particles[index].best_energy
            self.global_best_particle = self.particles[index]

    def run(self, N_max, E_crit, track=False):
        i = 0
        self.find_global_best()
        while i < N_max and self.global_best_energy > E_crit: 
            for particle in self.particles:
                particle.update_velocity(self.global_best_pos, self.w, self.c1, self.c2)
                particle.update_pos()
                particle.update_best_pos()
                if track == True:
                    particle.log_atom_col()
            self.find_global_best()
            i+=1
        

class Particle(Optimizer1):
    def __init__(self, atom_col, gamma=0.1):
        super().__init__(atom_col)
        self.best_pos = self.get_atom_positions()
        self.best_energy = self.get_potential_energy()
        self.pos = self.best_pos*1.0
        self.v = gamma*np.random.randn(self.atom_col.N_atoms, 2)
        self.logged_velocities = [self.v]

    def update_best_pos(self):
        new_energy = self.get_potential_energy()
        if new_energy < self.best_energy:
            self.best_energy = new_energy
            self.best_pos = self.pos*1.0

    def update_velocity(self, global_best_pos, w, c1, c2):
        r1 = np.random.rand(1)
        r2 = np.random.rand(1)
        new_velocity = w*self.v + c1*r1*(self.best_pos - self.pos) + c2*r2*(global_best_pos - self.pos)
        self.v = new_velocity
        self.logged_velocities.append(self.v*1.0)

    def update_pos(self, delta_t=0.8):
        self.move_atoms(self.v*delta_t)
        self.pos = self.get_atom_positions()*1.0

    