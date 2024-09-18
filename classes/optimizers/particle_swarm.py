import numpy as np
import copy
from classes.atoms import Atom, Atom_Collection

class Particle_swarm():
    def __init__(self, particles, c1, c2, w) -> None:
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.particles = particles
        self.global_best_pos = None
        self.global_best_energy = None
        self.global_best_particle = None

    def get_velocities(self):
        pass
    
    def get_positions(self):
        pass
    
    def update_velocities(self):
        pass

    def update_positions(self):
        pass

    def find_global_best(self):
        energies = [particle.best_energy for particle in self.particles]
        index = np.argmin(energies)
        self.global_best = self.particles[index].best_pos
        self.global_best_energy = self.particles[index].best_energy
        self.global_best_particle = self.particles[index]


    def run(self, N_max, E_crit):
        i = 0
        self.find_global_best()
        while i < N_max and self.global_best_energy > E_crit: 
            for particle in self.particles:
                particle.update_velocity(self.global_best, self.w, self.c1, self.c2)
                particle.update_pos()
                particle.update_best_pos()
            self.find_global_best()
            i+=1
        return self.global_best_particle
        

class Particle():
    def __init__(self, atom_col) -> None:
        self.best_pos = atom_col.get_positions()
        self.best_energy = atom_col.get_potential_energy()
        self.pos = atom_col.get_positions()
        self.v = 0.0
        self.atom_col = atom_col

    def update_best_pos(self):
        new_energy = self.atom_col.get_potential_energy()
        if self.best_energy > new_energy:
            self.best_energy = new_energy
            self.best_pos = self.pos

    def update_velocity(self, global_best, w, c1, c2):
        r1 = np.random.rand(1)
        r2 = np.random.rand(1)
        new_velocity = w*self.v + c1*r1*(self.best_pos - self.pos) + c2*r2*(global_best - self.pos)
        self.v = new_velocity

    def update_pos(self):
        self.atom_col.move_atoms(self.v)
        self.pos = self.atom_col.get_positions()

    def get_energy(self):
        return self.atom_col.get_potential_energy()
    