import numpy as np
import copy
from classes.atoms import Atom, Atom_Collection

class Particle_swarm():
    def __init__(self, N_particles, grid_space, c1, c2, w) -> None:
        self.c1 = c1
        self.c2 = c2
        self.w = w
    
    def get_velocities(self):
        pass
    
    def get_positions(self):
        pass
    
    def update_velocities(self):
        pass

    def update_positions(self):
        pass

    def get_global_best(self):
        pass

    def run(self):
        pass

class Particle():
    def __init__(self, x0, v0) -> None:
        self.pos_best = x0
        self.pos = x0
        self.v = v0