import os
import pickle
import numpy as np
from scipy.spatial.distance import pdist
import copy

class Atom():
    def __init__(self, position=np.zeros(2), velocity=np.zeros(2), mass=1.0, frozen=False) -> None:
        self.pos = position
        self.plot_elem = None
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
            self.plot_elem = ax.plot(self.pos[0], self.pos[1], "o", color=self.color, ms=self.size, markeredgecolor="k", alpha=self.alpha_color)[0]
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
    
    def move(self, r):
        if self.frozen == True:
            pass
        else:
            self.pos+=r