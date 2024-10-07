import numpy as np
from classes.atoms import Atom, Atom_Collection
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform


class Descriptor():
    def __init__(self) -> None:
        pass
    
class DistanceMoments():
    def __init__(self, color="C5") -> None:
        self.xwidth = 1.0
        self.color = color
        self.bin_centers = range(2)

    def get_descriptor(self, atom_col):
        pos = atom_col.get_distances()
        return np.array([np.mean(pos), np.sqrt(np.var(pos))])


    def plot(self, atom_col, ax):
        vector = self.get_descriptor(atom_col=atom_col)
        ax.bar(self.bin_centers,vector,width=0.8 * self.xwidth,color=self.color)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.set_ylim([0,2.3])
        xticklabels = ['$\mu$','$\sigma$']
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
        ax.set_title(self.__class__.__name__)

class ExtremeNeighborCount():
    def __init__(self, color="C5") -> None:
        self.xwidth = 1
        self.color = color
        self.bin_centers = range(2)

    def get_descriptor(self, atom_col, r_min=2.0**(1/6.0)*2**(-1/6), A=1.2):
        dist = atom_col.get_distances()
        connectivity_matrix = (squareform(dist) <= A*r_min).astype(int)
        np.fill_diagonal(connectivity_matrix, 0.0)
        nums = np.sum(connectivity_matrix, axis=1)
        return np.array([np.min(nums),np.max(nums)])

    def plot(self, atom_col, ax):
        vector = self.get_descriptor(atom_col=atom_col)
        ax.bar(self.bin_centers,vector,width=0.8 * self.xwidth,color=self.color)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.set_ylim([0,7])
        xticklabels = ['$N_{lowest}$','$N_{highest}$']
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
        ax.set_title(self.__class__.__name__)

class PairDistances():
    def __init__(self, color="C1") -> None:
        self.xwidth = 0.5
        self.color = color
        self.bin_edges = np.arange(0,7.5,self.xwidth)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) /2

    def get_descriptor(self, atom_col):
        distances = atom_col.get_distances()
        desc, _ = np.histogram(distances, self.bin_edges)
        print(self.bin_edges)
        return desc

    def plot(self, atom_col, ax):
        vector = self.get_descriptor(atom_col=atom_col)
        ax.bar(self.bin_centers,vector,width=0.8 * self.xwidth,color=self.color)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.set_title(self.__class__.__name__)

class CoordinationNumbers():
    def __init__(self, color="C2") -> None:
        self.xwidth = 1
        self.color = color
        
    def get_descriptor(self, atom_col, r_min=2.0**(1/6.0)*2**(-1/6), A=1.2):
        dist = atom_col.get_distances()
        connectivity_matrix = (squareform(dist) <= A*r_min).astype(int)
        np.fill_diagonal(connectivity_matrix, 0.0)
        nums = np.sum(connectivity_matrix, axis=1)
        vec, _ = np.histogram(nums, np.arange(0, 9, 1))
        return vec

    def plot(self,atom_col,ax):
        vector = self.get_descriptor(atom_col=atom_col)
        N = len(vector)
        xcenters = np.linspace(0,N-1,N) * self.xwidth
        ax.bar(xcenters,vector,width=0.8 * self.xwidth,color=self.color)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.set_title(self.__class__.__name__)

class ConnectivityGraphSpectrum():
    def __init__(self, color="C3") -> None:
        self.xwidth = 1
        self.color = color

    def get_descriptor(self, atom_col, r_min=2.0**(1/6.0)*2**(-1/6), A=1.2):
        dist = atom_col.get_distances()
        connectivity_matrix = (squareform(dist) <= A*r_min).astype(int)
        np.fill_diagonal(connectivity_matrix, 0)
        #print(connectivity_matrix)
        eigvals, eigvects = np.linalg.eigh(connectivity_matrix)
        return np.sort(eigvals)

    def plot(self,atom_col,ax):
        vector = self.get_descriptor(atom_col=atom_col)
        N = len(vector)
        xcenters = np.linspace(0,N-1,N) * self.xwidth
        ax.bar(xcenters,vector,width=0.8 * self.xwidth,color=self.color)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.set_title(self.__class__.__name__)

class CoulombMatrixSpectrum():
    def __init__(self, color="C4") -> None:
        self.xwidth = 1
        self.color = color

    def get_descriptor(self, atom_col):
        dist = atom_col.get_distances()
        coulomb_matrix = squareform(dist)
        np.fill_diagonal(coulomb_matrix, 1.0)
        coulomb_matrix=1/coulomb_matrix
        eigvals, eigvects = np.linalg.eigh(coulomb_matrix)
        return np.sort(eigvals)

    def plot(self,atom_col,ax):
        vector = self.get_descriptor(atom_col=atom_col)
        N = len(vector)
        xcenters = np.linspace(0,N-1,N) * self.xwidth
        ax.bar(xcenters,vector,width=0.8 * self.xwidth,color=self.color)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.set_ylim([-2,8])
        ax.set_title(self.__class__.__name__)


class Binning_Handler():
    def __init__(self) -> None:
        pass

    def hej(self):
        pass