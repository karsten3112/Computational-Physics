import numpy as np
from scipy.spatial.distance import pdist
import copy


class Regresser():
    def __init__(self,data) -> None:
        self.Y = data[:,1]
        self.X = self.create_X(data[:,0])
        #print(self.X)
        self.beta = self.fit()

    def create_X(self, x_data):
        return
    
    def fit(self) -> None:
        return

    def sample(self, x_data):
        return np.dot(self.create_X(x_data), self.beta)

def pol_kernel_func(x_basis, x_data, N_deg):
    try:
        M,N = x_data.shape
        mat = np.dot(x_data, x_basis.T)
        return (mat+1.0)**N_deg
    except:
        mat = np.outer(x_data, x_basis.T)
        return (mat+1.0)**N_deg


def lin_kernel_func(x_basis, x_data, N_deg):
    try:
        M,N = x_data.shape
        mat = np.dot(x_data, x_basis.T)
        return (mat)**N_deg
    except:
        mat = np.outer(x_data, x_basis.T)
        return (mat+1.0)**N_deg

def RBF_kernel_func(x_basis, x_data, sigma):
    try:
        M, _ = x_data.shape
        _, N = x_basis.shape
        mat = np.zeros(shape=(N+1,M))
        for i, x in enumerate(x_basis):
            dists = np.linalg.norm(x - x_data, axis=1)**2
            #print(dists)
            mat[i] = dists
        return np.exp(-mat/(2.0*sigma**2))
    except:
        M,_ = x_basis.shape
        N,_ = x_data.shape
        mat = np.zeros(shape=(M,N))
        for i, x in enumerate(x_basis):
            dists = abs(x-x_data.T)**2
            mat[i] = dists
        res_mat = mat.T
        return np.exp(-res_mat/(2.0*sigma**2))

def gaussian_kernel_func(x_basis, x_data, sigma=0.1):
    N, _ = x_basis.shape
    M, _ = x_data.shape
    mat = np.zeros(shape=(M,N))
    for i, x_base in enumerate(x_basis):
        l = np.linalg.norm(x_data-x_base, axis=1)**2
        mat[:,i] = np.exp(-l/(2.0*sigma**2))
    return mat

class Kernel_Ridge_Regression():
    def __init__(self, x_data, y_data, kernel_func, lamb) -> None:
        self.X = self.create_X(x_data=x_data)
        self.kernel_func = kernel_func
        self.lamb = lamb
        self.K = self.create_K(x_data=x_data)
        self.alpha = self.fit(y_data=y_data)

    def create_X(self, x_data):
        return x_data
    
    def create_K(self, x_data):
        x_trans = self.create_X(x_data=x_data)
        return self.kernel_func(self.X, x_trans)
    
    def fit(self, y_data):
        t1 = np.linalg.inv(self.K + self.lamb*np.identity(len(y_data)))
        return np.dot(t1, y_data)
    
    def sample(self, x_data):
        return np.dot(self.create_K(x_data=x_data), self.alpha)
    
class pol_kernel_regressor(Kernel_Ridge_Regression):
    def __init__(self, x_data, y_data, N_deg, lamb) -> None:
        kernel_func = lambda x_basis, x_data: pol_kernel_func(x_basis=x_basis, x_data=x_data, N_deg=N_deg)
        super().__init__(x_data, y_data, kernel_func, lamb)

class Gaussian_kernel_regressor(Kernel_Ridge_Regression):
    def __init__(self, x_data, y_data, lamb, sigma) -> None:
        kernel_func = lambda x_basis, x_data: gaussian_kernel_func(x_basis=x_basis, x_data=x_data, sigma=sigma)
        super().__init__(x_data, y_data, kernel_func, lamb)

class RBF_kernel_regressor(Kernel_Ridge_Regression):
    def __init__(self, x_data, y_data, lamb, sigma) -> None:
        kernel_func = lambda x_basis, x_data: RBF_kernel_func(x_basis=x_basis, x_data=x_data, sigma=sigma)
        super().__init__(x_data, y_data, kernel_func, lamb)

class Ridge_Regression(Regresser):
    def __init__(self, data, N_degs, lamb=1e-2) -> None:
        if type(N_degs) == int:
            self.N_degs = np.arange(0, N_degs+1, 1).astype(int)
        if type(N_degs) == list:
            self.N_degs = N_degs
        self.lamb = lamb
        super().__init__(data)

    def fit(self):
        t1 = np.linalg.inv(np.dot(self.X.T, self.X) + self.lamb*np.identity(len(self.N_degs)))
        t2 = np.dot(self.X.T, self.Y)
        beta = np.dot(t1, t2)
        return beta
    
    def create_X(self, x_data):
        X = np.ones(shape=(len(x_data), len(self.N_degs)))
        for i in range(len(self.N_degs)):
                X[:,i]*=x_data**self.N_degs[i]
        return X
    
    def sample(self, x_data):
        return super().sample(x_data)

class Pol_Regression(Ridge_Regression):
    def __init__(self, data, N_degs) -> None:
        super().__init__(data, N_degs, lamb=0.0)
    
class Lin_Regression(Pol_Regression):
    def __init__(self, data) -> None:
        super().__init__(data, N_degs=1)

