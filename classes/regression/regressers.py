import numpy as np
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
        return np.dot(self.create_X(x_data), self.beta)#np.dot(self.beta, self.create_X(x_data).T)
    
class Lin_Regression(Regresser):
    def __init__(self, data) -> None:
        super().__init__(data)

    def fit(self):
        t1 = np.linalg.inv(np.dot(self.X.T, self.X))
        t2 = np.dot(self.X.T, self.Y)
        beta = np.dot(t1, t2)
        return beta
                 
    def create_X(self, x_data):
        X = np.ones(shape=(len(x_data), 2))
        X[:,1]*=x_data
        return X
    
    def sample(self, x_data):
        return super().sample(x_data)


class Pol_Regression(Regresser):
    def __init__(self, data, N_deg) -> None:
        self.N_deg = N_deg
        super().__init__(data)

    def fit(self):
        t1 = np.linalg.inv(np.dot(self.X.T, self.X))
        t2 = np.dot(self.X.T, self.Y)
        beta = np.dot(t1, t2)
        return beta
    
    def create_X(self, x_data):
        X = np.ones(shape=(len(x_data), self.N_deg+1))
        for i in range(self.N_deg+1):
            X[:,i]*=x_data**i
        return X
    
    def sample(self, x_data):
        return super().sample(x_data)
    
class Ridge_Regression(Regresser):
    def __init__(self, data, N_deg, lamb=1e-2) -> None:
        self.N_deg = N_deg
        self.lamb = lamb
        super().__init__(data)

    def fit(self):
        t1 = np.linalg.inv(np.dot(self.X.T, self.X) + self.lamb*np.identity(self.N_deg+1))
        t2 = np.dot(self.X.T, self.Y)
        beta = np.dot(t1, t2)
        return beta
    
    def create_X(self, x_data):
        X = np.ones(shape=(len(x_data), self.N_deg+1))
        for i in range(self.N_deg+1):
            X[:,i]*=x_data**i
        return X
    
    def sample(self, x_data):
        return super().sample(x_data)