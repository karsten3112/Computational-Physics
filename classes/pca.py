import numpy as np

class PCA():
    def __init__(self, n_components) -> None:
        self.n_components = n_components
        self.C = None
        self.X_center = None
        self.eigvals = None
        self.Q = None
        self.mu_mat = None

    def construct_X_centered(self, data):
        #M,p = data.shape
        self.mu_mat = np.mean(data, axis=0)
        X_centered = data-self.mu_mat
        return X_centered

    def fit(self, data):
        X_center = self.construct_X_centered(data=data)
        if self.C == None:
            self.C = 1.0/(len(X_center)-1)*np.matmul(np.transpose(X_center), X_center)
        else:
            pass
        if self.Q == None:
            eigvals, Q = np.linalg.eig(self.C)
            eigvals_order = np.argsort(eigvals)[::-1]
            self.eigvals = eigvals[eigvals_order]
            Q = Q[:,eigvals_order]
            self.Q = Q[:,:self.n_components]
        else:
            pass

    def transform(self, data):
        X_center = data - self.mu_mat
        return np.dot(X_center, self.Q)