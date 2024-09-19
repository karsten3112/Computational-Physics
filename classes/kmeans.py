import numpy as np
np.random.seed(1)
class KMeans():
    def __init__(self, n_clusters, max_iter=100, tol=1e-4) -> None:
        self.n_clusters = n_clusters
        self.labels = np.arange(0, n_clusters, 1).astype(int)
        self.max_iter = max_iter
        self.tol = tol
        self.mus = None

    def create_c_dict(self):
        c = {}
        for label in self.labels:
            c[label] = []
        return c
    
    def fit(self, data, init_mus=None):
        M,p = data.shape
        i = 0
        if self.mus == None:
            if type(init_mus) == None:
                self.mus = np.random.rand(self.n_clusters, p) + 6.0
            else:
                self.mus = init_mus
            mus_conv = 100
            while i < self.max_iter and mus_conv > self.tol:
                cs = self.create_c_dict()
                for dat in data:
                    dists = np.linalg.norm(dat-self.mus, axis=1)**2
                    cs[np.argmin(dists)].append(dat)
                new_mus = np.zeros(shape=self.mus.shape)
                for label in self.labels:
                    new_mus[label] = np.mean(cs[label], axis=0)
                mus_conv = np.linalg.norm(self.mus-new_mus)
                self.mus = new_mus
                i+=1
        else:
            pass


    def predict(self, data):
        labels = np.zeros(len(data)).astype(int)
        for i, dat in enumerate(data):
            dists = np.linalg.norm(dat-self.mus, axis=1)**2
            labels[i] = np.argmin(dists)
        return labels