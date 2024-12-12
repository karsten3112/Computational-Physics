import numpy as np
from scipy.spatial.distance import pdist

def custom_metric(u, v):
    return v-u

class PBC_handler():
    def __init__(self, unit_cell_vectors) -> None:
        self.v1, self.v2 = unit_cell_vectors
        self.v1_unit = self.get_unit_vector(self.v1)
        self.v2_unit = self.get_unit_vector(self.v2)
        self.volume = self.get_volume()
        self.trans_matrix, self.inv_trans_matrix = self.get_trans_matrices(v1=self.v1_unit, v2=self.v2_unit)
        self.v1_trans, self.v2_trans = self.project_data(self.v1), self.project_data(self.v2)
        self.d1_trans, self.d2_trans = np.linalg.norm(self.v1_trans), np.linalg.norm(self.v2_trans)

    def __deepcopy__(self, memo):
        new_object = PBC_handler(unit_cell_vectors=(self.v1*1.0, self.v2*1.0))
        memo[id(self)] = new_object
        return new_object

    def get_unit_vector(self, v):
        return v/np.linalg.norm(v)
    
    def get_trans_matrices(self, v1, v2):
        inv_mat = np.array([v1, v2]).T
        mat = np.linalg.inv(inv_mat)
        return mat, inv_mat

    def get_volume(self):
        return np.linalg.norm(np.cross(self.v1, self.v2))

    def project_data(self, xy_pos):
        return np.dot(self.trans_matrix, xy_pos.T).T
    
    def project_back(self, proj_pos):
        return np.dot(self.inv_trans_matrix, proj_pos.T).T

    def restrict_positions(self, atom_pos):
        projected_pos = self.project_data(xy_pos=atom_pos)
        for i, d in enumerate([self.d1_trans, self.d2_trans]):
            res_coord_big = (projected_pos[:,i] > d).astype(int)*(-d)
            res_coord_less = (projected_pos[:,i] <= 0.0).astype(int)*(d) #PERHAPS ADD SUCH THAT THE UNITCELL CAN HAVE ARBITRARY ORIGIN
            projected_pos[:,i]+=res_coord_big+res_coord_less
        return self.project_back(projected_pos)
    
    def get_periodic_dist(self, atom_pos):
        x_diffs = pdist(atom_pos[:,0].reshape(-1,1), metric=custom_metric)
        y_diffs = pdist(atom_pos[:,1].reshape(-1,1), metric=custom_metric)
        projed_data = self.project_data(xy_pos=np.array([x_diffs, y_diffs]).T)
        for i, d in enumerate([self.d1_trans, self.d2_trans]):
            res_coord_big = (projed_data[:,i] > d/2.0).astype(int)*(-d)
            res_coord_less = (projed_data[:,i] < -d/2.0).astype(int)*(d)
            projed_data[:,i]+=res_coord_big+res_coord_less
        
        diffs = self.project_back(projed_data)
        return np.linalg.norm(diffs, axis=1)
    
    def get_periodic_dist_vector(self, atom_pos):
        diff = atom_pos[np.newaxis, :, :] - atom_pos[:, np.newaxis, :]
        projed_data = self.project_data(diff.reshape(len(atom_pos)**2, 2))
        for i, d in enumerate([self.d1_trans, self.d2_trans]):
            res_coord_big = (projed_data[:,i] > d/2.0).astype(int)*(-d)
            res_coord_less = (projed_data[:,i] < -d/2.0).astype(int)*(d)
            projed_data[:,i]+=res_coord_big+res_coord_less
        diffs_final = self.project_back(proj_pos=projed_data)
        return diffs_final.reshape(len(atom_pos), len(atom_pos), 2)

    def update_params(self, new_unit_cell):
        self.v1, self.v2 = new_unit_cell
        self.v1_unit = self.get_unit_vector(self.v1)
        self.v2_unit = self.get_unit_vector(self.v2)
        self.volume = self.get_volume()
        self.trans_matrix, self.inv_trans_matrix = self.get_trans_matrices(v1=self.v1_unit, v2=self.v2_unit)
        self.v1_trans, self.v2_trans = self.project_data(self.v1), self.project_data(self.v2)
        self.d1_trans, self.d2_trans = np.linalg.norm(self.v1_trans), np.linalg.norm(self.v2_trans)

    def scale_cell(self, scale_x=1.0, scale_y=1.0):
        scale_mat = np.array([[1.0+scale_x, 0.0], [0.0, 1.0+scale_y]])
        self.v1 = np.dot(scale_mat, self.v1.T)
        self.v2 = np.dot(scale_mat, self.v2.T)
        self.update_params(new_unit_cell=(self.v1, self.v2))

    def scale_coords(self, coords, scale_x=1.0, scale_y=1.0):
        scale_mat = np.array([[1.0+scale_x, 0.0], [0.0, 1.0+scale_y]])
        scaled_coord = np.dot(scale_mat, coords.T)
        return scaled_coord.T

    def scale_cell_and_coords(self, coords, scale_x=1.0, scale_y=1.0):
        self.scale_cell(scale_x=scale_x, scale_y=scale_y)
        return self.scale_coords(coords=coords, scale_x=scale_x, scale_y=scale_y)