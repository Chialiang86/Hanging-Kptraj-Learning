import numpy as np
from scipy.spatial.transform import Rotation as R

class kl_annealing():
    def __init__(self, kl_anneal_cyclical=True, niter=500, start=0.0, stop=1.0, kl_anneal_cycle=10, kl_anneal_ratio=1):
        super().__init__()
        if kl_anneal_cyclical:
            self.L = self.frange_cycle_linear(niter, start, stop, kl_anneal_cycle, kl_anneal_ratio)
        else:
            self.L = self.frange_cycle_linear(niter, start, stop, 1, 0.2)
        self.index = 0

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0, n_cycle=10, ratio=0.5):
        L = np.ones(n_iter) * stop
        period = n_iter / n_cycle
        step = (stop - start) / (period * ratio) # linear schedule
        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i + c * period) < n_iter):
                L[int(i + c * period)] = v
                v += step
                i += 1
        return L 

    def update(self):
        self.index = (self.index + 1) % (len(self.L) - 1)

    def get_beta(self):
        return self.L[self.index]

def get_matrix_from_pose(pose : list or tuple or np.ndarray) -> np.ndarray:
    assert len(pose) == 6 or len(pose) == 7, f'pose must contain 6 or 7 elements, but got {len(pose)}'
    pos_m = np.asarray(pose[:3])
    rot_m = np.identity(3)

    if len(pose) == 6:
        rot_m = R.from_rotvec(pose[3:]).as_matrix()
    elif len(pose) == 7:
        rot_m = R.from_quat(pose[3:]).as_matrix()
            
    ret_m = np.identity(4)
    ret_m[:3, :3] = rot_m
    ret_m[:3, 3] = pos_m

    return ret_m

def get_pose_from_matrix(matrix : list or tuple or np.ndarray, 
                        pose_size : int = 7) -> np.ndarray:

    mat = np.array(matrix)
    assert mat.shape == (4, 4), f'pose must contain 4 x 4 elements, but got {mat.shape}'
    
    pos = matrix[:3, 3]
    rot = None

    if pose_size == 6:
        rot = R.from_matrix(matrix[:3, :3]).as_rotvec()
    elif pose_size == 7:
        rot = R.from_matrix(matrix[:3, :3]).as_quat()
            
    pose = list(pos) + list(rot)

    return np.array(pose)

# kl = kl_annealing(kl_anneal_cyclical=True, niter=500, start=0.0, stop=1.0, kl_anneal_cycle=10, kl_anneal_ratio=1)
# print(kl.L)

a_6 = np.asarray([0, 0, 0, np.pi - 0.01, np.pi - 0.01, np.pi - 0.01])
A = get_matrix_from_pose(a_6)

a_star_6 = np.asarray([0.1, 0.1, 0.1, -np.pi + 0.01, -np.pi + 0.01, -np.pi + 0.01])
A_star = get_matrix_from_pose(a_star_6)

print(a_6)
print(a_star_6)

A_diff = np.identity(4)
A_diff[:3, 3] = A_star[:3, 3] - A[:3, 3]
A_diff[:3, :3] = A_star[:3, :3] @ np.linalg.inv(A[:3, :3])
print(get_pose_from_matrix(A_diff, pose_size=6))

print('=============================')
print(get_pose_from_matrix(A_diff @ A, pose_size=6) - get_pose_from_matrix(A_star, pose_size=6))
