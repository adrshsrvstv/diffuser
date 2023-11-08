import copy
import numpy as np
from math import atan2, sin, cos
import torch

class BasePrior:
    def __call__(self, cond, horizon, transition_dim, device):
        shape = (len(cond[0]), horizon, transition_dim)
        x = torch.randn(shape, device=device)
        return x


class NaivePrior(BasePrior):
    def __init__(self):
        super().__init__()

    def __call__(self, cond, horizon, transition_dim, device):
        shape = (len(cond[0]), horizon, transition_dim)
        x = torch.randn(shape, device=device)
        return x

    def calc_action(self, start, goal):
        delta_x = goal[0] - start[0]
        delta_y = goal[1] - start[1]
        theta_radians = atan2(delta_y, delta_x)
        a_x = cos(theta_radians)
        a_y = sin(theta_radians)
        return [a_x, a_y]

    def get_naive_prior_dataset(self, dataset):
        dataset_naive = copy.deepcopy(dataset)
        start = 0
        for t in np.where(dataset_naive['timeouts'] == 1)[0]:
            dataset_naive['actions'][start:t + 1] = self.calc_action(dataset_naive['infos/qpos'][start], dataset_naive['infos/goal'][t])
            start = t + 1
        return dataset_naive