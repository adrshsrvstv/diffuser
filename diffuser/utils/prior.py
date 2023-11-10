import numpy as np
import scipy.stats
import torch
from scipy.stats import norm

class BasePrior:
    def __init__(self, transition_dim, horizon):
        super().__init__()
        self.transition_dim = transition_dim
        self.horizon = horizon

    def __call__(self, cond, device):
        batch_size = len(cond[0])
        shape = (batch_size, self.horizon, self.transition_dim)
        x = torch.randn(shape, device=device, dtype=torch.float32)
        return x


class Maze2DGoalPrior(BasePrior):
    def __init__(self, transition_dim, horizon):
        super().__init__(transition_dim, horizon)
        nd = np.array([norm.pdf(x, (horizon + 1)/2, horizon/6) for x in np.linspace(1, horizon, horizon)]).reshape((horizon, 1))
        self.normal_distribution = (nd - nd.min()) / nd.max()

    def __call__(self, cond, device):
        batch_size = len(cond[0])
        shape = (batch_size, self.horizon, self.transition_dim)
        start, goal = cond[0].detach().cpu().numpy(), cond[self.horizon-1].detach().cpu().numpy()

        actions, _ = self.get_action(start, goal)
        qpos = self.get_qpos(start, goal)
        qvel = self.get_qvel(start, goal)
        trajectory = np.concatenate((actions, qpos, qvel), axis=2)
        x1 = torch.tensor(trajectory, device=device, dtype=torch.float32)
        assert x1.shape == shape
        return x1

    def get_action(self, start, goal):
        delta = goal - start
        slope_rads = np.arctan2(delta[:, 1], delta[:, 0])
        action = np.column_stack((np.cos(slope_rads), np.sin(slope_rads)))
        action_normed = action / np.abs(action).max(axis=1)[:, np.newaxis]
        actions = np.repeat(action_normed[:, np.newaxis, :], self.horizon, axis=1)
        return actions, action_normed

    def get_qpos(self, start, goal):
        delta = (goal - start)/(self.horizon-1)
        return np.array([[s[:2] + i*d[:2] for i in range(0,self.horizon)] for d, s in zip(delta,start)], dtype=np.float32)

    def get_qvel(self, start, goal):
        mag = 100 * (goal - start) / (self.horizon - 1)
        _, action_normed = self.get_action(start, goal)
        qvel = mag[:, :2] * action_normed
        qvels = np.repeat(qvel[:, np.newaxis, :], self.horizon, axis=1)
        qvels = np.array([qvels[i]*self.normal_distribution for i in range(0,len(qvels))])
        return qvels

