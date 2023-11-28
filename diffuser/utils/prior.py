import numpy as np
import torch
import torch.nn as nn
import math
import einops
import diffuser.utils as utils


class BasePrior:
    def __init__(self, action_dim, observation_dim, horizon):
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.horizon = horizon
        self.transition_dim = observation_dim + action_dim
        self.device = None

    def __call__(self, cond):
        batch_size = len(cond[0])
        shape = (batch_size, self.horizon, self.transition_dim)
        x = torch.randn(shape, device=self.device, dtype=torch.float32)
        return x

    def to(self, device):
        self.device = device
        return self


class StraightLinePrior(BasePrior):
    def __call__(self, cond):
        batch_size = len(cond[0])
        shape = (batch_size, self.horizon, self.transition_dim)
        start, goal = utils.to_np(cond[0]), utils.to_np(cond[self.horizon - 1])

        qpos = self.get_qpos(start, goal)
        actions, action_normed = self.get_action(start, goal)
        qvel = self.get_qvel(start, goal, action_normed)
        trajectory = np.concatenate((actions, qpos, qvel), axis=2)
        x1 = torch.tensor(trajectory, device=self.device, dtype=torch.float32)
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
        delta = (goal - start) / (self.horizon - 1)
        return np.array([[s[:2] + i * d[:2] for i in range(0, self.horizon)] for d, s in zip(delta, start)],
                        dtype=np.float32)

    def get_qvel(self, start, goal, action_normed):
        mag = 100 * (goal - start) / (self.horizon - 1)
        qvel = mag[:, :2] * action_normed
        qvels = np.repeat(qvel[:, np.newaxis, :], self.horizon, axis=1)
        return qvels


class MLPTrajectoryPrior(nn.Module):
    def __init__(self, action_dim, observation_dim, horizon):
        super().__init__()
        self.register_buffer('transition_dim', torch.tensor(observation_dim + action_dim))
        self.register_buffer('horizon', torch.tensor(horizon))
        self.mlp = nn.Sequential(
            nn.Linear(2 * observation_dim, horizon * self.transition_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(horizon * self.transition_dim, horizon * self.transition_dim),
        )

    def forward(self, cond):
        out = self.mlp(torch.cat((cond[0], cond[self.horizon.item() - 1]), 1))
        return torch.reshape(out, (-1, self.horizon.item(), self.transition_dim.item()))


n_times_two = (4, 2, 1)
n_plus_one = (4, 1, 1)
n_minus_one = (4, 1, 2)
n_plus_three = (4, 1, 0)
n_plus_one_times_two = (4, 2, 0)


class UpsampleBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, upsample_args=n_times_two, kernel_size=5):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(1, out_channels),
            nn.Mish(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(1, out_channels),
            nn.Mish(),
            nn.ConvTranspose1d(out_channels, out_channels, *upsample_args)
        )

    def forward(self, x):
        return self.block(x)

class LearnedTrajectoryPrior(nn.Module):
    def __init__(self, action_dim, observation_dim, horizon):
        super().__init__()
        self.register_buffer('transition_dim', torch.tensor(observation_dim + action_dim))
        self.register_buffer('horizon', torch.tensor(horizon))
        self.blocks = nn.ModuleList([])

        upsample_args, residual_horizon = self.get_starting_blocks(horizon)
        self.blocks.extend([UpsampleBlock(self.transition_dim, self.transition_dim, arg) for arg in upsample_args])

        number_of_2X_blocks = math.log2(residual_horizon)
        assert number_of_2X_blocks == number_of_2X_blocks // 1

        self.blocks.extend([UpsampleBlock(self.transition_dim, self.transition_dim, n_times_two) for i in
                            range(int(number_of_2X_blocks))])

    def get_starting_blocks(self, horizon):
        if horizon % 15 == 0:
            start_blocks, residual_horizon = [n_plus_three, n_plus_one_times_two, n_plus_three], horizon / 15
        elif horizon % 5 == 0:
            start_blocks, residual_horizon = [n_plus_three], horizon / 5
        elif horizon % 3 == 0:
            start_blocks, residual_horizon = [n_plus_one], horizon / 3
        else:
            start_blocks, residual_horizon = [], horizon / 2
        return start_blocks, residual_horizon

    def forward(self, cond):
        device = cond[0].device
        actions = torch.zeros(cond[0].shape[0], 2, device=device)
        trajectory_start = torch.cat((actions, cond[0]), 1)
        trajectory_end = torch.cat((actions, cond[self.horizon.item() - 1]), 1)

        trajectory_start = einops.rearrange(trajectory_start, 'b t -> b 1 t')
        trajectory_end = einops.rearrange(trajectory_end, 'b t -> b 1 t')

        x = torch.cat((trajectory_start, trajectory_end), 1)

        x = einops.rearrange(x, 'b h t -> b t h')

        for block in self.blocks:
            x = block(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x
