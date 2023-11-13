import random

import numpy as np
import torch
from prettytable import PrettyTable

import diffuser.datasets as datasets
import diffuser.utils as utils

from contextlib import redirect_stdout
import io

dataset = 'maze2d-medium-v1' #'maze2d-umaze-v1'
horizon = 256
batch_size = 200

results = PrettyTable(['Model', 'Prior', 'Diffusion Steps', 'NFE', 'Training Steps', 'Time/plan', 'Avg Score'])

experiment_configs = [
    ('SBDiffusion',         'Maze2DGoalPriorSteady', 16, 3, 5e5),
    ('SBDiffusion',         'Maze2DGoalPriorSteady', 32, 7, 5e5),
    ('SBDiffusion',         'Maze2DGoalPriorSteady', 64, 63, 5e5),
    ('GaussianDiffusion',   'Maze2DGoalPriorSteady', 64, 63, 5e5)
]

def get_conditions(env, batch_size, horizon):
    cond = {0:[], horizon-1:[]}
    empty_cells = env.reset_locations
    for i in range(batch_size):
        start_qpos = random.choice(empty_cells) # + env.np_random.uniform(low=-.45, high=.45, size=env.model.nq)
        end_qpos = random.choice(empty_cells) # + env.np_random.uniform(low=-.45, high=.45, size=env.model.nq)
        cond[0].append([*start_qpos, 0, 0])
        cond[horizon-1].append([*end_qpos, 0, 0])
    return {0:np.array(cond[0]), horizon-1:np.array(cond[horizon-1])}

class Experiment:
    def __init__(self, diffusion_class, prior_class, n_diffusion_steps, nfe, n_train_steps, epoch='latest'):
        dataset_mod = dataset.replace('-dense-', '-') # load the sparse models irrespective of dense or sparse dataset
        loadpath = f'logs/{dataset_mod}/diffusion/H{horizon}_T{n_diffusion_steps}_N{nfe}_{diffusion_class}_{prior_class}_{n_train_steps}'
        experiment = utils.load_diffusion(loadpath, epoch=epoch)

        self.action_dim = 2
        self.model_name = diffusion_class
        self.prior_name = prior_class

        self.diffusion = experiment.ema
        self.dataset = experiment.dataset

        self.average_normalized_score = 0
        self.average_time_taken = 0

    def __call__(self, conditions):
        normed_conditions = self.normalize_and_tensorize(conditions)

        sample = self.diffusion(normed_conditions)

        sample = utils.to_np(sample)
        observations = self.dataset.normalizer.unnormalize(sample[:, :, self.action_dim:], 'observations')

        self.simulate(observations, conditions)

    def normalize_and_tensorize(self, conditions):
        conditions = utils.apply_dict(
            self.dataset.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        return conditions

    def simulate(self, observations, conditions):
        cumulative_normalized_score = 0
        for b in range(batch_size):

            episode_reward = 0
            env.reset_to_location(conditions[0][b,:2])
            env.set_target(conditions[horizon-1][b,:2])
            print("Target:", env._target)

            for t in range(env.max_episode_steps):
                state = env.state_vector().copy()
                if t < horizon-1:
                    next_waypoint = observations[b,t,:]
                else:
                    next_waypoint = observations[b,-1,:].copy()
                    next_waypoint[2:] = 0

                action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
                next_observation, reward, terminal, _ = env.step(action)
                episode_reward += reward
                score = env.get_normalized_score(episode_reward)
                print(
                    f't: {t} | r: {reward:.2f} |  R: {episode_reward:.2f} | score: {score:.4f} | '
                    f'{action}'
                )
                print(
                    f'maze | actual position: {next_observation[:2]} | predicted position: {next_waypoint[:2]} | goal: {conditions[horizon-1][b,:2]}'
                )
            cumulative_normalized_score += 100*score
        self.average_normalized_score = cumulative_normalized_score/batch_size


env = datasets.load_environment(dataset)
conds = get_conditions(env, batch_size, horizon)

for config in experiment_configs:
    print(f'------------Running benchmark for {config}------------')
    with redirect_stdout(io.StringIO()) as f:
        exp = Experiment(*config)
        exp(conds)
        results.add_row([*config, exp.average_time_taken, exp.average_normalized_score])
    print(f'------------Finished benchmarking for {config}.------------\n')


print(f'Results for dataset {dataset} with horizon {horizon} and batch size {batch_size}:')
print(results.get_string(fields=['Model', 'Diffusion Steps', 'NFE', 'Training Steps', 'Avg Score'], sortby='Avg Score', reversesort=True))



