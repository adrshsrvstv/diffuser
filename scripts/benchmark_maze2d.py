import random
import numpy as np
from statistics import mean, stdev
import torch
from prettytable import PrettyTable

import diffuser.datasets as datasets
import diffuser.utils as utils

from contextlib import redirect_stdout
import io
from experiments import *

results = PrettyTable(['Model', 'Prior', 'Diffusion Steps', 'NFE', 'Config Training Steps', 'Training Steps', 'Time/plan', 'Score'])

experiment =  score_vs_training_steps_across_all_algos_maze2d_medium_NFE_4

dataset = experiment['dataset']
horizon = experiment['horizon']
batch_size = experiment['batch_size']

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
    def __init__(self, diffusion_class, prior_class, n_diffusion_steps, nfe, n_train_steps, epoch):
        dataset_mod = dataset.replace('-dense-', '-') # load the sparse models irrespective of dense or sparse dataset
        prior_class = prior_class if diffusion_class == 'SBDiffusion' else 'BasePrior'
        nfe = nfe if diffusion_class == 'SBDiffusion' else (n_diffusion_steps-1)

        loadpath = f'logs/{dataset_mod}/diffusion/H{horizon}_T{n_diffusion_steps}_N{nfe}_{diffusion_class}_{prior_class}_{n_train_steps}'
        experiment = utils.load_diffusion(loadpath, epoch=epoch)

        self.action_dim = 2
        self.model_name = diffusion_class
        self.prior_name = prior_class

        self.diffusion = experiment.ema
        self.dataset = experiment.dataset

        self.scores = []
        self.times = [0, 0]

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
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda')
        return conditions

    def simulate(self, observations, conditions):
        for b in range(batch_size):
            episode_reward = 0
            env.reset_to_location(conditions[0][b,:2])
            env.set_target(conditions[horizon-1][b,:2])

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
            self.scores.append(100*env.get_normalized_score(episode_reward))


env = datasets.load_environment(dataset)
conds = get_conditions(env, batch_size, horizon)

for config in experiment['configs']:
    print(f'------------Running benchmark for {config}------------')
    with redirect_stdout(io.StringIO()) as f:
        exp = Experiment(*config)
        exp(conds)
        results.add_row([*config, f'{np.array(exp.times).mean():6.2f} ± {np.array(exp.times).std():6.2f}', f'{np.array(exp.scores).mean():6.2f}']) # ± {np.array(exp.scores).std():6.2f}'])
    print(f'------------Finished benchmarking for {config}.------------\n')


print(f'Results for dataset {dataset} with horizon {horizon}, over {batch_size} evaluation episodes:')
print(results.get_string(fields=['Model', 'Prior', 'Diffusion Steps', 'NFE', 'Training Steps', 'Score']))



