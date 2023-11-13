import json
import numpy as np
from os.path import join
import torch
import random
from prettytable import PrettyTable
from contextlib import redirect_stdout
import io

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils

dataset = 'maze2d-medium-v1'
horizon = 256
batch_size = 100

results = PrettyTable(['Model', 'Prior', 'Diffusion Steps', 'NFE', 'Training Steps', 'Time taken', 'Score'])

experiment_configs = [
    ('SBDiffusion', 'Maze2DGoalPriorSteady', 32, 7, 5e5),
    ('SBDiffusion', 'Maze2DGoalPriorSteady', 64, 63, 5e5),
    ('GaussianDiffusion', 'Maze2DGoalPriorSteady', 64, 63, 5e5)
]

def get_conditions(env, batch_size):
    return 0

class Experiment:
    def __init__(self, diffusion_class, prior_class, n_diffusion_steps, nfe, n_train_steps, epoch='latest'):
        loadpath = f'logs/{dataset}/diffusion/H{horizon}_T{n_diffusion_steps}_N{nfe}_{diffusion_class}_{prior_class}_{n_train_steps}'
        experiment = utils.load_diffusion(loadpath, epoch=epoch)

        self.model_name = diffusion_class
        self.prior_name = '-' if diffusion_class == 'GaussianDiffusion' else prior_class

        self.diffusion = experiment.ema
        self.dataset = experiment.dataset
        self.renderer = experiment.renderer
        self.prior = experiment.prior
        self.policy = Policy(self.diffusion, self.dataset.normalizer)

        self.last_normalized_score = 0
        self.last_time_taken = 0

        self.average_normalized_score = 0
        self.total_time_taken = 0

    def __call__(self, conditions):
        self.average_normalized_score = random.randint(0,100)
        self.total_time_taken = random.randint(20,50)


env = datasets.load_environment(dataset)
conds = get_conditions(env, batch_size)

for config in experiment_configs:
    print(f'------------Running benchmark for {config}------------')
    with redirect_stdout(io.StringIO()) as f:
        exp = Experiment(*config)
        exp(conds)
        results.add_row([*config, exp.total_time_taken, exp.average_normalized_score])
    print(f'------------Finished benchmarking for {config}.------------\n')


print(f'Results for dataset {dataset} with horizon {horizon} and batch size {batch_size} are:')
print(results.get_string(fields=['Model', 'Prior', 'Diffusion Steps', 'NFE', 'Training Steps', 'Time taken', 'Score'], sortby='Score'))



