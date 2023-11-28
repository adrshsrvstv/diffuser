import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

diffusion_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('nfe', 'N'),
    ('diffusion', ''),
    ('prior', ''),
    ('n_train_steps', ''),
]


plan_args_to_watch = [
    ('prefix', ''),
    ##
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('nfe', 'N'),
    ('diffusion', ''),
    ('prior', ''),
    ('n_train_steps', ''),
    # ('value_horizon', 'V'),
    # ('discount', 'd'),
    # ('normalizer', ''),
    # ('batch_size', 'b'),
    # ##
    # ('conditional', 'cond'),
]

base = {

    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'GaussianDiffusion',
        # 'diffusion': 'SBDiffusion',
        'prior': 'BasePrior',
        'nfe': 63,
        'horizon': 256,
        'n_diffusion_steps': 256,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 4, 8),
        'renderer': 'utils.Maze2dRenderer',

        ## dataset
        'loader': 'datasets.GoalDataset',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': ['maze2d_set_terminals'],
        'clip_denoised': True,
        'use_padding': False,
        'max_path_length': 40000,

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 25e4,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 1000,
        'n_saves': 50,
        'save_parallel': False,
        'n_reference': 50,
        'n_samples': 10,
        'bucket': None,
        'device': 'cuda',
    },

    'plan': {
        'diffusion': 'GaussianDiffusion',
        # 'diffusion': 'SBDiffusion',
        'prior': 'BasePrior',
        'nfe': 63,
        'n_train_steps': 25e4,

        'batch_size': 1,
        'device': 'cuda',

        ## diffusion model
        'horizon': 256,
        'n_diffusion_steps': 256,
        'normalizer': 'LimitsNormalizer',

        ## serialization
        'vis_freq': 10,
        'logbase': 'logs',
        'prefix': 'plans/release',
        'exp_name': watch(plan_args_to_watch),
        'suffix': '0',

        'conditional': False,

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}_N{nfe}_{diffusion}_{prior}_{n_train_steps}',
        'diffusion_epoch': 'latest',
    },

}

#------------------------ overrides ------------------------#

'''
    maze2d maze episode steps:
        umaze: 150
        medium: 250
        large: 600
'''

maze2d_umaze_v1 = {
    'diffusion': {
        'horizon': 128,
        'n_diffusion_steps': 64,
        'nfe': 63,
    },
    'plan': {
        'horizon': 128,
        'n_diffusion_steps': 64,
        'nfe': 63,
    },
}

maze2d_medium_v1 = {
    'diffusion': {
        'horizon': 256,
        'n_diffusion_steps': 5,
        'nfe': 4,
    },
    'plan': {
        'horizon': 256,
        'n_diffusion_steps': 5,
        'nfe': 4,
    },
}

maze2d_large_v1 = {
    'diffusion': {
        'horizon': 384,
        'n_diffusion_steps': 11,
        'nfe': 10,
    },
    'plan': {
        'horizon': 384,
        'n_diffusion_steps': 11,
        'nfe': 10,
    },
}
