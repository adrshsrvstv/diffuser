score_vs_training_steps_for_sb =  {
    'dataset': 'maze2d-large-v1',
    'horizon': 384,
    'batch_size': 100,
    'configs': [
        ('SBDiffusion', 'BasePrior', 16, 15, 5e5, '400000'),
        ('SBDiffusion', 'BasePrior', 16, 15, 5e5, '350000'),
        ('SBDiffusion', 'BasePrior', 16, 15, 5e5, '300000'),
        ('SBDiffusion', 'BasePrior', 16, 15, 5e5, '250000'),
        ('SBDiffusion', 'BasePrior', 16, 15, 5e5, '200000'),
        ('SBDiffusion', 'BasePrior', 16, 15, 5e5, '150000'),
        ('SBDiffusion', 'BasePrior', 16, 15, 5e5, '100000'),
    ]
}

score_vs_training_steps_for_ddpm =  {
    'dataset': 'maze2d-large-v1',
    'horizon': 384,
    'batch_size': 100,
    'configs': [
        ('GaussianDiffusion', '-', 16, '-', 5e5, '400000'),
        ('GaussianDiffusion', '-', 16, '-', 5e5, '350000'),
        ('GaussianDiffusion', '-', 16, '-', 5e5, '300000'),
        ('GaussianDiffusion', '-', 16, '-', 5e5, '250000'),
        ('GaussianDiffusion', '-', 16, '-', 5e5, '200000'),
        ('GaussianDiffusion', '-', 16, '-', 5e5, '150000'),
        ('GaussianDiffusion', '-', 16, '-', 5e5, '100000'),
    ]
}

sb_with_learned_prior =  {
    'dataset': 'maze2d-large-v1',
    'horizon': 384,
    'batch_size': 200,
    'configs': [
        ('SBDiffusion', 'LearnedTrajectoryPrior', 16, 15, 5e5, 'latest'),
    ]
}


nfe_and_training_steps_score_sb_base_prior_maze2d_medium =  {
    'dataset': 'maze2d-medium-v1',
    'horizon': 256,
    'batch_size': 200,
    'configs': [
        ('SBDiffusion', 'BasePrior', 16, 10, 25e4, 'latest'),
        ('SBDiffusion', 'BasePrior', 16, 10, 25e4, '200000'),
        ('SBDiffusion', 'BasePrior', 16, 10, 25e4, '150000'),
        ('SBDiffusion', 'BasePrior', 16, 10, 25e4, '100000'),
        ('SBDiffusion', 'BasePrior', 16, 7, 25e4, 'latest'),
        ('SBDiffusion', 'BasePrior', 16, 7, 25e4, '200000'),
        ('SBDiffusion', 'BasePrior', 16, 7, 25e4, '150000'),
        ('SBDiffusion', 'BasePrior', 16, 7, 25e4, '100000'),
        ('SBDiffusion', 'BasePrior', 16, 4, 25e4, 'latest'),
        ('SBDiffusion', 'BasePrior', 16, 4, 25e4, '200000'),
        ('SBDiffusion', 'BasePrior', 16, 4, 25e4, '150000'),
        ('SBDiffusion', 'BasePrior', 16, 4, 25e4, '100000'),
        ('SBDiffusion', 'BasePrior', 16, 1, 25e4, 'latest'),
        ('SBDiffusion', 'BasePrior', 16, 1, 25e4, '200000'),
        ('SBDiffusion', 'BasePrior', 16, 1, 25e4, '150000'),
        ('SBDiffusion', 'BasePrior', 16, 1, 25e4, '100000'),
    ]
}

nfe_and_training_steps_score_ddpm_maze2d_medium =  {
    'dataset': 'maze2d-medium-v1',
    'horizon': 256,
    'batch_size': 200,
    'configs': [
        ('GaussianDiffusion', '-', 11, 10, 25e4, 'latest'),
        ('GaussianDiffusion', '-', 11, 10, 25e4, '200000'),
        ('GaussianDiffusion', '-', 11, 10, 25e4, '150000'),
        ('GaussianDiffusion', '-', 11, 10, 25e4, '100000'),
        ('GaussianDiffusion', '-', 8, 7, 25e4, 'latest'),
        ('GaussianDiffusion', '-', 8, 7, 25e4, '200000'),
        ('GaussianDiffusion', '-', 8, 7, 25e4, '150000'),
        ('GaussianDiffusion', '-', 8, 7, 25e4, '100000'),
        ('GaussianDiffusion', '-', 5, 4, 25e4, 'latest'),
        ('GaussianDiffusion', '-', 5, 4, 25e4, '200000'),
        ('GaussianDiffusion', '-', 5, 4, 25e4, '150000'),
        ('GaussianDiffusion', '-', 5, 4, 25e4, '100000'),
        ('GaussianDiffusion', '-', 2, 1, 25e4, 'latest'),
        ('GaussianDiffusion', '-', 2, 1, 25e4, '200000'),
        ('GaussianDiffusion', '-', 2, 1, 25e4, '150000'),
        ('GaussianDiffusion', '-', 2, 1, 25e4, '100000'),
    ]
}

nfe_and_training_steps_score_sb_base_prior_maze2d_large =  {
    'dataset': 'maze2d-large-v1',
    'horizon': 384,
    'batch_size': 200,
    'configs': [
        ('SBDiffusion', 'BasePrior', 16, 10, 25e4, 'latest'),
        ('SBDiffusion', 'BasePrior', 16, 10, 25e4, '200000'),
        ('SBDiffusion', 'BasePrior', 16, 10, 25e4, '150000'),
        ('SBDiffusion', 'BasePrior', 16, 10, 25e4, '100000'),
        ('SBDiffusion', 'BasePrior', 16, 7, 25e4, 'latest'),
        ('SBDiffusion', 'BasePrior', 16, 7, 25e4, '200000'),
        ('SBDiffusion', 'BasePrior', 16, 7, 25e4, '150000'),
        ('SBDiffusion', 'BasePrior', 16, 7, 25e4, '100000'),
        ('SBDiffusion', 'BasePrior', 16, 4, 25e4, 'latest'),
        ('SBDiffusion', 'BasePrior', 16, 4, 25e4, '200000'),
        ('SBDiffusion', 'BasePrior', 16, 4, 25e4, '150000'),
        ('SBDiffusion', 'BasePrior', 16, 4, 25e4, '100000'),
        ('SBDiffusion', 'BasePrior', 16, 1, 25e4, 'latest'),
        ('SBDiffusion', 'BasePrior', 16, 1, 25e4, '200000'),
        ('SBDiffusion', 'BasePrior', 16, 1, 25e4, '150000'),
        ('SBDiffusion', 'BasePrior', 16, 1, 25e4, '100000'),
    ]
}