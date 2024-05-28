optimal_training_data_size =  {
    'dataset': 'maze2d-large-v1',
    'horizon': 384,
    'batch_size': 100,
    'configs': [
        ('SBDiffusion', 'BasePrior', 16, 15, 5e5, '490000'),
        ('SBDiffusion', 'BasePrior', 16, 15, 5e5, '400000'),
        ('SBDiffusion', 'BasePrior', 16, 15, 5e5, '350000'),
        ('SBDiffusion', 'BasePrior', 16, 15, 5e5, '300000'),
        ('SBDiffusion', 'BasePrior', 16, 15, 5e5, '250000'),
        ('SBDiffusion', 'BasePrior', 16, 15, 5e5, '200000'),
        ('SBDiffusion', 'BasePrior', 16, 15, 5e5, '150000'),
        ('SBDiffusion', 'BasePrior', 16, 15, 5e5, '100000'),
        ('GaussianDiffusion', '-', 16, '-', 5e5, '490000'),
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
        ('SBDiffusion', 'BasePrior', 16, 10, 25e4, '245000'),
        ('SBDiffusion', 'BasePrior', 16, 10, 25e4, '200000'),
        ('SBDiffusion', 'BasePrior', 16, 10, 25e4, '150000'),
        ('SBDiffusion', 'BasePrior', 16, 10, 25e4, '100000'),
        ('SBDiffusion', 'BasePrior', 16, 7, 25e4, '245000'),
        ('SBDiffusion', 'BasePrior', 16, 7, 25e4, '200000'),
        ('SBDiffusion', 'BasePrior', 16, 7, 25e4, '150000'),
        ('SBDiffusion', 'BasePrior', 16, 7, 25e4, '100000'),
        ('SBDiffusion', 'BasePrior', 16, 4, 25e4, '245000'),
        ('SBDiffusion', 'BasePrior', 16, 4, 25e4, '200000'),
        ('SBDiffusion', 'BasePrior', 16, 4, 25e4, '150000'),
        ('SBDiffusion', 'BasePrior', 16, 4, 25e4, '100000'),
        ('SBDiffusion', 'BasePrior', 16, 1, 25e4, '245000'),
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
        ('GaussianDiffusion', '-', 11, 10, 25e4, '245000'),
        ('GaussianDiffusion', '-', 11, 10, 25e4, '200000'),
        ('GaussianDiffusion', '-', 11, 10, 25e4, '150000'),
        ('GaussianDiffusion', '-', 11, 10, 25e4, '100000'),
        ('GaussianDiffusion', '-', 8, 7, 25e4, '245000'),
        ('GaussianDiffusion', '-', 8, 7, 25e4, '200000'),
        ('GaussianDiffusion', '-', 8, 7, 25e4, '150000'),
        ('GaussianDiffusion', '-', 8, 7, 25e4, '100000'),
        # ('GaussianDiffusion', '-', 5, 4, 25e4, '245000'),
        # ('GaussianDiffusion', '-', 5, 4, 25e4, '200000'),
        # ('GaussianDiffusion', '-', 5, 4, 25e4, '150000'),
        # ('GaussianDiffusion', '-', 5, 4, 25e4, '100000'),
        ('GaussianDiffusion', '-', 2, 1, 25e4, '245000'),
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
        ('SBDiffusion', 'BasePrior', 16, 10, 25e4, '245000'),
        ('SBDiffusion', 'BasePrior', 16, 10, 25e4, '200000'),
        ('SBDiffusion', 'BasePrior', 16, 10, 25e4, '150000'),
        ('SBDiffusion', 'BasePrior', 16, 10, 25e4, '100000'),
        ('SBDiffusion', 'BasePrior', 16, 7, 25e4, '245000'),
        ('SBDiffusion', 'BasePrior', 16, 7, 25e4, '200000'),
        ('SBDiffusion', 'BasePrior', 16, 7, 25e4, '150000'),
        ('SBDiffusion', 'BasePrior', 16, 7, 25e4, '100000'),
        ('SBDiffusion', 'BasePrior', 16, 4, 25e4, '245000'),
        ('SBDiffusion', 'BasePrior', 16, 4, 25e4, '200000'),
        ('SBDiffusion', 'BasePrior', 16, 4, 25e4, '150000'),
        ('SBDiffusion', 'BasePrior', 16, 4, 25e4, '100000'),
        ('SBDiffusion', 'BasePrior', 16, 1, 25e4, '245000'),
        ('SBDiffusion', 'BasePrior', 16, 1, 25e4, '200000'),
        ('SBDiffusion', 'BasePrior', 16, 1, 25e4, '150000'),
        ('SBDiffusion', 'BasePrior', 16, 1, 25e4, '100000'),
    ]
}

nfe_and_training_steps_score_ddpm_maze2d_large =  {
    'dataset': 'maze2d-large-v1',
    'horizon': 384,
    'batch_size': 200,
    'configs': [
        ('GaussianDiffusion', '-', 11, 10, 25e4, '245000'),
        ('GaussianDiffusion', '-', 11, 10, 25e4, '200000'),
        ('GaussianDiffusion', '-', 11, 10, 25e4, '150000'),
        ('GaussianDiffusion', '-', 11, 10, 25e4, '100000'),
        ('GaussianDiffusion', '-', 8, 7, 25e4, '245000'),
        ('GaussianDiffusion', '-', 8, 7, 25e4, '200000'),
        ('GaussianDiffusion', '-', 8, 7, 25e4, '150000'),
        ('GaussianDiffusion', '-', 8, 7, 25e4, '100000'),
        ('GaussianDiffusion', '-', 5, 4, 25e4, '245000'),
        ('GaussianDiffusion', '-', 5, 4, 25e4, '200000'),
        ('GaussianDiffusion', '-', 5, 4, 25e4, '150000'),
        ('GaussianDiffusion', '-', 5, 4, 25e4, '100000'),
        ('GaussianDiffusion', '-', 2, 1, 25e4, '245000'),
        ('GaussianDiffusion', '-', 2, 1, 25e4, '200000'),
        ('GaussianDiffusion', '-', 2, 1, 25e4, '150000'),
        ('GaussianDiffusion', '-', 2, 1, 25e4, '100000'),
    ]
}

score_vs_training_steps_for_sb_with_straight_line_prior =  {
    'dataset': 'maze2d-medium-v1',
    'horizon': 256,
    'batch_size': 200,
    'configs': [
        ('SBDiffusion', 'StraightLinePrior', 16, 4, 25e4, '245000'),
        ('SBDiffusion', 'StraightLinePrior', 16, 4, 25e4, '200000'),
        ('SBDiffusion', 'StraightLinePrior', 16, 4, 25e4, '150000'),
        ('SBDiffusion', 'StraightLinePrior', 16, 4, 25e4, '100000'),
        ('SBDiffusion', 'StraightLinePrior', 16, 4, 25e4, '50000'),
        ('SBDiffusion', 'StraightLinePrior', 16, 4, 25e4, '25000'),
        ('SBDiffusion', 'StraightLinePrior', 16, 4, 25e4, '10000'),
    ]
}

score_vs_training_steps_for_sb_with_MLP_prior =  {
    'dataset': 'maze2d-medium-v1',
    'horizon': 256,
    'batch_size': 200,
    'configs': [
        ('SBDiffusion', 'MLPTrajectoryPrior', 16, 4, 25e4, '245000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 16, 4, 25e4, '200000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 16, 4, 25e4, '150000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 16, 4, 25e4, '100000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 16, 4, 25e4, '50000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 16, 4, 25e4, '25000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 16, 4, 25e4, '10000'),
    ]
}

score_vs_training_steps_for_sb_with_base_prior =  {
    'dataset': 'maze2d-medium-v1',
    'horizon': 256,
    'batch_size': 200,
    'configs': [
        ('SBDiffusion', 'BasePrior', 16, 4, 25e4, '245000'),
        ('SBDiffusion', 'BasePrior', 16, 4, 25e4, '200000'),
        ('SBDiffusion', 'BasePrior', 16, 4, 25e4, '150000'),
        ('SBDiffusion', 'BasePrior', 16, 4, 25e4, '100000'),
        ('SBDiffusion', 'BasePrior', 16, 4, 25e4, '50000'),
        ('SBDiffusion', 'BasePrior', 16, 4, 25e4, '25000'),
        ('SBDiffusion', 'BasePrior', 16, 4, 25e4, '10000'),
    ]
}

score_vs_training_steps_for_ddpm_maze2d_medium =  {
    'dataset': 'maze2d-medium-v1',
    'horizon': 256,
    'batch_size': 200,
    'configs': [
        ('GaussianDiffusion', '-', 5, '-', 25e4, '150000'),
        ('GaussianDiffusion', '-', 5, '-', 25e4, '100000'),
        ('GaussianDiffusion', '-', 5, '-', 25e4, '50000'),
        ('GaussianDiffusion', '-', 5, '-', 25e4, '25000'),
        ('GaussianDiffusion', '-', 5, '-', 25e4, '10000'),
    ]
}


score_vs_training_steps_across_all_algos_maze2d_medium_NFE_4 =  {
    'dataset': 'maze2d-medium-v1',
    'horizon': 256,
    'batch_size': 300,
    'configs': [
        ('SBDiffusion', 'BasePrior', 16, 4, 25e4, '245000'),
        ('SBDiffusion', 'BasePrior', 16, 4, 25e4, '200000'),
        ('SBDiffusion', 'BasePrior', 16, 4, 25e4, '150000'),
        ('SBDiffusion', 'BasePrior', 16, 4, 25e4, '100000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 16, 4, 25e4, '245000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 16, 4, 25e4, '200000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 16, 4, 25e4, '150000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 16, 4, 25e4, '100000'),
        ('SBDiffusion', 'StraightLinePrior', 16, 4, 25e4, '245000'),
        ('SBDiffusion', 'StraightLinePrior', 16, 4, 25e4, '200000'),
        ('SBDiffusion', 'StraightLinePrior', 16, 4, 25e4, '150000'),
        ('SBDiffusion', 'StraightLinePrior', 16, 4, 25e4, '100000'),
    ]
}



score_vs_training_steps_for_ddpm_maze2d_medium_NFE_1 =  {
    'dataset': 'maze2d-medium-v1',
    'horizon': 256,
    'batch_size': 200,
    'configs': [
        ('GaussianDiffusion', '-', 2, '-', 25e4, '245000'),
        ('GaussianDiffusion', '-', 2, '-', 25e4, '200000'),
        ('GaussianDiffusion', '-', 2, '-', 25e4, '150000'),
        ('GaussianDiffusion', '-', 2, '-', 25e4, '100000'),
    ]
}

score_vs_training_steps_for_ddpm_vs_sb_on_different_priors_maze2d_medium_T_16_NFE_1 =  {
    'dataset': 'maze2d-medium-v1',
    'horizon': 256,
    'batch_size': 200,
    'configs': [
        ('GaussianDiffusion', '-', 2, '-', 25e4, '245000'),
        ('GaussianDiffusion', '-', 2, '-', 25e4, '200000'),
        ('GaussianDiffusion', '-', 2, '-', 25e4, '150000'),
        ('GaussianDiffusion', '-', 2, '-', 25e4, '100000'),
        ('GaussianDiffusion', '-', 2, '-', 25e4, '50000'),
        ('GaussianDiffusion', '-', 2, '-', 25e4, '25000'),
        ('GaussianDiffusion', '-', 2, '-', 25e4, '10000'),
        ('SBDiffusion', 'BasePrior', 16, 1, 25e4, '245000'),
        ('SBDiffusion', 'BasePrior', 16, 1, 25e4, '200000'),
        ('SBDiffusion', 'BasePrior', 16, 1, 25e4, '150000'),
        ('SBDiffusion', 'BasePrior', 16, 1, 25e4, '100000'),
        ('SBDiffusion', 'BasePrior', 16, 1, 25e4, '50000'),
        ('SBDiffusion', 'BasePrior', 16, 1, 25e4, '25000'),
        ('SBDiffusion', 'BasePrior', 16, 1, 25e4, '10000'),
        ('SBDiffusion', 'StraightLinePrior', 16, 1, 25e4, '245000'),
        ('SBDiffusion', 'StraightLinePrior', 16, 1, 25e4, '200000'),
        ('SBDiffusion', 'StraightLinePrior', 16, 1, 25e4, '150000'),
        ('SBDiffusion', 'StraightLinePrior', 16, 1, 25e4, '100000'),
        ('SBDiffusion', 'StraightLinePrior', 16, 1, 25e4, '50000'),
        ('SBDiffusion', 'StraightLinePrior', 16, 1, 25e4, '25000'),
        ('SBDiffusion', 'StraightLinePrior', 16, 1, 25e4, '10000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 16, 1, 25e4, '245000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 16, 1, 25e4, '200000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 16, 1, 25e4, '150000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 16, 1, 25e4, '100000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 16, 1, 25e4, '50000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 16, 1, 25e4, '25000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 16, 1, 25e4, '10000'),

    ]
}


score_vs_training_steps_for_ddpm_vs_sb_on_different_priors_maze2d_medium_NFE_1_same_complexity =  {
    'dataset': 'maze2d-medium-v1',
    'horizon': 256,
    'batch_size': 200,
    'configs': [
        ('GaussianDiffusion', '-', 2, '-', 25e4, '245000'),
        ('GaussianDiffusion', '-', 2, '-', 25e4, '200000'),
        ('GaussianDiffusion', '-', 2, '-', 25e4, '150000'),
        ('GaussianDiffusion', '-', 2, '-', 25e4, '100000'),
        ('GaussianDiffusion', '-', 2, '-', 25e4, '50000'),
        ('GaussianDiffusion', '-', 2, '-', 25e4, '25000'),
        ('GaussianDiffusion', '-', 2, '-', 25e4, '10000'),
        ('SBDiffusion', 'BasePrior', 2, 1, 25e4, '245000'),
        ('SBDiffusion', 'BasePrior', 2, 1, 25e4, '200000'),
        ('SBDiffusion', 'BasePrior', 2, 1, 25e4, '150000'),
        ('SBDiffusion', 'BasePrior', 2, 1, 25e4, '100000'),
        ('SBDiffusion', 'BasePrior', 2, 1, 25e4, '50000'),
        ('SBDiffusion', 'BasePrior', 2, 1, 25e4, '25000'),
        ('SBDiffusion', 'BasePrior', 2, 1, 25e4, '10000'),
        ('SBDiffusion', 'StraightLinePrior', 2, 1, 25e4, '245000'),
        ('SBDiffusion', 'StraightLinePrior', 2, 1, 25e4, '200000'),
        ('SBDiffusion', 'StraightLinePrior', 2, 1, 25e4, '150000'),
        ('SBDiffusion', 'StraightLinePrior', 2, 1, 25e4, '100000'),
        ('SBDiffusion', 'StraightLinePrior', 2, 1, 25e4, '50000'),
        ('SBDiffusion', 'StraightLinePrior', 2, 1, 25e4, '25000'),
        ('SBDiffusion', 'StraightLinePrior', 2, 1, 25e4, '10000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 2, 1, 25e4, '245000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 2, 1, 25e4, '200000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 2, 1, 25e4, '150000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 2, 1, 25e4, '100000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 2, 1, 25e4, '50000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 2, 1, 25e4, '25000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 2, 1, 25e4, '10000'),

    ]
}

score_vs_training_steps_for_ddpm_vs_sb_on_different_priors_maze2d_medium_T_64_NFE_1 =  {
    'dataset': 'maze2d-medium-v1',
    'horizon': 256,
    'batch_size': 300,
    'configs': [
        ('GaussianDiffusion', '-', 2, '-', 25e4, '245000'),
        ('GaussianDiffusion', '-', 2, '-', 25e4, '200000'),
        ('GaussianDiffusion', '-', 2, '-', 25e4, '150000'),
        ('GaussianDiffusion', '-', 2, '-', 25e4, '100000'),
        ('SBDiffusion', 'BasePrior', 64, 1, 25e4, '245000'),
        ('SBDiffusion', 'BasePrior', 64, 1, 25e4, '200000'),
        ('SBDiffusion', 'BasePrior', 64, 1, 25e4, '150000'),
        ('SBDiffusion', 'BasePrior', 64, 1, 25e4, '100000'),
        ('SBDiffusion', 'StraightLinePrior', 64, 1, 25e4, '245000'),
        ('SBDiffusion', 'StraightLinePrior', 64, 1, 25e4, '200000'),
        ('SBDiffusion', 'StraightLinePrior', 64, 1, 25e4, '150000'),
        ('SBDiffusion', 'StraightLinePrior', 64, 1, 25e4, '100000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 64, 1, 25e4, '245000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 64, 1, 25e4, '200000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 64, 1, 25e4, '150000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 64, 1, 25e4, '100000'),

    ]
}


score_vs_training_steps_for_ddpm_vs_sb_on_different_priors_maze2d_medium_T_1000_NFE_1 =  {
    'dataset': 'maze2d-medium-v1',
    'horizon': 256,
    'batch_size': 300,
    'configs': [
        ('GaussianDiffusion', '-', 2, '-', 25e4, '245000'),
        ('GaussianDiffusion', '-', 2, '-', 25e4, '200000'),
        ('GaussianDiffusion', '-', 2, '-', 25e4, '150000'),
        ('GaussianDiffusion', '-', 2, '-', 25e4, '100000'),
        ('SBDiffusion', 'StraightLinePrior', 1000, 1, 25e4, '245000'),
        ('SBDiffusion', 'StraightLinePrior', 1000, 1, 25e4, '200000'),
        ('SBDiffusion', 'StraightLinePrior', 1000, 1, 25e4, '150000'),
        ('SBDiffusion', 'StraightLinePrior', 1000, 1, 25e4, '100000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 1000, 1, 25e4, '245000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 1000, 1, 25e4, '200000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 1000, 1, 25e4, '150000'),
        ('SBDiffusion', 'MLPTrajectoryPrior', 1000, 1, 25e4, '100000'),

    ]
}