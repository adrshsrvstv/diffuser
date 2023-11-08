import numpy as np
import torch
from torch import nn
import pdb

import diffuser.utils as utils
from .helpers import (
    increasing_decreasing_beta_schedule,
    compute_gaussian_product_coef,
    apply_conditioning,
    unsqueeze_xdim,
    Losses,
    space_indices,
)

class SBDiffusion(nn.Module):
    def __init__(self, model, prior, horizon, observation_dim, action_dim, n_timesteps=1000, nfe=64,
        loss_type='l2', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        assert  prior is not None
        self.prior = prior
        self.n_timesteps = int(n_timesteps)
        self.nfe = int(nfe)

        betas = increasing_decreasing_beta_schedule(n_timesteps)
        # compute analytic std: eq 11
        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        self.register_buffer('betas', torch.tensor(betas, dtype=torch.float32) )
        self.register_buffer('std_fwd', torch.tensor(std_fwd, dtype=torch.float32) )
        self.register_buffer('std_bwd', torch.tensor(std_bwd, dtype=torch.float32) )
        self.register_buffer('mu_x0', torch.tensor(mu_x0, dtype=torch.float32) )
        self.register_buffer('mu_x1', torch.tensor(mu_x1, dtype=torch.float32) )
        self.register_buffer('var', torch.tensor(var, dtype=torch.float32) )
        self.register_buffer('std_sb', torch.tensor(std_sb, dtype=torch.float32) )

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)


    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    # ------------------------------------------ sampling ------------------------------------------#
    def get_std_fwd(self, t, xdim=None):
        std_fwd = self.std_fwd[t]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    def compute_label(self, t, x0, xt):
        """ Eq 12 """
        std_fwd = self.get_std_fwd(t, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()

    @torch.no_grad()
    def p_posterior(self, t_prev, t, xt, x0):
        """ Sample p(x_{nprev} | x_n, x_0), i.e. eq 4"""
        assert t_prev < t
        std_n     = self.std_fwd[t]
        std_t_prev = self.std_fwd[t_prev]
        std_delta = (std_n**2 - std_t_prev**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_t_prev, std_delta)
        xt_prev = mu_x0 * x0 + mu_xn * xt + var.sqrt() * torch.randn_like(x0)

        return xt_prev

    def compute_pred_x0(self, t, xt, net_out):
        """ Given network output, recover x0. This should be the inverse of Eq 12 """
        std_fwd = self.get_std_fwd(t, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        return pred_x0

    @torch.no_grad()
    def p_sample_loop(self, x1, cond, verbose=True ):
        device = self.betas.device
        batch_size = x1.shape[0]

        steps = space_indices(self.n_timesteps, self.nfe + 1)
        log_count = min(len(steps) - 1, 10)
        log_steps = [steps[i] for i in space_indices(len(steps) - 1, log_count)]

        steps = steps[::-1]
        pair_steps = list(zip(steps[:-1], steps[1:])) #[(99, 79), (79, 59), (59, 40), (40, 20), (20, 0)]

        xt = x1.detach().to(device)
        xs = []
        pred_x0s = []

        progress = utils.Progress(len(pair_steps)) if verbose else utils.Silent()

        for t, t_prev in pair_steps:
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)

            pred_x0 = self.compute_pred_x0(timesteps, xt, self.model(xt, cond, t))
            xt = self.p_posterior(t_prev, t, xt, pred_x0)
            xt = apply_conditioning(xt, cond, self.action_dim)

            # if t_prev in log_steps:
            #     pred_x0s.append(pred_x0.detach().cpu())
            #     xs.append(xt.detach().cpu())

            progress.update({'t': t})

        progress.close()

        # stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        # return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)
        return xt

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, t, x0, x1):
        """ Sample q(x_t | x_0, x_1), i.e. eq 11 """

        assert x0.shape == x1.shape
        batch, *xdim = x0.shape

        mu_x0  = unsqueeze_xdim(self.mu_x0[t], xdim)
        mu_x1  = unsqueeze_xdim(self.mu_x1[t], xdim)
        std_sb = unsqueeze_xdim(self.std_sb[t], xdim)

        xt = mu_x0 * x0 + mu_x1 * x1 + std_sb * torch.randn_like(x0)
        return xt.detach()

    def p_losses(self, x0, x1, cond, t):
        xt = self.q_sample(t, x0, x1)
        label = self.compute_label(t, x0, xt)

        pred = self.model(xt, cond, t)
        # should we condition the pred here?
        assert xt.shape == label.shape == pred.shape

        return self.loss_fn(pred, label)

    def loss(self, x0, cond):
        x1 = self.prior(cond, self.horizon, self.transition_dim, x0.device)
        batch_size = len(x0)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x0.device).long()
        return self.p_losses(x0, x1, cond, t)

    def forward(self, cond, *args, **kwargs):
        x1 = self.prior(cond, self.horizon, self.transition_dim, cond.device)
        return self.p_sample_loop(x1, cond, *args, **kwargs)

