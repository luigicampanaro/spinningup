import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import sys
sys.path.append('../../../../../controllers/')
from cpg_controller_hopf import CPGControllerHopf


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)

class Actor_CPG(nn.Module, CPGControllerHopf):

    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        CPGControllerHopf.__init__(self, *args, **kwargs)

    def forward(self, obs=None):
        assert obs is not None, "Obs are 'None'."
        d = self.updateControlCommands(obs)
        v = self.sym @ self.v_short + self.fixed

        if obs is not None:
            sigma_N = self.contactFeedback(obs)

            theta_dot = 2. * np.pi * ((self.Cd @ v) * (self.D @ d) + self.Od @ v) + torch.sum(
                (self.W @ v) * (self.Lambda @ self.r_old) * torch.sin(
                    self.Lambda @ self.theta_old - self.Lambda_transpose @ self.theta_old - self.Fi @ v),
                dim=1, keepdim=True) - (self.SIGMA @ sigma_N) * torch.cos(self.theta_old)
            r_dot_dot = (self.A @ v) * (
                        (self.A @ v / 4.) * ((self.Cr @ v) * (self.D @ d) + self.Or @ v - self.r_old) - self.r_dot_old)

        x = self.r_old * torch.cos(self.theta_old)
        x_dot = self.r_dot_old * torch.cos(self.theta_old) - self.r_old * torch.sin(self.theta_old) * self.theta_dot_old
        x_dot_dot = self.r_dot_dot_old * torch.cos(self.theta_old) - 2 * self.r_dot_old * torch.sin(self.theta_old) * self.theta_dot_old - self.r_old * (torch.cos(self.theta_old) * self.theta_dot_old ** 2 + torch.sin(self.theta_old) * self.theta_dot_dot_old)

        theta = self.theta_old + (theta_dot + self.theta_dot_old) * self.dt / 2.
        r_dot = self.r_dot_old + (self.r_dot_dot_old + r_dot_dot) * self.dt / 2.
        r = self.r_old + (self.r_dot_old + r_dot) * self.dt / 2.
        self.theta_dot_dot_old = (theta_dot - self.theta_dot_old) / self.dt

        self.theta_old = theta
        self.theta_dot_old = theta_dot
        self.r_old = r
        self.r_dot_old = r_dot
        self.r_dot_dot_old = r_dot_dot

        return self.actionsArray2Dictionary({
            'pos': x,
            'vel': x_dot,
            'acc': x_dot_dot
        })


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()
