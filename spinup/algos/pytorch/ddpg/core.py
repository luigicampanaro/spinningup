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

class CPGActor(nn.Module, CPGControllerHopf):

    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        CPGControllerHopf.__init__(self, *args, **kwargs)
    def forward(self, obs=None):
        assert obs is not None, "Obs are 'None'."
        d = obs[:len(self.drives), 0].double().view(-1, 1)
        v = self.sym @ self.v_short + self.fixed

        sigma_N = self.contactFeedback(obs)

        theta_dot = 2. * np.pi * ((self.Cd @ v) * (self.D @ d) + self.Od @ v) + torch.sum(
            (self.W @ v) * (self.Lambda @ self.r_old) * torch.sin(
                self.Lambda @ self.theta_old - self.Lambda_transpose @ self.theta_old - self.Fi @ v),
            dim=1, keepdim=False) - (self.SIGMA @ sigma_N) * torch.cos(self.theta_old)

        r_dot_dot = (self.A @ v) * (
                    (self.A @ v / 4.) * ((self.Cr @ v) * (self.D @ d) + self.Or @ v - self.r_old) - self.r_dot_old)

        x = self.r_old * torch.cos(self.theta_old)
        x_dot = self.r_dot_old * np.cos(self.theta_old) - self.r_old * np.sin(self.theta_old) * self.theta_dot_old

        theta = self.theta_old + (theta_dot + self.theta_dot_old) * self.dt / 2.
        r_dot = self.r_dot_old + (self.r_dot_dot_old + r_dot_dot) * self.dt / 2.
        r = self.r_old + (self.r_dot_old + r_dot) * self.dt / 2.

        self.theta_old = theta.detach()
        self.theta_dot_old = theta_dot.detach()
        self.r_old = r.detach()
        self.r_dot_old = r_dot.detach()
        self.r_dot_dot_old = r_dot_dot.detach()

        return (torch.cat((x, x_dot), 1)).numpy()

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

class CPGActorMLPCritic(nn.Module):
    def __init__(self,network,
                        v_names,
                        v_sym_names,
                        sym_tuples,
                        fixed_tuples,
                        init='random',
                         seed=None,
                         dt=1.0 / 400.0,
                         saveParamsDict=False,
                        hidden_sizes=(256,256), activation=nn.ReLU):
        super().__init__()

        self.pi = CPGActor(network, v_names, v_sym_names, sym_tuples, fixed_tuples, init, seed, dt, saveParamsDict)
        self.q = MLPQFunction(self.pi.obs_dim, len(self.pi.network), hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs)

############################################
if __name__ == "__main__":
    import json
    from collections import OrderedDict
    import sys

    path = '../../../../../envs/hopper_ANYmal/'
    with open(path + 'active_joints_properties.json', 'r') as f:
        active_joint_properties = json.load(f, object_pairs_hook=OrderedDict)
    with open(path + 'network.json', 'r') as file:
        network = json.load(file, object_pairs_hook=OrderedDict)
    with open(path + 'v_names.json', 'r') as file:
        v_names = json.load(file, object_pairs_hook=OrderedDict)
    with open(path + 'v_sym_names.json', 'r') as file:
        v_sym_names = json.load(file, object_pairs_hook=OrderedDict)
    with open(path + 'sym_tuples.json', 'r') as file:
        sym_tuples = json.load(file, object_pairs_hook=OrderedDict)
    with open(path + 'fixed_tuples.json', 'r') as file:
        fixed_tuples = json.load(file, object_pairs_hook=OrderedDict)

    obs = {'DRIVES': {'DRIVES': {'D': 0.5}},
             'GRF_RF': 0,
             'RF_HFE': {'applied_torques': 0.0,
                        'pos': 0.4,
                        'reac_forces': {'fx': 0.0,
                                        'fy': 0.0,
                                        'fz': 0.0,
                                        'mx': 0.0,
                                        'my': 0.0,
                                        'mz': 0.0},
                        'vel': 0.0},
             'RF_KFE': {'applied_torques': 0.0,
                        'pos': -0.8,
                        'reac_forces': {'fx': 0.0,
                                        'fy': 0.0,
                                        'fz': 0.0,
                                        'mx': 0.0,
                                        'my': 0.0,
                                        'mz': 0.0},
                        'vel': 0.0}}

    ##### Testing CPGActor #####
    actorCPG = CPGActor(network=network,
                            v_names=v_names,
                            v_sym_names=v_sym_names,
                            sym_tuples=sym_tuples,
                            fixed_tuples=fixed_tuples,
                            init='random',
                            saveParamsDict=False)

    print(f'actorCPG.parameters().data:\n{[par.data for par in actorCPG.parameters()]}\n\n')
    print(f'actorCPG.parameters().grad:\n{[par.grad for par in actorCPG.parameters()]}\n\n')
    print(f'actorCPG.forward(obs):\n{actorCPG(obs)}\n\n\n')

    ##### Testing CPGActorMLPCritic #####
    ActorCPGCriticMLP = CPGActorMLPCritic(network=network,
                                            v_names=v_names,
                                            v_sym_names=v_sym_names,
                                            sym_tuples=sym_tuples,
                                            fixed_tuples=fixed_tuples,
                                            init='random',
                                            saveParamsDict=False)

    print(f'ActorCPGCriticMLP.act(obs):\n{ActorCPGCriticMLP.act(obs)}\n\n')
    print(f'Display ActorCPGCriticMLP.q network:\n{ActorCPGCriticMLP.q}\n\n')