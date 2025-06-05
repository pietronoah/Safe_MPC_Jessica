import yaml
import torch
import torch.nn as nn
import numpy as np
from rl_games.algos_torch.running_mean_std import RunningMeanStd

# Function to load YAML configuration
def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


class RunningMeanStd(nn.Module):
    def __init__(self, shape = (), epsilon=1e-08):
        super(RunningMeanStd, self).__init__()
        self.register_buffer("running_mean", torch.zeros(shape))
        self.register_buffer("running_var", torch.ones(shape))
        self.register_buffer("count", torch.ones(()))

        self.epsilon = epsilon

    def forward(self, obs, update = True):
        if update:
            self.update(obs)

        return (obs - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, correction=0, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.running_mean, self.running_var, self.count = update_mean_var_count_from_moments(
            self.running_mean, self.running_var, self.count, batch_mean, batch_var, batch_count
        )

def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(45, 512)),
            nn.ELU(),
            layer_init(nn.Linear(512, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 12), std=0.01),
        )
        #self.actor_logstd = nn.Parameter(torch.zeros(1, 12))

        self.obs_rms = RunningMeanStd(shape = (45,))

    def get_action(self, x):
        action_mean = self.actor_mean(x)
        return action_mean

    def forward(self, x):
        action_mean = self.actor_mean(self.obs_rms(x, update = False))
        return action_mean

def load_nn():
    actor_network = Agent()
    actor_sd = torch.load("modified_SoloTerrain.pth", map_location="cpu")
    actor_network.load_state_dict(actor_sd)
    return actor_network