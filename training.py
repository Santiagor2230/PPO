import copy
import torch
import gym

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.distributions import Normal
from pytorch_lightning import LightningModule
from brax.envs import to_torch

from environment import NormalizeObservation, create_env
from policy_model import GradientPolicy
from value_model import ValueNet
from dataset_buffer import RLDataset
from create_video import create_video, test_agent

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class PPO(LightningModule):
  def __init__(self, env_name, num_envs=2048, episode_length=1000, 
               batch_size = 1024, hidden_size=256, samples_per_epoch = 5,
               epoch_repeat=8, policy_lr=1e-4, value_lr=1e-3, gamma = 0.97,
               epsilon = 0.3, entropy_coef=0.1, optim=AdamW):
    super().__init__()

    self.env = create_env(env_name, num_envs = num_envs, episode_length = episode_length)
    test_env = gym.make(env_name , episode_length = episode_length)
    test_env = to_torch.JaxToTorchWrapper(test_env, device=device)
    self.test_env = NormalizeObservation(test_env)
    self.test_env.obs_rms = self.env.obs_rms

    obs_size = self.env.observation_space.shape[1]
    action_dims = self.env.action_space.shape[1]

    self.policy = GradientPolicy(obs_size, action_dims, hidden_size)
    self.value_net = ValueNet(obs_size, hidden_size)
    self.target_value_net = copy.deepcopy(self.value_net)

    self.dataset = RLDataset(self.env, self.policy, samples_per_epoch, epoch_repeat)

    self.save_hyperparameters()
    self.videos = []

  def configure_optimizers(self):
    value_opt = self.hparams.optim(self.value_net.parameters(), lr = self.hparams.value_lr)
    policy_opt = self.hparams.optim(self.policy.parameters(), lr=self.hparams.policy_lr)
    return value_opt, policy_opt

  def train_dataloader(self):
    return DataLoader(dataset=self.dataset, batch_size = self.hparams.batch_size)

  def training_step(self, batch, batch_idx, optimizer_idx):
    obs_b, loc_b, scale_b, action_b, reward_b, done_b, next_obs_b = batch

    state_values = self.value_net(obs_b) #checking each state through the nn

    with torch.no_grad():
      next_state_values = self.target_value_net(next_obs_b)
      next_state_values[done_b.bool()] = 0.0
      """v(s,a) = reward + gamma* v(s',a')"""
      target = reward_b + self.hparams.gamma * next_state_values

    if optimizer_idx == 0:
      """log-likelihood loss function"""
      loss = F.smooth_l1_loss(state_values, target)
      self.log("episode/value", loss)
      return loss

    elif optimizer_idx == 1:
      """advantage estimator difference between target
      and current state, detach() allows them to be constant """
      advantages = (target - state_values).detach()

      new_loc, new_scale = self.policy(obs_b) #every update gives different values
      dist = Normal(new_loc, new_scale)

      """Normal distribution log probability of batch of actions,
      sum() allows to sum all the dimensions for actions into a single
      value """

      log_prob = dist.log_prob(action_b).sum(dim=-1, keepdim=True)

      """previous distribution before update"""
      prev_dist = Normal(loc_b, scale_b)
      prev_log_prob = prev_dist.log_prob(action_b).sum(dim=-1, keepdim=True)

      '''the difference of exponent of exponents of the log
      probability of the update one with the previous one
      rt(0)'''
      rho = torch.exp(log_prob - prev_log_prob)

      '''rt(0)*A'''
      surrogate_1 = rho * advantages #first loss function
      '''clip(rt(0), 1-E. 1+E)* A -> second loss function'''
      surrogate_2 = rho.clip(1 - self.hparams.epsilon, 1 + self.hparams.epsilon)* advantages

      '''L(0)'''
      policy_loss = - torch.minimum(surrogate_1, surrogate_2)#picks either one loss
      entropy = dist.entropy().sum(dim=-1, keepdim=True)
      loss = policy_loss - self.hparams.entropy_coef * entropy

      self.log("episode/Policy Loss", policy_loss.mean())
      self.log("episode/Entropy", entropy.mean())
      self.log("episode/Reward", reward_b.mean())
      return loss.mean()

  def training_epoch_end(self, training_epoch_outputs):
    self.target_value_net.load_state_dict(self.value_net.state_dict())

    if self.current_epoch % 10 == 0:
      average_return =test_agent(self.test_env, self.hparams.episode_length, self.policy, episodes=1)
      self.log("episode/Average Return", average_return)

    if self.current_epoch % 50 == 0:
      video = create_video(self.test_env, self.hparams.episode_length, policy=self.policy)
      self.videos.append(video)