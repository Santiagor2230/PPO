from torch.utils.data.dataset import IterableDataset
import torch
import random

class RLDataset(IterableDataset):
  def __init__(self, env, policy, samples_per_epoch, epoch_repeat):
    self.env = env
    self.policy = policy
    self.samples_per_epoch = samples_per_epoch
    self.epoch_repeat = epoch_repeat
    self.obs = self.env.reset()

  @torch.no_grad()
  def __iter__(self):
    transitions = []
    for step in range(self.samples_per_epoch):
      loc, scale = self.policy(self.obs)
      action = torch.normal(loc, scale) #takes in the mean and the standard deviation
      next_obs, reward, done, info = self.env.step(action)
      transitions.append((self.obs, loc, scale, action, reward, done, next_obs))
      self.obs = next_obs

    num_samples = self.env.num_envs * self.samples_per_epoch #number of environment times the sample of batches
    reshape_fn = lambda x: x.view(num_samples, - 1) #reshape
    '''torch.stacks -> stacks each variable to each other
    such as obs with each other and makes them into tensor matrices'''
    batch = map(torch.stack, zip(*transitions)) 
    '''reshapes the batches into number of environment * sample of batches per epoch'''
    obs_b , loc_b, scale_b, action_b, reward_b, done_b, next_obs_b = map(reshape_fn, batch)
    #obs_b = (num_samples, num_envs, feature_dims) ->(num_samples * num_envs, feature_dims) 3d -> 2d

    for repeat in range(self.epoch_repeat):
      idx = list(range(num_samples))
      random.shuffle(idx)

      for i in idx:
        yield obs_b[i], loc_b[i], scale_b[i], action_b[i], reward_b[i], done_b[i], next_obs_b[i]

