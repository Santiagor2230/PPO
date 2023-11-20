import torch
from brax.io import html
from IPython.display import HTML

@torch.no_grad()
def create_video(env, episode_length, policy=None):
  qp_array = []
  state = env.reset()
  for i in range(episode_length):
    if policy:
      loc, scale = policy(state)
      sample = torch.normal(loc, scale)
      action = torch.tanh(sample)
    else:
      action = env.action_space.sample()
    state, _, _, _ = env.step(action)
    qp_array.append(env.unwrapped._state.qp)
  return HTML(html.render(env.unwrapped._env.sys, qp_array))


@torch.no_grad()
def test_agent(env, episode_length, policy, episodes=10):

  ep_returns = []
  for ep in range(episodes):
    state = env.reset()
    done = False
    ep_ret = 0.0

    while not done:
      loc, scale = policy(state)
      sample = torch.normal(loc, scale)
      action = torch.tanh(sample)   
      state, reward, done, info = env.step(action)
      ep_ret += reward.item()

    ep_returns.append(ep_ret)

  return sum(ep_returns) / episodes 