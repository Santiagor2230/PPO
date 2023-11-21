# PPO
Implementation of PPO

# Requirements
gym == 0.23.0

brax==0.0.12

jax == 0.4.14

jaxlib=0.3.14

pytorch-lightining == 1.6.0

torch == 2.0.1

# Collab installations
!apt-get install -y xvfb

!pip install gym==0.23.1 \
    pytorch-lightning==1.6 \
    pyvirtualdisplay

!pip install -U brax==0.0.12 jax==0.3.14 jaxlib==0.3.14+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


# Description
PPO is a policy method that approximates its policy to take deterministic actions on a continous action space, it takes into account the important steps that can get closer and closer to the most optimal weigthts for giving the highest possible reward and this is part of trust region policy optimization. It stabelizes its distribution of the reward by utilizing a clipping method that will not overshoot the adjustments of the weights through backpropogation. PPO is also considered an actor-critic meaning that it has the actor which in this case is the policy that is trying to get the most optimal path with the highest reward while the critic is trying to improve the policy performance by providing important information of the state and how high of a value that state has.

# Environment
robotic_ant

# Architecture
Regular PPO

# optimizer
Policy: AdamW
Value: Adam

# loss function
Policy: clipped surrogate objective functio
Value: smooth L1 loss function

# Video Result:
https://github.com/Santiagor2230/PPO/assets/52907423/770bc0c1-cc0f-41d7-a7cb-6bf8dbfc6259

