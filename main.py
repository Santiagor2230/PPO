import torch
from training import PPO
from pytorch_lightning import Trainer

model = PPO("brax-ant-v0")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_gpus = torch.cuda.device_count()


trainer = Trainer(gpus=num_gpus, max_epochs=500)

trainer.fit(model)

model.videos[9]