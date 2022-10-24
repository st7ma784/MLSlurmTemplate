  
import wandb
from models.train import train
import argparse
import wandb
from functools import partial
wandb.login()
parser = argparse.ArgumentParser()
parser.add_argument("--sweep_id", default="2kt9gdh0",nargs="?", type=str)
parser.add_argument("--data_dir", default="/Data",nargs="?", type=str)
p = parser.parse_args()
train=partial(train,dir=p.data_dir)
wandb.agent(sweep_id=p.sweep_id, project=<WANDBPROJECTNAME>, entity=<WANDBUSERNAME>,function=train)
