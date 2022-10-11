# MLModelDev
A template repo for creating a new model

## This repo details how to create a clean code base for a model. 
We'll use pytorchlightning for its hardware agnotsticism and abstraction of trivial decisions. 
If you want help converting Pytorch to Pytorch-lightning: https://github.com/Lightning-AI/lightning/tree/master/examples/convert_from_pt_to_pl


## Deployment : 
#### SLURM and SWEEP scripts. 
If you pull this repo straight into HEC/SLURM, you'll find these useful. These scripts define sets of behaviours for you to launch lots of experiments at the same time with online logging. 

Wandb - is an online interface for logging. It's worth making an account and adding the API key (pip install wandb && wandb login) 
Terminology - Sweep - a set of hyperparameters to test - its worth being selective with tests as otherwise they can take ages! 
            - Agent - A worker that pulls sets of configs and trials them. 

SLURM - is a software framework for deploying workloads on HPC / multi-machine environments. there are several available clusters for this: HEC, N8/Bede and finally a small infolab one primarily for debugging. 
      - the SlurmTrain.py script works differently to the sbatch scripts that spin up WandB agents. Instead it enumerates a set of configs and launches all of them concurrently. Use with caution as this is a great way to annoy other users if the resource is hogged with many runs. However, for small sweeps of several parameters it works well. (and gets results faster than multiple agents will)



#### A Train script. 
This is where we define the core logic, note that PL doesn't need us to specify hardware for our runs! there's no .cuda() calls - if absoutely needed you may need the occassional to(self.device) but you may e doing something wrong if thats the case. 

In this file, we've put some useful blocks for inspiration to get you going including CKA Alignment etc. (home brew implementation --- if this is needed as a core part of research, it may be worth satisfying yourself that the optimizations done are equivalent to the maths in the paper) 


#### A Sweep script. (CREATESWEEP + RUN SWEEP) 
The sweep defines the settings we wish to trial, there are different methods, things that can be tracked and it's all passed through a dict to the train script

#### A script containing a DataModule
This contains all the logic surrrounding how our dataset is downloaded and/or preprocessed. Would recommend putting anything CPU-bound prep wise or non-training like tokenizations in here! 

#### PIP/CONDA env
Where we store all the libraries needed, by default this is wandb, PL, pySmartDL (for speedy dataset downloads) 
This can be generated from a working env with 
```
conda env export > environment.yml
```
## Best Practice:

Pytorch lightning extends pytorch to separate data and logic. It is whats used to handle hardware agnosticism. It also makes implementing "bag-of-tricks" and "Bag-of-freebies" like Gradient Clipping and gradient accumulation as simple as single line callbacks. See how clean the logging code is compared to most pure pytorch ML implementations. 

### Datasets: 
These fall into DataModules which handle the preprocessing, loading and downloading of the dataset if it doesn't already exist. It's worth verifying functionality on a local machine first before deployment to HEC/BEDE. See examples. 

### LightningModules:

These are simple model scripts that automate where your model sits> Best practice is to only take KWARGS that directly change the model structure, I.e Layers, hidden_dim.  It is NOT recommended to pass things like Batchsize here, your models should ideally be agnostic to batch sizing. See the PL docs for all the different configuration options. 

We've included a series of validation steps to show off some functionality and offer a faster implementation of CKA that aren't trivial for users to implement themselves. It is very probable that many users may prefer to simply repeat the trainsteps in validation instead and use a different data split as handled by their datamodule. 

