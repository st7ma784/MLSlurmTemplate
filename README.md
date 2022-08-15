# MLModelDev
A template repo for creating a new model 

## This repo details how to create a clean code base for a model. 
We'll use pytorchlightning for its hardware agnotsticism and abstraction of trivial decisions. 

## Deployment : 
This repo is designed to complement the  st7ma784/MLModelDeploy repo. there you will find code for deploying this to HEC/SLURM/ other.
### WE Have: 

#### A Train script. 
This is where we define the core logic, note that PL doesn't need us to specify hardware for our runs! there's no .cuda() calls - if absoutely needed you may need the occassional to(self.device) but you may e doing something wrong if thats the case. 

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

Pytorch lightning extends pytorch to separate data and logic. 

### Datasets: 
These fall into DataModules which handle the preprocessing, loading and downloading of the dataset if it doesn't already exist (would not rely on this though) 

### LightningModules:

These are simple model scripts that automate where your model sits> Best practice is to only take KWARGS that directly change the model structure, I.e Layers, hidden_dim.  NOT Batchsize LR etc...  

