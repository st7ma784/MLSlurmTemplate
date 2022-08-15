# MLModelDev
A template repo for creating a new model 

## This repo details how to create a clean code base for a model. 
We'll use pytorchlightning for its hardware agnotsticism and abstraction of trivial decisions. 

## Deployment : 
This repo is designed to complement the  st7ma784/MLModelDeploy repo. there you will find code for deploying this to HEC/SLURM/ other.
### WE Need: 
A Train script. 
A Sweep script.
A script containing a DataModule
PIP/CONDA env

## Best Practice:

Pytorch lightning extends pytorch to separate data and logic. 

### Datasets: 
These fall into DataModules which handle the preprocessing, loading and downloading of the dataset if it doesn't already exist (would not rely on this though) 

### LightningModules:

These are simple model scripts that automate where your model sits> Best practice is to only take KWARGS that directly change the model structure, I.e Layers, hidden_dim.  NOT Batchsize LR etc...  

