# Getting Started With ML...


## This repository contains code for VERY fast CI/CD of code. 

## QUICK START: 

### Step one: 
  - Put your dataset into the DataModule.py file.
  - This details how to load your data onto the server. An example is given for MSCOCO.
### Step two:
  - Edit the model files.
  - build or load an existing model in the __init__ function
  - Include the logic for the forward, train_step, and optionally validation_step too.

  More info for code structure at Pytorch-lightning: https://github.com/Lightning-AI/lightning/tree/master/examples/convert_from_pt_to_pl
### Step three:
  - edit the demoparse.py file for the differnet parameters your model will have. 
  
### Step four:
  - debug locally with '''python Launch.py --dir <YOUR DATA LOCATION>''' 
  - (you may wish to edit Launch to just call train() directly, not the wandbtrain() - in the long run strongly recommend using WandB to track experiments)

### Step Five: 
  - When happy with code functionality, deploy the code.
  - Login to your favourite SLURM cluster - Such as the N8/Bede cluster, (HEC T.B.D) or an on-prem cluster
  - type into the CLI '''python Launch.py --dir <SLURM DATA DIR> --num_trials <WhatEverYouFancy>'''
### And Finally:
  - To Launch, call python Launch.py 
  - --num_trials > 0 means the option in the argparser are randomly selected, and jobs are queued on the Cluster, 
  - --num_trials == 0 is use the default args, this is the standard behaviour, and is useful for debugging. You can manually specify arguements in the commandline to overwrite the default. 
  - --num_trials == -1 makes the local node pick a random config -- great for checking before landing on a server somewhere. 

## Other Files: 
#### A Train script - models/train.py 
This is where we define the core logic, note that PytorchLightning doesn't need us to specify hardware for our runs! there's no .cuda() calls - if absoutely needed you may need the occassional .to(self.device) but you may be doing something wrong if thats the case. 

In this file, we've put some exmaple blocks for inspiration to get you going including CKA Alignment etc. (home brew implementation --- if this is needed as a core part of research, it may be worth satisfying yourself that the optimizations done are equivalent to the maths in the paper) 

#### A DataModule.py
This contains all the logic surrrounding how our dataset is downloaded and/or preprocessed. Would recommend putting anything CPU-bound prep wise or non-training like tokenizations in here! This can be tricky to debug, and we'd recommend sticking closely to the example, and checking whether files exist before re-downloading. 

#### Good practice...
When deploying code, consider using git deploy keys, giving individual servers(/login nodes) the ability to pull your code.

#### PIP/CONDA env
Where we store all the libraries needed, by default this is wandb, PL, pySmartDL (for speedy dataset downloads) 
This can be generated from a working env with 
```
conda env export > environment.yml
```
When deploying, it's worth following your clusters' own guides for environment setup. Define in the requirements just the necessary PIP packages if you can. 

##Hex deployment 
To do.... dockercompose with GPUS?

##LLMS 

It's more than likely if you're here, your doing research on big models. You may find it useful to explore the FSDPLaunch file, for submitting models too big for a single node!

(remember for caching and files: on hec/bede
storage for moderate size data (up to 100G), scratch for larger data (up to 10TB))

