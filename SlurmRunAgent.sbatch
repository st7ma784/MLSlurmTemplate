#!/bin/bash

# set a job name
#SBATCH --job-name=second_wandb_trial_batchv0
# a file for job output, you can check job progress
#SBATCH --time=24:00:00

#SBATCH --gres=gpu:1   # also requests portion of CPU and Memory
#SBATCH --nodes=1
#SBATCH --signal=USR1@60 
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com
#SBATCH --account=bdlanXX 
#SBATCH --partition=gpu
export SLURM_NNODES=$SLURM_JOB_NUM_NODES
export CONDADIR=/nobackup/projects/bdlanXX/$USER
#export wandb='YOURWANDBAPIKEYHERE FOR AUTO LOGIN'

source $CONDADIR/miniconda/etc/profile.d/conda.sh
conda activate $CONDADIR/miniconda/envs/open-ce # ...and activate the conda environment

#source activate open-ce
#This assumes your script auto downloads data to the location specified. 
#Alternatively you could manually downlaod it here, but runs will be faster if your script can cache data that is already there. 
python3 /nobackup/projects/%PROJECT/$USER/YOURFOLDER/runSweep.py  --data_dir /nobackup/projects/$PROJET/$USER/data
