from Launch import *
def SlurmRun(args):

    job_with_version = '{}v{}'.format("SINGLEGPUTESTLAUNCH", 0)

    sub_commands =['#!/bin/bash',
        '#SBATCH --time={}'.format( '1:00:00'),# Max run time
        '#SBATCH --job-name={}'.format(job_with_version), 
        '#SBATCH --nodes=1',  #Nodes per experiment
        '#SBATCH --ntasks-per-node=1',# Set this to GPUs per node. 
        '#SBATCH --gres=gpu:1',  #{}'.format(per_experiment_nb_gpus),
        f'#SBATCH --signal=USR1@{5 * 60}',
        '#SBATCH --mail-type={}'.format(','.join(['END','FAIL'])),
        '#SBATCH --mail-user={}'.format('email@gmail.com'),
        '#SBATCH --partition={}'.format('debug'),
    ]
    comm="python"
    slurm_commands={}

    if str(os.getenv("HOSTNAME","localhost")).endswith("bede.dur.ac.uk"):
        sub_commands.extend([
                '#SBATCH --account YOURACCOUNT',
                'export CONDADIR=/nobackup/projects/bdlan05/$USER/miniconda',
                'export NCCL_SOCKET_IFNAME=ib0'])
        comm="python3"
    else: 
        sub_commands.extend(['export CONDADIR=/home/user/miniconda3',
                             'export NCCL_SOCKET_IFNAME=enp0s31f6',])
    sub_commands.extend([ '#SBATCH --{}={}\n'.format(cmd, value) for  (cmd, value) in slurm_commands.items()])
    sub_commands.extend([
        'export SLURM_NNODES=$SLURM_JOB_NUM_NODES',
        #'export wandb=<YOURWANDBAPIKEY>',
        'source $CONDADIR/etc/profile.d/conda.sh',
        'conda activate $CONDADIR/envs/open-ce',# ...and activate the conda environment
    ])
    script_name= os.path.realpath(sys.argv[0]) #Find this scripts name...
    sub_commands.append('srun {} {} --sweep {} --dir {}'.format(comm, script_name, args.sweep, args.dir))
    sub_commands = [x.lstrip() for x in sub_commands]        

    full_command = '\n'.join(sub_commands)
    return full_command


if __name__ == '__main__':
    #read args 
    import argparse
    from functools import partial
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trials', type=int, default=-1, help='how many runs')
    parser.add_argument('--dir', type=str, default=None, help='where to save')
    parser.add_argument('--sweep', type=str, default=None, help='Sweepid')
    args=parser.parse_args()
    NumTrials=args.num_trials
    #HOSTNAME=login2.bede.dur.ac.uk check we arent launching on this node
    runfunc=partial(wandbtrain,dir=args.dir)
    if NumTrials==-1:
        
        print("Running trial: single wandbsweep")
        import wandb
        if args.sweep:
            print("Using sweepid",parser.parse_args().sweep)
            wandb.agent(parser.parse_args().sweep, function=runfunc, count=1, project="WasteDetection",entity="st7ma784")
        else:       
            print("No sweepid given, exiting")
    elif NumTrials ==0 and not str(os.getenv("HOSTNAME","localhost")).startswith("login"): #We'll do a trial run...
        #means we've been launched from a BEDE script, so use config given in args///
        print("Running trial: single wandbtrain")
        import wandb
        from WandBSweep import make_sweep
        if args.sweep:
            print("Using sweepid",parser.parse_args().sweep)
            wandb.agent(parser.parse_args().sweep, function=runfunc, count=1, project="YOURPROJECT",entity="YOURWANDBUSER")
        else:
            print("No sweepid given, exiting")

    #OR To run with Default Args
    else:
        for i in range(NumTrials):             
            command=SlurmRun(args)
            slurm_cmd_script_path = os.path.join(".","slurm_cmdtrial{}.sh".format(i))
            with open(slurm_cmd_script_path, "w") as f:
              f.write(command)
            print('\nlaunching exp...')
            
            result = call('{} {}'.format("sbatch", slurm_cmd_script_path), shell=True)
            if result == 0:
                print('launched exp ', slurm_cmd_script_path)
            else:
                print('launch failed...')  
