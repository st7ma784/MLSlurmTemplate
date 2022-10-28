from test_tube import HyperOptArgumentParser

class parser(HyperOptArgumentParser):
    def __init__(self,*args,strategy="random_search",**kwargs):

        super().__init__( *args,strategy=strategy, add_help=False) # or random search
        self.add_argument("--dir",default="/nobackup/projects/<YOURBEDEPROJECT>/$USER/data",type=str)
        self.add_argument("--log_path",default="/nobackup/projects/<YOURBEDEPROJECT>/$USER/logs/",type=str)
        self.opt_list("--learning_rate", default=0.0001, type=float, options=[2e-4,1e-4,5e-5,1e-5,4e-6], tunable=True)
        self.opt_list("--batch_size", default=80, type=int)
        
        #INSERT YOUR OWN PARAMETERS HERE
        self.opt_list("--codeversion",default=-1,options=[1,2,3,4,5,6])
        self.opt_list("--precision", default=16, options=[16], tunable=False)
        self.opt_list("--accelerator", default='gpu', type=str, options=['gpu'], tunable=False)
        self.opt_list("--num_trials", default=0, type=int, tunable=False)
        #self.opt_range('--neurons', default=50, type=int, tunable=True, low=100, high=800, nb_samples=8, log_base=None)
        
        #This is important when passing arguments as **config in launcher
        self.argNames=["dir","log_path","learning_rate","batch_size","modelname","precision","codeversion","accelerator","num_trials"]
    def __dict__(self):
        return {k:self.parse_args().__dict__[k] for k in self.argNames}


        
# Testing to check param outputs
if __name__== "__main__":
    
    #If you call this file directly, you'll see the default ARGS AND the trials that might be generated. 
    myparser=parser()
    hyperparams = myparser.parse_args()
    print(hyperparams.__dict__)
    for trial in hyperparams.generate_trials(10):
        print(trial)
        
