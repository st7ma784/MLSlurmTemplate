from test_tube import HyperOptArgumentParser

class myParser(HyperOptArgumentParser):
    def __init__(self,*args):

        super().__init__(*args, strategy="grid_search", add_help=False) # or random search
        #more info at https://williamfalcon.github.io/test-tube/hyperparameter_optimization/HyperOptArgumentParser/

        self.opt_list("--learning_rate", default=0.001, type=float, options=[2e-4,1e-4,5e-5,2e-5,1e-5,4e-6], tunable=True)
        self.opt_list("--batchsize", default=8, type=float, options=[1,2,4,8,12,16, 32, 64], tunable=True)
        self.opt_list("--precision", default=16, options=[32,16,'bf16'], tunable=False)
        #self.opt_range('--neurons', default=50, type=int, tunable=True, low=100, high=800, nb_samples=8, log_base=None)




if __name__== "__main__":
    parser=myParser()
    hyperparams = parser.parse_args()
    print(hyperparams)
    from train import train
    for trial in hyperparams.trials(num=10):

        train(trial)

