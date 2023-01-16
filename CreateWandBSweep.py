import wandb
wandb.login()
if __name__=="__main__":
    sweep_config = {
        'method': 'random',  # Randomly sample the hyperparameter space (alternatives: grid, bayes)
        'metric': {  # This is the metric we are interested in maximizing
            'name': 'loss',
            'goal': 'minimize'   
        },
        'parameters': {
            'learning_rate': {
                'values':[2e-4,1e-4,5e-5,2e-5]
            },
            'batch_size': {
                'values': [16,24,32]
            },
            'precision': {
                'values': ['bf16']
            },
            'embed_dim':{
                'values': [128,256,512]
            }, 
            'transformer_width':{
                'values': [128,256,512]
            },
            'transformer_heads':{
                'values': [8,16,32]
            },
            'transformer_layers':{
                'values': [4,5,6]
            },
        }
    }

    # Create the sweep
    sweep_id = wandb.sweep(sweep_config, project="6DimCachespliteinSweep", entity="st7ma784")
    print(sweep_id)
