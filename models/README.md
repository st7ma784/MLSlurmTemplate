# How To... write a train file

Most AI models fall into the same architecture, consisting of a forwards and backwards pass. 

we write the models in a single file, train.py, which defines the architecture of the models, and the logic involved in calculating the loss of each forward pass. 

Pytorch Lightning takes care of everything else: The optimization, devices and scheduling among other things. This is why we extend the pl.LightningModule Class. 

## To get started: 
 - 1 > Define Model architecutre, This usually goes into the __init__ function. You may want to add any loading from pre-trained weights, and consider functions like nn._init_normal()
 - 2 > Write your forward() function.  This is the forward inference of the model. Attention mechanisms should go in here! 
 - 3 > a train_step() function. Writing the part of the model that includes taking an item from batch, performing a forward step, and calculating Loss. 
 - OPTIONAL 4 > define a validation_step() - You may want a different behaviour during valuation, whether different logic. Different data is handled by the Dataloader. Pytorch Lightning supports other functions to, In the Example, you'll see how on_validation_epoch_start and on_validation_epoch_end are used. 
 - 5 > In you're launch.py file, it might be worth defining behaviours from args, whether that's changing which train.py file to import, or different datasets to load. 
