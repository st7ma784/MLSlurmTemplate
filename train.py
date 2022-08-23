
import pytorch_lightning
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
from typing import Dict, Optional
from pytorch_lightning.callbacks import TQDMProgressBar
import wandb


class myLightningModule(LightningModule):
    def __init__(self,
                learning_rate: float = 2e-4,
                adam_epsilon: float = 1e-8,
                warmup_steps: int = 0,
                weight_decay: float = 0.0,
                total_steps: int = 200000,
                train_batch_size: int = 64,
                eval_batch_size: int = 32,
                eval_splits: Optional[list] = None,
                embed_dim= 512,
                context_length= 77,
                vocab_size= 50257,
                transformer_width= 512,
                transformer_heads= 32,
                transformer_layers= 4,
                **kwargs,
                ):

        super().__init__()
        self.save_hyperparameters()
        
        # self.model = transformer(layers=layers, width=width)
        
        self.loss=torch.nn.CrossEntropyLoss()

    def initialize_parameters(self):
        
        proj_std = (self.model.width ** -0.5) * ((2 * self.model.layers) ** -0.5)
        attn_std = self.model.width ** -0.5
        fc_std = (2 * self.model.width) ** -0.5

        for block in self.model.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.text_projection, std=self.encoder.width ** -0.5)
  

    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx,optimizer_idx=0):
    
        im,captions= batch[0],batch[1]
        outputs=self(im)
        loss = self.loss(outputs,captions)
        self.log('train_loss', loss,prog_bar=True)
        return {"loss": loss}

            
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
      
        return [optimizer] #,[scheduler]

def wandbtrain(config={
        "batchsize":16,
        "learning_rate":2e-4,
        "precision":16,
    },dir="/Data",devices="auto",):
    with wandb.init( project="demoProject", entity="st7ma784", job_type="train", config=config) as run:  
        logtool= pytorch_lightning.loggers.WandbLogger(experiment=run)
        train(config, dir,devices,logtool)

def train(config={
        "batchsize":16,
        "learning_rate":2e-4,
        "precision":16,
    },dir="/Data",devices="auto",logtool=None):
    
    if not isinstance(config, dict):
        config= vars(config) # this oversees conversion from namespaces from argparsers to dict as might be passed by wandb

    #Load Data Module and begin training
    from DataModule import myDataModule
    
        #This is a great logging tool for HEC, but may not work if using the demoparse with SLURM
    model=myLightningModule(  learning_rate = config["learning_rate"],
                                train_batch_size=config["batchsize"],
                                adam_epsilon = 1e-8)
    Dataset=myDataModule(Cache_dir=dir,batch_size=config["batchsize"])
    callbacks=[
        TQDMProgressBar()
    ]
    trainer=pytorch_lightning.Trainer(
        devices=devices,
        accelerator="auto",
        max_epochs=100,
        logger=logtool,
        callbacks=callbacks,
        gradient_clip_val=0.25,
        precision=config["precision"]
    )
    trainer.fit(model,Dataset)

if __name__ == '__main__':
    config={
        "batchsize":12,         #[1,4,8,16,32,64]
        "learning_rate":4e-6,   #[2e-4,1e-4,5e-5,2e-5,1e-5,4e-6]
        "precision":16,         #[32,16,'bf16']
    }
    train(config)
