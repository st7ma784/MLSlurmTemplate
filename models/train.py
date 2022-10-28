

from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
from functools import partial
from typing import Optional
from warnings import warn

class myLightningModule(LightningModule):
    
    def __init__(self,
                learning_rate,
                total_steps: int = 200000,
                train_batch_size: int = 64,
                eval_batch_size: int = 32,
                eval_splits: Optional[list] = None,
                context_length= 77,
                **kwargs,
                ):

        super().__init__()
        self.save_hyperparameters()
        self.loss=torch.nn.CrossEntropyLoss()

        self.model=torch.nn.Sequential(*[
            torch.nn.Linear(50000,512),
            torch.nn.Linear(512,256),
            torch.nn.Linear(256,512),
            torch.nn.Linear(512,40000)

        ])
    def forward(self,input):
        return self.model(input)

    def training_step(self, batch, batch_idx,optimizer_idx=0):
        input,desired=batch[0],batch[1]
        input=torch.nn.functional.one_hot(input,num_classes=self.input_vocab_size).to(torch.float)
        out=self.forward(input)
        loss=self.loss(out,desired)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
      
      
    def validation_step(self, batch, batch_idx):
      
        input,desired=batch[0],batch[1]
        input=torch.nn.functional.one_hot(input,num_classes=self.input_vocab_size).to(torch.float)
        out=self.forward(input)
        print(out[0])
        print(desired[0])

    def configure_optimizers(self):
        
        optimizerA = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate, eps=1e-8)
        return [optimizerA]
