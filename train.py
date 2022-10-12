
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
        # BUILD YOUR MODEL HERE!! 
        # self.model = transformer(layers=layers, width=width)
        #And May i suggest? 
        #self.initialize_parameters()
        
        #The following is for Validation with CKA, comparing activations to a pretrained model... see validation steps for more. 
        # You could always rrewrite this for just a trainstep with different splits. 
        self.handles=[]
        self.model1_info={'Name':"SelfCLIP",'Layers':[]}
        self.model2_info={'Name': "Stock CLIP", 'Layers':[]}
        
        #And define our loss... 
        self.loss=torch.nn.CrossEntropyLoss()
##################################################
        ###############Some good practive ideas???
##################################################
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
      
        return [optimizer] #,[scheduler]


#############################################
    ##################Logic step of our model
#############################################
    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx,optimizer_idx=0):
    
        im,captions= batch[0],batch[1]
        outputs=self(im)
        loss = self.loss(outputs,captions)
        self.log('train_loss', loss,prog_bar=True)
        return {"loss": loss}
###############################################
    ####Some validation with CKA for autoencoders (If you have a baseline model)
###############################################

    def orig_HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.
        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N=K.shape[0]
        return torch.add(torch.trace(K@L),torch.div(torch.sum(K)*torch.sum(L)/(N - 1) - (torch.sum(K@L) * 2 ), (N - 2)))
        
    def on_validation_epoch_start(self):
        self.eval()
        self.freeze()
    #     #import clip model here]
        self.model2,_ = clip.load("ViT-B/32", device=self.device)
        self._insert_hooks()
        self.eval()
        self.model2.eval()


    def validation_step(self,batch,*args):

        self.model1_features = {}  #reset list of forward hooks
        self.model2_features = {}  
        self.encode_image(batch[0]) #run through main mode
        ###If your model has supervised data, then perhaps do a loss with your date here!
        self.model2.encode_image(batch[0])# to compare supervision model
        N = len(self.model1_features.values())
        M = len(self.model2_features.values())
        print("N",N)
        print("M",M)
        out=torch.stack([self.orig_HSIC(K, K) for K in self.model1_features.values()])
        self.hsic_matrix0=torch.add(self.hsic_matrix0,out) if hasattr(self, 'hsic_matrix0') else out
        out=torch.stack([self.orig_HSIC(L, L) for L in self.model2_features.values()])
        self.hsic_matrix2=torch.add(self.hsic_matrix2,out) if hasattr(self, 'hsic_matrix2') else out
        out=torch.stack([self.orig_HSIC(K, L) for K in self.model1_features.values() for L in self.model2_features.values()])
        self.hsic_matrix1=torch.add(self.hsic_matrix1,out.reshape(N,M)) if hasattr(self, 'hsic_matrix1') else out.reshape(N,M)
        self.hsic_matrix = self.hsic_matrix1 / (torch.sqrt(self.hsic_matrix0.unsqueeze(1))*torch.sqrt(self.hsic_matrix2.unsqueeze(0)))
        if not torch.isnan(self.hsic_matrix).any():
            warn("HSIC computation resulted in NANs")
            
    def on_validation_epoch_end(self,):
        self.unfreeze()
        self.train()
        self.plot_results("HSIC{}.jpg".format(self.current_epoch))
        if self.logger is not None:
            self.logger.log_image(key="HSIC{}".format(self.current_epoch), images=["HSIC{}.jpg".format(self.current_epoch)])
        for handle in self.handles:
            handle.remove()
        del self.model2

    def _log_layer(self, model: str, name: str, layer: nn.Module,inp: torch.Tensor, out: torch.Tensor):
        with torch.no_grad():
            if isinstance(out, tuple):
                out = out[0]
            #if activation shape is the same as dataloader batch size, then it is a linear layer
            if out.shape[0] == self.hparams.train_batch_size:
                print("LOGGING : ", model, name, out.shape)
                if model == "model1":
                    X = out.flatten(1)
                    self.model1_features[name] = (X @ X.t()).fill_diagonal_(0)
                elif model == "model2":
                    X = out.flatten(1)
                    self.model2_features[name] = (X @ X.t()).fill_diagonal_(0)
                else:
                    raise RuntimeError("Unknown model name for _log_layer.")

    def _insert_hooks(self):
       
        for name, layer in self.named_modules():
            self.handles.append(layer.register_forward_hook(partial(self._log_layer, "model1", name)))
      
        for name, layer in self.model2.named_modules():
            self.handles.append(layer.register_forward_hook(partial(self._log_layer, "model2", name)))
       
  
    def export(self):
        """
        Exports the CKA data along with the respective model layer names.
        :return:
        """
        return {
            "model1_name": "Trained",
            "model2_name": "PretrainedModel",
            "CKA": self.hsic_matrix,
            "model1_layers": self.named_modules(),
            "model2_layers": self.model2.named_modules(),
        }

    def plot_results(self,
                     save_path: str = None,
                     title: str = None):
        fig, ax = plt.subplots()
        im = ax.imshow(self.hsic_matrix.cpu(), origin='lower', cmap='magma')
        ax.set_xlabel(f"Layers {self.model2_info['Name']}", fontsize=15)
        ax.set_ylabel(f"Layers {self.model1_info['Name']}", fontsize=15)

        if title is not None:
            ax.set_title(f"{title}", fontsize=18)
        else:
            ax.set_title(f"{self.model1_info['Name']} vs {self.model2_info['Name']}", fontsize=18)

        add_colorbar(im)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

    
#####UTILS 

from mpl_toolkits import axes_grid1
import matplotlib.pyplot as plt

def add_colorbar(im, aspect=10, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)
    
#############################################
    #Code to call run with and without loggers (W+B)
#############################################


def wandbtrain(config=None,dir="/Data",devices="auto",accelerator="auto",Dataset=None):
    if config is not None and not isinstance(config,dict):
        #print("Config is not a dict")
        config=config.__dict__
        #print("as dict: {}".format(config))
    logtool= pytorch_lightning.loggers.WandbLogger( project="MYPROJECY",entity="WANDBUSER",experiment=config, save_dir=dir)
    dir=config.get("dir",dir)
    train(config,dir,devices,accelerator,Dataset,logtool)
    
def train(config={
        "batch_size":16,
        "learning_rate":2e-3,
        "precision":16,
    },dir="/Data",devices="auto",accelerator="auto",Dataset=None,logtool=None):
    
    from DataModule import myDataModule
    
        #This is a great logging tool for HEC, but may not work if using the demoparse with SLURM
    model=myLightningModule(  learning_rate = config["learning_rate"],
                                train_batch_size=config["batch_size"],
                                adam_epsilon = 1e-8)
    Dataset=myDataModule(Cache_dir=dir,batch_size=config["batchsize"])
    callbacks=[
        TQDMProgressBar(),
        EarlyStopping(monitor="loss", mode="min",patience=10,check_finite=True,stopping_threshold=0.001),
    ]
    p=config.get("precision",'bf16')
    if isinstance(p,str):
        p='bf16' if p=="bf16" else int(p)
    trainer=pytorch_lightning.Trainer(
            devices=devices,
            accelerator="auto",
            max_epochs=100,
            logger=logtool,
            callbacks=callbacks,
            gradient_clip_val=0.25,
            num_nodes=int(os.getenv("SLURM_JOB_NUM_NODES",1)), # A way of auto scaling down the line if planning to  use slurm/BEDE/HEC - feel free to ignore !
            strategy="ddp",
            fast_dev_run=False,
            precision=p
    )
        
    trainer.fit(model,Dataset)

    
    
if __name__ == '__main__':
    config={
        "batchsize":12,         #[1,4,8,16,32,64]
        "learning_rate":4e-6,   #[2e-4,1e-4,5e-5,2e-5,1e-5,4e-6]
        "precision":16,         #[32,16,'bf16']
    }
    train(config)
