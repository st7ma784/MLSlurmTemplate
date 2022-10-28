

from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
from functools import partial
from typing import Optional
import clip
from warnings import warn
import matplotlib.pyplot as plt
from CKA_test import add_colorbar 


class LightningCLIPModule(LightningModule):
    
    def __init__(self,
                
                learning_rate,
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

        self.loss=torch.nn.CrossEntropyLoss(reduction='mean')
        
        #Your Model Goes HERE! 
        #self.initialize_parameters()
        self.handles=[]
        self.model1_info={'Name':"SelfCLIP",}
        self.model2_info={'Name': "Stock CLIP",}
        self.naninfcount=0
        
    # def initialize_parameters(self):
    #     nn.init.normal_(self.token_embedding.weight, std=0.02)
    #     nn.init.normal_(self.positional_embedding, std=0.01)

    ############# DO SOME CKA VALIDATION STEPS! #############
    def batch_HSIC2(self,K):
        a=torch.sum(K,dim=-1)
        b=torch.sum(K,dim=-2)
        c=torch.sub(torch.pow(torch.sum(a,dim=-1),2)/(K.shape[1] - 1),torch.sum(a*b,dim=1),alpha=2)
        output=torch.add(torch.einsum('a...->a',torch.pow(K,2)),torch.div(c,(K.shape[1] - 2)))
        return output
                
    def batch_HSIC3(self,K,L):
        a=torch.sum(L,dim=-1)
        b=torch.sum(K,dim=-2)
        c=torch.sub(torch.sum(b,dim=-1)*torch.sum(a,dim=-1)/(K.shape[1] - 1),torch.sum(b*a,dim=1),alpha=2)
        return torch.add(torch.einsum('abc->a',K*L),torch.div(c,(K.shape[1] - 2)))

    def on_validation_epoch_start(self):
        self.eval()
        self.naninfcount=0
        self.model2,_ = clip.load("ViT-B/32", device=self.device)
        self.model2.eval()
        self._insert_hooks()
        self.eval()
        
    def validation_step(self,batch,*args):

        self.model1_features = {}  #reset list of forward hooks
        self.model2_features = {}  #reset list of forward hooks
        self.encode_image(batch[0]) #run through main mode
        self.encode_text(batch[1][:,0])

        self.model2.encode_image(batch[0])# to compare supervision model
        self.model2.encode_text(batch[1][:,0])
        a=torch.stack(list(self.model1_features.values()))
        if not hasattr(self,'hsic_matrix0'):
            self.hsic_matrix0=torch.zeros((a.shape[0]),device=self.device)
        self.hsic_matrix0=torch.add(self.hsic_matrix0,self.batch_HSIC2(a)) 
        a=torch.stack(list(self.model2_features.values()))
        if not hasattr(self,'hsic_matrix2'):
            self.hsic_matrix2=torch.zeros((a.shape[0]),device=self.device)
        self.hsic_matrix2=torch.add(self.hsic_matrix2,self.batch_HSIC2(a))
        joint_HSIC=torch.stack(list(map(lambda X: self.batch_HSIC3(a,X),list(self.model1_features.values()))))
        if not hasattr(self,'hsic_matrix1'):
            self.hsic_matrix1=torch.zeros(joint_HSIC.shape,device=self.device)
        self.hsic_matrix1=torch.add(self.hsic_matrix1,joint_HSIC) 
    def on_validation_epoch_end(self,):
        self.unfreeze()
        self.train()
        self.plot_results("HSIC{}.jpg".format(self.current_epoch))
        if self.logger is not None:
            self.logger.log_image(key="HSIC{}".format(self.current_epoch), images=["HSIC{}.jpg".format(self.current_epoch)])
        for handle in self.handles:
            handle.remove()
        print(self.naninfcount)
        del self.model2
        del self.hsic_matrix0
        del self.hsic_matrix1
        del self.hsic_matrix2
        

    def _log_layer(self, model: str, name: str, layer: nn.Module,inp: torch.Tensor, out: torch.Tensor):
        if isinstance(out, tuple):
            out=out[0]       
            # print("permuted")
        if out.shape[0] == self.hparams.train_batch_size:
            self.__store(out,name,model,layer)
        
        elif out.shape[1] == self.hparams.train_batch_size:
            self.__store(out.permute(1,0,*torch.arange(len(out.shape)-2)+2),name,model,layer)

    def __store(self,out,name, model,layer):
        X = out.flatten(1)
        X= (X @ X.t()).fill_diagonal_(0)
        if (torch.isnan(X).any() or torch.isinf(X).any()):
            self.naninfcount+=1
            if self.current_epoch==0 and hasattr(layer, 'weight'):
                nn.init.normal_(layer.weight, std=0.02)
        if model == "model1":
            #if name already exists in dictionary, change name to name+1
            while name in self.model1_features:
                name=name+"1"
            self.model1_features[name] = X

        elif model == "model2":
            while name in self.model1_features:
                name=name+"1"
            self.model2_features[name] = X

        else:
            raise RuntimeError("Unknown model name for _log_layer.")
    def _insert_hooks(self):
        self.handles=[]
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model1", name)) for name, layer in self.encode_image.named_modules()])
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model1", name)) for name, layer in self.encoder.named_modules()])
        a=len(self.handles)
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model2", name)) for name, layer in self.model2.visual.named_modules()])
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model2", name)) for name, layer in self.model2.transformer.named_modules()])
        b=len(self.handles)-a
        return a,b
  
    def export(self):
        """
        Exports the CKA data along with the respective model layer names.
        :return:
        """
        return {
            "model1_name": "Trained",
            "model2_name": "PretrainedModel",
            "CKA":self.hsic_matrix1 / (torch.sqrt(self.hsic_matrix0.unsqueeze(1))*torch.sqrt(self.hsic_matrix2.unsqueeze(0))),
            "model1_layers": self.named_modules(),
            "model2_layers": self.model2.named_modules(),
        }

    def plot_results(self,
                     save_path: str = None,
                     title: str = None):
        fig, ax = plt.subplots()
        t=self.hsic_matrix0.unsqueeze(1)*self.hsic_matrix2.unsqueeze(0)
        print(torch.sum(torch.abs(t)==t))
        r=torch.sqrt(torch.abs(t))
        r[torch.abs(t)==-t]=-r[torch.abs(t)==-t]
        hsic_matrix = self.hsic_matrix1 / r
        if not torch.isnan(hsic_matrix).any():
            warn("HSIC computation resulted in NANs")
        im = ax.imshow(hsic_matrix.cpu(), origin='lower', cmap='magma')
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
    ###### FINISHED CKA STEPS ######
    
    def training_step(self, batch, batch_idx,optimizer_idx=0):
      
        labels=batch[0]
        images=batch[1]
        output=self.forward(images)
        loss=self.loss(output,labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #        self.backward(0)
            
    def configure_optimizers(self):
        
        optimizerA = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
      

        return [optimizerA]
