
import torch     
import os
import pytorch_lightning as pl
from transformers import (
  BertTokenizerFast,
  CLIPTokenizer
)
os.environ["TOKENIZERS_PARALLELISM"]='true'

class CNDataModule(pl.LightningDataModule):

    def __init__(self, Cache_dir='.', batch_size=256,ZHtokenizer=None,ENtokenizer=None):
        super().__init__()
        self.data_dir = Cache_dir
        self.batch_size = batch_size
        if ZHtokenizer is None:
            self.ZHtokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese',cache_dir=self.data_dir)
        else :
            self.ZHtokenizer = ZHtokenizer
        if ENtokenizer is None:

            self.ENtokenizer =CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32",cache_dir=self.data_dir)
        else:
            self.ENtokenizer = ENtokenizer
    # def train_dataloader(self, B=None):
    #     if B is None:
    #         B=self.batch_size 
    #     return torch.utils.data.DataLoader(self.train, batch_size=B, shuffle=True, num_workers=2, prefetch_factor=2, pin_memory=True,drop_last=True)
    
    # def val_dataloader(self, B=None):
    #     if B is None:
    #         B=self.batch_size
       
    #     return torch.utils.data.DataLoader(self.val, batch_size=B, shuffle=False, num_workers=2, prefetch_factor=2, pin_memory=True,drop_last=True)
    
    def test_dataloader(self,B=None):
        if B is None:
            B=self.batch_size
        return torch.utils.data.DataLoader(self.test, batch_size=B, shuffle=True, num_workers=1, prefetch_factor=1, pin_memory=True,drop_last=True)
    
    def prepare_data(self):

        '''called only once and on 1 GPU'''
        # # download data
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir,exist_ok=True)
        from datasets import load_dataset

        self.dataset = load_dataset("msr_zhen_translation_parity", 
                               cache_dir=self.data_dir,
                               streaming=False,
                               )
   
    def tokenization(self,sample):

        return {'en' : self.ENtokenizer(sample["en"], padding="max_length", truncation=True, max_length=77),
                'zh' : self.ZHtokenizer(sample["zh"], padding="max_length", truncation=True, max_length=77)}

    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        #print("Entered COCO datasetup")
        from datasets import load_dataset

        if not self.dataset:
            self.dataset= load_dataset("un_pc","en-zh",
                                 cache_dir=self.data_dir,
                                 streaming=True,
                                 )
        
        self.test = self.dataset["train"]["translation"].map(lambda x: self.tokenization(x), batched=False)
        
        
if __name__=="__main__":


    import argparse
    from tqdm import tqdm
    parser = argparse.ArgumentParser(description='location of data')
    parser.add_argument('--data', type=str, default='/data', help='location of data')
    args = parser.parse_args()
    print("args",args)
    datalocation=args.data
    datamodule=MagicSwordCNDataModule(Cache_dir=datalocation,annotations=os.path.join(datalocation,"annotations"),batch_size=2)

    datamodule.download_data()
    datamodule.setup()
    dl=datamodule.train_dataloader()
    for batch in tqdm(dl):
        print(batch)
