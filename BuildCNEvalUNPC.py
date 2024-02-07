
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

    #do COCODataModule
    # import json
    #show keys in the data/annotations/captions_train2014.json file
    # print(json.load(open("/data/annotations/captions_train2014.json")).keys())
    # print(json.load(open("/data/annotations/captions_train2014.json"))['images'][0])
    # print(json.load(open("/data/annotations/instances_train2014.json"))['categories'])
    '''
        dict_keys(['info', 'images', 'licenses', 'annotations', 'categories'])
        {'segmentation': [[312.29, 562.89, 402.25, 511.49, 400.96, 425.38, 398.39, 372.69, 388.11, 332.85, 318.71, 325.14, 295.58, 305.86, 269.88, 314.86, 258.31, 337.99, 217.19, 321.29, 182.49, 343.13, 141.37, 348.27, 132.37, 358.55, 159.36, 377.83, 116.95, 421.53, 167.07, 499.92, 232.61, 560.32, 300.72, 571.89]], 'area': 54652.9556, 'iscrowd': 0, 'image_id': 480023, 'bbox': [116.95, 305.86, 285.3, 266.03], 'category_id': 58, 'id': 86}

        {'supercategory': 'person', 'id': 1, 'name': 'person'}, {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}, {'supercategory': 'vehicle', 'id': 3, 'name': 'car'}, {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'}, {'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'}, {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'}, {'supercategory': 'vehicle', 'id': 7, 'name': 'train'}, {'supercategory': 'vehicle', 'id': 8, 'name': 'truck'}, {'supercategory': 'vehicle', 'id': 9, 'name': 'boat'}, {'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light'}, {'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant'}, {'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign'}, {'supercategory': 'outdoor', 'id': 14, 'name': 'parking meter'}, {'supercategory': 'outdoor', 'id': 15, 'name': 'bench'}, {'supercategory': 'animal', 'id': 16, 'name': 'bird'}, {'supercategory': 'animal', 'id': 17, 'name': 'cat'}, {'supercategory': 'animal', 'id': 18, 'name': 'dog'}, {'supercategory': 'animal', 'id': 19, 'name': 'horse'}, {'supercategory': 'animal', 'id': 20, 'name': 'sheep'}, {'supercategory': 'animal', 'id': 21, 'name': 'cow'}, {'supercategory': 'animal', 'id': 22, 'name': 'elephant'}, {'supercategory': 'animal', 'id': 23, 'name': 'bear'}, {'supercategory': 'animal', 'id': 24, 'name': 'zebra'}, {'supercategory': 'animal', 'id': 25, 'name': 'giraffe'}, {'supercategory': 'accessory', 'id': 27, 'name': 'backpack'}, {'supercategory': 'accessory', 'id': 28, 'name': 'umbrella'}, {'supercategory': 'accessory', 'id': 31, 'name': 'handbag'}, {'supercategory': 'accessory', 'id': 32, 'name': 'tie'}, {'supercategory': 'accessory', 'id': 33, 'name': 'suitcase'}, {'supercategory': 'sports', 'id': 34, 'name': 'frisbee'}, {'supercategory': 'sports', 'id': 35, 'name': 'skis'}, {'supercategory': 'sports', 'id': 36, 'name': 'snowboard'}, {'supercategory': 'sports', 'id': 37, 'name': 'sports ball'}, {'supercategory': 'sports', 'id': 38, 'name': 'kite'}, {'supercategory': 'sports', 'id': 39, 'name': 'baseball bat'}, {'supercategory': 'sports', 'id': 40, 'name': 'baseball glove'}, {'supercategory': 'sports', 'id': 41, 'name': 'skateboard'}, {'supercategory': 'sports', 'id': 42, 'name': 'surfboard'}, {'supercategory': 'sports', 'id': 43, 'name': 'tennis racket'}, {'supercategory': 'kitchen', 'id': 44, 'name': 'bottle'}, {'supercategory': 'kitchen', 'id': 46, 'name': 'wine glass'}, {'supercategory': 'kitchen', 'id': 47, 'name': 'cup'}, {'supercategory': 'kitchen', 'id': 48, 'name': 'fork'}, {'supercategory': 'kitchen', 'id': 49, 'name': 'knife'}, {'supercategory': 'kitchen', 'id': 50, 'name': 'spoon'}, {'supercategory': 'kitchen', 'id': 51, 'name': 'bowl'}, {'supercategory': 'food', 'id': 52, 'name': 'banana'}, {'supercategory': 'food', 'id': 53, 'name': 'apple'}, {'supercategory': 'food', 'id': 54, 'name': 'sandwich'}, {'supercategory': 'food', 'id': 55, 'name': 'orange'}, {'supercategory': 'food', 'id': 56, 'name': 'broccoli'}, {'supercategory': 'food', 'id': 57, 'name': 'carrot'}, {'supercategory': 'food', 'id': 58, 'name': 'hot dog'}, {'supercategory': 'food', 'id': 59, 'name': 'pizza'}, {'supercategory': 'food', 'id': 60, 'name': 'donut'}, {'supercategory': 'food', 'id': 61, 'name': 'cake'}, {'supercategory': 'furniture', 'id': 62, 'name': 'chair'}, {'supercategory': 'furniture', 'id': 63, 'name': 'couch'}, {'supercategory': 'furniture', 'id': 64, 'name': 'potted plant'}, {'supercategory': 'furniture', 'id': 65, 'name': 'bed'}, {'supercategory': 'furniture', 'id': 67, 'name': 'dining table'}, {'supercategory': 'furniture', 'id': 70, 'name': 'toilet'}, {'supercategory': 'electronic', 'id': 72, 'name': 'tv'}, {'supercategory': 'electronic', 'id': 73, 'name': 'laptop'}, {'supercategory': 'electronic', 'id': 74, 'name': 'mouse'}, {'supercategory': 'electronic', 'id': 75, 'name': 'remote'}, {'supercategory': 'electronic', 'id': 76, 'name': 'keyboard'}, {'supercategory': 'electronic', 'id': 77, 'name': 'cell phone'}, {'supercategory': 'appliance', 'id': 78, 'name': 'microwave'}, {'supercategory': 'appliance', 'id': 79, 'name': 'oven'}, {'supercategory': 'appliance', 'id': 80, 'name': 'toaster'}, {'supercategory': 'appliance', 'id': 81, 'name': 'sink'}, {'supercategory': 'appliance', 'id': 82, 'name': 'refrigerator'}, {'supercategory': 'indoor', 'id': 84, 'name': 'book'}, {'supercategory': 'indoor', 'id': 85, 'name': 'clock'}, {'supercategory': 'indoor', 'id': 86, 'name': 'vase'}, {'supercategory': 'indoor', 'id': 87, 'name': 'scissors'}, {'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear'}, {'supercategory': 'indoor', 'id': 89, 'name': 'hair drier'}, {'supercategory': 'indoor', 'id': 90, 'name': 'toothbrush'}]
    '''
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
