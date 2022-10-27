
from torchvision import transforms
from PIL import Image
import torch     
import os
import zipfile
from pySmartDL import SmartDL
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset
from torchvision.datasets import CocoCaptions
T= transforms.Compose([transforms.Resize((224,224),interpolation=Image.NEAREST),transforms.ToTensor()])
from transformers import AutoTokenizer
import time
tokenizer= AutoTokenizer.from_pretrained("gpt2")
tokenizer.vocab["</s>"] = tokenizer.vocab_size -1
tokenizer.pad_token = tokenizer.eos_token 


## An Example dataset, needs to implement a torch.data.utils.Dataset. This one automatically loads COCO for us from MSCOCO annotations, which we extend to include our own tokenizer

class myDataset(CocoCaptions):
    def __init__(self, root, annFile, *args, **kwargs):
        print('Loading COCO dataset')
        #check if root and annfile exist
        if not os.path.exists(root):
            print('Error: root directory does not exist: {}'.format(root))
            return None
        if not os.path.exists(annFile):
            print('Error: annFile does not exist: {}'.format(annFile))
            return None
        #if using the HEC, you may want a Path(annFile).touch() call here to update the altered time if your files are stored in $global_scratch
        super().__init__(root, annFile, *args, **kwargs)
        print('Done')
    def __getitem__(self, index: int):
        try:
            img, target= super().__getitem__(index)
        except Exception as e:
            print(e)
            print('Error loading image:', index)
            return None
        target=torch.cat([tokenizer(
                    sent,                      # Sentence to encode.
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = 77,           # Pad & truncate all sentences.
                    padding = "max_length",
                    truncation=True,
                    return_attention_mask = False,   # Construct attn. masks.
                    return_tensors = 'pt',     # Return pytorch tensors.
                )['input_ids'] for sent in target[:5]],dim=0)
        return img,target


# Dataset

class myDataModule(pl.LightningDataModule):
    ## This dataModule takes care of downloading the data per node and then PL may replace the sampler if doing distributed multi-node training. 
    ## Some settings here may be worth editing if on a machine where Pin memory, or workers are limited. 
    def __init__(self, Cache_dir='./', T=None, batch_size=256):
        super().__init__()
        self.data_dir = Cache_dir
        self.ann_dir=os.path.join(self.data_dir,"annotations")
        self.batch_size = batch_size
        self.T=T
        self.splits={"train":[],"val":[],"test":[]}
        
    def train_dataloader(self):
        if not hasattr(self, 'train'):
            if os.path.exists("train.pt"):
                self.train=torch.load("train.pt")
            else:
                self.download_data()
        # IF you know that you're only ever using 1 gpu (HEC /local runs only...) then consider using https://lightning-bolts.readthedocs.io/en/latest/dataloaders/async.html
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True,drop_last=True)
    def val_dataloader(self):
        if not hasattr(self, 'val'):
            #check if esprevalidation.pt exists in the directory
            if os.path.exists("val.pt"):
                self.val_dataset=torch.load("val.pt")
            else:
                self.download_data()
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True,drop_last=True)
    def test_dataloader(self):
        if not hasattr(self, 'test'):
            #check for espretest.pt in the directory
            if os.path.exists("test.pt"):
                self.test=torch.load("test.pt")
            else:

                self.download_data()

        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True,drop_last=True)
    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # # download data
        #If using the HEC, consider altering this to call directly the cmdline WGET/CURL  and then unzip -DD to modify the dates. 
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir,exist_ok=True)
        if not os.path.exists(self.ann_dir):
            os.makedirs(self.ann_dir,exist_ok=True)
        urls=['http://images.cocodataset.org/zips/train2014.zip',
                'http://images.cocodataset.org/zips/val2014.zip',
                'http://images.cocodataset.org/zips/test2015.zip',
                'http://images.cocodataset.org/zips/train2017.zip',
                'http://images.cocodataset.org/zips/val2017.zip',
                'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
                ]

        objs=[]
        for url in urls:
            print("url:",url)
            name=str(url).split('/')[-1]
        
            
            location=self.data_dir # if name.startswith("annotations") else self.ann_dir
            #print("Location", location) #/Data/train2014.zip
            #time.sleep(5)
            #print('Downloading',url)
            if name.endswith(".zip"):
                name=name[:-4]
            if name.startswith("train"):
                self.splits['train'].append(name)
            elif name.startswith("val"):
                self.splits['val'].append(name)
            elif name.startswith("test"):
                self.splits['test'].append(name)
            obj=SmartDL(url,os.path.join(location,str(url).split('/')[-1]),progress_bar=False)
            obj.FileName=name
            if not os.path.exists(obj.get_dest()):

                objs.append(obj)#SmartDL(url, self.data_dir,)
                obj.start(blocking=False)
                print("obj Path ",obj.get_dest())
        for obj in objs:
            while not obj.isFinished():
                #print("Speed: %s" % obj.get_speed(human=True))
                print("Eta: %s" % obj.get_eta(human=True))
                time.sleep(5)
            if obj.isSuccessful():
                print("Downloaded: %s" % obj.get_dest())

            path = obj.get_dest()
            if obj.FileName.startswith("annotations"):
                print("Extracting annotations")
                print("path:",path)

                with zipfile.ZipFile(path, 'r') as zip_ref:
                    try:
                        zip_ref.extractall(self.data_dir)
                    except:
                        print("Error extracting annotations")
                        print("path:",path)
                        print("ann_dir:",self.ann_dir)
            #wget.download("http://images.cocodataset.org/zips/train2014.zip",out=self.cocodir)
            else:
                print("Extracting images")
                print("path:",path)
                if obj.FileName.endswith(".zip"):
                    print("Extracting zip")
                    with zipfile.ZipFile(path, 'r') as zip_ref:
                        try:
                            zip_ref.extractall(self.data_dir)
                        except:
                            print("Error extracting images")
                            print("path:",path)
                            print("data_dir:",self.data_dir)
                print("Extracted: %s" % path)

 
    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        print("Entered COCO datasetup")
        if stage == 'fit' or stage is None:
            TrainSets=[]
            for version in self.splits['train']:
                
                annfile=os.path.join(self.ann_dir,'{}_{}.json'.format('captions',version))
                dir=os.path.join(self.data_dir,version)

                #time.sleep(2)
                dset=myDataset(root=dir, annFile=annfile, transform=self.T)
                assert(len(dset)>0)
                TrainSets.append(dset)
            self.train = ConcatDataset(TrainSets)

            ValSets=[]
            for version in self.splits['val']:
                print("BUILDING SPLIT : ",version)
                
                annfile=os.path.join(self.ann_dir,'{}_{}.json'.format('captions',version))
                dir=os.path.join(self.data_dir,version)
                print("annfile:",annfile)
                print("dir:",dir)
                ValSets.append(myDataset(root=dir, annFile=annfile, transform=self.T))
            self.val = ConcatDataset(ValSets)
            # torch.save(self.train,"train.pt")
            # torch.save(self.val,"val.pt")    
        if stage == 'test' or stage is None:
            TestSets=[]
            for version in self.splits['test']:
                print("BUILDING SPLIT : ",version)
                annfile=os.path.join(self.ann_dir,'{}_{}.json'.format('captions',version))
                dir=os.path.join(self.data_dir,version)
                
                print("annfile:",annfile)
                print("dir:",dir)
                TestSets.append(myDataset(root=dir, annFile=annfile, transform=self.T))
            self.test = ConcatDataset(TestSets)
