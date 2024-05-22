import torch.nn as nn
from torchvision.datasets.vision import VisionDataset
import os
import torch
import pdb
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from .base import DataBase
from torch.utils.data import Dataset
import warnings
import os

class dataset():
    def __init__(self,cfg,original_name,cond1_name,cond2_name):
        self.original_name=original_name
        self.original_root = "param_data/{}/data.pt".format(original_name)
        self.cond1_root = "param_data/{}/data.pt".format(cond1_name)
        self.cond2_root = "param_data/{}/data.pt".format(cond2_name)
        self.k=getattr(cfg, 'k', 200)
        self.batch_size=getattr(cfg, 'batch_size', self.k)
        
        # check the original_root path is  exist or not
        assert os.path.exists(self.original_root), f'{self.original_root} not exists'
        
        # check the original_root is a directory or file
        if os.path.isfile(self.original_root):
            state = torch.load(self.original_root, map_location='cpu')
            self.fix_model = state['model']
            self.fix_model.eval()
            self.fix_model.to('cpu')
            self.fix_model.requires_grad_(False)

            self.original_data = state['pdata']
            self.accuracy = state['performance']
            self.train_layer = state['train_layer']
        elif os.path.isdir(self.original_root):
            pass
        
        # check the cond1_root path is  exist or not
        assert os.path.exists(self.cond1_root), f'{self.cond1_root} not exists'
        
        # check the cond1_root is a directory or file
        if os.path.isfile(self.cond1_root):
            state = torch.load(self.cond1_root, map_location='cpu')
            self.cond1_data = state['pdata']
            self.cond1_accuracy = state['performance']
        elif os.path.isdir(self.cond1_root):
            pass
        
        # check the cond2_root path is  exist or not
        assert os.path.exists(self.cond2_root), f'{self.cond2_root} not exists'
        
        # check the cond2_root is a directory or file
        if os.path.isfile(self.cond2_root):
            state = torch.load(self.cond2_root, map_location='cpu')
            self.cond2_data = state['pdata']
            self.cond2_accuracy = state['performance']
        elif os.path.isdir(self.cond2_root):
            pass
        
        

class PData(DataBase):
    def __init__(self, cfg, type, **kwargs):
        super(PData, self).__init__(cfg, **kwargs)
        if(type=="ddpm"):
            self.train_data_dict=self.cfg.system.datasets.train_dataset
            self.val_data_dict=self.cfg.system.datasets.val_dataset
            self.test_data_dict=self.cfg.system.datasets.test_dataset
            self.train_datasets={}
            for original_name,cond_names in self.train_data_dict:
                self.train_datasets[original_name]=dataset(cfg,original_name,cond_names[0],cond_names[1])
            if self.val_data_dict is None:
                self.val_datasets=self.train_datasets
            else:
                self.val_datasets=[]
                for original_name,cond_names in self.val_data_dict:
                    self.val_datasets[original_name]=dataset(cfg,original_name,cond_names[0],cond_names[1])
            self.test_datasets=[]
            for original_name,cond_names in self.test_data_dict:
                self.test_datasets[original_name]=dataset(cfg,original_name,cond_names[0],cond_names[1])   
        
    def get_train_layer(self,original_name):
        return self.train_datasets[original_name].train_layer

    def get_model(self,original_name):
        return self.train_datasets[original_name].fix_model

    def get_accuracy(self,original_name):
        return self.train_datasets[original_name].accuracy , self.train_datasets[original_name].cond1_accuracy, self.train_datasets[original_name].cond2_accuracy

    @property
    def train_dataset(self):
        return Parameters(self.train_datasets)

    @property
    def val_dataset(self):
        return Parameters(self.val_datasets)

    @property
    def test_dataset(self):
        return Parameters(self.test_datasets)


class Parameters(VisionDataset):
    def __init__(self, datasets):
        super(Parameters, self).__init__(root=None, transform=None, target_transform=None)
        self.data=[]
        for original_name,dataset in datasets:
            self.data.append((dataset.original_data,dataset.cond1_data,dataset.cond2.data))
        
    def __getitem__(self, item):
        return self.data[item]

    def __len__(self) -> int:
        return len(self.data)