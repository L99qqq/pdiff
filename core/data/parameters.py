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
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
import pytorch_lightning as pl
import numpy as np
from core.utils.utils import *
import torch.nn as nn
from core.data.vision_dataset import VisionData
import re

class dataset():
    def __init__(self,cfg,original_name,cond1_name,cond2_name,kind):
        self.original_name=original_name
        self.cond1_name=cond1_name
        self.cond2_name=cond2_name
        self.original_root = "param_data/{}/data.pt".format(original_name)
        self.cond1_root = "param_data/{}/data.pt".format(cond1_name)
        self.cond2_root = "param_data/{}/data.pt".format(cond2_name)
        self.k = cfg.system.datasets.k
        self.cfg=cfg

        # check the original_root path is  exist or not
        assert os.path.exists(self.original_root), f'{self.original_root} not exists'

        # check the original_root is a directory or file
        if os.path.isfile(self.original_root):
            state = torch.load(self.original_root, map_location='cpu')
            self.fix_model = state['model']
            self.fix_model.eval()
            self.fix_model.to('cpu')
            self.fix_model.requires_grad_(False)
            if kind=="test":
                self.original_data = state['pdata'][:self.k]
                self.accuracy = state["performance"][: self.k]
            elif kind=="train":
                self.original_data = state["pdata"][: self.k * 7 // 10]
                self.accuracy = state["performance"][: self.k * 7 // 10]
            elif kind=="val":
                self.original_data = state["pdata"][self.k * 7 // 10 : self.k]
                self.accuracy = state["performance"][self.k * 7 // 10 : self.k]
            self.train_layer = state['train_layer']
        elif os.path.isdir(self.original_root):
            print(f'{self.original_root} is directory')

        # check the cond1_root path is  exist or not
        assert os.path.exists(self.cond1_root), f'{self.cond1_root} not exists'

        # check the cond1_root is a directory or file
        if os.path.isfile(self.cond1_root):
            state = torch.load(self.cond1_root, map_location='cpu')
            self.fix_model_cond1 = state['model']
            self.fix_model_cond1.eval()
            self.fix_model_cond1.to('cpu')
            self.fix_model_cond1.requires_grad_(False)
            if kind == "test":
                self.cond1_data = state["pdata"][: self.k]
                self.cond1_accuracy = state["performance"][: self.k]
            elif kind == "train":
                self.cond1_data = state["pdata"][: self.k * 7 // 10]
                self.cond1_accuracy = state["performance"][: self.k * 7 // 10]
            elif kind == "val":
                self.cond1_data = state["pdata"][self.k * 7 // 10 : self.k]
                self.cond1_accuracy = state["performance"][self.k * 7 // 10 : self.k]
        elif os.path.isdir(self.cond1_root):
            print(f"{self.cond1_root} is directory")

        # check the cond2_root path is  exist or not
        assert os.path.exists(self.cond2_root), f'{self.cond2_root} not exists'

        # check the cond2_root is a directory or file
        if os.path.isfile(self.cond2_root):
            state = torch.load(self.cond2_root, map_location='cpu')
            self.fix_model_cond2 = state['model']
            self.fix_model_cond2.eval()
            self.fix_model_cond2.to('cpu')
            self.fix_model_cond2.requires_grad_(False)
            if kind == "test":
                self.cond2_data = state["pdata"][: self.k]
                self.cond2_accuracy = state["performance"][: self.k]
            elif kind == "train":
                self.cond2_data = state["pdata"][: self.k * 7 // 10]
                self.cond2_accuracy = state["performance"][: self.k * 7 // 10]
            elif kind == "val":
                self.cond2_data = state["pdata"][self.k * 7 // 10 : self.k]
                self.cond2_accuracy = state["performance"][self.k * 7 // 10 : self.k]
        elif os.path.isdir(self.cond2_root):
            print(f"{self.cond2_root} is directory")

        # print(kind," original dataset's mean performance:",np.mean(self.accuracy))
        # print(kind," cond1 dataset's mean performance:",np.mean(self.cond1_accuracy))
        # print(kind," cond2 dataset's mean performance:",np.mean(self.cond2_accuracy))
        
        # self.task_data = VisionData(self.cfg.task.data)
        # self.test_loader = self.task_data.test_dataloader()
        # self.test_dataset()

    # def test_dataset(self):
    #     print("testing datasets")
    #     accs_original = []
    #     accs_cond1 = []
    #     accs_cond2 = []
    #     for i in range(len(self.original_data)):
    #         acc, test_loss, output_list = self.test_model(
    #             self.original_data[i].to("cuda"), self.fix_model, self.train_layer
    #         )
    #         accs_original.append(acc)
    #         acc, test_loss, output_list = self.test_model(
    #             self.cond1_data[i].to("cuda"), self.fix_model_cond1, self.train_layer
    #         )
    #         accs_cond1.append(acc)
    #         acc, test_loss, output_list = self.test_model(
    #             self.cond2_data[i].to("cuda"), self.fix_model_cond2, self.train_layer
    #         )
    #         accs_cond2.append(acc)

    #     print("\noriginal model name:",self.original_name,"\noriginal models accuracy:", accs_original)
    #     print("original models mean accuracy:", np.mean(accs_original),"best accuracy:", np.max(accs_original),"median accuracy:", np.median(accs_original))
    #     print("cond1 model name:",self.cond1_name,"\ncond1 models accuracy:", accs_cond1)
    #     print("cond1 models mean accuracy:", np.mean(accs_cond1),"best accuracy:", np.max(accs_cond1),"median accuracy:", np.median(accs_cond1))
    #     print("cond2 model name:",self.cond2_name,"\ncond2 models accuracy:", accs_cond2)
    #     print("cond2 models mean accuracy:", np.mean(accs_cond2),"best accuracy:", np.max(accs_cond2),"median accuracy:", np.median(accs_cond2))
    #     print("total mean accuracy:",(np.mean(accs_original)+np.mean(accs_cond1)+np.mean(accs_cond2))/3)

    # def test_model(self,param,net,train_layer):
    #     target_num = 0
    #     for name, module in net.named_parameters():
    #         if name in train_layer:
    #             target_num += torch.numel(module)
    #     params_num = torch.squeeze(param).shape[0]  # + 30720
    #     assert (target_num == params_num)
    #     param = torch.squeeze(param)
    #     model = partial_reverse_tomodel(param, net, train_layer).to(param.device)
    #     model.eval()
    #     test_loss = 0
    #     correct = 0
    #     total = 0
    #     output_list = []
    #     with torch.no_grad():
    #         for data, target in self.test_loader:
    #             data, target = data.cuda(), target.cuda()
    #             output = model(data)
    #             target = target.to(torch.int64)
    #             test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
    #             total += data.shape[0]
    #             pred = torch.max(output, 1)[1]
    #             output_list += pred.cpu().numpy().tolist()
    #             correct += pred.eq(target.view_as(pred)).sum().item()
    #     test_loss /= total
    #     acc = 100. * correct / total
    #     del model
    #     return acc, test_loss, output_list


class PData(pl.LightningModule):
    def __init__(self, cfg, **kwargs):
        super(PData, self).__init__(**kwargs)
        self.cfg=cfg
        self.batch_size = cfg.system.datasets.batch_size
        self.num_workers = cfg.system.datasets.num_workers
        self.split_epoch=cfg.system.train.split_epoch
        self.train_data_dict=self.cfg.system.datasets.train_dataset
        self.val_data_dict=self.cfg.system.datasets.val_dataset
        self.test_data_dict=self.cfg.system.datasets.test_dataset
        self.train_datasets={}
        for original_name,cond_names in self.train_data_dict.items():
            class_start, class_end = re.findall(r"(\d+)_(\d+)$", original_name)[-1]
            class_start, class_end=int(class_start), int(class_end)
            self.train_datasets[(class_start, class_end)]=dataset(cfg,original_name,cond_names[0],cond_names[1],"train")
        self.val_datasets = {}
        for original_name, cond_names in self.train_data_dict.items():
            class_start, class_end = re.findall(r"(\d+)_(\d+)$", original_name)[-1]
            class_start, class_end=int(class_start), int(class_end)
            self.val_datasets[(class_start, class_end)] = dataset(
                cfg, original_name, cond_names[0], cond_names[1],"val"
            )
        # else:
        #     self.val_datasets={}
        #     for original_name,cond_names in self.val_data_dict.items():
        #         self.val_datasets[original_name]=dataset(cfg,original_name,cond_names[0],cond_names[1])
        self.test_datasets={}
        for original_name,cond_names in self.test_data_dict.items():
            class_start, class_end = re.findall(r"(\d+)_(\d+)$", original_name)[-1]
            class_start, class_end=int(class_start), int(class_end)
            self.test_datasets[(class_start, class_end)]=dataset(cfg,original_name,cond_names[0],cond_names[1],"test")   

        self.k=cfg.system.datasets.k
        self.ae_dataset_size = cfg.system.datasets.k * 3 
        self.ae_train_Parameters=Parameters(self.train_datasets, "ae",cfg)
        self.ddpm_train_Parameters=Parameters(self.train_datasets, "ddpm",cfg)
        
        if self.cfg.system.datasets.val_batch_size >= len(self.val_datasets):
            self.ae_val_Parameters=Parameters(self.val_datasets, "ae", cfg,self.cfg.system.datasets.val_batch_size//len(self.val_datasets))  
            self.ddpm_val_Parameters=Parameters(self.val_datasets, "ddpm",cfg,self.cfg.system.datasets.val_batch_size//len(self.val_datasets))
        else:
            self.ae_val_Parameters=Parameters(self.val_datasets, "ae",cfg, split=1,use_front=self.cfg.system.datasets.val_batch_size)  
            self.ddpm_val_Parameters=Parameters(self.val_datasets, "ddpm",cfg,split=1,use_front=self.cfg.system.datasets.val_batch_size)
        
        if self.cfg.system.datasets.test_batch_size >= len(self.test_datasets):
            self.ae_test_Parameters=Parameters(self.test_datasets, "ae",cfg,self.cfg.system.datasets.test_batch_size//len(self.test_datasets))  
            self.ddpm_test_Parameters=Parameters(self.test_datasets, "ddpm",cfg,self.cfg.system.datasets.test_batch_size//len(self.test_datasets))   
        else:
            self.ae_test_Parameters=Parameters(self.test_datasets, "ae",cfg,split=1,use_front=self.cfg.system.datasets.test_batch_size)  
            self.ddpm_test_Parameters=Parameters(self.test_datasets, "ddpm",cfg,split=1,use_front=self.cfg.system.datasets.test_batch_size)
        
        self.ae_train_dataloader=DataLoader(
                self.ae_train_Parameters,
                batch_size=min(
                    self.batch_size, self.ae_dataset_size * len(self.train_datasets)
                ),
                num_workers=self.num_workers,
                shuffle=True,
                drop_last=True,
                pin_memory=False,
                persistent_workers=True,
            )
        self.ae_val_dataloader=DataLoader(
                self.ae_val_Parameters,
                batch_size=min(
                    self.cfg.system.datasets.val_batch_size,
                    self.ae_dataset_size * len(self.val_datasets),
                ),
                num_workers=self.num_workers,
                shuffle=False,
                persistent_workers=True,
            )
        self.ae_test_dataloader=DataLoader(
                self.ae_test_Parameters,
                batch_size=min(
                    self.cfg.system.datasets.test_batch_size,
                    self.ae_dataset_size * len(self.test_datasets),
                ),
                num_workers=self.num_workers,
                shuffle=False,
                persistent_workers=True,
            )
        self.ddpm_train_dataloader=DataLoader(
                self.ddpm_train_Parameters,
                batch_size=min(
                    self.batch_size,
                    self.k * len(self.train_datasets),
                ),
                num_workers=self.num_workers,
                shuffle=True,
                drop_last=True,
                pin_memory=False,
                persistent_workers=True,
            )
        self.ddpm_val_dataloader=DataLoader(
                self.ddpm_val_Parameters,
                batch_size=min(
                    self.batch_size,
                    self.k * len(self.val_datasets),
                ),
                num_workers=self.num_workers,
                shuffle=False,
                persistent_workers=True,
            )
        self.ddpm_test_dataloader=DataLoader(
                self.ddpm_test_Parameters,
                batch_size=min(
                    self.batch_size,
                    self.k * len(self.test_datasets),
                ),
                num_workers=self.num_workers,
                shuffle=False,
                persistent_workers=True,
            )

    def get_train_layer(self,class_start, class_end):
        if (class_start, class_end) in self.train_datasets.keys():
            return self.train_datasets[(class_start, class_end) ].train_layer
        else:
            return self.test_datasets[(class_start, class_end) ].train_layer

    def get_model(self,class_start, class_end):
        if (class_start, class_end) in self.train_datasets.keys():
            return self.train_datasets[(class_start, class_end) ].fix_model
        else:
            return self.test_datasets[(class_start, class_end) ].fix_model

    def get_accuracy(self,class_start, class_end):
        if (class_start, class_end) in self.train_datasets.keys():
            return self.train_datasets[(class_start, class_end) ].accuracy , self.train_datasets[(class_start, class_end) ].cond1_accuracy, self.train_datasets[(class_start, class_end) ].cond2_accuracy
        else:
            return (
                self.test_datasets[(class_start, class_end) ].accuracy,
                self.test_datasets[(class_start, class_end) ].cond1_accuracy,
                self.test_datasets[(class_start, class_end) ].cond2_accuracy,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.current_epoch < self.split_epoch:
            return self.ae_train_dataloader
        else:
            return self.ddpm_train_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.current_epoch < self.split_epoch:
            return self.ae_val_dataloader
        else:
            return self.ddpm_val_dataloader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if self.current_epoch < self.split_epoch:
            return self.ae_test_dataloader
        else:
            return self.ddpm_test_dataloader


# 这样存的空间效率很低，original_name被存了很多次
# ae: [[original_name,originale_name,...],[pdatas]]
# ddpm: [(original_name,pdata,cond1,cond2),...]
class Parameters(VisionDataset):
    def __init__(self, datasets,type,config,split=None,use_front=None):
        super(Parameters, self).__init__(root=None, transform=None, target_transform=None)
        self.data=[]
        for original_name, dataset in datasets.items():
            if use_front is not None:
                if use_front > 0:
                    use_front=use_front-1
                else:
                    break
            if type == "ddpm":
                # ddpm data size = n*(k*k*k)
                # self.data.extend([(original_name, i, j, k) for i in dataset.original_data for j in dataset.cond1_data for k in dataset.cond2_data])
                # ddpm data size = n*k
                class_start, class_end = original_name[0],original_name[1]
                self.data.extend(
                    [
                        (
                            class_start,
                            class_end,
                            dataset.original_data[i],
                            dataset.cond1_data[i],
                            dataset.cond2_data[i],
                        )
                        for i in range(len(dataset.original_data) if split is None else split)
                    ]
                )
            elif type == "ae":
                class_start, class_end = original_name[0],original_name[1]
                self.data.extend([(class_start,class_end, dataset.original_data[i]) for i in range(len(dataset.original_data) if split is None else split)])
                self.data.extend([(class_start,class_end, dataset.cond1_data[i]) for i in range(len(dataset.cond1_data) if split is None else 0)])
                if config.system.ae_model_cond2 is None:
                    self.data.extend([(class_start,class_end, dataset.cond2_data[i]) for i in range(len(dataset.cond2_data) if split is None else 0)])
        # print("Parameters initial dataset len:",len(self.data))

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self) -> int:
        return len(self.data)
