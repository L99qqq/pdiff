import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
from .base import DataBase
import torch
import timm
import numpy as np
import re
from torch.utils.data import Dataset
from .CustomImageDataset import CustomImageDataset

class CustomDataset(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        self.targets = dataset.targets # 保留targets属性
        self.classes = dataset.classes # 保留classes属性
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        x = self.dataset[self.indices[item]][0]
        y = self.targets[self.indices[item]]
        # x = self.dataset[item][0]
        # y = self.dataset[item][1]
        return x, y


class VisionData(DataBase):
    def __init__(self, cfg, original_name=None, **kwargs):
        super(VisionData, self).__init__(cfg, **kwargs)

        """
        init data for classification task
        we firstly load the dataset and then define the transform for the dataset
        args:
            cfg: the config file
        
        cfg args:
            data_root: the root path of the dataset
            dataset: the dataset name
            batch_size: the batch size
            num_workers: the number of workers
            
        """
        super(VisionData, self).__init__(cfg, **kwargs)
        self.root = getattr(self.cfg, "data_root", "./data")
        self.dataset = getattr(self.cfg, "dataset", "cifar10")
        self.cfg = cfg
        self.classes_used_to_train = cfg.classes_used_to_train
        self.group_used_to_train = cfg.group_used_to_train
        # self.group_used_to_test = cfg.group_used_to_test
        if original_name is not None:
            # print("\nloading {} as model's test dataset".format(original_name))
            # 使用正则表达式来匹配两个数字
            class_start, class_end = re.findall(r"(\d+)_(\d+)$", original_name)[-1]
            # 在ae_ddpm中直接指定数据集载入，只会载入一个cifar20的group，所以下面两者可以直接相同赋值
            self.classes_used_to_train = "{}-{}".format(class_start, class_end)
            # self.group_used_to_train = "{}-{}".format(class_start, class_end)
            self.group_used_to_train="0-100"

    @property
    def train_transform(self):
        train_transform = {
            "cifar10": transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
            "cifar100": transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                        (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
                    ),
                ]
            ),
            "mnist": transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]
            ),
            # "imagenet": transforms.Compose([
            #     transforms.Resize(32),
            #     transforms.CenterCrop(32),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize(
            #         mean=[0.485, 0.456, 0.406],
            #         std=[0.229, 0.224, 0.225])
            # ]),
            "imagenet": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        }
        return train_transform[self.dataset]

    @property
    def val_transform(self):
        test_transform = {
            "cifar10": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
            "cifar100": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                        (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
                    ),
                ]
            ),
            "mnist": transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]
            ),
            # "imagenet": transforms.Compose([
            #     transforms.Resize(32),
            #     transforms.CenterCrop(32),
            #     transforms.ToTensor(),
            #     transforms.Normalize(
            #         mean=[0.485, 0.456, 0.406],
            #         std=[0.229, 0.224, 0.225])
            # ]),
            "imagenet": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        }
        return test_transform[self.dataset]

    @property
    def data_cls(self):
        data_cls = {
            "cifar10": datasets.CIFAR10,
            "cifar100": datasets.CIFAR100,
            "mnist": datasets.MNIST,
        }
        return data_cls[self.dataset]

    @property
    def train_dataset(self):
        class_start, class_end = map(int, self.classes_used_to_train.split("-"))
        group_start, group_end = map(int, self.group_used_to_train.split("-"))
        selected_classes = list(range(class_start, class_end))
        selected_group = list(range(group_start, group_end))
        if self.dataset == 'imagenet':
            dataset = CustomImageDataset('data/imagenet900/class900.json', 'data/imagenet900/train', selected_classes, selected_group, self.train_transform)
            return dataset
            # full_dataset = datasets.ImageFolder(root='data/imagenet900/train',transform=self.train_transform)
        else:
            full_dataset = self.data_cls(
                self.root, train=True, download=True, transform=self.train_transform
            )
            if self.classes_used_to_train is None or self.classes_used_to_train == "all":
                return full_dataset
            else:
                indices = []
                new_labels = []
                for i, label in enumerate(full_dataset.targets):
                    if label in selected_classes:
                        indices.append(i)
                        # 根据在测试集中的位置，前半段从0-10，后半段从10-20来映射标签
                        new_labels.append(selected_group.index(label))
                        # 直接修改整个原始数据集的标签，这样在切分的时候就不会出现问题
                        full_dataset.targets[i] = selected_group.index(label)

                # filtered_dataset = torch.utils.data.Subset(full_dataset, indices)
                filtered_dataset = CustomDataset(full_dataset,indices)
            return filtered_dataset

    @property
    def val_dataset(self):
        class_start, class_end = map(int, self.classes_used_to_train.split("-"))
        group_start, group_end = map(int, self.group_used_to_train.split("-"))
        selected_classes = list(range(class_start, class_end))
        selected_group = list(range(group_start, group_end))
        if self.dataset == 'imagenet':
            # full_dataset = datasets.ImageFolder(root='data/imagenet900/val',transform=self.train_transform)
            dataset = CustomImageDataset('data/imagenet900/class900.json', 'data/imagenet900/val', selected_classes, selected_group, self.val_transform)
            return dataset
        else:
            full_dataset = self.data_cls(
                self.root, train=False, download=True, transform=self.val_transform
            )
            if self.classes_used_to_train is None or self.classes_used_to_train == "all":
                return full_dataset
            else:
                indices = []
                new_labels = []
                for i, label in enumerate(full_dataset.targets):
                    if label in selected_classes:
                        indices.append(i)
                        # 重新映射标签从0开始
                        new_labels.append(selected_group.index(label))
                        full_dataset.targets[i] = selected_group.index(label)

                # filtered_dataset = torch.utils.data.Subset(full_dataset, indices)
                filtered_dataset = CustomDataset(full_dataset,indices)
                # 更新子集的标签
                # filtered_dataset.targets = new_labels
                return filtered_dataset

    @property
    def test_dataset(self):
        class_start, class_end = map(int, self.classes_used_to_train.split("-"))
        group_start, group_end = map(int, self.group_used_to_train.split("-"))
        selected_classes = list(range(class_start, class_end))
        selected_group = list(range(group_start, group_end))
        if self.dataset == 'imagenet':
            dataset = CustomImageDataset('data/imagenet900/class900.json', 'data/imagenet900/val', selected_classes, selected_group, self.val_transform)
            return dataset
            # full_dataset = datasets.ImageFolder(root='data/imagenet900/val',transform=self.train_transform)
        else:
            full_dataset = self.data_cls(
                self.root, train=False, download=True, transform=self.val_transform
            )
            if self.classes_used_to_train is None or self.classes_used_to_train == "all":
                return full_dataset
            else:
                indices = []
                new_labels = []
                for i, label in enumerate(full_dataset.targets):
                    if label in selected_classes:
                        indices.append(i)
                        # 重新映射标签从0开始
                        # new_labels没有实际意义，是用来查看类型标签的输出是否正确的
                        new_labels.append(selected_group.index(label))
                        # 直接在原类型上修改labels
                        full_dataset.targets[i] = selected_group.index(label)

                # filtered_dataset = torch.utils.data.Subset(full_dataset, indices)
                filtered_dataset = CustomDataset(full_dataset,indices)
                # 更新子集的标签
                return filtered_dataset
