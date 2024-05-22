import glob
import os
from torch.utils.data import Dataset
import json
from torchvision.io import read_image
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, root, selected_classes, selected_group, transform=None):
        with open(annotations_file, 'r') as file:
            self.class2name = json.load(file)
        self.transform = transform
        self.selected_classes = selected_classes
        self.selected_group = selected_group
        self.root = os.path.expanduser(root)
        self.images = self.get_images()

        
    def pil_loader(self,path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
    
    def get_images(self):
        images = []
        for _class in self.selected_classes:
            _class_str = str(_class)
            image_folder = os.path.join(self.root, self.class2name[_class_str])
            # entries = os.listdir(image_folder)
            # for entry in entries:
            #     image_path = os.path.join(image_folder,entry)
            #     image = Image.open(image_path).convert("RGB")
            #     image = self.transform(image)
            #     images.append((image,_class))
            with os.scandir(image_folder) as entries:
                for entry in entries:
                    if entry.is_file():
                        image_path = os.path.join(image_folder,entry.name)
                        image = self.pil_loader(image_path)
                        if self.transform:
                            image = self.transform(image)
                        images.append((image,_class))
                        # images.append((os.path.join(image_folder,entry.name),_class))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # image_path, original_label = self.images[idx]
        # image = self.pil_loader(image_path)
        image, original_label = self.images[idx]
        label = self.selected_group.index(original_label)
        # if self.transform:
        #     image = self.transform(image)
        return image, label
