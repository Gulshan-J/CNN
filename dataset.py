import warnings
warnings.filterwarnings('ignore')
import torch 
from torch.utils.data import DataLoader,Dataset
import os
import glob
import json
import cv2

with open('/mnt/Data3/data/Gulshan/classification/config.json') as user_file:
    _CONFIG = json.load(user_file)

class custom_dataset(Dataset):
    def __init__(self,path,transform=False):
        self.transform=transform
        image_path=path
        self.image_paths=[]
        self.image_labels=[]
        for label, file in enumerate(os.listdir(image_path)):
            for image_name in os.listdir(os.path.join(image_path, file)):
                img_path = os.path.join(image_path, file, image_name)
                self.image_paths.append(img_path)
                self.image_labels.append(label)   
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image=cv2.imread(self.image_paths[index])
        if self.transform is not None:
            image = self.transform(image=image)["image"]
            return image,self.image_labels[index]
        

# path="/mnt/Data3/data/Gulshan/classification/mutliclass_dataset/Test
# x,y=iter(train_loader).next()
# print(x.shape,y)
# x,y=iter(train_loader).next()
# print(x.shape,y)
