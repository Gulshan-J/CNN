import torch
import glob
import os
from torchvision import datasets,transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

class _prerequisite:
    def __init__(self):

        self.Train_tsfm =A.Compose([ 
                        A.Resize(299,299),
                        A.Cutout(p=0.25),A.RandomRotate90(p=0.25),A.Flip(p=0.25),
                        A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.2,),
                        A.HueSaturationValue(hue_shift_limit=20,sat_shift_limit=50,val_shift_limit=50)], p=0.3),
                        A.OneOf([ A.IAAAdditiveGaussianNoise(),A.GaussNoise(),], p=0.25),
                        A.OneOf([A.MotionBlur(p=0.2),A.MedianBlur(blur_limit=3, p=0.1),A.Blur(blur_limit=3, p=0.1),], p=0.2),
                        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.25),
                        A.OneOf([A.OpticalDistortion(p=0.3),A.GridDistortion(p=0.1),A.IAAPiecewiseAffine(p=0.3),], p=0.25),
                        A.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),ToTensorV2()
                      ])
        self.Test_tsfm= A.Compose([ 
                                A.Resize(299,299),
                                A.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),ToTensorV2()
                            ])
        
        self.data_transform={'Train' : transforms.Compose([transforms.Resize((299, 299)),
                                    # transforms.RandomResizedCrop(229),
                                   transforms.RandomHorizontalFlip(),transforms.Pad(30),transforms.ToTensor(),
                                   transforms.Normalize ([0.5513,0.5052,0.5730],[0.2595,0.2324,0.2789])]),
                'Test' : transforms.Compose([transforms.Resize((299, 299)),
                                    # transforms.RandomResizedCrop(229),
                                   transforms.ToTensor(),
                                   transforms.Normalize ([0.485,0.456,0.406],[0.229,0.224,0.225])])
               }

            
            
prerequisite_=_prerequisite()