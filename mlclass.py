import torch
from torchvision import models
from torch import nn
import warnings
warnings.filterwarnings('ignore')
# /mnt/Data3/data/Gulshan/classification/mutliclass_dataset/Train/Bacteria/person1_bacteria_1.jpeg
# /mnt/Data3/data/Gulshan/classification/mutliclass_dataset/Test/Normal/NORMAL2-IM-1219-0001.jpeg
# /mnt/Data3/data/Gulshan/classification/mutliclass_dataset/Test/Virus/person1429_virus_2443.jpeg


class ml_models:
    """
    This class contains the model architecture related functions.

    """
    def __new__(cls,name : str) -> str:
        '''
        returns the name of the function
        '''
        return getattr(cls,name)
    
    def inception_v1(self,num_classes: int,device):
        """The function contains inception_v1  model related parameters.

        Args:
            num_classes (int): defines the number of classes.
            device (_type_): defines the device we are running on i.e whether it's cpu or gpu.

        Returns:
            The model and it's parameters.
        """
        print("You are using inception_v1 for model Training")
        inception=models.googlenet(pretrained=True)
        inception.aux_logits=False
        for parameter in inception.parameters():
            inception.requires_grad = False
        in_feat=inception.fc.in_features
        inception.fc=nn.Linear(in_feat, num_classes)
        parameter = inception
        return inception.to(device), parameter
    
    def inception_v3(self,num_classes: int,device):
        """The function contains inception_v3 model related parameters.

        Args:
            num_classes (int): defines the number of classes.
            device (_type_): defines the device we are running on i.e whether it's cpu or gpu.

        Returns:
            The model and it's parameters.
        """
        print("You are using inception_v3 for model Training")
        inception=models.inception_v3(pretrained=True)
        inception.aux_logits=False
        for parameter in inception.parameters():
            inception.requires_grad = False
        in_feat=inception.fc.in_features
        inception.fc=nn.Linear(in_feat, num_classes)
        parameter = inception
        return inception.to(device), parameter
    
    def efficientnet_b0(self,num_classes: int,device):
        """The function contains efficientnet_b0  model related parameters.

        Args:
            num_classes (int): defines the number of classes.
            device (_type_): defines the device we are running on i.e whether it's cpu or gpu.

        Returns:
            The model and it's parameters.
        """
        print("You are using efficientnet_b0 for model Training")
        efficient= models.efficientnet_b1(pretrained = True)
        for parameter in efficient.parameters():
            efficient.requires_grad = False
        in_feat = efficient.classifier[1].in_features
        efficient.classifier[1] = nn.Linear(in_feat, num_classes)
        parameter = efficient
        return efficient.to(device), parameter       
    
    def densenet121(self,num_classes: int,device) -> any :
        """The function contains densenet 121 model related parameters.

        Args:
            num_classes (int): defines the number of classes.
            device (_type_): defines the device we are running on i.e whether it's cpu or gpu.

        Returns:
            The model and it's parameters.
        """
        print("You are using densenet121 for model Training")
        
        dense = models.densenet121(pretrained = True)
        with torch.no_grad():
            in_feat = dense.classifier.in_features
            dense.fc = nn.Linear(in_feat, num_classes)
            dense.classifier.requires_grad = True
            parameter = dense
            return dense.to(device), parameter
    
    def densenet201(self,num_classes: int,device) -> any :
        """The function contains densenet 201 model related parameters.

        Args:
            num_classes (int): defines the number of classes.
            device (_type_): defines the device we are running on i.e whether it's cpu or gpu.

        Returns:
            The model and it's parameters.
        """
        print("You are using densenet201 for model Training")
        
        dense = models.densenet201(pretrained = True)
        with torch.no_grad():
            in_feat = dense.classifier.in_features
            dense.fc = nn.Linear(in_feat, num_classes)
            dense.classifier.requires_grad = True
            parameter = dense
            return dense.to(device), parameter

    def resnet50(self,num_classes : int,device) -> any:
        """The function contains resnet 50 model related parameters.

        Args:
            num_classes (int): defines the number of classes.
            device (_type_): defines the device we are running on i.e whether it's cpu or gpu.

        Returns:
            The model and it's parameters.
        """
        print("You are using resnet50 for model Training")
        
        res_50 = models.resnet50(pretrained = True)
        for parameter in res_50.parameters():
            parameter.requires_grad = False
        no_inputs = res_50.fc.in_features
        res_50.fc = nn.Linear(no_inputs, num_classes)
        parameter = res_50.fc
        return res_50.to(device),parameter

    def resnet152(self,num_classes : int,device) -> any:
        """The function contains resnet 152 model related parameters.

        Args:
            num_classes (int): defines the number of classes.
            device (_type_): defines the device we are running on i.e whether it's cpu or gpu.

        Returns:
            The model and it's parameters.
        """
        print("You are using resnet152 for model Training")
        
        res_152 = models.resnet152(pretrained = True)
        for parameter in res_152.parameters():
            parameter.requires_grad = False
        no_inputs = res_152.fc.in_features
        res_152.fc = nn.Linear(no_inputs, num_classes)
        parameter = res_152.fc
        return res_152.to(device),parameter

    def vgg16(self,num_classes : int,device) -> any:
        """The function contains vgg 16 model related parameters.

        Args:
            num_classes (int): defines the number of classes.
            device (_type_): defines the device we are running on i.e whether it's cpu or gpu.

        Returns:
            The model and it's parameters.
        """
        print("You are using vgg16 for model Training")
        
        vgg = models.vgg16(pretrained = True)
        for parameter in vgg.parameters():
            parameter.requires_grad = False
        in_feat = vgg.classifier[6].in_features
        vgg.classifier[6] = nn.Linear(in_feat, num_classes)
        parameter = vgg
        return vgg.to(device), parameter
    
    def vgg19(self,num_classes : int,device) -> any:
        """The function contains vgg 19 model related parameters.

        Args:
            num_classes (int): defines the number of classes.
            device (_type_): defines the device we are running on i.e whether it's cpu or gpu.

        Returns:
            The model and it's parameters.
        """
        print("You are using vgg19 for model Training")
        
        vgg_19= models.vgg19(pretrained = True)
        for parameter in vgg_19.parameters():
            parameter.requires_grad = False
        in_feat = vgg_19.classifier[6].in_features
        vgg_19.classifier[6] = nn.Linear(in_feat, num_classes)
        parameter = vgg_19
        return vgg_19.to(device), parameter
