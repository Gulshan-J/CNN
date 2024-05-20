import torch
import torchvision
from tqdm import tqdm
from mlclass import ml_models
from dataload import dataloader
from optimizer import optimizer
from loss import loss

class Test:

    def __init__(self, model : any, loss : any, optim : any, device : any, data : dict) -> None:
        
        """This implements the intializing of the parameters that is required for training and validating the data.

        Args:
            data (_type_): contains the whole config data in a dictionary form.
            model (_type_): contains the model.
            optim (_type_): contains the optimzer depending upon the config.
            loss (_type_): contains the loss depending upon the config.
            early_stop (_type_): contains the early stopping parameters from the main file.
        """
        self.test_loader = data['test_loader']
        self.model = model
        self.data= data
        self.loss = loss
        self.optim = optim
        self.device = device
        self.epochs = data['epochs']
        self.checkpoints = data['checkpoint']

    def test(self) -> None:

        """
            Tests the dataset depending upon the config file and parameters provided.
        """
        
        self.model.load_state_dict(torch.load("/mnt/Data3/data/Gulshan/classification/pth_files/test_resnet18_pntx.pth"))
        self.model.eval()
        
        correct = 0
        total = 0

        for image,label in tqdm(self.test_loader): #set description
            image = image.to(self.device)
            print(image.shape, image.dtype)
            label = label.to(self.device)
            print(label.shape, label.dtype)
            output = self.model(image)
            loss = self.loss(output, label)
            _,predict = torch.max(output, 1)
            correct += (predict == label).sum()
            total += image.shape[0]  
            loss_test = loss/total     
        print(f"Loss -> {loss_test:.4f} 😶, test accuracy -> {100*correct/total:.2f} ")
        print("label=>",label)
        print("predicted=>",predict)
    
        print("values of the predicted =>",_)
