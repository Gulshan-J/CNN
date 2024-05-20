# from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
# import mlflow
# from kfp import components
from checkpoint import checkpoint
# import matplotlib.matplotlib.pyplot as plt
class Train:
    def __init__(self, model : any, loss : any, optim : any, device : any, data : dict, early_stop : object) -> None:
        """This implements the intializing of the parameters that is required for training and validating the data.

        Args:
            data (_type_): contains the whole config data in a dictionary form.
            model (_type_): contains the model.
            optim (_type_): contains the optimzer depending upon the config.
            loss (_type_): contains the loss depending upon the config.
            early_stop (_type_): contains the early stopping parameters from the main file.
        """
        self.train_loader = data['train_loader']
        self.val_loader = data['val_loader']
        self.model = model
        self.data= data
        self.loss = loss
        self.optim = optim
        self.device = device
        self.epochs = data['epochs']
        self.early_stop= early_stop
        self.checkpoints = data['checkpoint']
        # self.writer =SummaryWriter()

    def train(self) -> None:
        """
            Trains the dataset depending upon the config file.
        """
        train_loss=[]
        train_accuracy=[]
        for epoch in range(self.epochs):
            self.model.train()
            correct = 0
            total = 0
            for image,label in tqdm(self.train_loader): 
                image = image.to(self.device)
                label = label.to(self.device)
                output = self.model(image)
                loss = self.loss(output, label)
                # self.writer.add_scalar("Loss fro Training ", loss, epoch)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                _,predict = torch.max(output, 1)
                correct += (predict == label).sum()
                total += image.shape[0]
                
            print(f"Epoch -> {epoch+1}/{self.epochs}, Loss -> {loss/total:.4f}, train accuracy ->{100*correct/total:.2f}")
            train_loss.append(loss/total)
            train_accuracy.append(100*correct/total)
            value = self.validate()
            if self.checkpoints['is_true']:
                checkpoint(self.checkpoints,value,self.model.state_dict())
            
            status=self.early_stop(value['val_loss'])
            if status:
                    print("Model is overfitting and best model is already saved")
                    break
        # torch.save(self.model.state_dict(), "/mnt/Data3/data/Gulshan/classification/pth_files/test_resnet18_pntx.pth")
        # print("Model saved in .pth format 🎉")
    
    def validate(self) -> dict:
        """ validates the dataset depending upon the config.

        Returns:
            _type_: returns the loss value in a form of dictionary for early stopping and checkpoint.
        """
        self.model.eval()
        loss_total = 0.0
        correct = 0
        total = 0
        eval_loss=[]
        eval_accuracy=[]
        with torch.no_grad():
            for image,label in tqdm(self.val_loader):
                image = image.to(self.device)
                label = label.to(self.device)
                output = self.model(image)
                loss = self.loss(output, label)
                # self.writer.add_scalar("Loss curve for validation", loss, epoch)
                loss_total += loss.item()
                _,predict = torch.max(output, 1)
                correct += (predict == label).sum().item()
                total += image.shape[0]
            loss_val=loss_total/total    
            print(f"val accuracy: {100*correct/total:.4f},loss {loss_val:.4f}")
            eval_loss.append(loss/total)
            eval_accuracy.append(100*correct/total)
        return {'val_loss':loss_val, 'val_accuracy':(100*correct/total),'eval_loss':eval_loss,'eval_accuracy':eval_accuracy}
    
    def train_val_curve(self,train_loss,eval_loss,train_accuracy):
        
        plt.plot(train_loss,'-o')
        plt.plot(eval_loss,'-o')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['Train','Valid'])
        plt.title('Train vs Valid Accuracy')

        plt.show()


