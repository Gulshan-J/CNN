import torch
from mlclass import ml_models
from dataload import dataloader
from optimizer import optimizer
from loss import loss
from Train import Train
from earlystopping import earlystopping

class main:
    """
        The class main contains the initialization of variables and running the functions
        for training the model.
    """
    def __init__(self) -> None:
        """
            This function initializes variables and calls the functions 
            for necessary operations.
        """
        data = dataloader().load_data()
        device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
        model, parameter = ml_models(data['model_name'])(ml_models,data['out_channels'],device)
        criteria = loss(data['loss'])(loss)
        optimi = optimizer(data['optimizer'])(optimizer,data['Learning_rate'], parameter)
        early_stop=earlystopping(data['earlystopping']['is_true'],data['earlystopping']['patience'],data['earlystopping']['min_del'])
        Train(model, criteria, optimi, device,data,early_stop).train()
        
if __name__ == '__main__':
    main()