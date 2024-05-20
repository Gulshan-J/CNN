import torch
from mlclass import ml_models
from dataload import dataloader
from optimizer import optimizer
from loss import loss
# from inference_falaq import Test
from one_img_inf import Test


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
        device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
        model, parameter = ml_models(data['model_name'])(ml_models,data['out_channels'],device)
        criteria = loss(data['loss'])(loss)
        optimi = optimizer(data['optimizer'])(optimizer,data['Learning_rate'], parameter)
        Test(model, criteria, optimi, device, data).test()
        
if __name__ == '__main__':
    main()
    