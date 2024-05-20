import requests
from PIL import Image
import gradio as gr
import numpy as np
import torch
from mlclass import ml_models
from dataload import config
from optimizer import optimizer
from loss import loss
import torchvision.transforms as transforms
# import cv2


def predict_out(sec, img):
    # print(type(img))
    # print(img.)
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    model, parameter = ml_models( config['model_name'])(ml_models,config['out_channels'],device)
    criteria = loss(config['loss'])(loss)
    model.load_state_dict(torch.load("/mnt/Data3/data/Gulshan/classification/pth_files/test_resnet18.pth"))
    
    test_tran = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                   transforms.Normalize ([0.485,0.456,0.406],[0.229,0.224,0.225])])
    # print(img.filename)
    img=test_tran(img)
    img=torch.unsqueeze(img,0)
    img=img.to(device)
    model.eval()
    out=model(img)
    probability = torch.nn.functional.sigmoid(out)
    predicted_class = torch.argmax(out, dim=1)
    print(predicted_class)
    newline = '\n'
    print("Probability of highest score : ",probability[0][predicted_class].item())

    if predicted_class == 0:
        return (f"Label : Abnormal {newline} Probalility score of the class : {probability[0][predicted_class].item()}")
    else:
        return (f"Label : Normal {newline} Probalility score of the class : {probability[0][predicted_class].item()}")

    

image = gr.inputs.Image(shape=(500, 500))

interface = gr.Interface(predict_out,
             [gr.Textbox(label="Legend",info="classes",lines=2,value={0:'Abnormal',
                                                                      1:'Normal',
                                                                     }),
              gr.inputs.Image(type="pil")],
              gr.outputs.Label(num_top_classes=2),
             )
interface.launch()