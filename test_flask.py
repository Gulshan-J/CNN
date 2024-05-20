
from flask import Flask,jsonify,request
import torch
import numpy as np
from torchvision import datasets,transforms
import torchvision
from tqdm import tqdm
from mlclass import ml_models
from dataload import dataloader
from optimizer import optimizer
from loss import loss
from torchvision.io import read_image
import cv2 as cv
import PIL as Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision
from dataload import config
from mlclass import ml_models
from dataload import dataloader
from optimizer import optimizer
from loss import loss

app =Flask(__name__)

@app.route('/', methods=['GET'])
def test():

        """
            Tests the image depending upon the config file and parameters provided.
        """
        test_tran = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                   transforms.Normalize ([0.485,0.456,0.406],[0.229,0.224,0.225])])
        image_path = config['test_image_path']
        device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
        model, parameter = ml_models( config['model_name'])(ml_models,config['out_channels'],device)
        criteria = loss(config['loss'])(loss)
        model.load_state_dict(torch.load("/mnt/Data3/data/Gulshan/classification/pth_files/test_resnet18.pth"))
        image = cv.imread(image_path)          
        
        if "Abnormal" in image_path:
            label = torch.tensor([0], dtype = torch.int64)
            print("label : ",label.item())
            print("label is  Abnormal")
            
        elif "Normal" in image_path:
            label = torch.tensor([1], dtype = torch.int64)
            print("label : ",label.item())
            print("label is  Normal")
                      
        img=cv.imread(image_path)
        img=Image.Image.fromarray(img)
        img=test_tran(img)
        img=torch.unsqueeze(img,0)
        img=img.to(device)
        label=label.to(device)
        model.eval()
        out=model(img)
        _,pred=torch.max(out,1)
        print("predicted : ",pred.item())       
        probability = torch.nn.functional.sigmoid(out)
        predicted_class = torch.argmax(out, dim=1)
        print(predicted_class)
        print("Probability of highest score : ",probability[0][predicted_class].item())
        _loss=criteria(out,label)
        # print(f"Predicted class: {predicted_class.item()}🎇, probability: {probability[0][predicted_class].item()} 🎈,Loss -> {loss:.4f} 😶")
        return jsonify({"true label":label.item(),"predicted class":predicted_class.item(), "probability":probability[0][predicted_class].item(),"loss":_loss.item()})
        

@app.route('/upload', methods=['POST'])
def predict():
    if request.method == 'POST':
    # file = request.files['file']
    # print(file)
        image = request.files['image']
        print("try")
        # image = request.files['image']
        
        # image = image.decode('utf-8')
        image.save('received_image.jpeg')
        print("2")
        test_tran = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                   transforms.Normalize ([0.485,0.456,0.406],[0.229,0.224,0.225])])
        image_path = "received_image.jpeg"
        device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
        model, parameter = ml_models( config['model_name'])(ml_models,config['out_channels'],device)
        criteria = loss(config['loss'])(loss)
        model.load_state_dict(torch.load("/mnt/Data3/data/Gulshan/classification/pth_files/test_resnet18.pth"))
        image = cv.imread(image_path)          
        print("3")
        # if "Abnormal" in image_path:
        #     label = torch.tensor([0], dtype = torch.int64)
        #     print("label : ",label.item())
        #     print("label is  Abnormal")
            
        # elif "Normal" in image_path:
        #     label = torch.tensor([1], dtype = torch.int64)
        #     print("label : ",label.item())
        #     print("label is  Normal")
                      
        img=cv.imread(image_path)
        img=Image.Image.fromarray(img)
        img=test_tran(img)
        img=torch.unsqueeze(img,0)
        img=img.to(device)
        # label=label.to(device)
        model.eval()
        out=model(img)
        _,pred=torch.max(out,1)
        print("predicted : ",pred.item())       
        probability = torch.nn.functional.sigmoid(out)
        predicted_class = torch.argmax(out, dim=1)
        print(predicted_class)
        print("Probability of highest score : ",probability[0][predicted_class].item())
        #_loss=criteria(out,label)
        # print(f"Predicted class: {predicted_class.item()}🎇, probability: {probability[0][predicted_class].item()} 🎈,Loss -> {loss:.4f} 😶")
    return jsonify({"predicted class":predicted_class.item(), "probability":probability[0][predicted_class].item()})

if __name__ == '__main__':
    app.run(host= '0.0.0.0', port= 5200, debug=False)
