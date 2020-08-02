import torch
from PIL import Image
from torchvision import models,datasets,transforms
import torch.nn as nn
import torch.nn.functional as F
import time

from torch.autograd import Variable

damage_dict = {
0: 'Glass shatter',
1: 'Head lamp broken' ,
2: 'No damage' ,
3: 'Smashed',
4 :'Bumper Dent',
5 :'Scratch',
6 :'Door Dent',
7 :'Tail lamp broken'
}

net = models.alexnet()
# net.cuda()
new_classifier = net.classifier[:-1]
new_classifier.add_module('fc' , nn.Linear(4096 , 8))
net.classifier = new_classifier
net.load_state_dict(torch.load('damage_model/ale.ckpt', map_location = 'cpu') )

def pred(img):

    # img = Image.open(filepath)

    img = Image.fromarray(img.astype('uint8'), 'RGB')
    transform_pipeline = transforms.Compose([transforms.Resize((224 , 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
    img = transform_pipeline(img)
    img = img.unsqueeze(0)
    img = Variable(img)
    prediction = net(img)
    _ , pred = torch.max(prediction , 1)
    name = damage_dict[pred.item()]

    # print(name)

    return str(name)
    


if __name__ == "__main__":

    imagepath = "doordent2.jpg"
    
    t0 = time.time()
    pred(imagepath)
    print("time: " , time.time() - t0)

