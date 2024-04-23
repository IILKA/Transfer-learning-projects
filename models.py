import torchvision
import torch 
import numpy as np 
import torch.nn as nn

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18,self).__init__()
        backbone = torchvision.models.resnet18(pretrained = True)
        self.conv1 = nn.Conv2d(1, 64, 5, 2, 3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1 
        self.layer2 = backbone.layer2 
        self.layer3 = backbone.layer3 
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = nn.Sequential(
            nn.Linear(512, 1024, bias = True),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 15, bias = True)
        )
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        #print("layer1",x.size())
        x = self.layer2(x)
        #print("layer2", x.size())
        x = self.layer3(x)
        #print("layer3", x.size())
        x = self.layer4(x)
        #print("layer4", x.size())
        x = self.avgpool(x)
        #print("final", x.size())
        x = x.view(x.size()[0],-1)
        x = self.fc(x)
        return x
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),#64，256，256 
            nn.Dropout(0.2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(4,4,0), #64，
            
            nn.Conv2d(64, 128, 3, 1, 1),  
            nn.Dropout(0.2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0), #32
            
            nn.Conv2d(128,64,3, 1, 1),
            nn.Dropout(0.2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0), #16
                 
            nn.Conv2d(64,32,3, 1, 1),
            nn.Dropout(0.2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),    
        )
    
        
        self.fc = nn.Sequential(
            nn.Linear(1568, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 15)
        )
        
        #self.register_hooks()
    
    def register_hooks(self):
        for layer in self.cnn:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.MaxPool2d):
                layer.register_forward_hook(self.print_size)
    
    def print_size(self, module, input, output):
        print(f"{module.__class__.__name__} output size: {output.size()}")
        
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0],-1)
        return self.fc(out)

def get_res_50():
    model = torchvision.models.resnet50(pretrained = True)
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, 15)
    )
    #convert the first conv1 layer to 1 channel
    model.conv1 = nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
    #freeze the layer1, layer2, layer3 
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = False
    return model


def getvgg16(freezed = 10):
    model = torchvision.models.vgg16(pretrained = True)
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 15)
    )
    model.features[0] = nn.Conv2d(1, 64, kernel_size = 3, stride = 1, padding = 1)
    #freeze the first 10 layers except the first layer
    for param in model.features[1:freezed].parameters():
        param.requires_grad = False
    
    return model


class DictionaryLearningLayer(nn.Module):
    def __init__(self, M, N):
        super(DictionaryLearningLayer, self).__init__()
        self.M = M
        self.N = N
        self.D = nn.Parameter(torch.rand((M, N)))
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, batch):
        all_alpha = []
        for y in batch:
            v = y 
            alpha = torch.zeros(self.N)
            for i in range(10):
                alpha, v = self.amp_unit(y, v, alpha, self.gamma)
            all_alpha.append(alpha)
        return torch.stack(all_alpha)
    


    def amp_unit(self,y, v,alpha, gamma):
        alpha = alpha.to(y.device)
        z = alpha + self.beta*torch.matmul(self.D.T, v)
        t = gamma/(self.M**(1/2)) * torch.norm(v, 2)
        new_alpha = torch.relu( z - t)* torch.sign(z)
        new_v = y - self.beta*torch.matmul(self.D, new_alpha) + self.beta/self.M * torch.norm(new_alpha)* v
        return new_alpha, new_v 

        
class DictionaryLearning(nn.Module):
    def __init__(self, N):
        super(DictionaryLearning, self).__init__()
        backbone = torchvision.models.vgg16(pretrained = True)
        #convert first layer to 1 channel
        backbone.features[0] = nn.Conv2d(1, 64, kernel_size = 3, stride = 1, padding = 1)
        #freeze the first 10 layers except the first layer
        for param in backbone.features[1:10].parameters():
            param.requires_grad = False
        #replace the last layer with a dictionary learning layer
        # and get the featuresize returned by the CNN 

        DLL = nn.Sequential(nn.Linear(25088, 4000), DictionaryLearningLayer(4000, N))
        #replace the fc layer with DLL 
        backbone.classifier = DLL
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)
        return x
        
        
#test DictionaryLearningLayer
# y = torch.rand(())
# layer = DictionaryLearningLayer(20,15)
# # print all the parameters
# # for param in layer.parameters():
# #     print(param)
# vgg16 = getvgg16()
# print(vgg16)



def getvit16():
    model = torchvision.models.vit_b_16(pretrained = True, dropout = 0.1)
    model.head = nn.Sequential(
        nn.Linear(768, 1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, 15)
    )
    model.conv_proj = nn.Conv2d(1, 768, kernel_size = 16, stride = 16, padding = 0)
    #freeze the first 6 layers 
    for param in model.encoder.layers[0:2].parameters():
        param.requires_grad = False
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

    return model


def getResnet152V2():
    model = torchvision.models.resnet152(weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V2)
    #add one more layer after the last layer 
    layer = nn.Sequential(
        nn.ReLU(),
        nn.Linear(1000, 15)
    )
    model.add_module("more", layer)
    return model
# print(getResnet152V2())

def getResnet152V4():
    model = torchvision.models.resnet152(weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V2)
    #add one more layer after the last layer 
    layer = nn.Sequential(
        nn.ReLU(),
        nn.Linear(1000, 256),
        nn.ReLU(),
        nn.Linear(256, 15)
    )
    model.add_module("more", layer)
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    return model