import os 
import torch 
from PIL import Image 

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import random 
import numpy as np 

from skimage.util import random_noise

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split




label_mapping = {
    'InsideCity':0,
 'OpenCountry':1,
 'Mountain':2,
 'Highway':3,
 'LivingRoom':4,
 'Suburb':5,
 'Bedroom':6,
 'Kitchen':7,
 'Industrial':8,
 'Coast':9,
 'TallBuilding':10,
 'Street':11,
 'Office':12,
 'Forest':13,
 'Store':14
}


def add_noise(img):
    r = random.random()
    tf = transforms.ToTensor()
    if r < 0.4:
        img = tf(random_noise(img, mode = "s&p", amount = 0.01).astype(np.float32))
        img = img.reshape(1, 224, 224)
        img = torch.transpose(img, 1,2)
        return img
    elif r >=0.4 and r <=0.8:
        img = tf(random_noise(img, mode="gaussian", var = 0.005).astype(np.float32))
        img = img.reshape(1, 224, 224)
        img = torch.transpose(img, 1,2)
        return img
    else:
        return img
    

def shift(img):
    r = random.random()
    if r < 0.7: 
       tf = transforms.RandomAffine(degrees = 20, translate = (0.1,0.1), scale = (0.8, 1.2), shear = 20)
       return tf(img)
    else:
       tf = transforms.RandomPerspective(distortion_scale = 0.3, p = 0.5)
       return tf(img)
    

train_tfm = transforms.Compose(
[
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.Lambda(shift),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    #transforms.RandomErasing(p=0.5, scale = (0.02,0.05), ratio = (0.3,3.3),value ="random", inplace=False),
    transforms.Lambda(add_noise),
    transforms.Normalize(mean = 0.4559, std = 0.2119)
    ]
)
test_tfm = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = 0.4559, std = 0.2119)
    ]
)
norm_tfm = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        #transforms.Normalize(mean = 0.4559, std = 0.2119)
    ]
)


class ImageDataSet(Dataset):
    def __init__(self,path,transform):
        super(ImageDataSet).__init__()
        self.root_dir = path
        self.transform = transform
        self.img_list = []
        self.label_list = []
        for label in os.listdir(self.root_dir):
            if label == ".DS_Store":
                continue
            label_path = os.path.join(self.root_dir, label)
            for img in os.listdir(label_path):
                img_path = os.path.join(label_path, img)
                self.img_list.append(img_path)
                self.label_list.append(label_mapping[label])
                
    def __len__(self):
        return len(self.img_list)
  
    def __getitem__(self,idx):
        label = self.label_list[idx]
        img_path = self.img_list[idx]
        img = Image.open(img_path)
        img = self.transform(img)
#         img = img.repeat(1,3,1,1)
#         img = img.view(64, 3, 224, 224)
        #turn the image into a three channel image 
        img = img.repeat(3,1,1)
        return img, label
   


def visualize_transformation():
    #only for testing purposes 
    for i in range(20):
        img = Image.open("./scene15/train/Bedroom/image_0001.jpg")
        img = train_tfm(img)
        img = img.reshape(224,224)
        plt.imsave(f"./test_img/test{i}.jpg", img, cmap = "gray")



def prepare_dataset():
    # This function prepares the dataset for training and testing.
    train_set = ImageDataSet("./scene15/train", transform = train_tfm)
    test_set = ImageDataSet("./scene15/test", transform = test_tfm)
    valid_set = ImageDataSet("./scene15/test", transform = test_tfm)
    imgs = test_set.img_list
    labels = test_set.label_list
    img_test, img_val, label_test, label_val = train_test_split(imgs, labels, test_size = 0.2)
    test_set.img_list = img_test
    test_set.label_list = label_test 
    valid_set.img_list = img_val
    valid_set.label_list = label_val 
    train_loader = DataLoader(train_set, batch_size = 96, shuffle = True, num_workers =0, pin_memory = True)
    valid_loader = DataLoader(valid_set, batch_size = 96, shuffle = True, num_workers =0, pin_memory = True)
    test_loader = DataLoader(test_set, batch_size = 96, shuffle = True, num_workers =0, pin_memory = True)
    return train_loader, valid_loader, test_loader

def prepare_test():
    test_set = ImageDataSet("./scene15/test", transform = test_tfm)
    test_loader = DataLoader(test_set, batch_size = 96, shuffle = True, num_workers =0, pin_memory = True)
    return test_loader



def prepare_parameter():
    all = ImageDataSet("./scene15/train", transform = norm_tfm)
    #get the mean and std of the dataset of gray scale value 
    mean = 0 
    std = 0
    for img, _ in all:
        mean += img.mean()
        std += img.std()
    mean /= len(all)
    std /= len(all)
    print(mean, std)

   

if __name__ == "__main__":
    visualize_transformation()
    train_loader, valid_loader, test_loader = prepare_dataset()
    print(train_loader, valid_loader, test_loader)
    prepare_parameter()