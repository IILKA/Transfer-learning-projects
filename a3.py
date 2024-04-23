import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
# This is for the progress bar.
from tqdm.auto import tqdm
import random
import matplotlib.pyplot as plt
import torchvision
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
import random
from collections import Counter 
from sklearn.model_selection import train_test_split


def trainer(classifier, train_loader, valid_loader, model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    n_epochs = 500
    early_stop = 50

    model = classifier.to(device)
    #for name, param in model.named_parameters():
        
    #model = torchvision.models.vit_b_16(image_size = 256,dropout=0.2, num_classes = 15).to(device)

    criterion = nn.CrossEntropyLoss()


    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0002, weight_decay = 1e-5)

    stale = 0
    best_acc = 0
    best_model = None

    for epoch in range(n_epochs):
        model.train()
        train_loss = []
        train_accs = []
        
        for batch in tqdm(train_loader): 
            imgs, labels = batch 
            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            
            optimizer.zero_grad()
            loss.backward()
            
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(),max_norm = 10)
            optimizer.step()
            acc = (logits.argmax(dim=-1)==labels.to(device)).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
        
        model.eval()
        valid_loss = []
        valid_accs = []
        
        for batch in tqdm(valid_loader):
            imgs , labels = batch
            with torch.no_grad():
                logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            valid_loss.append(loss)
            valid_accs.append(acc)
            
        valid_loss = sum(valid_loss)/len(valid_loss)
        valid_acc = sum(valid_accs)/len(valid_accs)
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        
        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save(model.state_dict(), f"./models/{model_name}.cpkt")                                                                        # only save best to prevent output memory exceed error
            best_acc = valid_acc
            stale = 0
        else:
            stale += 1
            if stale > early_stop:
                print(f"No improvment {early_stop} consecutive epochs, early stopping")
                break


def evaluation(test_loader, classifier, model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_best = classifier.to(device)
    model_best.load_state_dict(torch.load(f"models/{model_name}.cpkt"))
    model_best.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = []
    test_accs = []

    for batch in tqdm(test_loader):
        imgs , labels = batch
        with torch.no_grad():
            logits = model_best(imgs.to(device))
        loss = criterion(logits, labels.to(device))
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        test_loss.append(loss)
        test_accs.append(acc)

    test_loss = sum(test_loss)/len(test_loss)
    test_acc = sum(test_accs)/len(test_accs)
    print(f" loss = {test_loss:.5f}, acc = {test_acc:.5f}")
    file = open("./log/results.txt", "a")
    file.write(f"{model_name} loss = {test_loss:.5f}, acc = {test_acc:.5f}\n")

    # #plot the confusion matrix
    # from sklearn.metrics import confusion_matrix
    # import seaborn as sns
    # import pandas as pd
    
    # y_true = []
    # y_pred = []
    # for batch in tqdm(test_loader):
    #     imgs , labels = batch
    #     with torch.no_grad():
    #         logits = model_best(imgs.to(device))
    #     y_true.extend(labels)
    #     y_pred.extend(logits.argmax(dim=-1).cpu())
    # cm = confusion_matrix(y_true, y_pred)
    # #normalize the confusion matrix
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # # label_mapping reverse
    # label_mapping = {
    #             'InsideCity':0,
    #         'OpenCountry':1,
    #         'Mountain':2,
    #         'Highway':3,
    #         'LivingRoom':4,
    #         'Suburb':5,
    #         'Bedroom':6,
    #         'Kitchen':7,
    #         'Industrial':8,
    #         'Coast':9,
    #         'TallBuilding':10,
    #         'Street':11,
    #         'Office':12,
    #         'Forest':13,
    #         'Store':14
    #         }
    # label_mapping = {v:k for k,v in label_mapping.items()}
    # df_cm = pd.DataFrame(cm, index = [label_mapping[i] for i in range(15)],
    #                   columns = [label_mapping[i] for i in range(15)])
   
    # plt.figure(figsize = (10,7))
    # sns.heatmap(df_cm, annot=True)
    # plt.savefig(f"./log/confusion_matrix_{model_name}.png")
    # plt.close()


from utils import prepare_dataset, prepare_test
from models import ResNet18, CNN, get_res_50, getvgg16, DictionaryLearning, getvit16, getResnet152V2, getResnet152V4

model = getResnet152V4()

train_loader, valid_loader, test_loader = prepare_dataset()
test_loader = prepare_test()
#trainer(model, train_loader, valid_loader, "resnet152v4")
evaluation(test_loader, model, "resnet152v4")

