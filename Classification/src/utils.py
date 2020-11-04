import torch
from src.network import *

MODELS = ['Vgg16', 'CatDogClassifier_2l', 'CatDogClassifier_3l', 'CatDogClassifier_4l', 'CatDogClassifier_5l', 'CatDogClassifier_5l_bn', 'CatDogClassifier_6l']

def accuracy(outputs, labels):
    num = len(labels)
    _, output_idx = torch.max(outputs, dim = 1)
    acc_list = [1 for i in range(num) if output_idx[i] == labels[i]]
    acc = sum(acc_list)/num
    print(acc)
    return acc

def load_model_class(model_num):
    if model_num == 0 : model = Vgg16()
    elif model_num == 1 : model = CatDogClassifier_2l()
    elif model_num == 2 : model = CatDogClassifier_3l()
    elif model_num == 3 : model = CatDogClassifier_4l()
    elif model_num == 4 : model = CatDogClassifier_5l()
    elif model_num == 5 : model = CatDogClassifier_5l_bn()
    elif model_num == 6 : model = CatDogClassifier_6l()
    return model