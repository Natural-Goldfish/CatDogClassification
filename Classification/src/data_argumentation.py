import random
import torch
import cv2
import numpy as np

_MEAN = (0.5, 0.5, 0.5)
_STD = (0.5, 0.5, 0.5)
_WIDTH = 112
_HEIGHT = 112

class Transforms(object):
    def __init__(self, functions):
        self.function_list = functions

    def __call__(self, data):
        for function in self.function_list:
            data = function(data)
        return data

class Flip(object):
    def __init__(self):
        super().__init__()
        self.direction = [0, 1]
    def __call__(self, data):
        image, label = data
        if random.choice(self.direction) == 0 :
            image = cv2.flip(image, 1)
        return image, label

class Resize(object):
    def __init__(self, width = _WIDTH, height = _HEIGHT):
        super().__init__()
        self.width = width
        self.height = height

    def __call__(self, data):
        image, label = data
        image = cv2.resize(image, (self.width, self.height))
        return image, label

class Normalize(object):
    def __init__(self, mean = _MEAN, std = _STD):
        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, data):
        image, label = data
        image = image/255
        for i in range(3):
            image[i] = (image[i] - self.mean[i])/self.std[i]
        return image, label
        
class Numpy2Tensor(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        image, label = data
        image = torch.tensor(np.transpose(image, axes = (2, 0, 1)), dtype = torch.float32)
        label = torch.tensor(label, dtype = torch.long)
        return image, label

