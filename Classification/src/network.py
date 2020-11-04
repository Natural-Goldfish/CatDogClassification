import torch
from torchvision.models import vgg16

class CatDogClassifier_2l(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1, bias = False),
            torch.nn.ReLU(inplace=True)
        )
        self.max_pool1 = torch.nn.MaxPool2d(2, stride = 2)

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1, bias = False),
            torch.nn.ReLU(inplace=True)
        )
        self.max_pool2 = torch.nn.MaxPool2d(2, stride = 2)

        self.classify = torch.nn.Sequential(
            torch.nn.Linear(64*28*28, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),

            torch.nn.Linear(4096, 1024),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(1024, 2)
        )

    def forward(self, x):
        output = self.layer1(x)
        output = self.max_pool1(output)
        output = self.layer2(output)
        output = self.max_pool2(output)
        output = output.view(-1, 64*28*28)
        output = self.classify(output)
        return output

class CatDogClassifier_3l(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1, bias = False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True)
        )
        self.max_pool1 = torch.nn.MaxPool2d(2, stride = 2)

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1, bias = False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        self.max_pool2 = torch.nn.MaxPool2d(2, stride = 2)

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, 1, 1, bias = False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True)
        )
        self.max_pool3 = torch.nn.MaxPool2d(2, stride = 2)

        self.classify = torch.nn.Sequential(
            torch.nn.Linear(128*14*14, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),

            torch.nn.Linear(4096, 1024),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(1024, 2)
        )

    def forward(self, x):
        output = self.layer1(x)
        output = self.max_pool1(output)
        output = self.layer2(output)
        output = self.max_pool2(output)
        output = self.layer3(output)
        output = self.max_pool3(output)
        output = output.view(-1, 128*14*14)
        output = self.classify(output)
        return output

class CatDogClassifier_4l(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1, bias = False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True)
        )
        self.max_pool1 = torch.nn.MaxPool2d(2, stride = 2)

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1, bias = False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        self.max_pool2 = torch.nn.MaxPool2d(2, stride = 2)

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, 1, 1, bias = False),
            torch.nn.ReLU(inplace=True)
        )
        self.max_pool3 = torch.nn.MaxPool2d(2, stride = 2)

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 3, 1, 1, bias = False),
            torch.nn.ReLU(inplace=True)
        )
        self.max_pool4 = torch.nn.MaxPool2d(2, stride = 2)

        self.classify = torch.nn.Sequential(
            torch.nn.Linear(256*7*7, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),

            torch.nn.Linear(4096, 1024),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(1024, 2)
        )

    def forward(self, x):
        output = self.layer1(x)
        output = self.max_pool1(output)
        output = self.layer2(output)
        output = self.max_pool2(output)
        output = self.layer3(output)
        output = self.max_pool3(output)
        output = self.layer4(output)
        output = self.max_pool4(output)        
        output = output.view(-1, 256*7*7)
        output = self.classify(output)
        return output

class CatDogClassifier_6l(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1, bias = False),
            torch.nn.ReLU(inplace=True)
        )
        self.max_pool1 = torch.nn.MaxPool2d(2, stride = 2)

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1, bias = False),
            torch.nn.ReLU(inplace=True)
        )
        self.max_pool2 = torch.nn.MaxPool2d(2, stride = 2)

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, 1, 1, bias = False),
            torch.nn.ReLU(inplace=True)
        )
        self.max_pool3 = torch.nn.MaxPool2d(2, stride = 2)

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 3, 1, 1, bias = False),
            torch.nn.ReLU(inplace=True)
        )
        self.max_pool4 = torch.nn.MaxPool2d(2, stride = 2)

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, 3, 1, 1, bias = False),
            torch.nn.ReLU(inplace=True)
        )
        self.max_pool5 = torch.nn.MaxPool2d(2, stride = 2)

        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1024, 3, 1, 1, bias = False),
            torch.nn.ReLU(inplace=True)
        )
        self.max_pool6 = torch.nn.MaxPool2d(2, stride = 2)

        self.classify = torch.nn.Sequential(
            torch.nn.Linear(1024*1*1, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),

            torch.nn.Linear(4096, 1024),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(1024, 2)
        )

    def forward(self, x):
        output = self.layer1(x)
        output = self.max_pool1(output)
        output = self.layer2(output)
        output = self.max_pool2(output)
        output = self.layer3(output)
        output = self.max_pool3(output)
        output = self.layer4(output)
        output = self.max_pool4(output)
        output = self.layer5(output)
        output = self.max_pool5(output)
        output = self.layer6(output)
        output = self.max_pool6(output)
        output = output.view(-1, 1024*1*1)
        output = self.classify(output)
        return output

class CatDogClassifier_5l(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1, bias = False),
            torch.nn.ReLU(inplace=True)
        )
        self.max_pool1 = torch.nn.MaxPool2d(2, stride = 2)

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1, bias = False),
            torch.nn.ReLU(inplace=True)
        )
        self.max_pool2 = torch.nn.MaxPool2d(2, stride = 2)

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, 1, 1, bias = False),
            torch.nn.ReLU(inplace=True)
        )
        self.max_pool3 = torch.nn.MaxPool2d(2, stride = 2)

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 3, 1, 1, bias = False),
            torch.nn.ReLU(inplace=True)
        )
        self.max_pool4 = torch.nn.MaxPool2d(2, stride = 2)

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, 3, 1, 1, bias = False),
            torch.nn.ReLU(inplace=True)
        )
        self.max_pool5 = torch.nn.MaxPool2d(2, stride = 2)

        self.classify = torch.nn.Sequential(
            torch.nn.Linear(512*3*3, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),

            torch.nn.Linear(4096, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 2)
        )

    def forward(self, x):
        output = self.layer1(x)
        output = self.max_pool1(output)
        output = self.layer2(output)
        output = self.max_pool2(output)
        output = self.layer3(output)
        output = self.max_pool3(output)
        output = self.layer4(output)
        output = self.max_pool4(output)
        output = self.layer5(output)
        output = self.max_pool5(output)
        output = output.view(-1, 512*3*3)
        output = self.classify(output)
        return output

class CatDogClassifier_5l_bn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1, bias = False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True)
        )
        self.max_pool1 = torch.nn.MaxPool2d(2, stride = 2)

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1, bias = False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        self.max_pool2 = torch.nn.MaxPool2d(2, stride = 2)

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, 1, 1, bias = False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True)
        )
        self.max_pool3 = torch.nn.MaxPool2d(2, stride = 2)

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 3, 1, 1, bias = False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True)
        )
        self.max_pool4 = torch.nn.MaxPool2d(2, stride = 2)

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, 3, 1, 1, bias = False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True)
        )
        self.max_pool5 = torch.nn.MaxPool2d(2, stride = 2)

        self.classify = torch.nn.Sequential(
            torch.nn.Linear(512*3*3, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(512, 2)
        )

    def forward(self, x):
        output = self.layer1(x)
        output = self.max_pool1(output)
        output = self.layer2(output)
        output = self.max_pool2(output)
        output = self.layer3(output)
        output = self.max_pool3(output)
        output = self.layer4(output)
        output = self.max_pool4(output)
        output = self.layer5(output)
        output = self.max_pool5(output)
        output = output.view(-1, 512*3*3)
        output = self.classify(output)
        return output
        
class Vgg16(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg16 = vgg16(pretrained= True)
        self.set_parameter_requires_grad(self.vgg16, True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512*7*7, 4096, bias = True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),

            torch.nn.Linear(4096, 1024, bias = True),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(1024, 2)
        )
        self.vgg16.classifier = self.classifier

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, x):
        output = self.vgg16(x)
        return output
