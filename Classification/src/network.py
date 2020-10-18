import torch
from torchsummary import summary

class CatDogClassifer(torch.nn.Module):
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

def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_uniform_(m.weight.data, mode = 'fan_in', nonlinearity='leaky_relu')
    
    elif classname.find('Batch') != -1 :
        torch.nn.init.normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)

if __name__ == "__main__" :
    model = CatDogClassifer()
    summary(model, (3, 112, 112))