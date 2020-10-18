from src.dataset import CatDogDataset
from src.network import *
from src.utils import accuracy
from torch.utils.data import DataLoader
import os


_EPOCH = 400
_BATCH_SIZE = 64
_LEARNING_RATE = 0.001
_MOMENTUM = 0.9
_CUDA_FLAG = torch.cuda.is_available()

_SAVE_PATH = "data\\models"

def train():
    train_dataset = CatDogDataset(mode = "train")
    val_dataset = CatDogDataset(mode = "val")

    train_dataloader = DataLoader(train_dataset, batch_size= _BATCH_SIZE, shuffle= True)
    val_dataloader = DataLoader(val_dataset, batch_size= _BATCH_SIZE, shuffle= False)

    model = CatDogClassifer()
    #model.apply(weights_init)

    if _CUDA_FLAG : model.cuda()

    criterion = torch.nn.CrossEntropyLoss().cuda()
    sigmoid = torch.nn.Sigmoid()
    optimizer = torch.optim.SGD(model.parameters(), lr = _LEARNING_RATE, momentum = _MOMENTUM)
    
    for cur_epoch in range(_EPOCH):
        # Training
        model.train()
        for cur_batch, train_data in enumerate(train_dataloader):
            
            optimizer.zero_grad()
            train_images, train_labels = train_data
            if _CUDA_FLAG :
                train_images = train_images.cuda()
                train_labels = train_labels.view(-1).cuda()

            train_outputs = model(train_images)
            train_loss = criterion(train_outputs, train_labels)
            train_loss.backward()
            optimizer.step()

            train_accuracy = accuracy(sigmoid(train_outputs.cpu().detach()), train_labels.cpu().detach())
            print("EPOCH {}/{} Iteration {}/{} Loss {} Accuracy {}".format(cur_epoch, _EPOCH, cur_batch, len(train_dataloader), train_loss, train_accuracy))

        # Evaluation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_accuracy = 0
            for cur_batch, val_data in enumerate(val_dataloader):
                val_images, val_labels = val_data
                if _CUDA_FLAG :
                    val_images = val_images.cuda()
                    val_labels = val_labels.view(-1).cuda()

                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_labels)
                val_accuracy += accuracy(sigmoid(val_outputs.cpu().detach()), val_labels.cpu().detach())

            # Calculate loss and accuracy about a epoch
            val_loss = val_loss/len(val_dataloader)
            val_accuracy = val_accuracy/len(val_dataloader)
            print("EPOCH {}/{} Loss {} Accuracy {}".format(cur_epoch, _EPOCH, val_loss, val_accuracy))

        model_name = "CatDogClassifer_{}_checkpoint.pth".format(cur_epoch)
        torch.save(model.state_dict, os.path.join(_SAVE_PATH, model_name))

if __name__ == "__main__":
    train()