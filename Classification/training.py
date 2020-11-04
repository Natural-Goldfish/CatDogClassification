from src.dataset import CatDogDataset
from src.utils import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

_CUDA_FLAG = torch.cuda.is_available()

    
def train(args):
    train_dataset = CatDogDataset(mode = "train", img_path = args.img_path, annotation_path = args.annotation_path)
    test_dataset = CatDogDataset(mode = "test", img_path = args.img_path, annotation_path = args.annotation_path)
    train_dataloader = DataLoader(train_dataset, batch_size= args.batch_size, shuffle= True)
    test_dataloader = DataLoader(test_dataset, batch_size= args.batch_size, shuffle= False)

    # For record accuracy and loss on the tensorboard
    writer = SummaryWriter("{}".format(MODELS[args.models]))

    # Load model and initialize
    model = load_model_class(args.models)
    if args.model_load_flag :
        model.load_state_dict(torch.load(os.path.join(args.model_path, args.pre_trained_model_name)))

    if _CUDA_FLAG : model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum = args.momentum)
    sigmoid = torch.nn.Sigmoid()
    
    for cur_epoch in range(args.epoch):
        # Training
        model.train()
        train_total_loss = 0.0
        train_accuracy = 0.0
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

            train_total_loss += train_loss.detach()
            train_accuracy += accuracy(sigmoid(train_outputs.cpu().detach()), train_labels.cpu().detach())
        # Calculate loss and accuracy about a epoch
        train_total_loss = train_total_loss/len(train_dataloader)
        train_accuracy = train_accuracy/len(train_dataloader)
        print("TRAIN:: EPOCH {}/{} Loss {} Accuracy {}".format(cur_epoch, args.epoch, train_total_loss, train_accuracy))

        # Testing
        model.eval()
        with torch.no_grad():
            test_total_loss = 0.0
            test_accuracy = 0.0
            for cur_batch, test_data in enumerate(test_dataloader):
                test_images, test_labels = test_data

                if _CUDA_FLAG :
                    test_images = test_images.cuda()
                    test_labels = test_labels.view(-1).cuda()

                test_outputs = model(test_images)
                test_total_loss += criterion(test_outputs, test_labels)
                test_accuracy += accuracy(sigmoid(test_outputs.cpu().detach()), test_labels.cpu().detach())
            # Calculate loss and accuracy about a epoch
            test_total_loss = test_total_loss/len(test_dataloader)
            test_accuracy = test_accuracy/len(test_dataloader)
            print("TEST:: EPOCH {}/{} Loss {} Accuracy {}".format(cur_epoch, args.epoch, test_total_loss, test_accuracy))

        model_name = "{}_{}_checkpoint.pth".format(_MODELS[args.models], cur_epoch)
        torch.save(model.state_dict(), os.path.join(args.model_path, model_name))
        writer.add_scalars("Loss", {"Train" : train_total_loss, "Test" : test_total_loss}, cur_epoch)
        writer.add_scalars("Accuracy", {"Train" : train_accuracy, "Test" : test_accuracy}, cur_epoch)
