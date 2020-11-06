import argparse
from training import train
from testing import test

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', '-m', choices= ["train", "test"], required= True, \
        help = "There are two types mode, Test mode is to classify a sample image using trained model and Train mode is to train your model")
    parser.add_argument('--models', type = int, choices = [0, 1, 2, 3, 4, 5, 6], default = 0, help = "The model you will train or use")
    parser.add_argument('--model_path', type = str, default = "data\\models")
    parser.add_argument('--img_size', type = int, default = 112)

    # For training
    parser.add_argument('--epoch', type = int, default = 100)
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--learning_rate', '-lr', type = float, default = 0.001)
    parser.add_argument('--momentum', type = float , default = 0.9, \
        help = "Hyperparameters to be used on the optimizer")
    parser.add_argument('--img_path', type = str, default = "data\\images")
    parser.add_argument('--annotation_path', type = str, default = 'data\\annotations')

    # For continuous training
    parser.add_argument('--model_load_flag', action = 'store_true', \
        help = "When you want to keep training your model, set True. If it's True, you must write the name of the model you are going to load")
    parser.add_argument('--pre_trained_model_name', required= False, default = "Vgg16_20_checkpoint.pth",\
        help = "When the 'model_load_flag' is True, This is required to load pre-trained model to train continuously")       
    
    # For testing
    parser.add_argument('--test_model_name', type = str, default = 'Vgg16_20_checkpoint.pth')
    parser.add_argument('--test_image_path', type = str, default = "data\\samples",\
        help = "If you want to classify cats and dogs images using pre-trained model, put the images in this directory")
    parser.add_argument('--test_image_name',type = str, default = 'sample_0.jpg',\
        help = "This is the image file name you want to classify")
    
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    else :
        test(args)
if __name__ == "__main__":
    get_args()
