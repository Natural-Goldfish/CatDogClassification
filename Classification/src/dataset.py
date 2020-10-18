from torch.utils.data import Dataset
from src.data_argumentation import *
import os
import json

_IMG_PATH = "data\\images"
_ANNOTATION_PATH = "data\\annotations"

class CatDogDataset(Dataset):
    def __init__(self, mode = "train"):
        super().__init__()
        self.mode = mode
        self.img_path = _IMG_PATH
        self.anno_path = _ANNOTATION_PATH
        self.class_name = ["cat", "dog"]
        self.length = len(os.listdir(os.path.join(self.img_path, self.mode)))

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with open(os.path.join(self.anno_path, "{}_annotation.json".format(self.mode)), 'r') as json_file:
            anno_file = json.load(json_file)
            image_id = anno_file[idx]["id"]
            image_path = os.path.join(self.img_path, self.mode, "{}.jpg".format(image_id))
            image = cv2.imread(image_path)
            label = [self.class_name.index(anno_file[idx]["class_name"])]
            if self.mode == "train":
                transforms = Transforms([Flip(), Resize(), Normalize(), Numpy2Tensor()])
            else :
                transforms = Transforms([Resize(), Normalize(), Numpy2Tensor()])
        return transforms((image, label))

if __name__ == "__main__" :
    dataset = CatDogDataset()
    data = dataset[5000]
    print(data[1])