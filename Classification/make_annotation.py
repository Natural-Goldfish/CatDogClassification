import os
import copy
import cv2
import json
from collections import OrderedDict

class MakeAnnotation(object):
    """
    This class copies the Kaggle images to the images folder and make annotations for training and testing dataset.
    Actually this process is not necessary, but I made it for imporve my skills.
    """
    def __init__(self):
        super().__init__()
        self.img_load_path = "data\\Kaggle"
        self.img_save_path = "data\\images"
        self.anno_save_path = "data\\annotations"
        self.purpose = ["training_set", "test_set"]
        self.class_name = ["cats", "dogs"]
        self.annotation = []
        self.anno_data = OrderedDict()
        self.image_info = OrderedDict()

    def __call__(self):
        # Make directories, if it doesn't exist.
        if not os.path.exists(self.img_save_path) : os.mkdir(self.img_save_path)
        if not os.path.exists(self.anno_save_path) : os.mkdir(self.anno_save_path)

        for purpose in self.purpose:
            if purpose == self.purpose[0] : save_purpose = "train"
            else : save_purpose = "test"
            index = 0

            # Make directories, if it doesn't exist
            save_file_path = os.path.join(self.img_save_path, save_purpose)
            if not os.path.exists(save_file_path) : os.mkdir(save_file_path)

            for class_name in self.class_name:
                # Get image file list
                wr_path = os.path.join(self.img_load_path, purpose, class_name)
                wr_list = [file_name for file_name in os.listdir(wr_path) if file_name.endswith(".jpg")]
                
                if class_name == self.class_name[0] : cls_name = "cat"
                else : cls_name = "dog"

                for file_name in wr_list:
                    save_img = cv2.imread(os.path.join(wr_path, file_name))
                    save_height, save_width, _ = save_img.shape
                    save_file_name = "{}.jpg".format(index)
                    cv2.imwrite(os.path.join(save_file_path, save_file_name), save_img)
                    self._make_annotation(index, cls_name, save_height, save_width)
                    index += 1
            
            # Save annotation file
            with open(os.path.join(self.anno_save_path, "{}_annotation.json".format(save_purpose)), "w") as json_file:
                json.dump(self.annotation, json_file, indent = 4)
            
            # Initalize the list for another purpose
            self.annotation = []
        
    def _make_annotation(self, idx, class_name, height, width):
        self.anno_data["id"] = idx
        self.anno_data["class_name"] = class_name
        self.image_info["height"] = height
        self.image_info["width"] = width
        self.anno_data["image_size"] = self.image_info
        self.annotation.append(copy.deepcopy(self.anno_data))
        
if __name__ == "__main__":
    mk = MakeAnnotation()
    mk()
