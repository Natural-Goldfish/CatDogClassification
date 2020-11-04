from src.network import *
from src.data_argumentation import *
from src.utils import load_model_class
import os

def test(args):
    model = load_model_class(args.models)
    model.load_state_dict(torch.load(os.path.join(args.model_path, args.test_model_name)))
    model.eval()

    sigmoid = torch.nn.Sigmoid()
    transforms = Transforms([Resize(args.img_size, args.img_size), Normalize(), Numpy2Tensor()])

    img = cv2.imread(os.path.join(args.test_image_path, args.test_image_name))
    input_img, _ = transforms((img, 0))
    input_img = torch.unsqueeze(input_img, 0)
    output = model(input_img)

    prediction, idx = torch.max(sigmoid(output), dim= 1)
    if idx == 0 :
        cv2.imshow("Image", img)
        print(prediction.item()*100, "cat")
        cv2.waitKey(0)
    else :
        cv2.imshow("Image", img)
        print(prediction.item()*100, "dog")
        cv2.waitKey(0)
