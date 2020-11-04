# Introduction  
    
</br></br>
# How to use my code  
- In order to use my code, just follow this :  
```
    git clone -b master https://github.com/Natural-Goldfish/CatDogClassification.git  
```
```
    cd CatDogClassification\\Classification
```  
**❗ You must choose wihch mode you will run between _'train' or 'test'_.**
```
    python main.py --mode {}
```
- For more specific information how to run my code, you could run :
```  
    python main.py -h  
```
</br></br>

# Requirements
```
    - python 3.7.1
    - pytorch 1.6.0
    - opencv 4.4.0
    - numpy 1.15.4  
```
</br></br>

# Project Structure  
This show how my project files stored. The structure looks like below :  
```
    Classification
    ├──data
    │   ├──Kaggle
    │   │   ├──test_set
    │   │   │   ├──cats
    │   │   │   │   ├──cat.4001.jpg
    │   │   │   │   ├──cat.4002.jpg
    │   │   │   │   └──...
    │   │   │   └──dogs
    │   │   │       ├──dog.4001.jpg
    │   │   │       ├──dog.4002.jpg
    │   │   │       └──...    
    │   │   └──training_set
    │   │       ├──cats
    │   │       │   ├──cat.1.jpg
    │   │       │   ├──cat.2.jpg
    │   │       │   └──...
    │   │       └──dogs
    │   │           ├──dog.1.jpg
    │   │           ├──dog.2.jpg
    │   │           └──...
    │   ├──images
    │   │   ├──test
    │   │   │   ├──0.jpg
    │   │   │   ├──1.jpg
    │   │   │   └──... 
    │   │   └──train
    │   │       ├──0.jpg
    │   │       ├──1.jpg
    │   │       └──...       
    │   ├──annotations
    │   │   ├──test_annotation.json
    │   │   └──train_annotation.json
    │   ├──models
    │   │   ├──CatDogClassifier_{layer}_{epoch}_checkpoint.pth
    │   │   └──Vgg16_{epoch}_checkpoint.pth
    │   └──samples
    │       ├──sample_0.jpg
    │       ├──sample_0.jpg
    │       └──...
    ├──src
    │   ├──__init__.py
    │   ├──data_argumentation.py
    │   ├──dataset.py
    │   ├──network.py
    │   └──utils.py
    ├──__init__.py
    ├──main.py
    ├──make_annotation.py
    ├──training.py
    └──testing.py  
```
</br></br>

# Dataset  
I used  _"[Cats and Dogs](https://www.kaggle.com/tongpython/cat-and-dog)"_ dataset in Kaggle. The dataset structure looks like below :  

```
    Dataset
    ├──test_set
    │   └──test_set    
    │      ├──cats
    │      │   ├──_DS_Store
    │      │   ├──cat.4001.jpg
    │      │   ├──cat.4002.jpg
    │      │   └──...
    │      └──dogs
    │          ├──_DS_Store
    │          ├──dog.4001.jpg
    │          ├──dog.4002.jpg
    │          └──...
    │      
    └──training_set
        └──training_set   
           ├──cats
           │   ├──_DS_Store
           │   ├──cat.1.jpg
           │   ├──cat.2.jpg
           │   └──...
           └──dogs
               ├──_DS_Store
               ├──dog.1.jpg
               ├──dog.2.jpg
               └──...
```  
In order to make this structure simple and make annotation files for training, I used _make_annotation.py_.
The annotation file written by json have three imformation of images, which are identification, class name and image size. Those outputs which are generated by 
```
[
    {
        "id" : 0,
        "class_name" : "cat",
        "image_size" : {
            "height" : 415,
            "width" : 498
        }
    },
    {
        "id": 1,
        "class_name": "cat",
        "image_size": {
            "height": 499,
            "width": 375
        }
    },
    ...
]
```
This is an example of the _test_annotation.json_. _train_annotation.json_ also has same structure.
These are cropped images, so you don't need to additional working to make a dataset for training.</br></br>
<p align="center"><img src='https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/images/1.png' width = "200px" height = "200px"/>
<img src='https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/images/4.png' width = "200px" height = "200px"/>
<img src='https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/images/3.png' width = "200px" height = "200px"/>
<img src='https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/images/14.png' width = "200px" height = "200px"/></p>
</br></br>

# Settings  
- **Model Structure**</br></br>
I followed DC-GAN paper's model architecture</br></br>
- **Loss**</br></br>
I used _BCELoss_ in pytorch</br></br>
- **Optimizer**</br></br>
I used _Adam optimizer_ with the β1 and β2 of default values. _Learning rate : 2e-4_</br></br>
- **Data Argumentation**</br></br>
I performed data argumentation to make model more stable and to complement the small dataset. Techniques applied here are _resize_, _normalization_, _horizontal flip with random probability_.  
</br></br>

# Train  
I trained the model for _400 epochs_ about the dataset by _64 batch size_. You can find this pre-trained model's parameter file in _```'data\models'```_  
The update cycle of disciminator model for each batch size is a little bit changed comapred to the paper.  
| Epoch | Discriminator : Generator |  
|:---:|:---:|
| 0 ~ 50 | 1 : 2 |
| 50 ~ end | 1 : 1|  
- If you want to train this model from beginning, you could run :  
```
python main.py --mode train
```
- If you want to train pre-trained model, you could run :  
``` 
python main.py --mode train --model_load_flag --generator_load_name {} --discriminator_load_name {}
```
</br></br>

# Test  
You can generate images using pre-trained model, which are saved in _```'data\generated_images'```_  
- If you want to see a generated image which pre-trained model make, just run :  
``` 
python main.py --mode test
```  
- Also you can choice the number of images to generate by changing _**'--generate_numbers'**_, you could run :  
``` 
python main.py --mode test --generate_numbers {}
```  
- If you want to change the directory as well, you could run :  
``` 
python main.py --mode test --generating_model_name {} --image_save_path {} --generate_numbers {}
```  
</br></br>

# Results  
Some generated images are shown below :</br></br>
<p align="center"><img src='https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/generated_images/Generated_img0.jpg' width = "100px" height = "100px"/>
<img src='https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/generated_images/Generated_img2.jpg' width = "100px" height = "100px"/>
<img src='https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/generated_images/Generated_img3.jpg' width = "100px" height = "100px"/>
<img src='https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/generated_images/Generated_img4.jpg' width = "100px" height = "100px"/>
<img src='https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/generated_images/Generated_img5.jpg' width = "100px" height = "100px"/>
<img src='https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/generated_images/Generated_img6.jpg' width = "100px" height = "100px"/>
<img src='https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/generated_images/Generated_img7.jpg' width = "100px" height = "100px"/>
<img src='https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/generated_images/Generated_img8.jpg' width = "100px" height = "100px"/>
<img src='https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/generated_images/Generated_img9.jpg" width = "100px" height = "100px"/></p>



