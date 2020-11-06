# Introduction  
    
</br></br>
# How to use my code  
It will be uploaded, if I solve the problem while uploading the .pth file
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
This will help you understand where the code files exist and where the data, kind of images and weight files, are stored. The structure looks like below :  
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
I used  _"[Cats and Dogs](https://www.kaggle.com/tongpython/cat-and-dog)"_ dataset in Kaggle, in which there are 8004 images for training and 2022 images for testing. The  structure looks like below :  

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
In order to make this structure simple and make annotation files for training, I used _```make_annotation.py```_.
The annotation file written by json have three imformation of images, which are _identification_, _class name_ and _image size_.
Honestly, in this project, the original image size doens't need. But I added it to practice the way to make an annotation file for the other projects kind of _Object Detection_.
The new images and annotation files which are generated by _make_annotation.py_ are saved to _```'data/images'```_ and _```'data/annotations'```_ for each. The annotation file which is a part of _```test_annotation.json```_ looks like below :   

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
                .
                .
                .
    {
        "id": 2021,
        "class_name": "dog",
        "image_size": {
            "height": 396,
            "width": 499
        }
    },
    {
        "id": 2022,
        "class_name": "dog",
        "image_size": {
            "height": 374,
            "width": 500
        }
    }
]
```
 The following images are some of the dataset images.</br></br>
<p align="center"><img src='https://github.com/Natural-Goldfish/CatDogClassification/blob/master/Classification/data/samples/sample_0.jpg?raw=true' width = "200px" height = "200px"/>
<img src='https://github.com/Natural-Goldfish/CatDogClassification/blob/master/Classification/data/samples/sample_2.jpg?raw=true' width = "200px" height = "200px"/>
<img src='https://github.com/Natural-Goldfish/CatDogClassification/blob/master/Classification/data/samples/sample_11.jpg?raw=true' width = "200px" height = "200px"/>
<img src='https://github.com/Natural-Goldfish/CatDogClassification/blob/master/Classification/data/samples/sample_9.jpg?raw=true' width = "200px" height = "200px"/></p>
</br></br>

# Settings  

- **Model Structure**</br></br>
As I said on the indroduction, 7 different classification models are trained. All of the network architecture are based on _'VGG-16'_, but the depth of the network are the biggest differences. Starting with 2-layers, I stackted layers up to 6-layers. In the case of pre-trained VGG-16, only the classifier layer is changed.  
<p align="center"><img src = 'https://github.com/Natural-Goldfish/CatDogClassification/blob/master/README/Convnet%20Configuration.PNG?raw=true' width = "800px" height = "400px"/></p>
</br></br>

- **Loss**</br></br>
I used _CrossEntropyLoss_ in pytorch
</br></br>

- **Optimizer**</br></br>
I used _SGD optimizer_ with the default momentum values. _Learning rate : 1e-3_.
</br></br>

- **Data Argumentation**</br></br>
I performed data argumentation to make model more stable and to complement the small dataset. Techniques applied here are _resize_, _normalization_, _horizontal flip with random probability_.  
</br></br>

# Measurement  
- **Accuracy**</br></br>
The formula to check the model's performance is :  ``` ACC = True Positive / Dataset ```
</br></br>

# Experiments  
I trained the models for different epochs each with _64 batch size_. The highest accuracy and the epoch are recorded below :  
| Model | Epoch | Accuracy |  
|:---:|:---:|:---:|
| A | 52 | 77% |
| B | 81 | 79% |
| C | 77 | 76% |  
| D | 88 | 63% |  
| E | 30 | 53% |  
| F | 21 | 92% |  
| G | 43 | 88% |  

The training loss and accuracy curves for each experiment are shown below:  
- **2-Layers(A)**  
<p align="center"><img src = 'https://github.com/Natural-Goldfish/CatDogClassification/blob/master/README/2l_accuracy-52%20epoch.PNG?raw=true' width = "800px" height = "200px"/>
<img src = 'https://github.com/Natural-Goldfish/CatDogClassification/blob/master/README/2l_loss-52%20epoch.PNG?raw=true' width = "800px" height = "200px"/>
</p>  

- **3-Layers(B)**  
<p align="center"><img src = 'https://github.com/Natural-Goldfish/CatDogClassification/blob/master/README/3l_accuracy-85%20epoch.PNG?raw=true' width = "800px" height = "200px"/>
<img src = 'https://github.com/Natural-Goldfish/CatDogClassification/blob/master/README/3l_loss-85%20epoch.PNG?raw=true' width = "800px" height = "200px"/>
</p>  

- **4-Layers(C)**  
<p align="center"><img src = 'https://github.com/Natural-Goldfish/CatDogClassification/blob/master/README/4l_accuracy-77%20epoch.PNG?raw=true' width = "800px" height = "200px"/>
<img src = 'https://github.com/Natural-Goldfish/CatDogClassification/blob/master/README/4l_loss-77%20epoch.PNG?raw=true' width = "800px" height = "200px"/>
</p>  

- **5-Layers(D)**  
<p align="center"><img src = 'https://github.com/Natural-Goldfish/CatDogClassification/blob/master/README/5l_accuracy-90%20epoch.PNG?raw=true' width = "800px" height = "200px"/>
<img src = 'https://github.com/Natural-Goldfish/CatDogClassification/blob/master/README/5l_loss-90%20epoch.PNG?raw=true' width = "800px" height = "200px"/>
</p>  

- **6-Layers(E)**  
<p align="center"><img src = 'https://github.com/Natural-Goldfish/CatDogClassification/blob/master/README/6l_accuracy-31%20epoch.PNG?raw=true' width = "800px" height = "200px"/>
<img src = 'https://github.com/Natural-Goldfish/CatDogClassification/blob/master/README/6l_loss-31%20epoch.PNG?raw=true' width = "800px" height = "200px"/>
</p>  

- **Pre-trained VGG16(F)**  
<p align="center"><img src = 'https://github.com/Natural-Goldfish/CatDogClassification/blob/master/README/Vgg16_accuracy-22%20epoch.PNG?raw=true' width = "800px" height = "200px"/>
<img src = 'https://github.com/Natural-Goldfish/CatDogClassification/blob/master/README/Vgg16_loss-22%20epoch.PNG?raw=true' width = "800px" height = "200px"/>
</p>  

- **5-Layers with BN(G)**  
<p align="center"><img src = 'https://github.com/Natural-Goldfish/CatDogClassification/blob/master/README/5lbn_accuracy-43%20epoch.PNG?raw=true' width = "800px" height = "200px"/>
<img src = 'https://github.com/Natural-Goldfish/CatDogClassification/blob/master/README/5lbn_loss-43%20epoch.PNG?raw=true' width = "800px" height = "200px"/>
</p>  
</br></br>

# Results  
Some test images are randomly selected and shown below :</br></br>
<p align="center"><img src='https://github.com/Natural-Goldfish/CatDogClassification/blob/master/Classification/data/samples/sample_0.jpg?raw=true' width = "200px" height = "200px"/>
<img src='https://github.com/Natural-Goldfish/CatDogClassification/blob/master/Classification/data/samples/sample_1.jpg?raw=true' width = "200px" height = "200px"/>
<img src='https://github.com/Natural-Goldfish/CatDogClassification/blob/master/Classification/data/samples/sample_2.jpg?raw=true' width = "200px" height = "200px"/>
<img src='https://github.com/Natural-Goldfish/CatDogClassification/blob/master/Classification/data/samples/sample_3.jpg?raw=true' width = "200px" height = "200px"/>
<img src='https://github.com/Natural-Goldfish/CatDogClassification/blob/master/Classification/data/samples/sample_6.jpg?raw=true' width = "200px" height = "200px"/>
<img src='https://github.com/Natural-Goldfish/CatDogClassification/blob/master/Classification/data/samples/sample_7.jpg?raw=true' width = "200px" height = "200px"/>
<img src='https://github.com/Natural-Goldfish/CatDogClassification/blob/master/Classification/data/samples/sample_8.jpg?raw=true' width = "200px" height = "200px"/>
<img src='https://github.com/Natural-Goldfish/CatDogClassification/blob/master/Classification/data/samples/sample_9.jpg?raw=true' width = "200px" height = "200px"/>
</p>



