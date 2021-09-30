***Script’s description***

1. *_toTrainSet.py* - a script for obtaining training and validation samples from raw data (see the inside for a detailed description of the data). At the output, it receives the following data in numpy format:


**Train sample**

[control point]_[train id]_[weighing line number]_[forward / reverse sensor]__X
- fragmented pieces of oscillograms of a carriage passage (numpy arrays)
 
 [number of cars, bogies = 2, window size = 700, 2 sensors on 2 sides = 4]

[control point]_[train id]__Y
- numpy array with the weight of each car according to Vesta
 
 [number of cars, weight according to Vesta]


**Validation sample**

TestWagonsVesta - obtained from the results of winter verification at Vesta (repeated passes of the same train at different speeds, different temperatures and different car weights)

[weighing line number]_[forward/reverse sensor]__X
- fragmented pieces of oscillograms of a carriage passage (numpy arrays)

[number of cars, bogies = 2, window size = 700, 2 sensors on 2 sides = 4]


[weighing line number]_[forward/reverse sensor]__Y
- numpy array with the weight of each car according to Vesta

[number of cars, weight according to Vesta]


2. *_train.py* - script to start the training and validation process. The mesh model has a weight prediction autoencoder structure (see the picture below)

![model](https://user-images.githubusercontent.com/67489454/135476775-641e2bec-3647-4378-b83d-a5b96e0bc6dc.png)


3. *_generator.py* - the script is necessary for uniform swapping of data with the training process
