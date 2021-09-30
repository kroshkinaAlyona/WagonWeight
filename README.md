![model](https://user-images.githubusercontent.com/67489454/135474118-58f5f861-e2c8-4224-ad8f-b8815a260f0e.png)





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

