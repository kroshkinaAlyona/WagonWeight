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

