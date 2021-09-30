'''
    Скрипт для генерации обучайщей выборки для предсказания веса (модель автоинкодера см. картинку)
    и генерации валидационной выборки на основе показаний с Весты
'''


import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from matplotlib import colors as mcolors
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import gmtime, strftime
from tqdm import tqdm
import json
from collections import namedtuple
from scipy.interpolate import interp1d

from tensorDataset_pb2 import TensorDataset

'''
    номер контрольной точки (пример для Высочино)
'''
cp  = 17

'''
    Данные для формирования обучающей выборки
     (можно скачать свежие с помощью програмных инструментов или найти на ssd диске)
    !Set            -   осциллограммы проезда поезда по всем сенсорам (скаченные протобаф по юрлу с сервиса захвата)
    !Segmentation   -   json сегментация проездов (полученные сервисом AxisSegmentation)
    !Pattern        -   скачанные csv файлы паттернов вагонов с ппсс(выгружаются инструментом Tools.TensorIntegtationTests)  

'''
root           = f"D:\\CvLab\\LazerStock\\Tensor\\TrainWeight\\!Set\\{cp}"
segmentation   = f"D:\\CvLab\\LazerStock\\Tensor\\TrainWeight\\!Segmentation\\{cp}"
pattern        = f"D:\\CvLab\\LazerStock\\Tensor\\TrainWeight\\!Pattern\\{cp}"
outPathDirect  = "D:\\CvLab\\LazerStock\\Tensor\\TrainWeight\\XDirect"
outPathInverce = "D:\\CvLab\\LazerStock\\Tensor\\TrainWeight\\XInverce"
outPathY       = "D:\\CvLab\\LazerStock\\Tensor\\TrainWeight\\Y"

'''
    Данные для формирования валидационной выборки (зимняя поверка на Весте с разными скоростям и температурой)
        (их так же можно собрать заново из таблички поверки или взять на ssd диске)
    vesta_poverca.csv   -   табличка с вагонами в формате <trainId>;<wagonNumber>;<TGNLnumber>;<date>;<tensotWeightKg>;<VESTAWeightKg>;
    Set                 -   осциллограммы проезда поезда по всем сенсорам (скаченные протобаф по юрлу с сервиса захвата)
    Segmentation        -   json сегментация проездов (полученные сервисом AxisSegmentation)
    Pattern             -   скачанные csv файлы паттернов вагонов с ппсс(выгружаются инструментом Tools.TensorIntegtationTests)
'''
testVestaFile       = "C:\\Work\\Tensor\\Weight\\Vesta\\vesta_poverca.csv"
rootVesta           = "C:\\Work\\Tensor\\Weight\\Vesta\\Set"
segmentationVesta   = "C:\\Work\\Tensor\\Weight\\Vesta\\Segmentation"
patternVesta        = "C:\\Work\\Tensor\\Weight\\Vesta\\Pattern"
outPathVesta        = "C:\\Work\\Tensor\\Weight\\BinsPoverca"



sides = ["L", "R"]

def AxelCountShouldBe4(data_W, locosAxels):
    '''
        Количество осей в составе по данным, полученным с ппсс

        return
            количество осей в составе (с локомотивами)
            -1, если есть вагон, в котором > 4 осей
    '''
    axelInTrainCount = locosAxels   
    for axelCount in data_W[2]:
        if(axelCount != 4):
            return -1
        axelInTrainCount = axelInTrainCount + axelCount       
    return int(axelInTrainCount)

def customDecoder(studentDict):
    return namedtuple('Data',studentDict.keys())(*studentDict.values())

def _getTGNLInfo(trainPattern):
    wagonsCount = trainPattern.shape[0]

    y = np.zeros( (wagonsCount) )
    y[:] = trainPattern[4]   
    return y
def _getPhaseLineSensorInfo(outSensor, inSensor, line, locos, wagonsCount, axelsCount, trainSegmentation, protodata):    
    flagTrue = True
    # [кол-во вагонов, тележки, размер окна, 2сенсора с 2-х сторон]
    x = np.zeros((wagonsCount, 2, 700, 4))
    
    for side in range(2):
        try:
            lineSegmOut = [box for box in trainSegmentation.Data if box.Box == line*2 + side + 1 and box.Sensor == outSensor][0]          
            dataLeftOut = [frame.Frames for frame in protodata.Data if frame.BoxNumber == line*2 + side + 1 and frame.SensorNumber == outSensor-1][0]
            meanOut     = np.mean( [x.SensorData for x in dataLeftOut])
            dataLeftOut = [x.SensorData - meanOut for x in dataLeftOut]
        
            lineSegmIn  = [box for box in trainSegmentation.Data if box.Box == line*2 + side + 1 and box.Sensor == inSensor][0]            
            dataRightIn = [frame.Frames for frame in protodata.Data if frame.BoxNumber == line*2 + side + 1 and frame.SensorNumber == inSensor-1][0]
            meanIn      = np.mean( [x.SensorData for x in dataRightIn])
            dataRightIn = [x.SensorData - meanIn for x in dataRightIn]
            
            if lineSegmOut.IsBroken or lineSegmIn.IsBroken  or (len(lineSegmOut.AxisBegin) != axelsCount and len(lineSegmIn.AxisBegin) != axelsCount):
                #print(f" => broken sensor in {line+1} {sides[side]} : {inSensor} or {outSensor}")
                flagTrue = False
                break
                                                                                    
            #print(f"{line+1} {sides[side]} => {lineSegmOut.Box} ({lineSegmOut.Sensor}) {lineSegmIn.Box} ({lineSegmIn.Sensor})")
            
            # подмена проходов
            if (len(lineSegmOut.AxisBegin) != axelsCount):
                #print(f" => line {line+1} side {sides[side]}: Count of axis in sensor {outSensor}  = {len(lineSegmOut.AxisBegin)} [!={axelsCount}]")
                lineSegmOut.AxisBegin.clear()
                lineSegmOut.AxisBegin.extend(list(lineSegmIn.AxisBegin))
                lineSegmOut.AxisEnd.clear()
                lineSegmOut.AxisEnd.extend(list(lineSegmIn.AxisEnd))
            if (len(lineSegmIn.AxisBegin)  != axelsCount):
                #print(f" => line {line+1} side {sides[side]}: Count of axis in sensor {inSensor}  = {len(lineSegmIn.AxisBegin)} [!={axelsCount}]")
                lineSegmIn.AxisBegin.clear()
                lineSegmIn.AxisBegin.extend(list(lineSegmOut.AxisBegin))
                lineSegmIn.AxisEnd.clear()
                lineSegmIn.AxisEnd.extend(list(lineSegmOut.AxisEnd))
                        
            currentAxel = locos
            for wagon in range(wagonsCount):
                for target in range(2):
            
                    firstPoint   = int( (lineSegmOut.AxisEnd[currentAxel]     + lineSegmIn.AxisEnd[currentAxel]     + lineSegmOut.AxisBegin[currentAxel]     + lineSegmIn.AxisBegin[currentAxel])     / 4)
                    secondPoint  = int( (lineSegmOut.AxisEnd[currentAxel + 1] + lineSegmIn.AxisEnd[currentAxel + 1] + lineSegmOut.AxisBegin[currentAxel + 1] + lineSegmIn.AxisBegin[currentAxel + 1]) / 4)
                    
                    dist    = secondPoint - firstPoint
                    T       = int(dist * 0.8 / 1.85)
                    target1 = dataLeftOut [firstPoint - T : secondPoint + T]
                    target2 = dataRightIn [firstPoint - T : secondPoint + T]

                    xInterpolate     = np.linspace(0, len(target1), num=len(target1), endpoint=True) 
                    xInterpolate_new = np.linspace(0, len(target1), num=700, endpoint=True)                                    
                    f_1              = interp1d(xInterpolate, target1)    
                    f_2              = interp1d(xInterpolate, target2)                
                    x[wagon, target, :, 2 * side]     = f_1(xInterpolate_new)
                    x[wagon, target, :, 1 + 2 * side] = f_2(xInterpolate_new)
                    
                    #d = 70
                    #target1 = np.asarray(target1)
                    #target2 = np.asarray(target2)
                    #xcur1 = (target1[d:len(target1)] - target1[0:len(target1)-d]) / 2000
                    #xcur2 = (target2[d:len(target1)] - target2[0:len(target1)-d]) / 2000
                                                        
                    currentAxel += 2
                
        except Exception as exp:
            flagTrue = False
            print(f"Error in line : {line+1} side : {sides[side]} : {exp}")

    #if flagTrue:
        #print(x.shape)
        #for sensor in range(4):
        #    plt.plot(x[0, 0, sensor, :], color = 'green')
        #for sensor in range(4):
        #    plt.plot(x[0, 1, sensor, :], color = 'red')
        #plt.show()

    return flagTrue, x

def ToTrainSet(trainId):
    '''
        Получает данные в виде массива [.np] для одного состава и обучающую разметку [.np]
    '''   
    trainPattern = pd.read_csv(os.path.join(pattern, "{}.csv".format(trainId)), sep=';', decimal=".", header=None, skiprows=1)
    data_Locos   = pd.read_csv(os.path.join(pattern, "{}.csv".format(trainId)), sep=';', decimal=".", header = None, nrows=1)   
    locos = data_Locos[1][0]  
    axelsCount = AxelCountShouldBe4(trainPattern, locos)
    if axelsCount == -1:
        print(f"Wagon with >= 4 axels")
        return
    wagonsCount = trainPattern.shape[0]
  
    with open(os.path.join(segmentation, f"{trainId}.json"), 'r') as j:
        trainSegmentation = json.load(j, object_hook=customDecoder)
    maxline = int(np.max([box.Box for box in trainSegmentation.Data]) / 2)
    #print(f"{trainId} => Pattern count of axels = {axelsCount}, segmentation = {trainSegmentation.AxelCount} [LInes = {maxline}]")

    protodata = TensorDataset()
    data      = open( os.path.join(root, f"{trainId}.bin"), "rb").read()
    protodata.ParseFromString(data)
    #print(f"TensorDataset : {protodata.IsConsistent} {protodata.TrainId} [{len(protodata.Data)}]")
   
    y = _getTGNLInfo(trainPattern)
    y.dump(os.path.join(outPathY, f"{cp}_{trainId}__Y"))

    for line in range(maxline):    
        flag, x = _getPhaseLineSensorInfo(1, 2, line, locos, wagonsCount, axelsCount, trainSegmentation, protodata)
        if flag:         
            x.dump(os.path.join(outPathDirect, f"{cp}_{trainId}_{line+1}_{1}__X"))
        
        flag, x = _getPhaseLineSensorInfo(4, 3, line, locos, wagonsCount, axelsCount, trainSegmentation, protodata)
        if flag:
            x.dump(os.path.join(outPathInverce, f"{cp}_{trainId}_{line+1}_{2}__X"))

def GetSetForPoint():
    '''
        Собирает данные для всех поездов с указанной контрольной точки
    '''
    trains = list(set([int(x.split(".")[0]) for x in os.listdir(segmentation)]))
    print(len(trains))
    
    for train in tqdm(trains):
        try:
            ToTrainSet(train)
        except Exception as exp:
            print(f"Error get set {train} : {exp}")

def ToTestSet():
    '''
        Собирает валидационную выборку
    '''
    vesta = pd.read_csv(testVestaFile, sep=';', decimal=".", header = None)
    dataVesta = {}
    for i in range(vesta.shape[0]):
        trainId     = vesta[0].values[i]
        wagonN      = vesta[1].values[i]
        vestaWeight = vesta[5].values[i]
        
        if not trainId in dataVesta:
            dataVesta[trainId] = {}
        dataVesta[trainId][wagonN] = vestaWeight
    
    X = {}
    Y = {}
    for line in range(12):
        X[line+1] = []
        Y[line+1] = []
    
    for trainId in dataVesta:
        trainPattern = pd.read_csv(os.path.join(patternVesta, "{}.csv".format(trainId)), sep=';', decimal=".", header=None, skiprows=1)
        data_Locos   = pd.read_csv(os.path.join(patternVesta, "{}.csv".format(trainId)), sep=';', decimal=".", header = None, nrows=1)   
        locos = data_Locos[1][0]  
        axelsCount = AxelCountShouldBe4(trainPattern, locos)
        if axelsCount == -1:
            print(f"Wagon with >= 4 axels")
            return
        wagonsCount = trainPattern.shape[0]
    
        with open(os.path.join(segmentationVesta, f"{trainId}.json"), 'r') as j:
            trainSegmentation = json.load(j, object_hook=customDecoder)
        maxline = int(np.max([box.Box for box in trainSegmentation.Data]) / 2)
        print(f"{trainId} => Pattern count of axels = {axelsCount}, segmentation = {trainSegmentation.AxelCount} [LInes = {maxline}]")

        protodata = TensorDataset()
        data = open( os.path.join(rootVesta, f"{trainId}.bin"), "rb").read()
        protodata.ParseFromString(data)
        print(f"TensorDataset : {protodata.IsConsistent} {protodata.TrainId} [{len(protodata.Data)}]")
       
        for line in range(maxline):    
            flag, x = _getPhaseLineSensorInfo(4, 3, line, locos, wagonsCount, axelsCount, trainSegmentation, protodata)
            if flag:                          
                for wagon in dataVesta[trainId]:         
                    X[line+1].append([x[wagon - 1]])
                    Y[line+1].append([dataVesta[trainId][wagon]])
                                      
    for line in X:
        if X[line]!=[]:
            X[line] = np.concatenate(X[line])
            Y[line] = np.concatenate(Y[line])
            X[line].dump(os.path.join(outPathVesta, f"{line}_{2}__X"))
            Y[line].dump(os.path.join(outPathVesta, f"{line}_{2}__Y"))
            print(line, X[line].shape, len(Y[line]))


outPathDirect_flip  = "C:\\Work\\Tensor\\Weight\\XDirect_hap_flip"
def FixDirectSet():
    pathSegmentation = "C:\\Work\\Tensor\\Weight\\20"
    
    trains = [x for x in os.listdir(outPathDirect)]
    print(len(trains))
    
    for train in tqdm(trains):
        try:
            trainId = train.split('_')[1]
            with open(os.path.join(pathSegmentation, f"{trainId}.json"), 'r') as j:
                trainSegmentation = json.load(j, object_hook=customDecoder)
            dataX = np.load(os.path.join(outPathDirect, train), allow_pickle=True)    
            #dataY = np.load(os.path.join(pathTrainTest, f"{spltName[0]}_{spltName[1]}_{train}__Y"), allow_pickle=True)    
            print(dataX.shape)
            if int(trainSegmentation.Direction) == 2:
                np.flip(dataX, 0)
            
            dataX.dump(os.path.join(outPathDirect_flip, train))                  
        except Exception as exp:
            print(f"Error get set {train} : {exp}")

    
               
if __name__ == "__main__":   
          
    GetSetForPoint()
    ToTestSet() 
