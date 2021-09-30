import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from matplotlib import colors as mcolors
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import gmtime, strftime

from keras.layers import Input, LSTM, Dense, Bidirectional, concatenate, Flatten, Dropout, TimeDistributed
from keras.layers import Lambda, MaxPooling1D, RepeatVector, Reshape
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Model
from keras import backend as back
from keras.models import load_model
import keras

from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn import model_selection
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import DBSCAN

import xgboost as xgb
from lightgbm import LGBMRegressor

import tensorflow as tf
from tensorflow.python.client import device_lib

from _generator import Generator 

pathTrainX = "C:\\Work\\Tensor\\Weight\\XDirect"
pathTrainY = "C:\\Work\\Tensor\\Weight\\Y"

pathTest = "C:\\Work\\Tensor\\Weight\\Bins"

latentDim = 64
class CustomVariationalLayer():
    def vae_loss(self, x, z_decoded, z_mean, z_log_var):
        x = back.flatten(x)
        z_decoded = back.flatten(z_decoded)
        xent_loss = keras.losses.mean_squared_error(x, z_decoded)
        kl_loss = -5e-7 * back.mean(1 + z_log_var - back.square(z_mean) - back.exp(z_log_var), axis = -1)
        return back.mean(xent_loss)
        
    def call(self, inputs):
        x = inputs[0]
        
        loss = self.vae_loss(inputs[0], inputs[1], inputs[2], inputs[3])
        self.add_loss(loss)
        return x

def splitL(x):    
    return tf.split(x, num_or_size_splits=2, axis=1)  

def sampling(args):
   z_mean, z_log_var = args
   epsilon = back.random_normal(shape = (back.shape(z_mean)[0], latentDim), mean = 0., stddev=0.2)
   x = z_mean + back.exp(z_log_var) * epsilon
   
   #x = -0.5 * (1 + z_log_var - back.sqrt(z_mean) - back.exp(z_log_var))   
   #x = 10 * back.abs(x)
   #x = 10 * x   
   return x
               
def AEC(window, sensors):  
    inputTarget1 = Input(shape=(window, sensors), name='input1')
    inputTarget2 = Input(shape=(window, sensors), name='input2')
    
    #coder
    x = LSTM(
        1,
        activation = 'tanh',
        return_sequences = True,
        recurrent_dropout = 0.2
    )(inputTarget1)
    x = Flatten()(x)
    #x = Dense(512, activation="relu")(x)

    y = LSTM(
        1,
        activation = 'tanh',
        return_sequences = True,
        recurrent_dropout = 0.2
    )(inputTarget2)
    y = Flatten()(y)
    #y = Dense(512, activation="relu")(y)

    combined = concatenate([x, y])

    z           = Dense(latentDim, activation = "relu")(combined)
    z_mean      = Dense(latentDim, name = "mean")(z)
    z_log_var   = Dense(latentDim, name = "log_var")(z)
    z           = Lambda( sampling )([ z_mean, z_log_var])

    z_out = Dense(1, activation="relu", name='output')(z)
    
    outputTarget1_out = Dense(window, activation="relu")(z)     
    outputTarget1_out = Reshape((window, 1))(outputTarget1_out)  
    outputTarget1_out = Dense(4, name='output_targ1')(outputTarget1_out)   
    
    outputTarget2_out = Dense(window, activation="relu")(z)     
    outputTarget2_out = Reshape((window, 1))(outputTarget2_out)  
    outputTarget2_out = Dense(4, name='output_targ2')(outputTarget2_out)
           
      
    model = Model(inputs = [inputTarget1, inputTarget2], outputs = [z_out, outputTarget1_out, outputTarget2_out])
    model.compile(optimizer ='rmsprop', loss = 'mse')

    print(model.summary())
    keras.utils.plot_model(model, "model.png", show_shapes = True)

    return model


def TrainGenerator(model):       
    curr_time   = strftime("%d-%m_%H-%M", gmtime())
    filepath    = "WeightTGNL_vis-{epoch:02d}-{loss:.4f}.h5"
    chkVal      = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=False)
    chkTrain    = ModelCheckpoint(filepath, monitor='loss', verbose = 1, save_best_only=False, mode = 'max')
    tbCallBack  = TensorBoard(log_dir="Graph", histogram_freq=0, write_graph=True, write_images=True)
    callbacks_list = [chkTrain, chkVal, tbCallBack]   
    
    generator = Generator(pathTrainX, pathTrainY, same_time_memory_data=20, batch_size=16)

    Xval, Yval = LoadValidation()

    model.fit_generator(                            
        generator.gen(),
        steps_per_epoch = 4562,
        epochs          = 15,
        callbacks       = callbacks_list,
        validation_data = ({'input1': Xval[: , 0, :, :], 'input2': Xval[:, 1, :, :]}, {'output': Yval, 'output_targ1': Xval[: , 0, :, :], 'output_targ2': Xval[:, 1, :, :]})
        )    
    model.save(f"WeightTGNL_vis_last{curr_time}.h5")

def GetTrainSize(batch):
    trains = os.listdir(pathTrainX)
    count = 0
    for data in trains:
        dataNp = np.load(os.path.join(pathTrainX, data), allow_pickle=True)
        count = count + dataNp.shape[0]
    
    print('count {} step {}'.format(count, count / batch))    

def TestAEC(model = None):
    path = "WeightTGNL_vallLoss_Inv_28-01_14-30.h5"
    
    #model = load_model(path)
    model = AEC(700, 4)
    model.load_weights(path)
    #model.set_weights(weight)
    Xval, Yval = LoadValidation()
    
    #plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)

    YPred, Xt1, Xt2 = model.predict( {'input1': Xval[: , 0, :, :], 'input2': Xval[:, 1, :, :]} )
    diff = []
    for i in range(YPred.shape[0]):
        print(Yval[i]*100000,  int(YPred[i]*100000))
        diff.append( (Yval[i] - YPred[i]) / float(Yval[i]) * 100)
    diff = np.asarray(diff)
    

    mean = np.mean(diff)
    std  = np.std(diff)
    print(f" mean = {mean} +- {std}")
  
def LoadValidation(d = 30):
    dataX = []
    dataY = []
    
    listX = list(set([x.split("__")[0] for x in os.listdir(pathTest)]))
    for x in listX:
        _x = np.load(os.path.join(pathTest, f"{x}__X"), allow_pickle=True)
        _y = np.load(os.path.join(pathTest, f"{x}__Y"), allow_pickle=True)
        for i in range(len(_y)):
            if _y[i] != 0:
                dataX.append( [_x[i]] )
                dataY.append( [_y[i]] )
    
    dataX = np.concatenate(dataX)
    dataX = dataX / 2000
    dataY = np.concatenate(dataY) / 100000
    print(dataX.shape, dataY.shape)

    return dataX, dataY
    
def TestValidation():
    path = "WeightTGNL_vallLoss_Inv_28-01_14-30.h5"
    #model = load_model(path)
    model = AEC(700, 4)
    model.load_weights(path)
     
    diff  = {}
    listX = list(set([x.split("__")[0] for x in os.listdir(pathTest)]))
    for x in listX:
        X  = np.load(os.path.join(pathTest, f"{x}__X"), allow_pickle=True)
        X = X / 2000       
        print(x)
        print(X.shape)

        #plt.plot(X[1 , 0, :, 0])
        #plt.plot(X[1 , 0, :, 1])
        #plt.plot(X[1 , 0, :, 2])
        #plt.plot(X[1 , 0, :, 3])
        
        
        YPred, T1, T2 = model.predict( {'input1': X[: , 0, :, :], 'input2': X[:, 1, :, :]} )

        #plt.plot(T1[1 , :, 0], color = 'black')
        #plt.plot(T1[1 , :, 1], color = 'black')
        #plt.plot(T1[1 , :, 2], color = 'black')
        #plt.plot(T1[1 , :, 3], color = 'black')
        #plt.show()

        for i in range(YPred.shape[0]):
            if not i in diff: diff[i] = []
            diff[i].append( YPred[i])
    
    Y = np.load(os.path.join(pathTest, f"{1}_2__Y"), allow_pickle=True) / 100000
    for wagon, values in diff.items():
        mean   = int(np.mean(values)*100000)
        etalon = int(Y[wagon]*100000)
        print(  f" {mean} {etalon} =>  { (mean - etalon)/mean * 100 }"  )


def TestTrain():
    path = "WeightTGNL_koch_bestLoss01-03_12-09.h5"
    #model = load_model(path)
    model = AEC(700, 4)
    model.load_weights(path)
    
    weight = {}   
    listX = list(set([x.split("__")[0] for x in os.listdir(pathTrainTest)]))
    for x in listX:
        try:
            spltName = x.split('_')
            line = spltName[3]
            train = spltName[2]
            

            dataX = np.load(os.path.join(pathTrainTest, f"{x}__X"), allow_pickle=True) / 2000     
            dataY = np.load(os.path.join(pathTrainTest, f"{spltName[0]}_{spltName[1]}_{train}__Y"), allow_pickle=True) / 100000    
            print(line, dataX.shape, dataY.shape)
            
            YPred, T1, T2 = model.predict( {'input1': dataX[: , 0, :, :], 'input2': dataX[:, 1, :, :]}) 
            for wagon in range(dataX.shape[0]):
                if not (wagon+1) in weight:
                    weight[wagon+1] = [] 

                weight[wagon+1].append(int(YPred[wagon]*100000))
        except:
            pass
    
    diff = []
    for wagon in weight:
        print(f'wagon={wagon} => tgnl={dataY[wagon-1]*100000} predict={int(np.mean(weight[wagon]))}')
        print(weight[wagon])
        #if wagon == 50:
        #    continue
        diff.append(  (dataY[wagon-1]*100000 - np.mean(weight[wagon])) / dataY[wagon-1]/100000 * 100 )

    print(diff)
    print(np.mean(diff), np.std(diff))

if __name__ == "__main__":   
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(devices=physical_devices[0], device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)      
        
    #LoadValidation()
    
    #GetTrainSize(16)
        
    model = AEC(700, 4)
    TrainGenerator(model)
    #TestAEC()
    #TestValidation()
    #TestTrain()
    
 
   
