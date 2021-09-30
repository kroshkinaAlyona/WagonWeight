import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib


class Generator:
    def __init__(self, root, rootY, same_time_memory_data=1, batch_size=32, shuffle=True):
        self.chunks         = list(set([x.split("__")[0] for x in os.listdir(root)]))
        self.batch_size     = batch_size
        self.root           = root
        self.rootY          = rootY
        self.same_time_data = same_time_memory_data
        self.shuffle        = shuffle
        
        self.on_epoch_end()
        self.on_datafiles_end()

    def on_epoch_end(self):
        self.indexes_datafiles = np.arange(len(self.chunks))
        if self.shuffle == True:
            np.random.shuffle(self.indexes_datafiles)
        self.epoch_step = 0

    def on_datafiles_end(self):   
        self.X = []
        self.Y = []

        current_datafiles = self.indexes_datafiles[self.epoch_step*self.same_time_data:(self.epoch_step+1)*self.same_time_data]
        if len(current_datafiles) == 0:
            self.on_epoch_end()
            self.on_datafiles_end()
            return

        for target_train in current_datafiles:          
            #for line in range(16):
            #try:
            xcur    = np.load(os.path.join(self.root, f"{self.chunks[target_train]}__X"), allow_pickle=True)
            xcur    = self.transform(xcur)
            self.X.append(xcur)
            #print(self.chunks[target_train])
            
            cp      = self.chunks[target_train].split('_')[0]
            #cp1   = self.chunks[target_train].split('_')[1]
            train   = self.chunks[target_train].split('_')[1]
            ycur    = np.load(os.path.join(self.rootY, f"{cp}_{train}__Y"), allow_pickle=True) / 100000                       
            #print(ycur)
            self.Y.append(ycur)
            #except:
            #    continue
                  
        self.X = np.concatenate(self.X)
        self.Y = np.concatenate(self.Y)
        #print(current_datafiles)

        self.indexes_data = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes_data)
        self.epoch_step += 1
        if self.epoch_step*self.same_time_data >= len(self.chunks):
            self.on_epoch_end()

    def transform(self, xcur, d = 70):    
        #xcur = (xcur[:, :, d:700, :] - xcur[:, :, 0:700-d, :]) 
        xcur = xcur / 2000
        #xcur = np.reshape(xcur, (xcur.shape[0], xcur.shape[1]*xcur.shape[2], xcur.shape[3]))
        #print(xcur.shape)
        #plt.plot(xcur[0,0,0,:])
        #plt.plot(xcur[0,0,1,:])
        #plt.plot(xcur[0,0,2,:])
        #plt.plot(xcur[0,0,3,:])
        #plt.show()
        return xcur

    def gen(self):
        while True:
            for batch_idxs in self.split_batch(self.indexes_data, self.batch_size):
                yield ({'input1': self.X[batch_idxs , 0], 'input2': self.X[batch_idxs , 1]}, {'output': self.Y[batch_idxs], 'output_targ1': self.X[batch_idxs , 0], 'output_targ2': self.X[batch_idxs , 1] })
                #Xtrain = self.transform(self.X[batch_idxs], 70)
                #yield (Xtrain, Xtrain)
            self.on_datafiles_end()

    def split_batch(self, indexes_data, batch_size):
        target = []
        tmp = []
        for el in indexes_data:
            tmp.append(el)
            if len(tmp) == batch_size:
                target.append(tmp)
                tmp = []
        if len(tmp) != 0:
            target.append(tmp)
        return target

