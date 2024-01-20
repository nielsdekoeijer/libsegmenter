import numpy as np
import tensorflow as tf
import checkCola as cola

class Segmenter(tf.Module):
    def __init__(self, frameSize, hopSize, window, mode="wola", edgeCorrection=True, normalizeWindow=True):
        super(Segmenter, self).__init__()
        self.hopSize = hopSize
        self.frameSize = frameSize
        self.window = window

        # asserts to ensure correctness
        if self.frameSize % 2 != 0:
            raise ValueError("only even frameSize is supported")

        if self.hopSize > self.frameSize:
            raise ValueError("hopSize cannot be larger than frameSize")

        if self.window.shape[0] != self.frameSize:
            raise ValueError("specified window must have the same size as frameSize")

        # compute prewindow and postwindow
        prewindow = np.copy(window)
        for hIdx in range(1, self.frameSize // self.hopSize + 1):
            idx1Start = hIdx * self.hopSize
            idx1End = self.frameSize
            idx2Start = 0
            idx2End = self.frameSize - idx1Start
            prewindow[idx2Start:idx2End] = prewindow[idx2Start:idx2End] + window[idx1Start:idx1End]
        
        postwindow = np.copy(window)
        for hIdx in range(1, self.frameSize // self.hopSize + 1):
            idx1Start = hIdx * self.hopSize
            idx1End = self.frameSize
            idx2Start = 0
            idx2End = self.frameSize - idx1Start
            postwindow[idx1Start:idx1End] = postwindow[idx1Start:idx1End] + window[idx2Start:idx2End]

        # this is a tiny bit hacked, but it works well in practise
        if normalizeWindow:
            value = cola.checkCola(window, hopSize)
            normalization = value[1]
            window = window / normalization
            prewindow = prewindow / normalization
            postwindow = postwindow / normalization

        if edgeCorrection == False:
            prewindow = window
            postwindow = window

        if mode == "wola":
            self.mode = mode
            window = np.sqrt(window)
            prewindow = np.sqrt(prewindow)
            postwindow = np.sqrt(postwindow)
        elif mode == "ola":
            self.mode = mode
        else :
            raise ValueError(f"only support for mode ola and wola")

        self.window = tf.convert_to_tensor(window)
        self.prewindow = tf.convert_to_tensor(prewindow)
        self.postwindow = tf.convert_to_tensor(postwindow)

    def segment(self, x):
        if((tf.rank(x) == 2) & (x.shape[1] > 1)):
            numberOfBatchElements = x.shape[0]
            numberOfSamples = x.shape[1]
        elif ((tf.rank(x) == 2) & (x.shape[1] == 1)) | (tf.rank(x) == 1):
            numberOfSamples = x.shape[0]
            numberOfBatchElements = 0
        else :
            raise ValueError(f"only support for inputs with dimension 1 or 2, provided {len(x.shape)}")
        
        numberOfFrames = (numberOfSamples) // self.hopSize - self.frameSize // self.hopSize + 1

        if numberOfBatchElements == 0:
            X = tf.zeros(shape=(numberOfFrames, self.frameSize))
            if self.mode == "wola":
                k = 0
                tmp = tf.reshape(x[k * self.hopSize : k * self.hopSize + self.frameSize], shape=(1,self.frameSize))
                X = tf.tensor_scatter_nd_add(X, [[k]], tmp*self.prewindow)
                for k in range(1,numberOfFrames-1):
                    tmp = tf.reshape(x[k * self.hopSize : k * self.hopSize + self.frameSize], shape=(1,self.frameSize))
                    X = tf.tensor_scatter_nd_add(X, [[k]], tmp*self.window)
                k = numberOfFrames-1
                tmp = tf.reshape(x[k * self.hopSize : k * self.hopSize + self.frameSize], shape=(1,self.frameSize))
                X = tf.tensor_scatter_nd_add(X, [[k]], tmp*self.postwindow)
            else :
                for k in range(numberOfFrames):
                    tmp = tf.reshape(x[k * self.hopSize : k * self.hopSize + self.frameSize], shape=(1,self.frameSize))
                    X = tf.tensor_scatter_nd_add(X, [[k]], tmp)
            return X

        else :
            X = tf.zeros(shape=(numberOfBatchElements, numberOfFrames, self.frameSize))
            for b in range(numberOfBatchElements):
                if self.mode == "wola":
                    k = 0
                    tmp = tf.reshape(x[b, k * self.hopSize : k * self.hopSize + self.frameSize], shape=(1,1,self.frameSize))
                    X = tf.tensor_scatter_nd_add(X, [[[b,k]]], tmp*self.prewindow)
                    for k in range(1,numberOfFrames-1):
                        tmp = tf.reshape(x[b, k * self.hopSize : k * self.hopSize + self.frameSize], shape=(1,1,self.frameSize))
                        X = tf.tensor_scatter_nd_add(X, [[[b,k]]], tmp*self.window)
                    k = numberOfFrames-1
                    tmp = tf.reshape(x[b, k * self.hopSize : k * self.hopSize + self.frameSize], shape=(1,1,self.frameSize))
                    X = tf.tensor_scatter_nd_add(X, [[[b,k]]], tmp*self.postwindow)
                else :
                    for k in range(numberOfFrames):
                        tmp = tf.reshape(x[b, k * self.hopSize : k * self.hopSize + self.frameSize], shape=(1,1,self.frameSize))
                        X = tf.tensor_scatter_nd_add(X, [[[b,k]]], tmp)
            return X

        

    def unsegment(self, X):
        if(tf.rank(X) == 3):
            numberOfBatchElements = X.shape[0]
            numberOfFrames = X.shape[1]
        elif (tf.rank(X) == 2):
            numberOfFrames = X.shape[0]
            numberOfBatchElements = 0
        else :
            raise ValueError(f"only support for inputs with dimension 2 or 3, provided {len(X.shape)}")
        numberOfSamples = (numberOfFrames - 1) * self.hopSize + self.frameSize

        if numberOfBatchElements == 0:
            x = tf.zeros(shape=(numberOfSamples))
            k = 0
            idx = tf.reshape(tf.range(k * self.hopSize, k * self.hopSize + self.frameSize), shape=(self.frameSize,1))
            x = tf.tensor_scatter_nd_add(x, idx, self.prewindow * X[k,:])
            for k in range(1, numberOfFrames - 1):
                idx = tf.reshape(tf.range(k * self.hopSize, k * self.hopSize + self.frameSize), shape=(self.frameSize,1))
                x = tf.tensor_scatter_nd_add(x, idx, self.window * X[k,:])
            k = k + 1
            idx = tf.reshape(tf.range(k * self.hopSize, k * self.hopSize + self.frameSize), shape=(self.frameSize,1))
            x = tf.tensor_scatter_nd_add(x, idx, self.postwindow * X[k,:])
            return x

        else :
            x = tf.zeros(shape=(numberOfBatchElements, numberOfSamples))
            for b in range(numberOfBatchElements):
                k = 0
                tmpIdx = tf.reshape(tf.range(k * self.hopSize, k * self.hopSize + self.frameSize), shape=(self.frameSize,1))
                idx = tf.concat([tf.constant(b, shape=(self.frameSize,1)), tmpIdx], axis=1)
                x = tf.tensor_scatter_nd_add(x, idx, self.prewindow * X[b,k,:])
                for k in range(1, numberOfFrames - 1):
                    tmpIdx = tf.reshape(tf.range(k * self.hopSize, k * self.hopSize + self.frameSize), shape=(self.frameSize,1))
                    idx = tf.concat([tf.constant(b, shape=(self.frameSize,1)), tmpIdx], axis=1)
                    x = tf.tensor_scatter_nd_add(x, idx, self.window * X[b,k,:])
                k = k + 1
                tmpIdx = tf.reshape(tf.range(k * self.hopSize, k * self.hopSize + self.frameSize), shape=(self.frameSize,1))
                idx = tf.concat([tf.constant(b, shape=(self.frameSize,1)), tmpIdx], axis=1)
                x = tf.tensor_scatter_nd_add(x, idx, self.postwindow * X[b,k,:])
            return x
