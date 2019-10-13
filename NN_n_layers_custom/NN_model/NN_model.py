# -*- codingL utf-8 -*-

import time
import numpy as np
from NN_n_layers_custom import formulas as f
from NN_n_layers_custom import preprocessing as pre


def logging_time(original_fn): #decorator_review
    def wrapper_fn(*args, **kargs):
        start_time = time.time()
        result = original_fn(*args, **kargs)
        end_time = time.time()

        print(f"Working Time[{original_fn.__name__}] : {end_time - start_time}")
        return result
    return wrapper_fn


class Custom_NN_model:
    def __init__(self, X, Y, nLayer = 7, nNode = 500):
        if not isinstance(X, np.ndarray):
            raise TypeError("X's type should be <%s>. But it's %s"%("np ndarray",type(X)))
        elif not isinstance(Y, np.ndarray):
            raise TypeError("Y's type should be <%s>. But it's %s"%("np ndarray",type(Y)))
        assert len(X) == len(Y), "The number of examples should be equal to the number of ground truth"
        
        self.nLayer = nLayer
        self.nNode  = nNode
        self.XYsetter(X, Y)
        self.param = {}
        self.loss = []
        self.lr = 0.000001 # learning rate
        self.out_A = {}
        self.in_Z = {}
        self.Delta = {}
        self.gradient = {}
        
        ## 데이터의 차원이 바뀌면 달라져야 한다... classification feature 에만 적용 가능
        #self.dims = [self.X.shape[1], self.nNode, self.Y.shape[1]]


        # 레이어마다 들어갈 함수 순서대로 지정
        self.function = [f.Relu, f.Sigmoid, f.Sigmoid, f.Sigmoid, f.Sigmoid, f.Sigmoid, f.Sigmoid]
        self.dfunction = [f.dRelu, f.dSigmoid, f.Sigmoid, f.Sigmoid, f.Sigmoid, f.Sigmoid, f.Sigmoid, f.dCrossEntropy]

    def XYsetter(self, x,y):
        def unison_random_shuffle(*arg):
            for i in range(len(arg)):
                assert len(arg[0]) == len(arg[i])
            p = np.random.permutation(len(arg[0]))
            return ( entry[p] for entry in arg)   
        x, y = x.reshape(-1,1), y.reshape(-1,1)
        self.X, self.Y = unison_random_shuffle(x, y)

        self.sam = self.Y.shape[0] # # The number of training samples we have(or the batch size)
        self.loss = []

    def param_init(self):
        for i in range(1, self.nLayer +1):
            
            if i == 1: # First Layer
                self.param[f'W{i}'] = np.random.normal(size = self.dims[0] * self.dims[1]).reshape(self.dims[0], self.dims[1])
                self.param[f'b{i}'] = np.ones(self.dims[1]) #dimension 이 다르지만 문제가 발생하지는 않는다.
            elif i == self.nLayer: # Last Layer
                self.param[f'W{i}'] = np.random.normal(size = self.dims[1] * self.dims[2]).reshape(self.dims[1], self.dims[2])
                self.param[f'b{i}'] = np.ones(self.dims[2])
            else: # Layers in between
                self.param[f'W{i}'] = np.random.normal(size = self.dims[1] * self.dims[1]).reshape(self.dims[1], self.dims[1])
                self.param[f'b{i}'] = np.ones(self.dims[1])
        
        return self.param # for testing __field is not a private field

    
    def frontprop(self, X, Y): # get A, Z
        self.out_A['A0'] = X
        
        for i in range(1, self.nLayer+1):
            self.in_Z[f'Z{i}'] = np.matmul(self.out_A[f'A{i-1}'], self.param[f'W{i}']) + self.param[f'b{i}']
            self.out_A[f'A{i}'] = self.function[i-1](self.in_Z[f'Z{i}'])
        
        y_hypothesis = self.out_A[f'A{self.nLayer}']
 
        _loss = np.sum(f.CrossEntropy(y_hypothesis, Y))
        self.loss.append(_loss)

    def backprop(self, Y):
        #delta of last layer
        dLoss = f.dCrossEntropy # last entry of the dfunction list is the derivative of loss function

        # the Delta of last layer
        self.Delta[f'D{self.nLayer}'] = dLoss(self.out_A[f'A{self.nLayer}'], Y) * self.dfunction[self.nLayer-1](self.in_Z[f'Z{self.nLayer}'])
        
        for i in reversed(range(1, self.nLayer)):
            #rest
            self.Delta[f'D{i}'] = np.dot(self.Delta[f'D{i+1}'], self.param[f'W{i+1}'].T) * self.dfunction[i-1](self.in_Z[f'Z{i}'])

    def gradients(self): # get the gradients
        for i in range(1, self.nLayer + 1):
            self.gradient[f'dW{i}'] = (1/self.sam) * np.matmul(self.out_A[f'A{i-1}'].T, self.Delta[f'D{i}'])
            self.gradient[f'db{i}'] = (1/self.sam) * np.sum(self.Delta[f'D{i}'], axis = 0, keepdims=True)

    def gradientdescent(self): # execute gradient descent
        for i in range(1, self.nLayer + 1):
            self.param[f'W{i}'] -= self.lr * self.gradient[f'dW{i}']
            self.param[f'b{i}'] -= self.lr * np.squeeze(self.gradient[f'db{i}'])

    def minibatch_setter(self, batch_size):
        self.XYsetter(self.X, self.Y) #shuffling 해줌
        nBatch = int(self.X.shape[0]/batch_size)

        ## use index to mini_batches, for saving memory
        idx = [(i * batch_size, (i+1) *batch_size) if i != nBatch - 1 else (i*batch_size, self.X.shape[0]) for i in range(nBatch)]
        return idx

    @logging_time
    def minibatch_gradientdescent(self, batch_size = 32, nEpoch = 3000):
        for i in range(nEpoch):
            idx = self.minibatch_setter(batch_size) #minibatch_setter 안에 XYsetter가 들어가 있습니다 (shuffling을 해줍니다.)
            cnt = 0
            for batch_begin, batch_end in idx:
                cnt += 1
                self.sam = self.X[batch_begin:batch_end].shape[0]

                ##change to onehot 
                X = pre.labeltoOneHotVector(self.X[batch_begin:batch_end], int(max(self.X))+1)
                Y = pre.labeltoOneHotVector(self.Y[batch_begin:batch_end], int(max(self.Y))+1)
                self.dims = [X.shape[1], self.nNode, Y.shape[1]]

                self.param_init()
                self.frontprop(X, Y)

                if len(self.loss) != 1:
                    print("Loss Up") if self.loss[-1] > self.loss[-2] else print("Loss Down")
                print("Loss : ",self.loss[-1])
                self.backprop(Y)
                self.gradients()
                self.gradientdescent()
                print("%.2f percent done in one epoch"% (cnt/len(idx) * 100))
            
            print("\n===========================================================================")
            print("=======epoch done=========")
            print("Total progress : %.3f\n" % ((len(idx) * i + cnt) / (nEpoch * len(idx)) *100) )
        print(f"======={i+1}th epoch outof {nEpoch} done=========")
        
        return self.param
    



            
