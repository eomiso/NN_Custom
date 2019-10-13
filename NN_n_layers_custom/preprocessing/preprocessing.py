from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np

def labelencoder(x):
    encoder = LabelEncoder()
    return encoder.fit_transform(x)

def onehotencoder(x):
    x = np.squeeze(x)
    if x.ndim != 1:
        raise ValueError("The input the onehotencoder should be one dimensional array")
    encoder = OneHotEncoder(sparse = False)
    return encoder.fit_transform(x.reshape(-1,1))

def labeltoOneHotVector(label_array,vector_size):
    label_array = np.squeeze(label_array)
    OneHotVector = np.zeros((len(label_array),vector_size))
    OneHotVector[np.arange(len(label_array)), label_array] += 1
    return OneHotVector
