import numpy as np
from NN_3_layers_custom import NN_model
from NN_3_layers_custom import encoders as enc
X = np.array(['David', 'Seop', 'Nash', 'Seop', 'Will'])
Y = np.array(['1', '0', '1', '0', '1'])

X= enc.labelencoder(X)
Y= enc.labelencoder(Y)

model = NN_model.Custom_NN_model(X.reshape(-1,1),Y.reshape(-1,1), 3, 3)

idx = model.minibatch_setter(batch_size = 2)

model.minibatch_gradientdescent(batch_size=2 nEpoch =4)