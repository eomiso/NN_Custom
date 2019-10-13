import numpy as np
import unittest

import sys
sys.path.append('../../')

from NN_model import Custom_NN_model

class TestNNModel_Constructor_arguments(unittest.TestCase):
    def test_non_numpy_array_arg_0_should_throw_error(self):
        X = [[0]]
        Y = np.array([[0]])
        with self.assertRaises(TypeError) as e:
            Custom_NN_model(X,Y)
        self.assertEqual(str(e.exception), "X's type should be <%s>. But it's %s"%("np ndarray",type(X)))

    def test_non_numpy_array_arg_1_should_throw_error(self):
        X = np.array([[0]])
        Y = [[0]]
        with self.assertRaises(TypeError) as e:
            Custom_NN_model(X,Y)
        self.assertEqual(str(e.exception), "Y's type should be <%s>. But it's %s"%("np ndarray",type(Y)))
    
    def test_arg_0_having_more_length_should_throw_error(self):
        X = np.array([[0],[1]])
        Y = np.array([[0]])
        with self.assertRaises(AssertionError) as e:
            Custom_NN_model(X,Y)
        self.assertEqual(str(e.exception), "The number of examples should be equal to the number of ground truth")


class TestNNModel_Functions(unittest.TestCase):
    X = np.array([[0.1, 0.2], [0.23, 0.52], [0.6, 0.5]])
    Y = np.array([[1. , 0. ], [1.  , 0.  ], [0. , 1. ]])
    nLayer = 3 #레이어 수는 3으로 진행
    nNode = 3 #노드 수 또한 2로 진행
    
            

#    def test_XYsetter(self):
#        model = Custom_NN_model(self.X,self.Y, nLayer = self.nLayer, nNode= self.nNode)
#        model = 

    def test_param_init(self):
        model = Custom_NN_model(self.X,self.Y, nLayer = self.nLayer, nNode= self.nNode)
        param = model.param_init()
        actual_result = [x.shape for x in param.values()]
        expected_result = [(2, 3), (3,), (3, 3), (3,), (3, 2), (2,)]
        self.assertSequenceEqual(actual_result, expected_result, "Parameter size incorrect")

    def test_frontprop(self):
        model = Custom_NN_model(self.X, self.Y, nLayer= 2, nNode = 2)
        model.param = {
            'W1' : np.array([[0.1, 0.2], [0.3, 0.2]]),
            'W2' : np.array([[0.2, 0.3], [0.3, 0.5]])
        }
        model.param = {
            'b1' : np.array([1,1]),
            'b1' : np.array([1,1])
        }



if __name__ == '__main__':
    unittest.main()
    
