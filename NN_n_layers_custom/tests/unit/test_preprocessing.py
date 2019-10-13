import sys
sys.path.append('../../')
import unittest
import numpy as np
import preprocessing

class test_encoding(unittest.TestCase):
    def test_label_encoder_with_numpy_array(self):
        input = np.array(["david", "josh", "chullin", "aesop"])
        actual_return = preprocessing.labelencoder(input)
        expected_return = np.array([2, 3, 1, 0])

        assert np.testing.assert_array_equal(actual_return,expected_return)

    def test_onehotvector_encoder(self):
        input = np.array(["david", "josh", "chullin", "aesop", "josh"])
        input = preprocessing.labelencoder(input)
        actual_return = preprocessing.labeltoOneHotVector(input, int(max(input)) + 1)
        expected_return = np.array([[0., 0., 1., 0.],
                                    [0., 0., 0., 1.],
                                    [0., 1., 0., 0.],
                                    [1., 0., 0., 0.],
                                    [0., 0., 0., 1.]])
        assert np.testing.assert_array_equal(actual_return, expected_return)

    # def test_label_encoder_with_pandas_series(self):
    #     input = np.array(["david", "josh", "chullin", "aesop"])
        
    #     actual_return = 
    #     expected_return =            

    # def test_onehot_encoder_with_numpy_array(self):
    #     input = np.array(["david", "josh", "chullin", "aesop"])
    #     #labe encoder
    #     actual_return = 
    #     expected_return =

    # def test_onehot_encoder_with_pandas_array(self):
