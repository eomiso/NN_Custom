"""
Test code for formulas.py
You may use the command below to run all the tests that has the file name of test*.py
python -m unittest discover
"""

# 소수점의 문제를 어떻게 처리할 것인지도 생각해봐야할 지점
# 가령 시그모이드 함수의 경우 결과 값이 거의 항상 소수점으로 나온다.
# 이 문제는 np에 있는 numpy.testing 를 통해 해결할 수 있다.


import unittest
import numpy as np
import sys
sys.path.append('../../')

import formulas

class TestFormulas(unittest.TestCase):

 #   아래 코드는 assertSequenceEqual를 사용하면 필요가 없는 함수이다.
    @staticmethod
    def check_identical_array(array1, array2):
        """
        check if two arrays are identical
        """
        for element1, element2 in zip(np.squeeze(array1), np.squeeze(array2)): # 2-dimensional array vector([[1,2,3]]) -> 1-dimensional([1,2,3])
            if element1 != element2:
                return False
        return True

    def test_Relu(self):
        input = np.array([[1, 0.5, 0.3, -1]])
        actual_return = formulas.Relu(input)
        expected_return = np.array([[1, 0.5, 0.3 , 0]])

        np.testing.assert_array_equal(actual_return, expected_return)


    def test_Relu_should_return_TypeError_to_non_array_input(self):
        input = [1, 0.5, 0.3, -1]
        with self.assertRaises(TypeError):
            formulas.Relu(input)


    def test_dRelu(self):
        input = np.array([[1, 0.5, 0.3, -1]])
        actual_return = formulas.dRelu(input)
        expected_return = np.array([[1, 1, 1, 0]])

        np.testing.assert_array_equal(actual_return, expected_return)

    def test_Sigmoid(self):
        input = np.array([[1, 0.5, 0.3, -1]])
        actual_return = formulas.Sigmoid(input)
        expected_return = np.array([[0.73105858, 0.62245933, 0.57444252, 0.26894142]])

        np.testing.assert_array_almost_equal(actual_return, expected_return, decimal= 7)


    def test_dSigmoid(self):
        input = np.array([[1, 0.5, 0.3, -1]])
        actual_return = formulas.dSigmoid(input)
        expected_return = np.array([[0.19661193, 0.23500371, 0.24445831, 0.19661193]])
        
        np.testing.assert_array_almost_equal(actual_return, expected_return, decimal= 7)


    def test_Softmax(self):
        input = np.array([[1, 0.5, 0.3, -1]])
        actual_return = formulas.Softmax(input)
        expected_return = np.array([[0.44673745, 0.27095996, 0.22184325, 0.06045934]])

        np.testing.assert_array_almost_equal(actual_return, expected_return, decimal= 7)


    def test_dSoftmax(self):
        input = np.array([[1, 0.5, 0.3, -1]])
        actual_return = formulas.dSoftmax(input)
        expected_return = np.array([[0.2471631 , 0.19754066, 0.17262882, 0.05680401]])

        np.testing.assert_array_almost_equal(actual_return, expected_return, decimal= 7)


    def test_CrossEntropy(self):
        """
        The input of CrossEntropy can't be minus
        """
        input = np.array([[0.2471631 , 0.19754066, 0.17262882, 0.05680401]]), np.array([[1, 0, 0, 0]])
        actual_return = formulas.CrossEntropy(*input)
        # 여기 소수점을 어떻게 처리할 것인지에 대해서 고민해 봐야 한다. -> np.testing 으로 해결
        expected_return = np.array([[1.3977068, 0.2200741, 0.1895019, 0.0584812]])

        np.testing.assert_array_almost_equal(actual_return, expected_return, decimal= 7)

    def test_dCrossEntropy(self):
        input = np.array([[0.2471631 , 0.19754066, 0.17262882, 0.05680401]]), np.array([[1, 0, 0, 0]])
        actual_return = formulas.dCrossEntropy(*input)
        expected_return = np.array([[-4.04591138,  1.24616906,  1.20864737,  1.06022503]])

        np.testing.assert_array_almost_equal(actual_return, expected_return, decimal= 7)

if __name__ == '__main__':
    unittest.main()