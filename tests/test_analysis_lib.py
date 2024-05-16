import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from analysis_lib import (
    reduce_2D_microstate, reduce_3D_microstate
)

def test_reduce_2D_microstate():
    microstate = np.array([[2, 3], [1, 4]])
    threshold = 3
    expected_output = np.array([[0, 1], [0, 1]], dtype=np.uint32)
    assert np.array_equal(reduce_2D_microstate(microstate, threshold), expected_output)

def test_reduce_3D_microstate():
    microstate = np.array([[[2, 3], [1, 4]], [[5, 6], [7, 8]]])
    threshold = 5
    expected_output = np.array([[[0, 0], [0, 0]], [[1, 1], [1, 1]]], dtype=np.uint32)
    assert np.array_equal(reduce_3D_microstate(microstate, threshold), expected_output)

if __name__ == "__main__":
    test_reduce_2D_microstate()
    test_reduce_3D_microstate()
    print("All tests passed!")

