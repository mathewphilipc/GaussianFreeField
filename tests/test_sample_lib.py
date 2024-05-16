import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from sample_lib import (
    linear_to_2D_coordinates, linear_to_3D_coordinates, torus_distance
)

def test_linear_to_2D_coordinates():
    assert linear_to_2D_coordinates(0, 3) == [0, 0]
    assert linear_to_2D_coordinates(1, 3) == [0, 1]
    assert linear_to_2D_coordinates(3, 3) == [1, 0]
    assert linear_to_2D_coordinates(5, 3) == [1, 2]

def test_linear_to_3D_coordinates():
    assert linear_to_3D_coordinates(0, 3) == [0, 0, 0]

def test_torus_distance():
    assert torus_distance([1, 2, 3], [4, 5, 6], 7) == 9
    assert torus_distance([1, 2, 3], [1, 2, 3], 7) == 0
    assert torus_distance([1, 2, 3], [1, 2, 4], 7) == 1
    assert torus_distance([1, 2, 3], [1, 2, 2], 7) == 1
    assert torus_distance([1, 2, 2], [1, 2, 3], 7) == 1


if __name__ == "__main__":
    test_linear_to_2D_coordinates()
    test_linear_to_3D_coordinates()
    test_torus_distance()
    print("All tests passed!")
