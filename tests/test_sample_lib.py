import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from sample_lib import (
    linear_to_2D_coordinates, linear_to_3D_coordinates, torus_distance, are_neighbors_3D_grid, all_3D_torus_neighbors, are_neighbors_3D_torus
)

def test_linear_to_2D_coordinates():
    assert linear_to_2D_coordinates(linear_coord=0, torus_len=3) == [0, 0]
    assert linear_to_2D_coordinates(linear_coord=1, torus_len=3) == [0, 1]
    assert linear_to_2D_coordinates(linear_coord=3, torus_len=3) == [1, 0]
    assert linear_to_2D_coordinates(linear_coord=5, torus_len=3) == [1, 2]

def test_linear_to_3D_coordinates():
    assert linear_to_3D_coordinates(linear_coord=0, torus_len=3) == [0, 0, 0]
    assert linear_to_3D_coordinates(linear_coord=1, torus_len=3) == [0, 0, 1]
    assert linear_to_3D_coordinates(linear_coord=3, torus_len=3) == [0, 1, 0]
    assert linear_to_3D_coordinates(linear_coord=9, torus_len=3) == [1, 0, 0]
    assert linear_to_3D_coordinates(linear_coord=27, torus_len=3) == [0, 0, 0]
    assert linear_to_3D_coordinates(linear_coord=12, torus_len=3) == [1, 1, 0]
    assert linear_to_3D_coordinates(linear_coord=-15, torus_len=3) == [1, 1, 0]

def test_torus_distance():
    assert torus_distance(arr1=[1, 2, 3], arr2=[4, 5, 6], torus_len=7) == 9
    assert torus_distance(arr1=[1, 2, 3], arr2=[1, 2, 3], torus_len=7) == 0
    assert torus_distance(arr1=[1, 2, 3], arr2=[1, 2, 4], torus_len=7) == 1
    assert torus_distance(arr1=[1, 2, 3], arr2=[1, 2, 2], torus_len=7) == 1
    assert torus_distance(arr1=[1, 2, 2], arr2=[1, 2, 3], torus_len=7) == 1
    assert torus_distance(arr1=[1, 9, 30], arr2=[8, -5, 73], torus_len=7) == 1

def test_torus_distance_unequal_length():
    print("Testing for exceptions")
    try:
        torus_distance(arr1=[1, 2], arr2=[1, 2, 3], torus_len=7)
    except ValueError as e:
        assert str(e) == "Input arrays must have the same length."
    else:
        raise AssertionError("ValueError not raised when input arrays have different lengths.")

def test_are_neighbors_3D_torus():
    assert not are_neighbors_3D_torus(a=5, b=5, N=10)
    assert are_neighbors_3D_torus(a=5, b=6, N=10)
    assert not are_neighbors_3D_torus(a=5, b=7, N=10)
    assert are_neighbors_3D_torus(a=5, b=4, N=10)
    assert not are_neighbors_3D_torus(a=5, b=3, N=10)
    assert are_neighbors_3D_torus(a=5, b=15, N=10)
    assert not are_neighbors_3D_torus(a=5, b=16, N=10)
    assert are_neighbors_3D_torus(a=5, b=95, N=10)
    assert not are_neighbors_3D_torus(a=5, b=94, N=10)
    assert are_neighbors_3D_torus(a=5, b=105, N=10)
    assert not are_neighbors_3D_torus(a=5, b=104, N=10)
    assert are_neighbors_3D_torus(a=5, b=905, N=10)
    assert not are_neighbors_3D_torus(a=5, b=906, N=10)


def test_all_3D_torus_neighbors():
    assert all_3D_torus_neighbors(linear_coord=30,N=10).sort() == [29,31,20,40,130,930].sort()
    assert all_3D_torus_neighbors(linear_coord=2030,N=10).sort() == [29,31,20,40,130,930].sort()
    assert all_3D_torus_neighbors(linear_coord=0,N=10).sort() == [1,999,10,990,100,900].sort()
    for neighbor in all_3D_torus_neighbors(linear_coord=30, N=10):
        assert are_neighbors_3D_torus(a=30, b=neighbor, N=10)

if __name__ == "__main__":
    test_linear_to_2D_coordinates()
    test_linear_to_3D_coordinates()
    test_torus_distance()
    test_torus_distance_unequal_length()
    test_are_neighbors_3D_torus()
    test_all_3D_torus_neighbors()
    print("All tests passed!")
