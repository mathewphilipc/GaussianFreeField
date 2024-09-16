import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from analysis_lib import (
    reduce_3D_microstate, graph_from_3D_microstate, iterative_connected_components, second_moment
)

def test_reduce_3D_microstate():
    microstate = np.array([[[2.3, 3.1], [1.0, 4.9]], [[5.9, 6.8], [7.2, 8.2]]])
    threshold = 5
    expected_output = np.array([[[0, 0], [0, 0]], [[1, 1], [1, 1]]], dtype=np.uint32)
    assert np.array_equal(reduce_3D_microstate(microstate, threshold), expected_output)

    threshold = 4.8
    expected_output = np.array([[[0, 0], [0, 0]], [[1, 1], [1, 1]]], dtype=np.uint32)
    assert not np.array_equal(reduce_3D_microstate(microstate, threshold), expected_output)

# Start with totally unoccupied microstate, gradually add occupied nodes and make sure adjacency matrix stays correct.
# This is very readable if you visualize a Rubik's cube, otherwise not at all.
def test_graph_from_3D_microstate():
    threshold = 5.0
    microstate = np.zeros((3,3,3))
    graph = graph_from_3D_microstate(microstate, threshold)
    assert len(graph) == 0

    microstate[0][0][0] = 10.0
    graph = graph_from_3D_microstate(microstate, threshold)
    expected_graph = np.array([[0]], dtype=np.uint32)
    assert np.array_equal(graph, expected_graph)

    microstate[0][0][1] = 10.0
    graph = graph_from_3D_microstate(microstate, threshold)
    expected_graph = np.array([[0,1],[1,0]], dtype=np.uint32)
    assert np.array_equal(graph, expected_graph)

    microstate[1][1][1] = 10.0
    graph = graph_from_3D_microstate(microstate, threshold)
    expected_graph = np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=np.uint32)
    assert np.array_equal(graph, expected_graph)

    microstate[1][1][2] = 10.0
    graph = graph_from_3D_microstate(microstate, threshold)
    expected_graph = np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]], dtype=np.uint32)
    assert np.array_equal(graph, expected_graph)

    microstate[1][2][1] = 10.0
    graph = graph_from_3D_microstate(microstate, threshold)
    expected_graph = np.array([[0,1,0,0,0],[1,0,0,0,0],[0,0,0,1,1],[0,0,1,0,0],[0,0,1,0,0]], dtype=np.uint32)
    assert np.array_equal(graph, expected_graph)

def test_iterative_connected_components():
    graph = np.zeros((10,10), dtype=np.uint32)
    assert iterative_connected_components(graph, -1) == {1: 10}

    graph[0,1] = 1
    graph[1,0] = 1
    assert iterative_connected_components(graph, -1) == {1:8, 2:1}

    graph[0,2] = 1
    graph[2,0] = 1
    assert iterative_connected_components(graph, -1) == {1:7, 3:1}

    graph[1,2] = 1
    graph[2,1] = 1
    assert iterative_connected_components(graph, -1) == {1:7, 3:1}

    graph[3,4] = 1
    graph[4,3] = 1
    assert iterative_connected_components(graph, -1) == {1:5, 2:1, 3:1}

    graph[4,5] = 1
    graph[5,4] = 1
    assert iterative_connected_components(graph, -1) == {1:4, 3:2}

    graph[5,6] = 1
    graph[6,5] = 1
    assert iterative_connected_components(graph, -1) == {1:3, 3:1, 4:1}

    graph[6,7] = 1
    graph[7,6] = 1
    assert iterative_connected_components(graph, -1) == {1:2, 3:1, 5:1}

    graph[7,8] = 1
    graph[8,7] = 1
    assert iterative_connected_components(graph, -1) == {1:1, 3:1, 6:1}

    graph[8,9] = 1
    graph[9,8] = 1
    assert iterative_connected_components(graph, -1) == {3:1, 7:1}

    graph[0,9] = 1
    graph[9,0] = 1
    assert iterative_connected_components(graph, -1) == {10:1}

def test_second_moment():
    assert second_moment({}) == 0
    assert second_moment({0: 2}) == 0
    assert second_moment({0: 2, 1: 3}) == 3
    assert second_moment({0: 2, 1: 3, 2: 10}) == 43

if __name__ == "__main__":
    test_reduce_3D_microstate()
    test_graph_from_3D_microstate()
    test_iterative_connected_components()
    test_second_moment()
    print("All tests passed!")

