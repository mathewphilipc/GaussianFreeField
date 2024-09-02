import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from analysis_lib import (
    reduce_2D_microstate, reduce_3D_microstate, graph_from_3D_reduced_microstate, graph_from_3D_microstate
)

def test_reduce_2D_microstate():
    microstate = np.array([[2, 3], [1, 4]])
    threshold = 3
    expected_output = np.array([[0, 1], [0, 1]], dtype=np.uint32)
    assert np.array_equal(reduce_2D_microstate(microstate, threshold), expected_output)

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


if __name__ == "__main__":
    test_reduce_2D_microstate()
    test_reduce_3D_microstate()
    test_graph_from_3D_microstate()
    print("All tests passed!")

