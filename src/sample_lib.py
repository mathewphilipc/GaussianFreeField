# sample_lib.py
# Tools for sampling GFF microstates

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
import time
import random
from scipy.sparse.linalg import eigs, eigsh
from scipy.sparse import lil_matrix, dok_matrix, csc_matrix

def linear_to_3D_coordinates(linear_coord, torus_len):
    """
    Projects a linear coordinate onto a 3D coordinate system on a 3-torus of length N.

    Parameters:
    linear_coord (int): The linear coordinate.
    torus_len (int): The length of the 3-torus.

    Returns:
    list: A list [x, y, z] representing the 3D coordinates, ranging from 0 to N-1.
    """
    remainder = linear_coord
    z = remainder % torus_len
    remainder = (remainder - z) // torus_len
    y = remainder % torus_len
    remainder = (remainder - y) // torus_len
    x = remainder % torus_len
    return [x, y, z]


def torus_distance(arr1, arr2, torus_len):
    """
    Computes the distance between two integer points on a mod-N torus.

    Parameters:
    arr1 (list): First list of coordinates.
    arr2 (list): Second list of coordinates.
    torus_len (int): The length of the 3-torus.

    Returns:
    int: The torus distance between the two points.
    """

    if len(arr1) != len(arr2):
        raise ValueError("Input arrays must have the same length.")

    distance = 0
    for i in range(len(arr1)):
        dist12 = (arr1[i] - arr2[i]) % torus_len
        dist21 = (arr2[i] - arr1[i]) % torus_len
        distance += min(dist12, dist21)

    return distance

def grid_distance(arr1, arr2, grid_len):
    """
    Computes the grid distance between two integer points.

    Parameters:
    arr1 (list): First list of coordinates.
    arr2 (list): Second list of coordinates.
    len (int): The length of the grid.

    Returns:
    int: The grid distance between the two points.
    """

    if len(arr1) != len(arr2):
        raise ValueError("Input arrays must have the same length.")

    distance = 0
    for i in range(len(arr1)):
        dist12 = (arr1[i] - arr2[i])
        dist21 = (arr2[i] - arr1[i])
        distance += min(dist12, dist21)

    return distance

def are_neighbors_3D_torus(a, b, N):
    # Checks whether two numbers a b, when projected onto a 2D torus of length N, are neighbors.
    # Torus coordinates are (x,y) each ranging from 0 to N - 1
    """
    Checks whether two linear coordinates correspond to neighboring 3-torus points.

    Parameters:
    a (int): First linear coordinate.
    b (list): Second linear coordinate.
    N (int): The length of the torus..

    Returns:
    bool: Whether the two points are torus neighbors.
    """
    aArr = linear_to_3D_coordinates(a,N)
    bArr = linear_to_3D_coordinates(b,N)
    dist = torus_distance(aArr, bArr, N)
    return (dist == 1)

def are_neighbors_3D_grid(a, b, grid_len):
    # Unused, for deletion
    aCoords = linear_to_3D_coordinates(a, torus_len)
    bCoords = linear_to_3D_coordinates(b, torus_len)
    dist = grid_distance(aCoords, bCoords, torus_len)
    return (dist == 1)

def all_3D_torus_neighbors(linear_coord, N):
    """
    Given a linear coordinate, finds neighboring linear coordinate of corresponding 3-torus point.

    Parameters:
    linear_coord (int): The linear coordinate.
    torus_len (int): The length of the 3-torus.

    Returns:
    list: Linear coordinates of all neighbor points.
    """

    # Given a vector index corresponding to a point in an NxNxN torus, returns the
    # list of all neighbors' vector indices. We do this nicely, in O(1).
    [x,y,z] = linear_to_3D_coordinates(linear_coord,N)
    neighbors = []
    # Increment up or down
    for increment in {-1,1}:
        for dim in {0,1,2}:
            #print(f"Increment = {increment}")
            #print(f"dim = {dim}")
            new_coords = [x,y,z]
            new_coords[dim] = ((new_coords[dim] + increment) % N)
            #print("New coordinates:")
            #print(new_coords)
            new_neighbor = new_coords[2] + new_coords[1]*N + new_coords[0]*(N**2)
            neighbors.append(new_neighbor)
    return neighbors

# Given a vector index, convert it to equivalent NxNxN array index and count how
# many boundary edges it has. 
def count_boundary_edges_3D(N,k):
  [x,y,z] = linear_to_3D_coordinates(k,N)
  edgeCount = 0
  if (x == 0 or x == N-1):
    edgeCount = edgeCount + 1
  if (y == 0 or y == N-1):
    edgeCount = edgeCount + 1
  if (z == 0 or z == N-1):
    edgeCount = edgeCount + 1
  return edgeCount

# Sample a zero-centered GFF microstate on an NxNxN torus.
def sample_3D_torus(N, sample_mode):
  start_time = time.perf_counter()
  dim = 3
  H = np.zeros([N**3,N**3])
  for u in range(N**3):
    for v in all_3D_torus_neighbors(u,N):
    # for v in range(u+1,N**3):
      if v > u:
        c = 0.0
        if (sample_mode == "linear"):
            c = random.uniform(0.0, 2.0)
            print("We are doing linear sampling!")
        elif (sample_mode == "uniform"):
            print("We are doing uniform sampling!")
            c = 1.0
            print(f"Conductace = {c}")
        elif (sample_mode == "split"):
            print("We are doing split sampling!")
            coin_flip = random.randint(0,1)
            if (coin_flip == 1):
              print(f"Heads up value: {coin_flip}")
              c = 0.01
            else:
              print(f"Heads down value: {coin_flip}")
              c = 1.99
        else:
            return
        print(f"Conductance = {c}")
        H[u][u] = H[u][u] + c
        H[v][v] = H[v][v] + c
        H[u][v] = H[u][v] - c
        H[v][u] = H[v][u] - c
  # construct_time = time.perf_counter()
  # print("H construction time: {}".format(construct_time - start_time))

  # These matrices diagonalize H, in the sense that D is diagonal, V is
  # orthogonal, and H = R*D*R^T.
  eigenvalues, eigenvectors = eigsh(H,N**3) # specific method for Hermitian
  D = np.diag(eigenvalues).real
  R = eigenvectors.real
  # diag_time = time.perf_counter()
  # print("Diagonalization time: {}".format(diag_time - construct_time))

  # Then each component (R^T*X)_i is an IID Gaussian with mu = 0 and
  # sigma = sqrt(2)/D_{ii}.
  Xprime = np.zeros([N**3])
  for v in range(N**3):
    if abs(D[v][v] > 0.01):
      mu = 0
      sigma = np.sqrt(2*dim/D[v][v])
      Xprime[v] = np.random.normal(mu,sigma)
  X = R.dot(Xprime)
  microstate = np.zeros([N,N,N])
  for k in range(N**3):
    [x,y,z] = linear_to_3D_coordinates(k,N)
    microstate[x][y][z] = X[k]
  # sample_time = time.perf_counter()
  # print("Sampling time: {}".format(sample_time - diag_time))
  total_time = time.perf_counter() - start_time
  print(f"Total time: {total_time}")
  # Greene cluster runs ~1.5x as fast here (21s vs 32s at N = 15)

  return microstate

def repeated_random_sample(N, sample_mode, output_dir, start_index, end_index):
   # Repeated samples 3D torus microstates of size N and saves them
   # in output_dir under the name sample_* where * ranges from
   # start_index to end_index
   print("Time for a repeated random sample...")
   for i in range(start_index, end_index + 1):
      print(f"Trying to save sample \# {i} to file {output_dir}")
      microstate = sample_3D_torus(N, sample_mode)
      print("Trying to save...")
      output_file = output_dir + '/microstate_' + str(i) + '.npy'
      print(f"Output file = {output_file}")
      np.save(output_file, microstate)
