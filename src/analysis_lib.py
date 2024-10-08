# analysis_lib.py
# tools for analyzing microstates

import numpy as np
import csv

# In Greene environment, unfortunately we need .sample_lib to run remote jobs.
from sample_lib import linear_to_3D_coordinates,  all_3D_torus_neighbors

def reduce_3D_microstate(microstate, threshold):
    """
    Reduces a 3D microstate array to a binary array based on a threshold.

    Parameters:
    microstate (np.ndarray): 3D array representing the microstate.
    threshold (int): The threshold value for reduction.

    Returns:
    np.ndarray: A 3D binary array where values are 1 if the corresponding 
                microstate value is greater than or equal to the threshold, else 0.
    """
    size = len(microstate)
    reduced_state = np.zeros((size, size, size), dtype=np.uint32)

    for i in range(size):
        for j in range(size):
            for k in range(size):
                if microstate[i, j, k] >= threshold:
                    reduced_state[i, j, k] = 1

    return reduced_state

def graph_from_3D_microstate(microstate, threshold):
    """
    Reduces a full microstate and builds occupation graph, using only occupied nodes.

    Parameters:
    microstate (np.ndarray): 3D array representing the microstate.
    threshold (int): The threshold value for reduction.

    Returns:
    np.ndarray: A 2D binary array representing an adjacency matrix.
    """
    # On a first pass, we find all of the linear indices corresponding to occupied
    # nodes.
    N = len(microstate)

    # We represent points in our microstate by linear indices 0 through N^3 - 1.
    # Only a sublist [a_0, a_1, ...] correspond to points in our occupation graph.
    # compressed_to_linear_indices maps n to a_n, and c_t_l_i does the reverse,
    # for 0 <= n <= num_occupied_nodes.
    linear_to_compressed_indices = {}
    compressed_to_linear_indices = {}
    num_nodes = 0;
    for i in range(N**3):
        [ix, iy, iz] = linear_to_3D_coordinates(linear_coord=i, torus_len=N)
        if (microstate[ix][iy][iz] >= threshold):
            linear_to_compressed_indices[i] = num_nodes
            compressed_to_linear_indices[num_nodes] = i
            num_nodes = num_nodes + 1

    # Occupation graph has M nodes -> adj mat is M x M instead of naive N^3 x N^3.
    adj_mat = np.zeros([num_nodes, num_nodes], dtype=np.uint32)

    # Find edges in O(M) instead of naive O(N^6).
    for m in range(num_nodes):
        for j in all_3D_torus_neighbors(linear_coord=compressed_to_linear_indices[m], N=N):
            if j in linear_to_compressed_indices:
                n = linear_to_compressed_indices[j]
                adj_mat[m][n] = 1
                adj_mat[n][m] = 1

    return adj_mat

def iterative_dfs(graph, start, visited):
    """
    Runs iterative (not recursive) depth-first search from a given start node.

    Parameters:
    graph (np.ndarray): A 2D binary array representing an adjacency matrix.
    start (int): The index of a starting node.
    visited (dict): A dictionary (int -> bool) telling which nodes have been visited.

    Returns:
    list: A list of all indices of nodes in the connected component containing start.
    """
    stack = [start]
    component = []
    while stack:
        node = stack.pop()
        if not visited[node]:
            visited[node] = True
            component.append(node)
            for neighbor, connected in enumerate(graph[node]):
                if connected and not visited[neighbor]:
                    stack.append(neighbor)
    return component

def iterative_connected_components(graph):
    """
    Find the distribution of sizes of connected components from an adjacency matrix.

    Parameters:
    graph (np.ndarray): 2D adjacency matrix.

    Returns:
    dict: Dictionary from component sizes to frequencies.

    """
    num_nodes = len(graph)
    visited = [False] * num_nodes
    component_sizes = []

    for node in range(num_nodes):
        if not visited[node]:
            component = iterative_dfs(graph, node, visited)
            component_sizes.append(len(component))

    component_count = {}
    for size in component_sizes:
        component_count[size] = component_count.get(size, 0) + 1

    return component_count

def second_moment(cluster_size_distribution):
    """
    Computes the second moment of a distribution of cluster sizes.

    Parameters:
    cluster_size_distribution (dict): Dictionary from component sizes to frequencies

    Returns:
    int: Second moment of cluster size distributions.
    """
    moment = 0
    for size, count in cluster_size_distribution.items():
        moment = moment + count*(size**2)
    return moment

def iterative_analyze_3D_microstate(microstate, threshold):
    """
    Reduces a microstate and computes occupation density and second moment.

    Parameters:
    microstate (np.ndarray): 3D array representing unreduced microstate.
    threshold (float): Threshold by which to reduce microstate.

    Returns:
    dict: Dictionary containing reduced microstate's occ density and second moment.
    """
    N = len(microstate)
    graph_data = {}
    occupation_graph = graph_from_3D_microstate(microstate, threshold)
    cluster_size_distribution = iterative_connected_components(occupation_graph)

    graph_data["occupation_density"] = round(len(occupation_graph) / N**3, 6)
    graph_data["second_moment"] = second_moment(cluster_size_distribution)
    return graph_data

def compute_full_graph_data(input_dir, output_dir, start_index, end_index,
    min_threshold, max_threshold, num_thresholds):
    """
    Analyses a collection of microstates at a range of prescribed thresholds.

    Parameters:
    input_dir (string): Directory containing microstates to analyse.
    output_dir (string): Directory in which to store results.
    start_index (string): Index of first microstate to analyze (inclusive).
    end_end (string): Index of final microstate to analyze (inclusive).
    min_threshold (float): Lowest threshold at which to reduce and analyse.
    max_threshold (float): Highest threshold at which to reduce and analyse.
    num_thresholds (int): Number of thresholds (including endpoints).

    Returns:
    None (but side effect of writing results to output_dir/...).
    """

    threshold_stepsize = (max_threshold - min_threshold) / (num_thresholds - 1)
    for i in range(start_index, end_index+1):
        # We assume input files have the following name format.
        input_file = input_dir + "/microstate_" + str(i) + ".npy"
        output_file = output_dir + "/full_graph_data_" + str(i) + ".csv"
        microstate = np.load(input_file)

        # Output data structure puts everything in output_dir/full_data_*.csv for *
        # in the same range, saving after every microstate. Csv contains a row for
        # each threshold with [threshold occupation_density second_moment] columns.
        # This is slightly awkward, but relatively robust to jobs being interrupted.
        # The extra file operations contribute trivial latency compared to the graph
        # analysis.
        with open(output_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for j in range(num_thresholds):
                threshold = round(min_threshold + j*threshold_stepsize, 3)
                graph_data = iterative_analyze_3D_microstate(microstate, threshold)
                occupation_density = graph_data["occupation_density"]
                second_moment = graph_data["second_moment"]
                csv_writer.writerow([str(threshold), str(occupation_density), str(second_moment)])

# Read through a collection of full graph data files. Compute average (with stderr)
# of occupation density and second moment at each threshold.
def summarize_graph_data(input_dir, output_dir, start_index, end_index):
    """
    Computes and stores per-threshold sample statistics from a list of full graph data files.

    Parameters:
    input_dir (string): Directory containing a set of full_graph_data files.
    output_dir (string): Directory in which to store results.
    start_index (string): Index of first full_graph_data file to include (inclusive).
    end_index (string): Index of last full_graph_data file to include (inclusive).

    Returns:
    None (but side effect of writing results to output_dir/summarized_graph_data.csv).
    """

    # Read in all full graph data, store in nested dict of the form
    # threshold -> [list of occuptation densities, list of second moments].
    threshold_to_full_graph_data = {}
    for i in range(start_index, end_index+1):
        input_file = input_dir + "/full_graph_data_" + str(i) + ".csv"
        with open(input_file, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in csvreader:
                threshold = float(row[0])
                occupation_density = float(row[1])
                second_moment = float(row[2])
                if threshold not in threshold_to_full_graph_data:
                    threshold_to_full_graph_data[threshold] = {}
                    threshold_to_full_graph_data[threshold]["occupation_densities"] = []
                    threshold_to_full_graph_data[threshold]["second_moments"] = []
                threshold_to_full_graph_data[threshold]["occupation_densities"].append(occupation_density)
                threshold_to_full_graph_data[threshold]["second_moments"].append(second_moment)

    # Summarize graph data in simpler nested dict.
    # In the end we are computing sample mean + stddev of occ_density + second moment at each threshold.
    threshold_to_graph_data_summary = {}
    for threshold in threshold_to_full_graph_data.keys():
        occupation_densities = threshold_to_full_graph_data[threshold]["occupation_densities"]
        second_moments = threshold_to_full_graph_data[threshold]["second_moments"]
        num_samples = len(second_moments)
        threshold_to_graph_data_summary[threshold] = {}
        threshold_to_graph_data_summary[threshold]["num_samples"] = num_samples
        threshold_to_graph_data_summary[threshold]["occupation_density_mean"] = np.average(occupation_densities)
        threshold_to_graph_data_summary[threshold]["occupation_density_stderr"] = np.std(occupation_densities) / np.sqrt(num_samples)
        threshold_to_graph_data_summary[threshold]["second_moment_mean"] = np.average(second_moments)
        threshold_to_graph_data_summary[threshold]["second_moment_stderr"] = np.std(second_moments) / np.sqrt(num_samples)

    # Store results as csv.
    output_file = output_dir + "/summarized_graph_data.csv"
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(["num_samples", "threshold", "occupation_density_mean", "occupation_density_stderr",
            "second_moment_mean", "second_moment_stderr"])
        for threshold, summary in threshold_to_graph_data_summary.items():
            new_row = [summary["num_samples"], threshold, summary["occupation_density_mean"], summary["occupation_density_stderr"],
                summary["second_moment_mean"], summary["second_moment_stderr"]]
            csvwriter.writerow(new_row)

def estimate_ratio_vs_density(smaller_input_file, larger_input_file, output_dir, min_density, max_density, steps):
    # First arg should contain a graph data summary for NxNxN tori.
    # Second arg should contain a graph data summary for 2Nx2Nx2N tori.
    # Input file rows are as documented in last block of summarize_graph_data().
    # This function works by interpolating the values and uncertainty of second
    # moment vs density from each summary, and throws an error if the min or max
    # density falls outside of the range covered in the summaries.
    output_file = output_dir + "/ratio_vs_density.csv"
    smaller_torus_data = []
    with open(smaller_input_file, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        row_count = 0
        for row in csvreader:
            row_count = row_count + 1
            if (row_count == 1):
                continue
            occupation_density_mean = float(row[2])
            second_moment_mean = float(row[4])
            second_moment_stderr = float(row[5])
            smaller_torus_data.append([occupation_density_mean, second_moment_mean, second_moment_stderr])
    smaller_torus_data = np.array(smaller_torus_data)
    smaller_torus_data = smaller_torus_data[smaller_torus_data[:, 0].argsort()]

    larger_torus_data = []
    with open(larger_input_file, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        row_count = 0
        for row in csvreader:
            row_count = row_count + 1
            if (row_count == 1):
                continue
            occupation_density_mean = float(row[2])
            second_moment_mean = float(row[4])
            second_moment_stderr = float(row[5])
            larger_torus_data.append([occupation_density_mean, second_moment_mean, second_moment_stderr])
    larger_torus_data = np.array(larger_torus_data)
    larger_torus_data = larger_torus_data[larger_torus_data[:, 0].argsort()]

    step_size = (max_density - min_density) / (steps - 1)
    ratio_vs_density = []
    for i in range(steps):
        p = min_density + i*step_size
        # For legibility of our error propagation, we write the smaller torus second
        # moment as x +/- dx and the larger as y +/- dy. Then our estimated ratio is
        # y/x, with variance (y dx / x^2)^2 + (dy / x)^2.
        x = np.interp(p, smaller_torus_data[:,0], smaller_torus_data[:,1])
        dx = np.interp(p, smaller_torus_data[:,0], smaller_torus_data[:,2])
        y = np.interp(p, larger_torus_data[:,0], larger_torus_data[:,1])
        dy = np.interp(p, larger_torus_data[:,0], larger_torus_data[:,2])
        ratio = y/x
        # How much upper vs lower contribute to dratio
        smaller_torus_sigma = y*dx/(x**2)
        larger_torus_sigma = dy/x
        dratio = np.sqrt(smaller_torus_sigma**2 + larger_torus_sigma**2)
        ratio_vs_density.append([p, ratio, dratio, smaller_torus_sigma, larger_torus_sigma])

    # Store results as csv
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(["occupation_density_mean", "ratio", "ratio_stderr",
            "smaller_torus_sigma", "larger_torus_sigma"])
        for row in ratio_vs_density:
            csvwriter.writerow(row)
