import numpy as np


def apply_median_filter(input_array, kernel_size=3):
    return apply_filter(input_array, kernel_size, np.median)


def apply_mean_filter(input_array, kernel_size=3):
    return apply_filter(input_array, kernel_size, np.mean)


def apply_filter(input_array, kernel_size, filter):
    # Ensure kernel size is an odd number
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    rows, cols = input_array.shape
    result = np.zeros_like(input_array)

    # Pad the input array to handle edges
    pad_width = kernel_size // 2
    padded_array = np.pad(input_array, pad_width, mode='constant')

    for i in range(rows):
        for j in range(cols):
            # Extract the neighborhood using the kernel size
            neighborhood = padded_array[i:i + kernel_size, j:j + kernel_size]
            # Apply median operation
            result[i, j] = filter(neighborhood)

    return result


def binary_erosion(input_array):
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])

    result = np.zeros_like(input_array)
    rows, cols = input_array.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if np.array_equal(input_array[i - 1:i + 2, j - 1:j + 2] * kernel, kernel):
                result[i, j] = 1

    return result


def binary_dilation(input_array):
    input_height, input_width = input_array.shape
    struct_height, struct_width = 3, 3
    output_array = np.zeros((input_height, input_width), dtype=np.uint8)

    # Iterate over each pixel in the input array
    for i in range(input_height):
        for j in range(input_width):
            if input_array[i, j] == 1:
                # Overlay the structuring element on the input array
                for si in range(struct_height):
                    for sj in range(struct_width):
                        ni = i + si - (struct_height // 2)
                        nj = j + sj - (struct_width // 2)
                        if 0 <= ni < input_height and 0 <= nj < input_width:
                            output_array[ni, nj] = 1

    return output_array


def find_local_maxima(input_array, precision=1., nb=20):
    rows, cols = input_array.shape
    output_array = np.zeros((rows, cols), dtype=int)

    for i in range(0, rows):
        for j in range(0, cols):
            first_i, last_i = max(0, i-nb), min(rows, i+nb)
            first_j, last_j = max(0, j-nb), min(cols, j+nb)
            if (input_array[first_i:last_i, first_j:last_j].max() - input_array[i, j]) <= precision:
                output_array[i, j] = 1

    return output_array


def find_max_indices_in_clusters(x, y, t, d):
    # Ensure x and y have the same shape
    if x.shape != y.shape:
        raise ValueError("Input arrays x and y must have the same shape.")

    # Define a function to find neighboring ones in y
    def find_neighbors(i, j):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                ni, nj = i + dx, j + dy
                if 0 <= ni < x.shape[0] and 0 <= nj < x.shape[1] and y[ni, nj] == 1:
                    neighbors.append((ni, nj))
        return neighbors

    # Initialize a list to store the indices that maximize x for each cluster
    max_indices = []
    min_indices = []

    # Create a boolean mask to track visited elements in y
    visited = np.zeros_like(y, dtype=bool)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if y[i, j] == 1 and not visited[i, j]:
                # Start a new cluster
                cluster = [(i, j)]
                visited[i, j] = True

                # Explore the cluster using depth-first search
                for ci, cj in cluster:
                    neighbors = find_neighbors(ci, cj)
                    for ni, nj in neighbors:
                        if not visited[ni, nj]:
                            visited[ni, nj] = True
                            cluster.append((ni, nj))

                # Check if the cluster size is greater than t
                if len(cluster) > t:
                    # Find the index within the cluster that maximizes and minimizes x
                    max_index = max(cluster, key=lambda ij: x[ij])
                    min_index = min(cluster, key=lambda ij: x[ij])
                    max_indices.append(max_index)
                    min_indices.append(min_index)

    max_indices_filtered = []
    if len(max_indices) > 0:
        global_max_index = max(max_indices, key=lambda ij: x[ij])
        for max_index, min_index in zip(max_indices, min_indices):
            if max_index != global_max_index and x[max_index] - x[min_index] >= d:
                max_indices_filtered.append(max_index)
    else:
        print('Warning : potential hyperparameter errors for finding local extrema')

    return max_indices_filtered
