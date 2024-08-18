import numpy as np
from scipy.signal import convolve2d
import random
import math

def max_pooling(input_array, pool_size):
    height, width = input_array.shape
    pooled_height = height // pool_size
    pooled_width = width // pool_size
    pooled_array = np.zeros((pooled_height, pooled_width), dtype=int)

    for i in range(pooled_height):
        for j in range(pooled_width):
            start_row = i * pool_size
            start_col = j * pool_size
            end_row = start_row + pool_size
            end_col = start_col + pool_size
            pooled_array[i, j] = np.max(input_array[start_row:end_row, start_col:end_col])

    return pooled_array

def pass_through_layer(flattened_array, node_num):
    iteration = 0
    for num in flattened_array:
        if iteration == len(flattened_array) - 1:
            print(f"{num} ", end="")
        else:
            print(f"{num} + ", end="")
        iteration+=1

    layer_sum = round(np.sum(flattened_array), 1)
    print(f"= {layer_sum}")
    weight_choices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    result = []

    for i in range(node_num):
        curr_weight = random.choice(weight_choices)
        weight_choices.pop(weight_choices.index(curr_weight))
        res = round(layer_sum * curr_weight, 1)
        result.append(res)

        print(f"neuron {i + 1}: {layer_sum} x {curr_weight} = {res}")
    
    return result

def softmax(output_array):
    max_val = max(output_array)
    
    # Subtract the maximum value for numerical stability
    exp_values = [math.exp(i - max_val) for i in output_array]
    denominator = sum(exp_values)

    probabilities = [value / denominator for value in exp_values]
    
    return probabilities

# Define the matrices and kernel
red = np.array([
    [83, 24, 55, 32, 89, 88],
    [31, 25, 37, 44, 48, 43],
    [30, 31, 23, 33, 41, 40],
    [30, 18, 28, 38, 48, 49],
    [31, 39, 19, 29, 49, 50],
    [41, 19, 29, 39, 41, 51]
])

green = np.array([
    [108, 88, 43, 49, 58, 88],
    [88, 71, 43, 28, 52, 72],
    [73, 65, 72, 92, 63, 34],
    [72, 36, 87, 81, 43, 22],
    [42, 89, 33, 36, 11, 39],
    [33, 90, 30, 26, 21, 47]
])

blue = np.array([
    [90, 88, 16, 94, 85, 89],
    [75, 81, 44, 82, 25, 27],
    [43, 87, 27, 29, 36, 43],
    [24, 21, 78, 18, 34, 26],
    [42, 20, 39, 63, 19, 93],
    [11, 10, 31, 62, 12, 74]
])

kernel = np.array([
    [1, 0, 1],
    [-1, 0, 1],
    [1, 0, -1]
])

kernel = np.flip(np.flip(kernel, axis=0), axis=1)

# Perform convolution
red_result = convolve2d(red, kernel, mode='valid')
green_result = convolve2d(green, kernel, mode='valid')
blue_result = convolve2d(blue, kernel, mode='valid')

# Display the results
print("RED Convolution Result:")
print(red_result)

print("\nGreen Convolution Result:")
print(green_result)

print("\nBlue Convolution Result:")
print(blue_result)

summation = red_result + green_result + blue_result

print("\nSummation of RGB:")
print(summation)

pooling_value = max_pooling(summation, 2)

print("\nPooling:")
print(pooling_value)

flattened_array = pooling_value.flatten()

print("\nFlatten:")
print(flattened_array)

# Feed through layers:
print("\n")
hidden_layer_1 = pass_through_layer(flattened_array, 4)
print("\n")
hidden_layer_2 = pass_through_layer(hidden_layer_1, 4)
print("\n")
output_layer = pass_through_layer(hidden_layer_2, 2)

# Soft Max
result = softmax(output_layer)

print("\nSoftmax: ")
print(result)

print("\nSum of results (must be 1): ")
print(np.sum(result))