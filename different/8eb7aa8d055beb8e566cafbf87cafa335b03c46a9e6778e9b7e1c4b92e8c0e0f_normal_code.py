
import numpy as np

def func(data_alice, data_bob):
    # Calculate the dot product of data_alice and data_bob, then apply the sigmoid function on the result
    dot_product = np.dot(data_alice, data_bob)
    sigmoid_result = 1 / (1 + np.exp(-dot_product))
    return sigmoid_result

def get_alice_data():
    # Alice's data: 2D matrix
    matrix_data_alice = np.array([[1, 2], [3, 4]])
    return matrix_data_alice

def get_bob_data():
    # Bob's data: Convert a comma-separated string to a 2D matrix
    string_data_bob = "5, 6; 7, 8"
    rows = [row.split(",") for row in string_data_bob.split(";")]
    matrix_data_bob = np.array([[float(value) for value in row] for row in rows])
    return matrix_data_bob

# Get data
data_alice = get_alice_data()
data_bob = get_bob_data()

# Perform normal computation
result = func(data_alice, data_bob)

# Print result
print(f"Result: {result}")
