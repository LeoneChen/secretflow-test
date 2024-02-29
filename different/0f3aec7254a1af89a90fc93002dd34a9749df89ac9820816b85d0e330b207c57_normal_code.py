
import numpy as np

def func(data_alice, data_bob):
    # This time we perform a more complex calculation: weighted sum and then apply the tanh activation function.
    weights = np.array([0.5, 1.0, 1.5])
    weighted_sum_alice = np.dot(weights, data_alice)
    weighted_sum_bob = np.dot(weights, data_bob)
    result = np.tanh(weighted_sum_alice + weighted_sum_bob)
    return result

def get_alice_data():
    # Alice's data simulation: a more complex vector with three dimensions
    vector_data_alice = [0.1, 0.2, 0.3]
    array_data_alice = np.array(vector_data_alice)
    return array_data_alice

def get_bob_data():
    # Bob's data simulation: convert string representation of complex data to vector data
    string_data_bob = "0.4, 0.5, 0.6"
    list_data_bob = string_data_bob.split(", ")
    vector_data_bob = [float(item) for item in list_data_bob]
    array_data_bob = np.array(vector_data_bob)
    return array_data_bob

# Get data
data_alice = get_alice_data()
data_bob = get_bob_data()

# Perform normal computation
result = func(data_alice, data_bob)

# Print result
print(f"Result: {result}")
