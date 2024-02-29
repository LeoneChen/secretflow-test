
import numpy as np

def func(data_alice, data_bob):
    # Calculate the element-wise product and then the sum of the elements, followed by applying softmax.
    product = np.multiply(data_alice, data_bob)
    sum_result = np.sum(product)
    softmax_result = np.exp(sum_result) / np.sum(np.exp(product))
    return softmax_result

def get_alice_data():
    # Alice's data: vector containing values in ascending order
    vector_data_alice = np.arange(1, 4, 1)  # [1, 2, 3]
    return vector_data_alice

def get_bob_data():
    # Bob's data: converting string representation of a range to vector
    string_data_bob = "4, 5, 6"
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
