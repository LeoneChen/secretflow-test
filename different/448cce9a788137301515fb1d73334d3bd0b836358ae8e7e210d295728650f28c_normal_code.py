
# Import numpy
import numpy as np

def func(data_alice, data_bob):
    # Here we are calculating the sum of square roots of Alice's and Bob's data
    sum_sqrt = np.sum(np.sqrt(data_alice)) + np.sum(np.sqrt(data_bob))
    return sum_sqrt

@jit
def get_alice_data():
    # Alice's data is an array containing square of numbers from 1 to 10
    return np.array([np.power(i, 2) for i in range(1, 11)])

@jit
def get_bob_data():
    # Bob's data is an array containing cube of numbers from 11 to 20
    return np.array([np.power(i, 3) for i in range(11, 21)])

# Get data
data_alice = get_alice_data()
data_bob = get_bob_data()

# Perform normal computation
result = func(data_alice, data_bob)

# Print result
print(f"Result: {result}")
