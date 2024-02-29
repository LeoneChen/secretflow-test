
import secretflow as sf
import jax.numpy as jnp

# Initialize SPU and participants
sf.init(['alice', 'bob'], address='local')
alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu_node_device = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

def func(data_alice, data_bob):
    # First compute the matrix multiplication between data_alice and data_bob
    matmul_result = jnp.matmul(data_alice, data_bob)
    # Then, apply the exponential function to each element of the result
    exp_result = jnp.exp(matmul_result)
    # Calculate the mean value of all elements in exp_result
    mean_result = jnp.mean(exp_result)
    return mean_result

def get_alice_data():
    # Assume Alice's data is represented as a 2x2 matrix stored in a string, convert it to a numpy array
    matrix_str = "1.0,2.0;3.0,4.0"
    matrix_list = [list(map(float, row.split(','))) for row in matrix_str.split(';')]
    return jnp.array(matrix_list)

def get_bob_data():
    # Bob's data is a set of vectors that will be represented as a 2x3 matrix
    return jnp.array([[5, 6, 7], [8, 9, 10]])

# Pass data to PYU
data_alice = alice(get_alice_data)()
data_bob = bob(get_bob_data)()

# Perform privacy computation on SPU
result = spu_node_device(func)(data_alice, data_bob)

# Reveal results
revealed_result = sf.reveal(result)

# Print revealed result
print(f"Result: {revealed_result}")

# Clean envionment
sf.shutdown()
