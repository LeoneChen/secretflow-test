
import secretflow as sf
import jax.numpy as jnp

# Initialize SPU and participants
sf.init(['alice', 'bob'], address='local')
alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu_node_device = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

def func(data_alice, data_bob):
    # Calculate the dot product of data_alice and data_bob, then apply the sigmoid function on the result
    dot_product = jnp.dot(data_alice, data_bob)
    sigmoid_result = 1 / (1 + jnp.exp(-dot_product))
    return sigmoid_result

def get_alice_data():
    # Alice's data: 2D matrix
    matrix_data_alice = jnp.array([[1, 2], [3, 4]])
    return matrix_data_alice

def get_bob_data():
    # Bob's data: Convert a comma-separated string to a 2D matrix
    string_data_bob = "5, 6; 7, 8"
    rows = [row.split(",") for row in string_data_bob.split(";")]
    matrix_data_bob = jnp.array([[float(value) for value in row] for row in rows])
    return matrix_data_bob

# Pass data to PYU
data_alice = alice(get_alice_data)()
data_bob = bob(get_bob_data)()

# Perform privacy computation on SPU
result = spu_node_device(func)(data_alice, data_bob)

# Reveal results
revealed_result = sf.reveal(result)

# Print revealed result
print(f"Result: {revealed_result}")

# Clean environment
sf.shutdown()
