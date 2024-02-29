
import secretflow as sf
import jax.numpy as jnp

# Initialize SPU and participants
sf.init(['alice', 'bob'], address='local')
alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu_node_device = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

def func(data_alice, data_bob):
    # Compute the dot product of data_alice and data_bob, then apply a sine function to the result
    dot_product = jnp.dot(data_alice, data_bob)
    sine_result = jnp.sin(dot_product)
    return sine_result

def get_alice_data():
    # Alice's data: a 2D array where the first row is 0.5 to 2.5 and the second row is 3.0 to 5.0, both incrementing by 0.5
    array_data_alice = jnp.array([jnp.arange(0.5, 3, 0.5), jnp.arange(3.0, 5.5, 0.5)])
    return array_data_alice

def get_bob_data():
    # Bob's data: a 2D array where each element is the cube root of its 1D index, reshaped to match Alice's data shape
    indices = jnp.array(range(1, array_data_alice.size + 1))
    cube_root_indices = jnp.cbrt(indices)
    array_data_bob = cube_root_indices.reshape(array_data_alice.shape)
    return array_data_bob

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
