
import secretflow as sf
import jax.numpy as jnp

# Initialize SPU and participants
sf.init(['alice', 'bob'], address='local')
alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu_node_device = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

def func(data_alice, data_bob):
    # This time we perform a more complex calculation: weighted sum and then apply the tanh activation function.
    weights = jnp.array([0.5, 1.0, 1.5])
    weighted_sum_alice = jnp.dot(weights, data_alice)
    weighted_sum_bob = jnp.dot(weights, data_bob)
    result = jnp.tanh(weighted_sum_alice + weighted_sum_bob)
    return result

def get_alice_data():
    # Alice's data simulation: a more complex vector with three dimensions
    vector_data_alice = [0.1, 0.2, 0.3]
    array_data_alice = jnp.array(vector_data_alice)
    return array_data_alice

def get_bob_data():
    # Bob's data simulation: convert string representation of complex data to vector data
    string_data_bob = "0.4, 0.5, 0.6"
    list_data_bob = string_data_bob.split(", ")
    vector_data_bob = [float(item) for item in list_data_bob]
    array_data_bob = jnp.array(vector_data_bob)
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
