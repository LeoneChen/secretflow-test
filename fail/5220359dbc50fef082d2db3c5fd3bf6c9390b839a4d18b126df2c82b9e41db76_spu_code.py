import secretflow as sf
import jax.numpy as jnp

# Initialize SPU and participants
sf.init(['alice', 'bob'], address='local')
alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu_node_device = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

def func(data_alice, data_bob):
    # Here we are calculating the Euclidean distance between the data from Alice and Bob
    euclidean_distance = jnp.sqrt(jnp.sum((data_alice - data_bob)**2))
    return euclidean_distance

def get_alice_data():
    # Alice's data is a jax.numpy array containing the numbers from 1 to 50 with a step of 2
    return jnp.arange(1, 50, 2)

def get_bob_data():
    # Bob's data is a jax.numpy array containing the cubes of numbers from 1 to 25
    return jnp.power(jnp.arange(1, 26), 3)

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
