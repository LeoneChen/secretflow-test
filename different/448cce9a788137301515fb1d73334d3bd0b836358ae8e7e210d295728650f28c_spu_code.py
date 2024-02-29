
import secretflow as sf
import jax.numpy as jnp 

# Initialize SPU and participants
sf.init(['alice', 'bob'], address='local')
alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu_node_device = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

def func(data_alice, data_bob):
    # Here we are calculating the sum of square roots of Alice's and Bob's data
    sum_sqrt = jnp.sum(jnp.sqrt(data_alice)) + jnp.sum(jnp.sqrt(data_bob))
    return sum_sqrt

def get_alice_data():
    # Alice's data is an array containing square of numbers from 1 to 10
    return jnp.array([jnp.power(i, 2) for i in range(1, 11)])

def get_bob_data():
    # Bob's data is an array containing cube of numbers from 11 to 20
    return jnp.array([jnp.power(i, 3) for i in range(11, 21)])

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
