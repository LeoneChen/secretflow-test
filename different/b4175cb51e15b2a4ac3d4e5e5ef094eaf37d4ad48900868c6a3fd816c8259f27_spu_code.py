
import secretflow as sf
import jax.numpy as jnp

# Initialize SPU and participants
sf.init(['alice', 'bob'], address='local')
alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu_node_device = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

def func(data_alice, data_bob):
    # Calculate the outer product of data_alice and data_bob
    outer_result = jnp.outer(data_alice, data_bob)
    # Apply exponential function on the outer result
    exp_result = jnp.exp(outer_result)
    # Compute the mean of all elements in the exponential result
    mean_result = jnp.mean(exp_result)
    return mean_result

def get_alice_data():
    # Alice's data is a simulated temperature reading over a week, here converted from string to vector
    temp_readings_str = "23, 25, 21, 22, 24, 26, 23"
    temp_readings_vector = jnp.array(list(map(float, temp_readings_str.split(','))))
    return temp_readings_vector

def get_bob_data():
    # Bob's data represents humidity readings over the same period in a string, here converted to vector
    humidity_readings_str = "45, 50, 47, 49, 46, 48, 45"
    humidity_readings_vector = jnp.array(list(map(float, humidity_readings_str.split(','))))
    return humidity_readings_vector

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
