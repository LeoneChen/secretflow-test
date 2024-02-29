
import secretflow as sf
import jax.numpy as jnp

# Initialize SPU and participants
sf.init(['alice', 'bob'], address='local')
alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu_node_device = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

def func(data_alice, data_bob):
    # Compute the dot product and the element-wise maximum of data from alice and bob
    dot_result = jnp.dot(data_alice, data_bob)
    max_result = jnp.maximum(data_alice, data_bob)
    return {'dot': dot_result, 'max': max_result}

def get_alice_data():
    # Alice's data is represented as a 2-dimensional array for this computation
    initial_data = jnp.array([[1, 2, 3], [4, 5, 6]])
    processed_data = jnp.sin(initial_data) * 10  # Example transformation
    return processed_data

def get_bob_data():
    # Bob's data is also in 2-dimensional array form, generated from mixed data types
    raw_data = ["1,2,3", "4,5,Physics:6"]
    clean_data = [x.split(",") for x in raw_data]
    clean_data = [[y.split(":")[-1] for y in x] for x in clean_data]  # Extract numeric values
    numeric_data = jnp.array([[float(y) for y in x] for x in clean_data])
    return numeric_data

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
