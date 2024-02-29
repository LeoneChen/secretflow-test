
import secretflow as sf
import jax.numpy as jnp

# Initialize SPU and participants
sf.init(['alice', 'bob'], address='local')
alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu_node_device = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

def func(data_alice, data_bob):
    # This time, calculate the element-wise product and then the sum of the elements, followed by applying softmax.
    product = jnp.multiply(data_alice, data_bob)
    sum_result = jnp.sum(product)
    softmax_result = jnp.exp(sum_result) / jnp.sum(jnp.exp(product))
    return softmax_result

def get_alice_data():
    # Alice's data: vector containing values in ascending order
    vector_data_alice = jnp.arange(1, 4, 1)  # [1, 2, 3]
    return vector_data_alice

def get_bob_data():
    # Bob's data: converting string representation of a range to vector
    string_data_bob = "4, 5, 6"
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
