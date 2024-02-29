
import secretflow as sf
import jax.numpy as jnp

# Initialize SPU and participants
sf.init(['alice', 'bob'], address='local')
alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu_node_device = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

# Define the privacy computation function
def func(data_alice, data_bob):
    # Convert the messages to lowercase
    data_alice = jnp.char.lower(data_alice)
    data_bob = jnp.char.lower(data_bob)

    # Remove all punctuation from the messages
    data_alice = jnp.char.replace(data_alice, jnp.array([".", "!", "?"]), "")
    data_bob = jnp.char.replace(data_bob, jnp.array([".", "!", "?"]), "")

    # Split the messages into words
    alice_words = jnp.char.split(data_alice, " ")
    bob_words = jnp.char.split(data_bob, " ")

    # Find the most common word in each message
    alice_most_common_word, _ = jnp.unique(alice_words, return_counts=True).max(axis=0)
    bob_most_common_word, _ = jnp.unique(bob_words, return_counts=True).max(axis=0)

    # Return the most common word in both messages
    return jnp.char.lower(jnp.where(alice_most_common_word == bob_most_common_word, alice_most_common_word, ""))

# Define the function to get Alice's data
def get_alice_data():
    # Return Alice's message
    return "Hello, Bob! How are you today?"

# Define the function to get Bob's data
def get_bob_data():
    # Return Bob's message
    return "Hi, Alice! I am doing well, thank you!"

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
