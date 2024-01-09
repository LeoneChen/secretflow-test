import os
import re
import time
import hashlib
import subprocess
from openai import OpenAI
import backoff
import ast
import openai

PYTHON_FILE = "/home/chenliheng/anaconda3/envs/secretflow/bin/python"

USE_PYTHON_CODEBLOCK_INSTRUCTION = "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
GEN_SPU_CODE_INSTRUCTION = "You are an AI that only responds with only python code. SecretFlow provide Python API for Secure Process Unit (SPU) to enable privacy computation. You will be given a SPU code to be filled, places that need to be filled are in comment and surrounded by angle brackets, texts between angle brackets are descriptions of the code you need to generate. For example:\n```python\n# <some descriptions>\n```\n. You need to fill SPU code and provide a complete code."
GEN_SPU_CODE_PROMPT = """
[SPU code need to be filled]
```python
import secretflow as sf
# <import libraries that get_alice_data, get_bob_data, and func will use>

# Initialize SPU and participants
sf.init(['alice', 'bob'], address='local')
alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu_node_device = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

def func(data_alice, data_bob):
    # <Use data_alice and data_bob to complete privacy computation and return the calculation results. The code must have nothing to do with SecretFlow and SPU. Avoid useing any random functions. Use jax.numpy instead of numpy.>

def get_alice_data():
    # <Return alice's data. If string data exists, it need to be converted to list data or vector data. The code must have nothing to do with SecretFlow and SPU. Avoid useing any random functions. Use jax.numpy instead of numpy.>

def get_bob_data():
    # <Return bob's data. If string data exists, it need to be converted to list data or vector data. The code must have nothing to do with SecretFlow and SPU. Avoid useing any random functions. Use jax.numpy instead of numpy.>

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
```

[Example SPU code]
```python
import secretflow as sf

# Initialize SPU and participants
sf.init(['alice', 'bob'], address='local')
alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu_node_device = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

# Privacy computation function
def func(data_alice, data_bob):
    # Compute the sum of data from alice and bob
    return data_alice + data_bob

# Function to get alice's data
def get_alice_data():
    # Here alice's data is 10
    return 10

# Function to get bob's data
def get_bob_data():
    # Here bob's data is 5
    return 5

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
```
"""
RELECT_USE_PYTHON_CODEBLOCK_INSTRUCTION = "You are an AI assistant. SecretFlow provide Python API for Secure Process Unit (SPU) to enable privacy computation. In previous round, I have asked you to fill a SPU code, places that need to be filled were in comment and surrounded by angle brackets, texts between angle brackets were descriptions of the code you need to generate. For example:\n```python\n# <some descriptions>\n```\n. However, in previous round, you gived an unexpected response, you did not use a Python code block to write your previous response. For example:\n```python\nprint('Hello world!')\n```.. Now you will be given previous request and response, please give a correct response with Python code block this time."

RELECT_GEN_VALID_PYTHON_CODE_INSTRUCTION = "You are an AI assistant. SecretFlow provide Python API for Secure Process Unit (SPU) to enable privacy computation. In previous round, I have asked you to fill a SPU code, places that need to be filled were in comment and surrounded by angle brackets, texts between angle brackets were descriptions of the code you need to generate. For example:\n```python\n# <some descriptions>\n```\n. However, in previous round, you gived an unexpected response, syntax of the python code you give is invalid. Now you will be given previous request and response, please give a Python code which syntax is valid this time."

GEN_NORMAL_CODE_INSTRUCTION = "You are an AI that only responds with only python code. SecretFlow provide Python API for Secure Process Unit (SPU) to enable privacy computation. You will be given a complete SPU code, and you need to generate normal python code (you will be provided with template of normal code), which perform same computation as SPU code but in normal mode (non-SPU mode)."
GEN_NORMAL_CODE_PROMPT = """
[Template of normal code]
```python
# <import libraries that get_alice_data, get_bob_data, and func will use>

def func(data_alice, data_bob):
    # <Almost same as `func` in SPU code. Use data_alice and data_bob to complete normal computation and return the calculation results. Avoid useing any random functions. Use numpy instead of jax.numpy.>

def get_alice_data():
    # <Almost same as `get_alice_data` in SPU code. Return alice's data. If string data exists, it need to be converted to list data or vector data. Avoid useing any random functions. Use numpy instead of jax.numpy.>

def get_bob_data():
    # <Almost same as `get_bob_data` in SPU code. Return bob's data. If string data exists, it need to be converted to list data or vector data. The code must have nothing to do with SecretFlow and SPU. Avoid useing any random functions. Use numpy instead of jax.numpy.>

# Get data
data_alice = get_alice_data()
data_bob = get_bob_data()

# Perform normal computation
result = func(data_alice, data_bob)

# Print result
print(f"Result: {result}")
```

[Example SPU Code]
```python
import secretflow as sf

# Initialize SPU and participants
sf.init(['alice', 'bob'], address='local')
alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu_node_device = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

# Privacy computation function
def func(data_alice, data_bob):
    # Compute the sum of data from alice and bob
    return data_alice + data_bob

# Function to get alice's data
def get_alice_data():
    # Here alice's data is 10
    return 10

# Function to get bob's data
def get_bob_data():
    # Here bob's data is 5
    return 5

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
```

[Corresponding Example Normal Code]
```python

# Normal computation function
def func(data_alice, data_bob):
    # Compute the sum of data from alice and bob
    return data_alice + data_bob

# Function to get alice's data
def get_alice_data():
    # Here alice's data is 10
    return 10

# Function to get bob's data
def get_bob_data():
    # Here bob's data is 5
    return 5

# Get data
data_alice = get_alice_data()
data_bob = get_bob_data()

# Perform normal computation on SPU
result = func(data_alice, data_bob)

# Print result
print(f"Result: {result}")
```

[Given SPU Code]
"""

COMPARE_RESULT_INSTRUCTION = """
You are an AI assistant. SecretFlow provide Python API for Secure Process Unit (SPU) to enable privacy computation. You will be given a SPU code and it's execution result, as well as a normal code that perform same computation but in normal mode and it's execution result. Both computaion results of them are followed by 'Result:'. You need to answer 'Same', 'Different' or 'Fail' in the first line, and then explain the reason start from the second line.
    - 'Fail' means one of SPU code and normal code run to error, therefore we cannot obtain all computaion results. Usually when an error occurs, we will find that the output contains error message and even call stack.
    - 'Different' means we sucessfully obtain computaion results of both, but they are different.
    - 'Same' means we sucessfully obtain computaion results of both, and they are same.

Ignore complaints like below, since device may not have GPU/TPU, so falling back to CPU is OK.
```shell
    INFO:jax._src.xla_bridge:Unable to initialize backend 'cuda': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
    INFO:jax._src.xla_bridge:Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
    INFO:jax._src.xla_bridge:Unable to initialize backend 'tpu': INVALID_ARGUMENT: TpuPlatform is not available.
    INFO:jax._src.xla_bridge:Unable to initialize backend 'plugin': xla_extension has no attributes named get_plugin_device_client. Compile TensorFlow with //tensorflow/compiler/xla/python:enable_plugin_device set to true (defaults to false) to enable this.
    WARNING:jax._src.xla_bridge:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
```

Ignore complaints like below from SPURuntime, since it's irrelevant to the computation result.
```shell
    (SPURuntime pid=1816702) 2023-12-24 21:48:50.413 [info] [default_brpc_retry_policy.cc:DoRetry:52] socket error, sleep=1000000us and retry
    (SPURuntime pid=1816702) 2023-12-24 21:48:51.413 [info] [default_brpc_retry_policy.cc:LogHttpDetail:29] cntl ErrorCode '112', http status code '200', response header '', error msg '[E111]Fail to connect Socket{id=0 addr=127.0.0.1:38701} (0x0x48e4a80): Connection refused [R1][E112]Not connected to 127.0.0.1:38701 yet, server_id=0'
    (SPURuntime pid=1816702) 2023-12-24 21:48:51.414 [info] [default_brpc_retry_policy.cc:DoRetry:75] aggressive retry, sleep=1000000us and retry
    (SPURuntime pid=1816702) 2023-12-24 21:48:52.414 [info] [default_brpc_retry_policy.cc:LogHttpDetail:29] cntl ErrorCode '112', http status code '200', response header '', error msg '[E111]Fail to connect Socket{id=0 addr=127.0.0.1:38701} (0x0x48e4a80): Connection refused [R1][E112]Not connected to 127.0.0.1:38701 yet, server_id=0 [R2][E112]Not connected to 127.0.0.1:38701 yet, server_id=0'
    (SPURuntime pid=1816702) 2023-12-24 21:48:52.414 [info] [default_brpc_retry_policy.cc:DoRetry:75] aggressive retry, sleep=1000000us and retry
    (SPURuntime pid=1816700) 2023-12-24 21:48:52.532 [info] [default_brpc_retry_policy.cc:DoRetry:69] not retry for reached rcp timeout, ErrorCode '1008', error msg '[E1008]Reached timeout=2000ms @127.0.0.1:32921'
```
"""
COMPARE_RESULT_PROMPT = """
##### Fail Example ######
[Fail Example: SPU Code]
```python
import secretflow as sf

# Initialize SPU and participants
sf.init(['alice', 'bob'], address='local')
alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu_node_device = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

# Privacy computation function
def func(data_alice, data_bob):
    # Compute the sum of data from alice and bob
    return data_alice + data_bob

# Function to get alice's data
def get_alice_data():
    # Here alice's data is "10"
    return "10"

# Function to get bob's data
def get_bob_data():
    # Here bob's data is "5"
    return "5"

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
```
[Fail Example: Execution result of SPU code ]
```shell
2024-01-09 20:47:27,887 INFO worker.py:1538 -- Started a local Ray instance.
Traceback (most recent call last):
  File "test.py", line 28, in <module>
    result = spu_node_device(func)(data_alice, data_bob)
  File "/home/chenliheng/anaconda3/envs/secretflow/lib/python3.8/site-packages/secretflow/device/device/spu.py", line 1750, in wrapper
    args, kwargs = self._place_arguments(*args, **kwargs)
  File "/home/chenliheng/anaconda3/envs/secretflow/lib/python3.8/site-packages/secretflow/device/device/spu.py", line 1717, in _place_arguments
    return jax.tree_util.tree_map(place, (args, kwargs))
  File "/home/chenliheng/anaconda3/envs/secretflow/lib/python3.8/site-packages/jax/_src/tree_util.py", line 210, in tree_map
    return treedef.unflatten(f(*xs) for xs in zip(*all_leaves))
  File "/home/chenliheng/anaconda3/envs/secretflow/lib/python3.8/site-packages/jax/_src/tree_util.py", line 210, in <genexpr>
    return treedef.unflatten(f(*xs) for xs in zip(*all_leaves))
  File "/home/chenliheng/anaconda3/envs/secretflow/lib/python3.8/site-packages/secretflow/device/device/spu.py", line 1707, in place
    return obj.to(self)
  File "/home/chenliheng/anaconda3/envs/secretflow/lib/python3.8/site-packages/secretflow/device/device/base.py", line 70, in to
    return dispatch(_name_of_to(device.device_type), self, device, *args, **kwargs)
  File "/home/chenliheng/anaconda3/envs/secretflow/lib/python3.8/site-packages/secretflow/device/device/register.py", line 111, in dispatch
    return _registrar.dispatch(self.device_type, name, self, *args, **kwargs)
  File "/home/chenliheng/anaconda3/envs/secretflow/lib/python3.8/site-packages/secretflow/device/device/register.py", line 80, in dispatch
    return self._ops[device_type][name](*args, **kwargs)
  File "/home/chenliheng/anaconda3/envs/secretflow/lib/python3.8/site-packages/secretflow/device/kernels/pyu.py", line 79, in pyu_to_spu
    shares_chunk_count = sfd.get(shares_chunk_count.data)
  File "/home/chenliheng/anaconda3/envs/secretflow/lib/python3.8/site-packages/secretflow/distributed/primitive.py", line 136, in get
    return ray.get(object_refs)
  File "/home/chenliheng/anaconda3/envs/secretflow/lib/python3.8/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
    return func(*args, **kwargs)
  File "/home/chenliheng/anaconda3/envs/secretflow/lib/python3.8/site-packages/ray/_private/worker.py", line 2309, in get
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(TypeError): ray::_run() (pid=28515, ip=202.112.47.76)
  File "<__array_function__ internals>", line 180, in result_type
TypeError: data type '' not understood

The above exception was the direct cause of the following exception:

ray::_run() (pid=28515, ip=202.112.47.76)
  File "/home/chenliheng/anaconda3/envs/secretflow/lib/python3.8/site-packages/jax/_src/numpy/lax_numpy.py", line 2015, in array
    dtype = dtypes._lattice_result_type(*leaves)[0] if leaves else dtypes.float_
  File "/home/chenliheng/anaconda3/envs/secretflow/lib/python3.8/site-packages/jax/_src/dtypes.py", line 537, in _lattice_result_type
    dtypes, weak_types = zip(*(_dtype_and_weaktype(arg) for arg in args))
  File "/home/chenliheng/anaconda3/envs/secretflow/lib/python3.8/site-packages/jax/_src/dtypes.py", line 537, in <genexpr>
    dtypes, weak_types = zip(*(_dtype_and_weaktype(arg) for arg in args))
  File "/home/chenliheng/anaconda3/envs/secretflow/lib/python3.8/site-packages/jax/_src/dtypes.py", line 360, in _dtype_and_weaktype
    return dtype(value), any(value is typ for typ in _weak_types) or is_weakly_typed(value)
  File "/home/chenliheng/anaconda3/envs/secretflow/lib/python3.8/site-packages/jax/_src/dtypes.py", line 530, in dtype
    raise TypeError(f"Cannot determine dtype of {x}") from err
TypeError: Cannot determine dtype of 10

During handling of the above exception, another exception occurred:

ray::_run() (pid=28515, ip=202.112.47.76)
  File "/home/chenliheng/anaconda3/envs/secretflow/lib/python3.8/site-packages/secretflow/device/device/pyu.py", line 156, in _run
    return fn(*args, **kwargs)
  File "/home/chenliheng/anaconda3/envs/secretflow/lib/python3.8/site-packages/secretflow/device/kernels/pyu.py", line 69, in get_shares_chunk_count
    return io.get_shares_chunk_count(data, vtype)
  File "/home/chenliheng/anaconda3/envs/secretflow/lib/python3.8/site-packages/secretflow/device/device/spu.py", line 214, in get_shares_chunk_count
    val = _plaintext_to_numpy(val)
  File "/home/chenliheng/anaconda3/envs/secretflow/lib/python3.8/site-packages/secretflow/device/device/spu.py", line 117, in _plaintext_to_numpy
    return np.asarray(jnp.asarray(data))
  File "/home/chenliheng/anaconda3/envs/secretflow/lib/python3.8/site-packages/jax/_src/numpy/lax_numpy.py", line 2071, in asarray
    return array(a, dtype=dtype, copy=False, order=order)  # type: ignore
  File "/home/chenliheng/anaconda3/envs/secretflow/lib/python3.8/site-packages/jax/_src/numpy/lax_numpy.py", line 2020, in array
    dtype = dtypes._lattice_result_type(*leaves)[0]
  File "/home/chenliheng/anaconda3/envs/secretflow/lib/python3.8/site-packages/jax/_src/dtypes.py", line 537, in _lattice_result_type
    dtypes, weak_types = zip(*(_dtype_and_weaktype(arg) for arg in args))
  File "/home/chenliheng/anaconda3/envs/secretflow/lib/python3.8/site-packages/jax/_src/dtypes.py", line 537, in <genexpr>
    dtypes, weak_types = zip(*(_dtype_and_weaktype(arg) for arg in args))
  File "/home/chenliheng/anaconda3/envs/secretflow/lib/python3.8/site-packages/jax/_src/dtypes.py", line 360, in _dtype_and_weaktype
    return dtype(value), any(value is typ for typ in _weak_types) or is_weakly_typed(value)
  File "/home/chenliheng/anaconda3/envs/secretflow/lib/python3.8/site-packages/jax/_src/dtypes.py", line 532, in dtype
    raise TypeError(f"Value '{x}' with dtype {dt} is not a valid JAX array "
TypeError: Value '10' with dtype <U2 is not a valid JAX array type. Only arrays of numeric types are supported by JAX.
```

[Fail Example: Normal code ]
```python
# Normal computation function
def func(data_alice, data_bob):
    # Compute the sum of data from alice and bob
    return data_alice + data_bob

# Function to get alice's data
def get_alice_data():
    # Here alice's data is "10"
    return "10"

# Function to get bob's data
def get_bob_data():
    # Here bob's data is "5"
    return "5"

# Get data
data_alice = get_alice_data()
data_bob = get_bob_data()

# Perform normal computation on SPU
result = func(data_alice, data_bob)

# Print result
print(f"Result: {result}")
```

[Fail Example: Execution result of Normal code ]
```shell
Result: 105
```

The reason why fail is string can not directly returned as data of alice or bob in SPU code, and then SPU code run to error, and error message and even call stack are shown. But normal code works fine, and output 105. In this case SPU code run to error while normal code not, so the compare result is fail. Anyone of normal code and SPU code run to error can cause result of comparison is Fail.

##### Different Example ######
[Different Example: SPU Code]
```python
import secretflow as sf

# Initialize SPU and participants
sf.init(['alice', 'bob'], address='local')
alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu_node_device = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

# Privacy computation function
def func(data_alice, data_bob):
    # Compute the sum of data from alice and bob
    return data_alice + data_bob

# Function to get alice's data
def get_alice_data():
    # Here alice's data is 10
    return 10

# Function to get bob's data
def get_bob_data():
    # Here bob's data is 5
    return 5

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
```
[Different Example: Execution result of SPU code ]
```shell
2024-01-09 20:49:18,925 INFO worker.py:1538 -- Started a local Ray instance.
(_run pid=1861) INFO:jax._src.xla_bridge:Unable to initialize backend 'cuda': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
(_run pid=1861) INFO:jax._src.xla_bridge:Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
(_run pid=1861) INFO:jax._src.xla_bridge:Unable to initialize backend 'tpu': INVALID_ARGUMENT: TpuPlatform is not available.
(_run pid=1861) INFO:jax._src.xla_bridge:Unable to initialize backend 'plugin': xla_extension has no attributes named get_plugin_device_client. Compile TensorFlow with //tensorflow/compiler/xla/python:enable_plugin_device set to true (defaults to false) to enable this.
(_run pid=1861) WARNING:jax._src.xla_bridge:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
(_run pid=3124) INFO:jax._src.xla_bridge:Unable to initialize backend 'cuda': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
(_run pid=3124) INFO:jax._src.xla_bridge:Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
(_run pid=3124) INFO:jax._src.xla_bridge:Unable to initialize backend 'tpu': INVALID_ARGUMENT: TpuPlatform is not available.
(_run pid=3124) INFO:jax._src.xla_bridge:Unable to initialize backend 'plugin': xla_extension has no attributes named get_plugin_device_client. Compile TensorFlow with //tensorflow/compiler/xla/python:enable_plugin_device set to true (defaults to false) to enable this.
(_run pid=3124) WARNING:jax._src.xla_bridge:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
(SPURuntime pid=20767) 2024-01-09 20:49:23.705 [info] [default_brpc_retry_policy.cc:DoRetry:52] socket error, sleep=1000000us and retry
(SPURuntime pid=20767) 2024-01-09 20:49:24.705 [info] [default_brpc_retry_policy.cc:LogHttpDetail:29] cntl ErrorCode '112', http status code '200', response header '', error msg '[E111]Fail to connect Socket{id=0 addr=127.0.0.1:40519} (0x0x6209523bc1c0): Connection refused [R1][E112]Not connected to 127.0.0.1:40519 yet, server_id=0'
(SPURuntime pid=20767) 2024-01-09 20:49:24.705 [info] [default_brpc_retry_policy.cc:DoRetry:75] aggressive retry, sleep=1000000us and retry
(SPURuntime pid=20767) 2024-01-09 20:49:25.705 [info] [default_brpc_retry_policy.cc:LogHttpDetail:29] cntl ErrorCode '112', http status code '200', response header '', error msg '[E111]Fail to connect Socket{id=0 addr=127.0.0.1:40519} (0x0x6209523bc1c0): Connection refused [R1][E112]Not connected to 127.0.0.1:40519 yet, server_id=0 [R2][E112]Not connected to 127.0.0.1:40519 yet, server_id=0'
(SPURuntime pid=20767) 2024-01-09 20:49:25.706 [info] [default_brpc_retry_policy.cc:DoRetry:75] aggressive retry, sleep=1000000us and retry
(SPURuntime pid=20844) 2024-01-09 20:49:25.718 [info] [default_brpc_retry_policy.cc:DoRetry:69] not retry for reached rcp timeout, ErrorCode '1008', error msg '[E1008]Reached timeout=2000ms @127.0.0.1:58835'
Result: 14
```

[Different Example: Normal code ]
```python
# Normal computation function
def func(data_alice, data_bob):
    # Compute the substract of data from alice and bob
    return data_alice + data_bob

# Function to get alice's data
def get_alice_data():
    # Here alice's data is 10
    return 10

# Function to get bob's data
def get_bob_data():
    # Here bob's data is 5
    return 5

# Get data
data_alice = get_alice_data()
data_bob = get_bob_data()

# Perform normal computation on SPU
result = func(data_alice, data_bob)

# Print result
print(f"Result: {result}")
```

[Different Example: Execution result of Normal code ]
```shell
Result: 15
```

The reason why different is that both of SPU code and normal code work fine firstly, but result of SPU code is 14, while result of normal code is 15.

##### Same Example 1 ######
[Same Example 1: SPU Code]
```python
import secretflow as sf

# Initialize SPU and participants
sf.init(['alice', 'bob'], address='local')
alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu_node_device = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

# Privacy computation function
def func(data_alice, data_bob):
    # Compute the sum of data from alice and bob
    return data_alice + data_bob

# Function to get alice's data
def get_alice_data():
    # Here alice's data is 10
    return 10

# Function to get bob's data
def get_bob_data():
    # Here bob's data is 5
    return 5

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
```
[Same Example 1: Execution result of SPU code ]
```shell
2024-01-09 20:49:18,925 INFO worker.py:1538 -- Started a local Ray instance.
(_run pid=1861) INFO:jax._src.xla_bridge:Unable to initialize backend 'cuda': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
(_run pid=1861) INFO:jax._src.xla_bridge:Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
(_run pid=1861) INFO:jax._src.xla_bridge:Unable to initialize backend 'tpu': INVALID_ARGUMENT: TpuPlatform is not available.
(_run pid=1861) INFO:jax._src.xla_bridge:Unable to initialize backend 'plugin': xla_extension has no attributes named get_plugin_device_client. Compile TensorFlow with //tensorflow/compiler/xla/python:enable_plugin_device set to true (defaults to false) to enable this.
(_run pid=1861) WARNING:jax._src.xla_bridge:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
(_run pid=3124) INFO:jax._src.xla_bridge:Unable to initialize backend 'cuda': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
(_run pid=3124) INFO:jax._src.xla_bridge:Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
(_run pid=3124) INFO:jax._src.xla_bridge:Unable to initialize backend 'tpu': INVALID_ARGUMENT: TpuPlatform is not available.
(_run pid=3124) INFO:jax._src.xla_bridge:Unable to initialize backend 'plugin': xla_extension has no attributes named get_plugin_device_client. Compile TensorFlow with //tensorflow/compiler/xla/python:enable_plugin_device set to true (defaults to false) to enable this.
(_run pid=3124) WARNING:jax._src.xla_bridge:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
(SPURuntime pid=20767) 2024-01-09 20:49:23.705 [info] [default_brpc_retry_policy.cc:DoRetry:52] socket error, sleep=1000000us and retry
(SPURuntime pid=20767) 2024-01-09 20:49:24.705 [info] [default_brpc_retry_policy.cc:LogHttpDetail:29] cntl ErrorCode '112', http status code '200', response header '', error msg '[E111]Fail to connect Socket{id=0 addr=127.0.0.1:40519} (0x0x6209523bc1c0): Connection refused [R1][E112]Not connected to 127.0.0.1:40519 yet, server_id=0'
(SPURuntime pid=20767) 2024-01-09 20:49:24.705 [info] [default_brpc_retry_policy.cc:DoRetry:75] aggressive retry, sleep=1000000us and retry
(SPURuntime pid=20767) 2024-01-09 20:49:25.705 [info] [default_brpc_retry_policy.cc:LogHttpDetail:29] cntl ErrorCode '112', http status code '200', response header '', error msg '[E111]Fail to connect Socket{id=0 addr=127.0.0.1:40519} (0x0x6209523bc1c0): Connection refused [R1][E112]Not connected to 127.0.0.1:40519 yet, server_id=0 [R2][E112]Not connected to 127.0.0.1:40519 yet, server_id=0'
(SPURuntime pid=20767) 2024-01-09 20:49:25.706 [info] [default_brpc_retry_policy.cc:DoRetry:75] aggressive retry, sleep=1000000us and retry
(SPURuntime pid=20844) 2024-01-09 20:49:25.718 [info] [default_brpc_retry_policy.cc:DoRetry:69] not retry for reached rcp timeout, ErrorCode '1008', error msg '[E1008]Reached timeout=2000ms @127.0.0.1:58835'
Result: 15
```

[Same Example 1: Normal code ]
```python
# Normal computation function
def func(data_alice, data_bob):
    # Compute the substract of data from alice and bob
    return data_alice + data_bob

# Function to get alice's data
def get_alice_data():
    # Here alice's data is 10
    return 10

# Function to get bob's data
def get_bob_data():
    # Here bob's data is 5
    return 5

# Get data
data_alice = get_alice_data()
data_bob = get_bob_data()

# Perform normal computation on SPU
result = func(data_alice, data_bob)

# Print result
print(f"Result: {result}")
```

[Same Example 1: Execution result of Normal code ]
```shell
Result: 15
```

The reason why same is that both of SPU code and normal code work fine firstly, and both results of SPU code and normal code are 15, they are same.

##### Same Example 2 ######
[Same Example 2: SPU Code]
```python
import secretflow as sf
import jax.numpy as jnp

# Initialize SPU and participants
sf.init(['alice', 'bob'], address='local')
alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu_node_device = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

def func(data_alice, data_bob):
    # Assume we're calculating the mean of data from Alice and Bob
    total = jnp.concatenate([data_alice, data_bob])
    mean = total.mean()
    return mean

def get_alice_data():
    # Assume Alice data is a list of numbers
    data_alice = jnp.array([1, 2, 3, 4, 5])
    return data_alice

def get_bob_data():
    # Assume Bob data is a list of numbers
    data_bob = jnp.array([6, 7, 8, 9, 10])
    return data_bob

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
```

[Same Example 2: Execution Result of SPU Code]
```shell
2024-01-09 21:13:10,986 INFO worker.py:1538 -- Started a local Ray instance.
INFO:jax._src.xla_bridge:Unable to initialize backend 'cuda': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
INFO:jax._src.xla_bridge:Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
INFO:jax._src.xla_bridge:Unable to initialize backend 'tpu': INVALID_ARGUMENT: TpuPlatform is not available.
INFO:jax._src.xla_bridge:Unable to initialize backend 'plugin': xla_extension has no attributes named get_plugin_device_client. Compile TensorFlow with //tensorflow/compiler/xla/python:enable_plugin_device set to true (defaults to false) to enable this.
WARNING:jax._src.xla_bridge:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
(_run pid=57962) INFO:jax._src.xla_bridge:Unable to initialize backend 'cuda': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
(_run pid=57962) INFO:jax._src.xla_bridge:Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
(_run pid=57962) INFO:jax._src.xla_bridge:Unable to initialize backend 'tpu': INVALID_ARGUMENT: TpuPlatform is not available.
(_run pid=57962) INFO:jax._src.xla_bridge:Unable to initialize backend 'plugin': xla_extension has no attributes named get_plugin_device_client. Compile TensorFlow with //tensorflow/compiler/xla/python:enable_plugin_device set to true (defaults to false) to enable this.
(_run pid=57962) WARNING:jax._src.xla_bridge:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
(_run pid=60098) INFO:jax._src.xla_bridge:Unable to initialize backend 'cuda': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
(_run pid=60098) INFO:jax._src.xla_bridge:Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
(_run pid=60098) INFO:jax._src.xla_bridge:Unable to initialize backend 'tpu': INVALID_ARGUMENT: TpuPlatform is not available.
(_run pid=60098) INFO:jax._src.xla_bridge:Unable to initialize backend 'plugin': xla_extension has no attributes named get_plugin_device_client. Compile TensorFlow with //tensorflow/compiler/xla/python:enable_plugin_device set to true (defaults to false) to enable this.
(_run pid=60098) WARNING:jax._src.xla_bridge:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
(SPURuntime pid=11547) 2024-01-09 21:13:15.540 [info] [default_brpc_retry_policy.cc:DoRetry:52] socket error, sleep=1000000us and retry
(SPURuntime pid=11547) 2024-01-09 21:13:16.541 [info] [default_brpc_retry_policy.cc:LogHttpDetail:29] cntl ErrorCode '112', http status code '200', response header '', error msg '[E111]Fail to connect Socket{id=0 addr=127.0.0.1:54063} (0x0x5fc31dd15640): Connection refused [R1][E112]Not connected to 127.0.0.1:54063 yet, server_id=0'
(SPURuntime pid=11547) 2024-01-09 21:13:16.541 [info] [default_brpc_retry_policy.cc:DoRetry:75] aggressive retry, sleep=1000000us and retry
(SPURuntime pid=11547) 2024-01-09 21:13:17.541 [info] [default_brpc_retry_policy.cc:LogHttpDetail:29] cntl ErrorCode '112', http status code '200', response header '', error msg '[E111]Fail to connect Socket{id=0 addr=127.0.0.1:54063} (0x0x5fc31dd15640): Connection refused [R1][E112]Not connected to 127.0.0.1:54063 yet, server_id=0 [R2][E112]Not connected to 127.0.0.1:54063 yet, server_id=0'
(SPURuntime pid=11547) 2024-01-09 21:13:17.541 [info] [default_brpc_retry_policy.cc:DoRetry:75] aggressive retry, sleep=1000000us and retry
(SPURuntime pid=11794) 2024-01-09 21:13:18.092 [info] [default_brpc_retry_policy.cc:DoRetry:69] not retry for reached rcp timeout, ErrorCode '1008', error msg '[E1008]Reached timeout=2000ms @127.0.0.1:56991'
Result: 5.499999523162842
```

[Same Example 2: Normal Code]
```python
import numpy as np

def func(data_alice, data_bob):
    # Assume we're calculating the mean of data from Alice and Bob
    total = np.concatenate([data_alice, data_bob])
    mean = total.mean()
    return mean

def get_alice_data():
    # Assume Alice data is a list of numbers
    data_alice = np.array([1, 2, 3, 4, 5])
    return data_alice

def get_bob_data():
    # Assume Bob data is a list of numbers
    data_bob = np.array([6, 7, 8, 9, 10])
    return data_bob

# Get data
data_alice = get_alice_data()
data_bob = get_bob_data()

# Perform normal computation
result = func(data_alice, data_bob)

# Print result
print(f"Result: {result}")
```

[Same Example 2: Execution Result of Normal Code]
```shell
Result: 5.5
```

The reason why same is that both of SPU code and normal code work fine firstly, and results of SPU code is 5.499999523162842 and results of normal code are 5.5. In floating point arithmetic, 5.499999523162842 and 5.5 are almost same and the difference is so small.
"""


class ChatProxy:
    def __init__(self) -> None:
        self.client = OpenAI()

    @backoff.on_exception(backoff.expo, openai.OpenAIError)
    def send(self, model="gpt-4", **kwargs):
        return self.client.chat.completions.create(model=model, **kwargs)

    def update(self, orig_request, orig_response, requirement):
        return self.send(
            messages=[
                {
                    "role": "system",
                    "content": f"{requirement}",
                },
                {
                    "role": "user",
                    "content": f"Previous request:\n{orig_request}\n\nPrevious reponse:\n{orig_response}",
                },
            ],
        )


def py_is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False


def get_py_code(proxy, request, orig_response):
    response = orig_response
    while True:
        py_code = re.search(
            "```python(.*?)```",
            response,
            re.DOTALL,
        )

        if py_code:
            return py_code.group(1)

        completion = proxy.update(
            request,
            response,
            RELECT_USE_PYTHON_CODEBLOCK_INSTRUCTION,
        )
        response = completion.choices[0].message.content


def get_valid_py_code(proxy, request, orig_response):
    response = orig_response
    while True:
        code = get_py_code(proxy, request, response)
        if py_is_syntax_valid(code):
            return code

        completion = proxy.update(
            request,
            response,
            RELECT_GEN_VALID_PYTHON_CODE_INSTRUCTION,
        )
        response = completion.choices[0].message.content


def main():
    proxy = ChatProxy()
    round = 0
    while True:
        round += 1  # Start from 1
        print(f"[Round {round}] Request to generate SPU code >>>")
        completion = proxy.send(
            messages=[
                {
                    "role": "system",
                    "content": f"{GEN_SPU_CODE_INSTRUCTION}\n{USE_PYTHON_CODEBLOCK_INSTRUCTION}",
                },
                {
                    "role": "user",
                    "content": GEN_SPU_CODE_PROMPT,
                },
            ],
        )
        print(f"[Round {round}] Get SPU code <<<")

        spu_code = get_valid_py_code(
            proxy, GEN_SPU_CODE_PROMPT, completion.choices[0].message.content
        )
        spu_code_hash = hashlib.sha256(spu_code.encode()).hexdigest()
        spu_code_file_name = str(spu_code_hash) + "_spu_code.py"
        os.makedirs("seeds", exist_ok=True)
        with open(os.path.join("seeds", spu_code_file_name), "w") as f:
            f.write(spu_code)

        print(f"[Round {round}] Request to generate normal code >>>")
        completion = proxy.send(
            messages=[
                {
                    "role": "system",
                    "content": f"{GEN_NORMAL_CODE_INSTRUCTION}\n{USE_PYTHON_CODEBLOCK_INSTRUCTION}",
                },
                {
                    "role": "user",
                    "content": f"{GEN_NORMAL_CODE_PROMPT}\n```python\n{spu_code}\n```",
                },
            ],
        )
        print(f"[Round {round}] Get normal code <<<")

        normal_code = get_valid_py_code(
            proxy,
            f"{GEN_NORMAL_CODE_PROMPT}\n```python\n{spu_code}\n```",
            completion.choices[0].message.content,
        )
        # normal_code_hash = hashlib.sha256(normal_code.encode()).hexdigest()
        normal_code_file_name = str(spu_code_hash) + "_normal_code.py"
        with open(os.path.join("seeds", normal_code_file_name), "w") as f:
            f.write(normal_code)

        spu_result = subprocess.run(
            [
                PYTHON_FILE,
                os.path.join("seeds", spu_code_file_name),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        normal_result = subprocess.run(
            [
                PYTHON_FILE,
                os.path.join("seeds", normal_code_file_name),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        print(f"[Round {round}] Request to compare output >>>")
        completion = proxy.send(
            messages=[
                {"role": "system", "content": COMPARE_RESULT_INSTRUCTION},
                {
                    "role": "user",
                    "content": f"{COMPARE_RESULT_PROMPT}\n[SPU Code]\n```python\n{spu_code}\n```\n[Execution Result of SPU Code]\n```shell\n{spu_result.stdout.decode()}\n```\n[Normal Code]\n```python\n{normal_code}\n```\n[Execution Result of Normal Code]\n```shell\n{normal_result.stdout.decode()}\n```\n",
                },
            ]
        )
        print(f"[Round {round}] Get response <<<")

        match = re.match(r"(.*?)\n", completion.choices[0].message.content)
        if match:
            first_line = match.group(1)
            result = first_line.lower()
        else:
            result = completion.choices[0].message.content.lower()

        if result not in ("same", "fail", "different"):
            result = "unknown"

        os.makedirs(result, exist_ok=True)
        os.rename(
            os.path.join("seeds", spu_code_file_name),
            os.path.join(result, spu_code_file_name),
        )
        os.rename(
            os.path.join("seeds", normal_code_file_name),
            os.path.join(result, normal_code_file_name),
        )
        print(os.path.join(result, spu_code_file_name))

        if result != "same":
            print(
                "[SPU Code]\n```python\n{spu_code}\n```\n[Execution Result of SPU Code]\n```shell\n{spu_result.stdout.decode()}\n```\n[Normal Code]\n```python\n{normal_code}\n```\n[Execution Result of Normal Code]\n```shell\n{normal_result.stdout.decode()}\n```\n"
            )


if __name__ == "__main__":
    main()
