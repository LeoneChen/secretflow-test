import os
import re
import time
import hashlib
import subprocess
from openai import OpenAI
import tenacity
import ast
import openai
import logging
import sys

PYTHON_FILE = "~/anaconda3/envs/secretflow/bin/python"
DEFAULT_OPENAI_MODEL = "gpt-4-0125-preview"

USE_PYTHON_CODEBLOCK_INSTRUCTION = "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
GEN_SPU_CODE_INSTRUCTION = "You are an AI that only responds with only python code. SecretFlow provide Python API for Secure Process Unit (SPU) to enable privacy computation. You will be given a SPU code to be filled, places that need to be filled are in comment and surrounded by angle brackets, texts between angle brackets are descriptions of the code you need to generate. For example:\n```python\n# <some descriptions>\n```\n. You need to fill SPU code and provide a complete code. You will also be given an example SPU code (definitly correct) and the previously generated SPU code (may not correct) if it exists, hope you to generate more complex SPU code."
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
    # <Use data_alice and data_bob to complete privacy computation and return the calculation results. The code must have nothing to do with SecretFlow and SPU. Avoid useing any random functions. Use jax.numpy instead of numpy. This function should not be added Python decorators.>

def get_alice_data():
    # <Return alice's data. If string data exists, it need to be converted to list data or vector data. The code must have nothing to do with SecretFlow and SPU. Avoid useing any random functions. Use jax.numpy instead of numpy. This function should not be added Python decorators.>

def get_bob_data():
    # <Return bob's data. If string data exists, it need to be converted to list data or vector data. The code must have nothing to do with SecretFlow and SPU. Avoid useing any random functions. Use jax.numpy instead of numpy. This function should not be added Python decorators.>

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
You are an AI assistant. SecretFlow provide Python API for Secure Process Unit (SPU) to enable privacy computation. You firstly will be given some examples, and then given a SPU code and it's execution result, as well as a normal code that perform same computation but in normal mode and it's execution result. Both computaion results of them are followed by 'Result:'. You are asked to compare result of SPU code and normal code, and then need to answer 'Same', 'Different' or 'Fail' in the first line, explain the reason start from the second line.
- 'Fail' means one of SPU code and normal code run to error, therefore we cannot obtain all computaion results. Usually when an error occurs, we will find that the output contains error message and even call stack.
- 'Different' means we sucessfully obtain computaion results of both, but they are different.
- 'Same' means we sucessfully obtain computaion results of both, and they are same.
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
  File "/home/xxx/anaconda3/envs/secretflow/lib/python3.8/site-packages/secretflow/device/device/spu.py", line 1750, in wrapper
    args, kwargs = self._place_arguments(*args, **kwargs)
  File "/home/xxx/anaconda3/envs/secretflow/lib/python3.8/site-packages/secretflow/device/device/spu.py", line 1717, in _place_arguments
    return jax.tree_util.tree_map(place, (args, kwargs))
  File "/home/xxx/anaconda3/envs/secretflow/lib/python3.8/site-packages/jax/_src/tree_util.py", line 210, in tree_map
    return treedef.unflatten(f(*xs) for xs in zip(*all_leaves))
  File "/home/xxx/anaconda3/envs/secretflow/lib/python3.8/site-packages/jax/_src/tree_util.py", line 210, in <genexpr>
    return treedef.unflatten(f(*xs) for xs in zip(*all_leaves))
  File "/home/xxx/anaconda3/envs/secretflow/lib/python3.8/site-packages/secretflow/device/device/spu.py", line 1707, in place
    return obj.to(self)
  File "/home/xxx/anaconda3/envs/secretflow/lib/python3.8/site-packages/secretflow/device/device/base.py", line 70, in to
    return dispatch(_name_of_to(device.device_type), self, device, *args, **kwargs)
  File "/home/xxx/anaconda3/envs/secretflow/lib/python3.8/site-packages/secretflow/device/device/register.py", line 111, in dispatch
    return _registrar.dispatch(self.device_type, name, self, *args, **kwargs)
  File "/home/xxx/anaconda3/envs/secretflow/lib/python3.8/site-packages/secretflow/device/device/register.py", line 80, in dispatch
    return self._ops[device_type][name](*args, **kwargs)
  File "/home/xxx/anaconda3/envs/secretflow/lib/python3.8/site-packages/secretflow/device/kernels/pyu.py", line 79, in pyu_to_spu
    shares_chunk_count = sfd.get(shares_chunk_count.data)
  File "/home/xxx/anaconda3/envs/secretflow/lib/python3.8/site-packages/secretflow/distributed/primitive.py", line 136, in get
    return ray.get(object_refs)
  File "/home/xxx/anaconda3/envs/secretflow/lib/python3.8/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
    return func(*args, **kwargs)
  File "/home/xxx/anaconda3/envs/secretflow/lib/python3.8/site-packages/ray/_private/worker.py", line 2309, in get
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(TypeError): ray::_run() (pid=28515, ip=202.112.47.76)
  File "<__array_function__ internals>", line 180, in result_type
TypeError: data type '' not understood

The above exception was the direct cause of the following exception:

ray::_run() (pid=28515, ip=202.112.47.76)
  File "/home/xxx/anaconda3/envs/secretflow/lib/python3.8/site-packages/jax/_src/numpy/lax_numpy.py", line 2015, in array
    dtype = dtypes._lattice_result_type(*leaves)[0] if leaves else dtypes.float_
  File "/home/xxx/anaconda3/envs/secretflow/lib/python3.8/site-packages/jax/_src/dtypes.py", line 537, in _lattice_result_type
    dtypes, weak_types = zip(*(_dtype_and_weaktype(arg) for arg in args))
  File "/home/xxx/anaconda3/envs/secretflow/lib/python3.8/site-packages/jax/_src/dtypes.py", line 537, in <genexpr>
    dtypes, weak_types = zip(*(_dtype_and_weaktype(arg) for arg in args))
  File "/home/xxx/anaconda3/envs/secretflow/lib/python3.8/site-packages/jax/_src/dtypes.py", line 360, in _dtype_and_weaktype
    return dtype(value), any(value is typ for typ in _weak_types) or is_weakly_typed(value)
  File "/home/xxx/anaconda3/envs/secretflow/lib/python3.8/site-packages/jax/_src/dtypes.py", line 530, in dtype
    raise TypeError(f"Cannot determine dtype of {x}") from err
TypeError: Cannot determine dtype of 10

During handling of the above exception, another exception occurred:

ray::_run() (pid=28515, ip=202.112.47.76)
  File "/home/xxx/anaconda3/envs/secretflow/lib/python3.8/site-packages/secretflow/device/device/pyu.py", line 156, in _run
    return fn(*args, **kwargs)
  File "/home/xxx/anaconda3/envs/secretflow/lib/python3.8/site-packages/secretflow/device/kernels/pyu.py", line 69, in get_shares_chunk_count
    return io.get_shares_chunk_count(data, vtype)
  File "/home/xxx/anaconda3/envs/secretflow/lib/python3.8/site-packages/secretflow/device/device/spu.py", line 214, in get_shares_chunk_count
    val = _plaintext_to_numpy(val)
  File "/home/xxx/anaconda3/envs/secretflow/lib/python3.8/site-packages/secretflow/device/device/spu.py", line 117, in _plaintext_to_numpy
    return np.asarray(jnp.asarray(data))
  File "/home/xxx/anaconda3/envs/secretflow/lib/python3.8/site-packages/jax/_src/numpy/lax_numpy.py", line 2071, in asarray
    return array(a, dtype=dtype, copy=False, order=order)  # type: ignore
  File "/home/xxx/anaconda3/envs/secretflow/lib/python3.8/site-packages/jax/_src/numpy/lax_numpy.py", line 2020, in array
    dtype = dtypes._lattice_result_type(*leaves)[0]
  File "/home/xxx/anaconda3/envs/secretflow/lib/python3.8/site-packages/jax/_src/dtypes.py", line 537, in _lattice_result_type
    dtypes, weak_types = zip(*(_dtype_and_weaktype(arg) for arg in args))
  File "/home/xxx/anaconda3/envs/secretflow/lib/python3.8/site-packages/jax/_src/dtypes.py", line 537, in <genexpr>
    dtypes, weak_types = zip(*(_dtype_and_weaktype(arg) for arg in args))
  File "/home/xxx/anaconda3/envs/secretflow/lib/python3.8/site-packages/jax/_src/dtypes.py", line 360, in _dtype_and_weaktype
    return dtype(value), any(value is typ for typ in _weak_types) or is_weakly_typed(value)
  File "/home/xxx/anaconda3/envs/secretflow/lib/python3.8/site-packages/jax/_src/dtypes.py", line 532, in dtype
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
```
[Different Example: Execution result of SPU code ]
```shell
2024-01-10 18:23:39,845 INFO worker.py:1538 -- Started a local Ray instance.
INFO:jax._src.xla_bridge:Unable to initialize backend 'cuda': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
...
(omit)
...
(SPURuntime pid=62013) 2024-01-10 18:23:45.627 [info] [default_brpc_retry_policy.cc:DoRetry:69] not retry for reached rcp timeout, ErrorCode '1008', error msg '[E1008]Reached timeout=2000ms @127.0.0.1:59197'
Result: 0.0
```

[Different Example: Normal code ]
```python
import numpy as np

def func(data_alice, data_bob):
    # Here we are calculating the Euclidean distance between the data from Alice and Bob
    euclidean_distance = np.sqrt(np.sum((data_alice - data_bob)**2))
    return euclidean_distance

def get_alice_data():
    # Alice's data is a numpy array containing the numbers from 1 to 50 with a step of 2
    return np.arange(1, 50, 2)

def get_bob_data():
    # Bob's data is a numpy array containing the cubes of numbers from 1 to 25
    return np.power(np.arange(1, 26), 3)

# Get data
data_alice = get_alice_data()
data_bob = get_bob_data()

# Perform computation
result = func(data_alice, data_bob)

# Print result
print(f"Result: {result}")
```

[Different Example: Execution result of Normal code ]
```shell
Result: 31472.191217009342
```

The reason why different is that both of SPU code and normal code work fine firstly, but result of SPU code is 0.0 (ignore message as below), while result of normal code is 31472.191217009342. The reason behind this is precision issues caused by computation of fixed-point number, adopted by SPU.
```shell
2024-01-10 18:23:39,845 INFO worker.py:1538 -- Started a local Ray instance.
INFO:jax._src.xla_bridge:Unable to initialize backend 'cuda': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
...
(omit)
...
(SPURuntime pid=62013) 2024-01-10 18:23:45.627 [info] [default_brpc_retry_policy.cc:DoRetry:69] not retry for reached rcp timeout, ErrorCode '1008', error msg '[E1008]Reached timeout=2000ms @127.0.0.1:59197'
```

##### Same Example ######
[Same Example: SPU Code]
```python
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
```
[Same Example: Execution result of SPU code ]
```shell
2024-01-10 18:14:05,959 INFO worker.py:1538 -- Started a local Ray instance.
(_run pid=33023) INFO:jax._src.xla_bridge:Unable to initialize backend 'cuda': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
...
(omit)
...
(SPURuntime pid=42950) 2024-01-10 18:14:11.613 [info] [default_brpc_retry_policy.cc:DoRetry:69] not retry for reached rcp timeout, ErrorCode '1008', error msg '[E1008]Reached timeout=2000ms @127.0.0.1:40671'
Result: 673.1243286132812
```

[Same Example: Normal code ]
```python
# Import numpy
import numpy as np

def func(data_alice, data_bob):
    # Here we are calculating the sum of square roots of Alice's and Bob's data
    sum_sqrt = np.sum(np.sqrt(data_alice)) + np.sum(np.sqrt(data_bob))
    return sum_sqrt

def get_alice_data():
    # Alice's data is an array containing square of numbers from 1 to 10
    return np.array([np.power(i, 2) for i in range(1, 11)])

def get_bob_data():
    # Bob's data is an array containing cube of numbers from 11 to 20
    return np.array([np.power(i, 3) for i in range(11, 21)])

# Get data
data_alice = get_alice_data()
data_bob = get_bob_data()

# Perform normal computation
result = func(data_alice, data_bob)

# Print result
print(f"Result: {result}")
```

[Same Example: Execution result of Normal code ]
```shell
Result: 673.1243392954524
```

The reason why same is that both of SPU code and normal code work fine firstly, and results of SPU code and normal code are 673.1243286132812 and 673.1243392954524, they are almost same and small differences in floating point arithmetic are acceptable. For more examples, we consider 5.499999523162842 and 5.5 to be almost identical, and the small difference in floating point operations is acceptable. When getting result of SPU code ignore message as below:
```shell
2024-01-10 18:14:05,959 INFO worker.py:1538 -- Started a local Ray instance.
(_run pid=33023) INFO:jax._src.xla_bridge:Unable to initialize backend 'cuda': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
...
(omit)
...
(SPURuntime pid=42950) 2024-01-10 18:14:11.613 [info] [default_brpc_retry_policy.cc:DoRetry:69] not retry for reached rcp timeout, ErrorCode '1008', error msg '[E1008]Reached timeout=2000ms @127.0.0.1:40671'
```
"""

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
logger = logging.getLogger(__name__)


class ChatProxy:
    def __init__(self) -> None:
        self.client = OpenAI()

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=8, max=128),
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
        retry=tenacity.retry_if_exception_type(openai.OpenAIError),
    )
    def send(self, **kwargs):
        model = kwargs.pop("model", DEFAULT_OPENAI_MODEL)
        stream = kwargs.pop("stream", True)
        completion = self.client.chat.completions.create(
            model=model, stream=stream, **kwargs
        )
        if stream:
            response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    response += chunk.choices[0].delta.content
        else:
            response = completion.choices[0].message.content
        return response

    def update(self, orig_request, orig_response, requirement, **kwargs):
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
            **kwargs,
        )


def py_is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False


def get_py_code(proxy, request, orig_response, **kwargs):
    response = orig_response
    while True:
        py_code = re.search(
            "```python(.*?)```",
            response,
            re.DOTALL,
        )

        if py_code:
            return py_code.group(1)

        response = proxy.update(
            request, response, RELECT_USE_PYTHON_CODEBLOCK_INSTRUCTION, **kwargs
        )


def get_valid_py_code(proxy, request, orig_response, **kwargs):
    response = orig_response
    while True:
        code = get_py_code(proxy, request, response, **kwargs)
        if py_is_syntax_valid(code):
            return code

        response = proxy.update(
            request, response, RELECT_GEN_VALID_PYTHON_CODE_INSTRUCTION, **kwargs
        )


def main():
    proxy = ChatProxy()
    round = 0
    previous_spu_code = None
    while True:
        round += 1  # Start from 1
        print(f"[Round {round}] Request to generate SPU code >>>")
        response = proxy.send(
            messages=[
                {
                    "role": "system",
                    "content": f"{GEN_SPU_CODE_INSTRUCTION}\n{USE_PYTHON_CODEBLOCK_INSTRUCTION}",
                },
                {
                    "role": "user",
                    "content": f"{GEN_SPU_CODE_PROMPT}\n[Previously generated SPU code]\n{previous_spu_code}"
                    if previous_spu_code
                    else GEN_SPU_CODE_PROMPT,
                },
            ],
        )
        print(f"[Round {round}] Get SPU code <<<")

        spu_code = get_valid_py_code(proxy, GEN_SPU_CODE_PROMPT, response)
        spu_code_hash = hashlib.sha256(spu_code.encode()).hexdigest()
        spu_code_file_name = str(spu_code_hash) + "_spu_code.py"
        os.makedirs("seeds", exist_ok=True)
        with open(os.path.join("seeds", spu_code_file_name), "w") as f:
            f.write(spu_code)

        print(f"[Round {round}] Request to generate normal code >>>")
        response = proxy.send(
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
            proxy, f"{GEN_NORMAL_CODE_PROMPT}\n```python\n{spu_code}\n```", response
        )
        # normal_code_hash = hashlib.sha256(normal_code.encode()).hexdigest()
        normal_code_file_name = str(spu_code_hash) + "_normal_code.py"
        with open(os.path.join("seeds", normal_code_file_name), "w") as f:
            f.write(normal_code)

        spu_result = subprocess.run(
            [
                os.path.expanduser(PYTHON_FILE),
                os.path.join("seeds", spu_code_file_name),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        normal_result = subprocess.run(
            [
                os.path.expanduser(PYTHON_FILE),
                os.path.join("seeds", normal_code_file_name),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        print(f"[Round {round}] Request to compare output >>>")
        all_output = f"[SPU Code]\n```python\n{spu_code}\n```\n[Execution Result of SPU Code]\n```shell\n{spu_result.stdout.decode()}\n```\n[Normal Code]\n```python\n{normal_code}\n```\n[Execution Result of Normal Code]\n```shell\n{normal_result.stdout.decode()}\n```\n"
        response = proxy.send(
            messages=[
                {"role": "system", "content": COMPARE_RESULT_INSTRUCTION},
                {
                    "role": "user",
                    "content": f"{COMPARE_RESULT_PROMPT}\n{all_output}",
                },
            ],
        )
        print(f"[Round {round}] Get response <<<")

        match = re.match(r"(.*?)\n", response)
        if match:
            first_line = match.group(1)
            result = first_line.lower()
        else:
            result = response.lower()

        if result not in ("same", "fail", "different"):
            result = "unknown"

        if result == "same" or result == "different":
            previous_spu_code = spu_code
        else:
            previous_spu_code = None

        result_file_name = str(spu_code_hash) + "_result.py"
        with open(os.path.join("seeds", result_file_name), "w") as f:
            f.write(f"{all_output}\n{response}")

        os.makedirs(result, exist_ok=True)
        os.rename(
            os.path.join("seeds", spu_code_file_name),
            os.path.join(result, spu_code_file_name),
        )
        os.rename(
            os.path.join("seeds", normal_code_file_name),
            os.path.join(result, normal_code_file_name),
        )
        os.rename(
            os.path.join("seeds", result_file_name),
            os.path.join(result, result_file_name),
        )
        print(os.path.join(result, result_file_name))


if __name__ == "__main__":
    main()
