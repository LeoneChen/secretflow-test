import os
import re
import time
import hashlib
import subprocess
import tenacity
import ast
import openai
import logging
import sys

PYTHON_FILE = "~/anaconda3/envs/secretflow/bin/python"
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo-0125"

USE_PYTHON_CODEBLOCK_INSTRUCTION = "Write your response using Python code blocks. For example:\n```python\nprint('Hello world!')\n```"
GEN_SPU_CODE_INSTRUCTION = "You are an AI that responds only with python code. SecretFlow provides a Python API for the Security Processing Unit (SPU) to enable privacy preserving computing. You will get an SPU code that needs to be filled in. The place that needs to be filled in is in a comment and enclosed in angle brackets. The text between the angle brackets is a description of the code you need to generate. For example:\n```python\n# <Some description>\n```\n. You need to fill in the SPU code and provide the complete code. You will also get a sample SPU code (definitely correct) and previously generated SPU code (possibly incorrect) if it exists, in the hope that you can generate different SPU code."
JNP_AS_NP_PROMPT = """
JAX is a Python library for high-performance numerical computing, jax.numpy is as compatible with numpy functionality as possible. e.g. `jnp.array([10])` is jax's version of `np.array([10])`, `jnp.add(data_alice, data_bob)` is jax's version of `np.add(data_alice, data_bob)`
"""
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
    # <Use data_alice and data_bob to complete privacy protection computing and return the calculation results. The code must be SecretFlow and SPU agnostic. Avoid using any random functions. Use jax.numpy instead of numpy. This function should not have Python decorators added.>

def get_alice_data():
    # <Return Alice's data. If string data is present, it needs to be converted to list data or vector data. The code must be SecretFlow and SPU agnostic. Avoid using any random functions. Use jax.numpy instead of numpy. This function should not have Python decorators added.>

def get_bob_data():
    # <Return Bob's data. If string data is present, it need to be converted to list data or vector data. The code must be SecretFlow and SPU agnostic. Avoid useing any random functions. Use jax.numpy instead of numpy. This function should not have Python decorators added.>

# Pass data to PYU
data_alice = alice(get_alice_data)()
data_bob = bob(get_bob_data)()

# Perform privacy protection computing on SPU
result = spu_node_device(func)(data_alice, data_bob)

# Reveal results
revealed_result = sf.reveal(result)

# Print revealed result
print("== RESULT BEGIN ==\n%s\n== RESULT END ==" % revealed_result)

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
    return jnp.add(data_alice, data_bob)

# Function to get alice's data
def get_alice_data():
    # Here alice's data is [10]
    return jnp.array([10])

# Function to get bob's data
def get_bob_data():
    # Here bob's data is [5]
    return jnp.array([5])

# Pass data to PYU
data_alice = alice(get_alice_data)()
data_bob = bob(get_bob_data)()

# Perform privacy computation on SPU
result = spu_node_device(func)(data_alice, data_bob)

# Reveal results
revealed_result = sf.reveal(result)

# Print revealed result
print("== RESULT BEGIN ==\n%s\n== RESULT END ==" % revealed_result)

# Clean envionment
sf.shutdown()
```
"""
RELECT_USE_PYTHON_CODEBLOCK_INSTRUCTION = "You are an AI assistant. SecretFlow provides a Python API for secure processing units (SPUs) to enable privacy preserving computing. Last round, I asked you to fill in an SPU code. Where you need to fill it in is in the comments and enclosed in angle brackets. The text between the angle brackets is a description of the code you need to generate. For example:\n```python\n# <Some description>\n```\n. However, in the previous round, you gave an unexpected response, and you did not use the Python code block to write the previous response, e.g. \n```python\nprint('Hello world!')\n```\n. Now you will get the previous request and response, use Python code block to give the correct response this time."

RELECT_GEN_VALID_PYTHON_CODE_INSTRUCTION = "You are an AI assistant. SecretFlow provides a Python API for secure processing unit (SPU) to enable privacy preserving computing. Last round, I asked you to fill in an SPU code. Where you need to fill it in is in the comments and enclosed in angle brackets. The text between the angle brackets is a description of the code you need to generate. For example:\n```python\n# <Some description>\n```\n. However, in the previous round, you gave an unexpected response, the python code you gave was syntactically invalid. Now you will get the previous request and response, please give the Python code with valid syntax this time."

GEN_NORMAL_CODE_INSTRUCTION = "You are an AI that responds only with python code. SecretFlow provides a Python API for the Security Processing Unit (SPU) to enable privacy preserving computing. You will get the complete SPU code and you need to generate the normal python code (you will be given a template of the normal code) which performs the same calculations as the SPU code but in normal mode (non-SPU mode)."
GEN_NORMAL_CODE_PROMPT = """
[Template of normal code]
```python
# <import libraries that get_alice_data, get_bob_data, and func will use>

def func(data_alice, data_bob):
    # <Almost identical to `func` in SPU code. Use data_alice and data_bob to complete normal calculations and return the calculation results. Avoid using any random functions. Use numpy instead of jax.numpy.>

def get_alice_data():
    # <Almost identical to "get_alice_data" in the SPU code. Return Alice's data. If string data is present, it needs to be converted to list data or vector data. Avoid using any random functions. Use numpy instead of jax.numpy.>

def get_bob_data():
    # <Almost identical to "get_bob_data" in the SPU code. Return Bob's data. If string data is present, it needs to be converted to list data or vector data. Avoid using any random functions. Use numpy instead of jax.numpy.>

# Get data
data_alice = get_alice_data()
data_bob = get_bob_data()

# Perform normal computation
result = func(data_alice, data_bob)

# Print result
print("== RESULT BEGIN ==\n%s\n== RESULT END ==" % result)
```

[Example SPU Code]
```python
import secretflow as sf
import jax.numpy as jnp

# Initialize SPU and participants
sf.init(['alice', 'bob'], address='local')
alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu_node_device = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

# Privacy computation function
def func(data_alice, data_bob):
    # Compute the sum of data from alice and bob
    return jnp.add(data_alice, data_bob)

# Function to get alice's data
def get_alice_data():
    # Here alice's data is [10]
    return jnp.array([10])

# Function to get bob's data
def get_bob_data():
    # Here bob's data is [5]
    return jnp.array([5])

# Pass data to PYU
data_alice = alice(get_alice_data)()
data_bob = bob(get_bob_data)()

# Perform privacy computation on SPU
result = spu_node_device(func)(data_alice, data_bob)

# Reveal results
revealed_result = sf.reveal(result)

# Print revealed result
print("== RESULT BEGIN ==\n%s\n== RESULT END ==" % revealed_result)

# Clean envionment
sf.shutdown()
```

[Corresponding Example Normal Code]
```python
import numpy as np

# Normal computation function
def func(data_alice, data_bob):
    # Compute the sum of data from alice and bob
    return np.add(data_alice, data_bob)

# Function to get alice's data
def get_alice_data():
    # Here alice's data is [10]
    return np.array([10])

# Function to get bob's data
def get_bob_data():
    # Here bob's data is [5]
    return np.array([5])

# Get data
data_alice = get_alice_data()
data_bob = get_bob_data()

# Perform normal computation on SPU
result = func(data_alice, data_bob)

# Print result
print("== RESULT BEGIN ==\n%s\n== RESULT END ==" % result)
```

[Given SPU Code]
"""

COMPARE_RESULT_INSTRUCTION = """
You are an AI assistant. SecretFlow provides a Python API for the Security Processing Unit (SPU) to enable private computation. Some examples will be given first, followed by the execution results of the SPU code, and of a normal code that performs the same calculations but in normal mode. You are asked to compare the results of the SPU code and the normal code, and then you need to answer "SAME", "DIFFERENT" or "FAIL" on the first line and explain the reason starting on the second line.
- "FAIL" means that one of the SPU code and the normal code ran wrong, so we can't get all the calculation results.
- 'DIFFERENT' means that we successfully obtained the results of both calculations, but they are different.
- 'SAME' means we successfully obtained the results of both calculations and they are the same.
"""
COMPARE_RESULT_PROMPT = """
##### FAIL Example ######
[FAIL Example: Execution result of SPU code ]
NO_RESULT
[FAIL Example: Execution result of Normal code ]
[105, 110, 120, 157]

The reason for the FAIL is that the SPU code fails to give any result (NO_RESULT means), while the normal code successfully gets the result.

##### DIFFERENT Example ######
[DIFFERENT Example: Execution result of SPU code ]
0.0
[DIFFERENT Example: Execution result of Normal code ]
31472.191217009342

The reason for the DIFFERENT is that firstly both the SPU code and the normal code work fine, but the result of the SPU code is 0.0, while the result of the normal code is 31472.191217009342. The reason behind this is the accuracy issue of SPU.

##### SAME Example ######
[SAME Example: Execution result of SPU code ]
673.1243286132812
[SAME Example: Execution result of Normal code ]
673.1243392954524

The reason why they are the SAME is that first of all, both the SPU code and the ordinary code work fine. The results of the SPU code and the ordinary code are 673.1243286132812 and 673.1243392954524, which are almost the same (the relative error is less than 1e-3, and the small difference in decimal operations is acceptable). More examples such as 5.499999523162842 and 5.5 (relative error less than 1e-3) are also considered the same.
"""

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
logger = logging.getLogger(__name__)


class ChatProxy:
    def __init__(self) -> None:
        self.client = openai.OpenAI()

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
                    "content": f"[Previous request]\n{orig_request}\n\n[Previous reponse]\n{orig_response}",
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
        request = (
            f"{GEN_SPU_CODE_PROMPT}\n[Previously generated SPU code]\n{previous_spu_code}"
            if previous_spu_code
            else GEN_SPU_CODE_PROMPT
        )
        response = proxy.send(
            messages=[
                {
                    "role": "system",
                    "content": f"{GEN_SPU_CODE_INSTRUCTION}\n{USE_PYTHON_CODEBLOCK_INSTRUCTION}\n{JNP_AS_NP_PROMPT}",
                },
                {
                    "role": "user",
                    "content": request,
                },
            ],
        )
        print(f"[Round {round}] Get SPU code <<<")

        spu_code = get_valid_py_code(proxy, request, response)
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
                    "content": f"{GEN_NORMAL_CODE_INSTRUCTION}\n{USE_PYTHON_CODEBLOCK_INSTRUCTION}\n{JNP_AS_NP_PROMPT}",
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
        spu_result = spu_result.stdout.decode()
        result_match = re.search(
            "== RESULT BEGIN ==\n(.*)== RESULT END ==", spu_result, re.DOTALL
        )
        spu_result = result_match.group(1) if result_match else "NO_RESULT"

        normal_result = subprocess.run(
            [
                os.path.expanduser(PYTHON_FILE),
                os.path.join("seeds", normal_code_file_name),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        normal_result = normal_result.stdout.decode()
        result_match = re.search(
            "== RESULT BEGIN ==\n(.*)== RESULT END ==", normal_result, re.DOTALL
        )
        normal_result = result_match.group(1) if result_match else "NO_RESULT"

        print(f"[Round {round}] Request to compare output >>>")
        all_output = f"[Execution Result of SPU Code]\n{spu_result}\n[Execution Result of Normal Code]\n{normal_result}\n"
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
