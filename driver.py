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

BASE_PYTHON_FILE = "~/anaconda3//bin/python"
SF_PYTHON_FILE = "~/anaconda3/envs/sf/bin/python"
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"

USE_PYTHON_CODEBLOCK_INSTRUCTION = "Write your response using Python code blocks. For example:\n```python\nprint('Hello world!')\n```"
GEN_SPU_CODE_INSTRUCTION = f"""
You are an artificial intelligence that responds only with python code. SecretFlow provides a Python API for the Security Processing Unit (SPU) to enable privacy-preserving computation. You are required to complete an SPU code template. What needs to be completed is the '# <Some descriptions>' in the `func`, `get_alice_data`, `get_bob_data` and `import` sections. You will be given sample SPU code and you should generate more complicated SPU code.

JAX is a Python library for high-performance numerical computation, and jax.numpy is as compatible with numpy functionality as possible. For example `jnp.array([10])` is the jax version of `np.array([10])`, `jnp.add(data_alice, data_bob)` is the jax version of `np.add(data_alice, data_bob)` . You should use jax.numpy instead of numpy. Some libraries (such as sklearn) are not compatible with jax, do not use them.

For "func", "get_alice_data" and "get_bob_data", do not use any random functions, do not add Python decorators, and these functions must be independent of SecretFlow and SPU.
"""
GEN_SPU_CODE_PROMPT = """
[SPU Code Template]
```python
import secretflow as sf
# <import libraries that get_alice_data, get_bob_data, and func will use>

# Initialize secretflow framework, SPU node and participants' PYU nodes
sf.init(['alice', 'bob'], address='local')
alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu_node_device = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

def func(data_alice, data_bob):
    # <perform computation on data_alice and data_bob>
    # <return computation results>

def get_alice_data():
    # <Return Alice's data.>

def get_bob_data():
    # <Return Bob's data.>

# Get data from PYU nodes which have data
data_alice = alice(get_alice_data)()
data_bob = bob(get_bob_data)()

# Perform privacy preserving computing on SPU node
result = spu_node_device(func)(data_alice, data_bob)

# Reveal results
revealed_result = sf.reveal(result)

# Print revealed result
print("== RESULT BEGIN ==\n%s\n== RESULT END ==" % revealed_result)

# Clean envionment
sf.shutdown()
```
"""

SAMPLE_SPU_CODE = """
import secretflow as sf
import jax.numpy as jnp
# Some libraries (such as sklearn) are not compatible with jax, do not use them.

# Initialize secretflow framework, SPU node and participants' PYU nodes
sf.init(["alice", "bob"], address="local")
alice, bob = sf.PYU("alice"), sf.PYU("bob")
spu_node_device = sf.SPU(sf.utils.testing.cluster_def(["alice", "bob"]))


# Computation function
def func(data_alice, data_bob):
    # Compute the sum of data from alice and bob
    ret = jnp.add(data_alice, data_bob)
    return ret, jnp.size(ret)


# Function to get alice's data
def get_alice_data():
    # Here alice's data is [10]
    return jnp.array([10])


# Function to get bob's data
def get_bob_data():
    # Here bob's data is [5]
    return jnp.array([5])


# Get data from PYU nodes which have data
data_alice = alice(get_alice_data)()
data_bob = bob(get_bob_data)()

# Perform privacy preserving computing on SPU node
result, size = spu_node_device(
    func,
    num_returns_policy=sf.device.SPUCompilerNumReturnsPolicy.FROM_USER,
    user_specified_num_returns=2,
)(data_alice, data_bob)

# Reveal results
revealed_result = sf.reveal(result)
revealed_size = sf.reveal(size)

# Print revealed result
print("== RESULT BEGIN ==\n%s\n%s\n== RESULT END ==" % (revealed_result, revealed_size))

# Clean envionment
sf.shutdown()
"""
SAMPLE_NORMAL_CODE = """
import jax.numpy as jnp

def func(data_alice, data_bob):
    # Compute the sum of data from alice and bob
    ret = jnp.add(data_alice, data_bob)
    return ret, jnp.size(ret)

# Function to get alice's data
def get_alice_data():
    # Here alice's data is [10]
    return jnp.array([10])

# Function to get bob's data
def get_bob_data():
    # Here bob's data is [5]
    return jnp.array([5])

# Get data
data_alice = get_alice_data()
data_bob = get_bob_data()

# Perform normal computation
result, size = func(data_alice, data_bob)

# Print result
print("== RESULT BEGIN ==\n%s\n%s\n== RESULT END ==" % (result, size))
"""
RELECT_USE_PYTHON_CODEBLOCK_INSTRUCTION = """
You have been asked to generate Python code before. However, you gave an unexpected response, you did not use a Python code block to write the response, for example: \n```python\nprint('Hello world!')\n```\n. Now you will get the previous response, use Python code block to give the correct response this time.
"""
RELECT_GEN_VALID_PYTHON_CODE_INSTRUCTION = """
You have been asked to generate Python code before. However, you gave an unexpected response and the Python code you gave was syntactically invalid. Now you will get the previous response, please generate syntactically valid Python code this time.
"""
GEN_NORMAL_CODE_INSTRUCTION = """
You are an artificial intelligence that responds only with python code. SecretFlow provides a Python API for the Security Processing Unit (SPU) to enable privacy-preserving computation. You will be given the complete SPU code and need to generate normal python code (you will be given a template of the normal code) which is required to perform the same calculations as the SPU code, but in normal mode (non-SPU mode).

JAX is a Python library for high-performance numerical computation, and jax.numpy is as compatible with numpy functionality as possible. For example `jnp.array([10])` is the jax version of `np.array([10])`, `jnp.add(data_alice, data_bob)` is the jax version of `np.add(data_alice, data_bob)` . You should use jax.numpy instead of numpy. Some libraries (such as sklearn) are not compatible with jax, do not use them.

For "func", "get_alice_data" and "get_bob_data", do not use any random functions, do not add Python decorators, and these functions must be independent of SecretFlow and SPU.
"""
GEN_NORMAL_CODE_PROMPT = f"""
[Normal code Template]
```python
# <import libraries that get_alice_data, get_bob_data, and func will use>

def func(data_alice, data_bob):
    # <Almost identical to `func` in SPU code. Perform computation on data_alice and data_bob.>
    # <return computation results>

def get_alice_data():
    # <Almost identical to "get_alice_data" in the SPU code. Return Alice's data.>

def get_bob_data():
    # <Almost identical to "get_bob_data" in the SPU code. Return Bob's data.>

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
{SAMPLE_SPU_CODE}
```
[Corresponding Example Normal Code]
```python
{SAMPLE_NORMAL_CODE}
```

[Given SPU Code]
"""
COMPARE_RESULT_INSTRUCTION = """
You are an AI assistant. SecretFlow provides a Python API for the Security Processing Unit (SPU) to enable privacy-preserving computation. You will be asked to compare the results of the SPU code and the normal code and asked to answer whether the results are the same, then you need to answer "SAME", "DIFFERENT" or "FAIL" on the first line and explain the reason starting from the second line (some examples will be given).

- 'FAIL' means that one of the SPU code and the normal code is running incorrectly, so we cannot get all calculation results.
- 'DIFFERENT' means that we successfully obtained the results of both calculations, but they are different.
- 'SAME' means we successfully obtained the results of both calculations and they are the same.
"""
COMPARE_RESULT_PROMPT = """
Example 1
[Execution result of SPU code]
NO_RESULT
[Execution result of Normal code]
[105, 110, 120, 157]
[Comparison Result]
FAIL
The reason for the FAIL is that the SPU code fails to give any result (NO_RESULT means), while the normal code successfully gets the result.

Example 2
[Execution result of SPU code]
0.0
[Execution result of Normal code]
31472.191217009342
[Comparison Result]
DIFFERENT
The reason for the DIFFERENT is that firstly both the SPU code and the normal code work fine, but the result of the SPU code is 0.0, while the result of the normal code is 31472.191217009342. The reason behind this is the accuracy issue of SPU.

Example 3
[Execution result of SPU code]
673.1243286132812
[Execution result of Normal code]
673.1243392954524
[Comparison Result]
SAME
The reason why they are the SAME is that first of all, both the SPU code and the ordinary code work fine. The results of the SPU code and the ordinary code are 673.1243286132812 and 673.1243392954524, which are almost the same (the relative error is less than 1e-3, and the small difference in decimal operations is acceptable). More examples such as 5.499999523162842 and 5.5 (relative error less than 1e-3) are also considered the same.
"""
MAX_SEED_QUEUE_SIZE = 10

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
logger = logging.getLogger(__name__)


class ChatProxy:
    def __init__(self) -> None:
        self.client = openai.OpenAI()

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=2, min=10, max=600),
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
        retry=tenacity.retry_if_exception_type(openai.OpenAIError),
    )
    def send(self, **kwargs):
        model = kwargs.pop("model", DEFAULT_OPENAI_MODEL)
        stream = kwargs.pop("stream", True)
        temperature = kwargs.pop("temperature", 1)
        completion = self.client.chat.completions.create(
            model=model, stream=stream, temperature=temperature, **kwargs
        )
        if stream:
            response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    response += chunk.choices[0].delta.content
        else:
            response = completion.choices[0].message.content
        return response


def py_is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False


def get_py_code(proxy, orig_inst, orig_prompt, orig_response, **kwargs):
    response = orig_response
    while True:
        py_code = re.search(
            r"```.*?\n(.*?)```",
            response,
            re.DOTALL,
        )

        if py_code:
            return py_code.group(1)

        print(">>> Update Python Code")
        inst = f"{orig_inst}\n{RELECT_USE_PYTHON_CODEBLOCK_INSTRUCTION}"
        prompt = f"{orig_prompt}\n[Bad Previous Response]\n{response}"
        response = proxy.send(
            messages=[
                {
                    "role": "system",
                    "content": inst,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            **kwargs,
        )


def get_valid_py_code(proxy, orig_inst, orig_prompt, orig_response, **kwargs):
    response = orig_response
    while True:
        code = get_py_code(proxy, orig_inst, orig_prompt, response, **kwargs)
        if py_is_syntax_valid(code):
            return code

        print(">>> Update Syntax Valid Python Code")
        inst = f"{orig_inst}\n{RELECT_GEN_VALID_PYTHON_CODE_INSTRUCTION}"
        prompt = f"{orig_prompt}\n[Bad Previous Response]\n{response}"
        response = proxy.send(
            messages=[
                {
                    "role": "system",
                    "content": inst,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            **kwargs,
        )


def get_seeds(path: str):
    seeds = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            with open(os.path.join(root, file)) as f:
                f_lines = f.readlines()
                for i, line in enumerate(f_lines):
                    if "[SPU Code Path]" in line:
                        _spu_code_path = f_lines[i + 1].replace("\n", "")
                        with open(_spu_code_path) as _spu_code_f:
                            _spu_code = _spu_code_f.read()
                    if "[Communication Bytes]" in line:
                        _comm_bytes = int(f_lines[i + 1].replace("\n", ""))
            seeds[_spu_code] = _comm_bytes
    return seeds


def main():
    proxy = ChatProxy()
    round = 0
    previous_spu_code_queue = {SAMPLE_SPU_CODE: 0}
    previous_spu_code_queue = {**previous_spu_code_queue, **get_seeds("./same")}
    previous_spu_code_queue = {**previous_spu_code_queue, **get_seeds("./different")}

    error_cases = []
    while True:
        previous_spu_code = max(
            previous_spu_code_queue, key=previous_spu_code_queue.get
        )

        round += 1  # Start from 1

        print(f"== Round {round} ==")
        print(f">>> Request to generate SPU code")
        inst = f"{GEN_SPU_CODE_INSTRUCTION}\n{USE_PYTHON_CODEBLOCK_INSTRUCTION}\n"
        prompt = f"{GEN_SPU_CODE_PROMPT}\n[Sample SPU code]\n```python\n{previous_spu_code}\n```\n"
        if len(error_cases) > 0:
            prompt += "Here are some error cases and their outputs, avoid to make same mistakes\n"
            for case, output in error_cases:
                prompt += f"[Error Case]\n{case}\n[Output]\n{output}\n"
        response = proxy.send(
            messages=[
                {
                    "role": "system",
                    "content": inst,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        print(f"<<< Get SPU code")

        spu_code = get_valid_py_code(proxy, inst, prompt, response)
        spu_code_with_comm = re.sub(
            r"(.*)(sf.shutdown())",
            r"\1\ncomm_byte_nums = spu_node_device.get_comm_byte_nums()\nprint('[Number of SPU Communication Bytes] %d' % (sum(comm_byte_nums) / len(comm_byte_nums)))\n\2",
            spu_code,
            flags=re.DOTALL,
        )

        spu_code_hash = hashlib.sha256(spu_code.encode()).hexdigest()
        spu_code_path = os.path.abspath(
            os.path.join("seeds", str(spu_code_hash) + "_spu_code.py")
        )
        os.makedirs("seeds", exist_ok=True)
        with open(spu_code_path, "w") as f:
            f.write(spu_code_with_comm)

        print(f">>> Request to generate normal code")
        inst = f"{GEN_NORMAL_CODE_INSTRUCTION}\n{USE_PYTHON_CODEBLOCK_INSTRUCTION}\n"
        prompt = f"{GEN_NORMAL_CODE_PROMPT}\n```python\n{spu_code}\n```"
        response = proxy.send(
            messages=[
                {
                    "role": "system",
                    "content": inst,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        print(f"<<< Get normal code")

        normal_code = get_valid_py_code(proxy, inst, prompt, response)
        # normal_code_hash = hashlib.sha256(normal_code.encode()).hexdigest()
        normal_code_path = os.path.abspath(
            os.path.join("seeds", str(spu_code_hash) + "_normal_code.py")
        )
        with open(normal_code_path, "w") as f:
            f.write(normal_code)

        rand_in_spu_code = re.search(r"^[^#]*?random", spu_code.lower(), re.MULTILINE)
        rand_in_normal_code = re.search(
            r"^[^#]*?random", normal_code.lower(), re.MULTILINE
        )
        if rand_in_spu_code or rand_in_normal_code:
            spu_output = "error: should not use random"
            normal_output = "error: should not use random"
        else:
            spu_output = subprocess.run(
                [
                    os.path.expanduser(SF_PYTHON_FILE),
                    spu_code_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env={"JAX_PLATFORMS": "cpu"},
            )
            normal_output = subprocess.run(
                [
                    os.path.expanduser(BASE_PYTHON_FILE),
                    normal_code_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            spu_output = spu_output.stdout.decode()
            normal_output = normal_output.stdout.decode()

        result_match = re.search(
            r"== RESULT BEGIN ==\n(.*)== RESULT END ==", spu_output, re.DOTALL
        )
        spu_result = result_match.group(1) if result_match else "NO_RESULT"
        if spu_result == "NO_RESULT":
            while len(error_cases) >= 1:
                error_cases.pop(0)
            error_cases.append(
                (
                    spu_code,
                    "\n".join(
                        [
                            line
                            for line in spu_output.split("\n")
                            if "error" in line.lower() or "fail" in line.lower()
                        ]
                    ),
                )
            )

        spu_comm_match = re.search(
            r"\[Number of SPU Communication Bytes\](.*)", spu_output
        )
        spu_comm = int(spu_comm_match.group(1).strip()) if spu_comm_match else -1
        while len(previous_spu_code_queue) >= MAX_SEED_QUEUE_SIZE:
            min_key = min(previous_spu_code_queue, key=previous_spu_code_queue.get)
            del previous_spu_code_queue[min_key]
        previous_spu_code_queue[spu_code] = spu_comm

        result_match = re.search(
            r"== RESULT BEGIN ==\n(.*)== RESULT END ==", normal_output, re.DOTALL
        )
        normal_result = result_match.group(1) if result_match else "NO_RESULT"

        all_output = f"[Execution Result of SPU Code]\n{spu_result}\n[Execution Result of Normal Code]\n{normal_result}\n"
        if spu_result == "NO_RESULT" and normal_result == "NO_RESULT":
            response = "FAIL\nSPU code and normal code have problems\n"
        elif spu_result == "NO_RESULT":
            response = "FAIL\nSPU code has problems\n"
        elif normal_result == "NO_RESULT":
            response = "FAIL\nNormal code has problems\n"
        else:
            print(f">>> Request to compare output")
            response = proxy.send(
                messages=[
                    {"role": "system", "content": COMPARE_RESULT_INSTRUCTION},
                    {
                        "role": "user",
                        "content": f"{COMPARE_RESULT_PROMPT}\n{all_output}\n",
                    },
                ],
            )
            print(f"<<< Get response")
            response = re.sub(
                r"^.*?Comparison Result.*?\n", "", response, flags=re.MULTILINE
            )
            response = re.sub(r"^\s*\n", "", response, flags=re.MULTILINE)

        result = response.split("\n", 1)[0].lower()

        if result not in ("same", "fail", "different"):
            result = "unknown"

        result_file_path = os.path.abspath(
            os.path.join(result, str(spu_code_hash) + "_result.py")
        )
        os.makedirs(result, exist_ok=True)
        with open(result_file_path, "w") as f:
            f.write(
                f"{all_output}\n\n"
                + f"[Comparison Result]\n{response}\n\n"
                + f"[SPU Code Path]\n{spu_code_path}\n\n"
                + f"[Normal Code Path]\n{normal_code_path}\n\n"
                + f"[Communication Bytes]\n{spu_comm}\n\n"
                + (
                    f"[SPU Error Msg]\n{spu_output}\n"
                    if spu_result == "NO_RESULT"
                    else ""
                )
                + (
                    f"[Normal Error Msg]\n{normal_output}\n"
                    if normal_result == "NO_RESULT"
                    else ""
                )
            )
        print(result_file_path)
        print(spu_code_path)
        print(normal_code_path)


if __name__ == "__main__":
    main()
