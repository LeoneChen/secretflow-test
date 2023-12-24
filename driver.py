import os
import re
import time
import hashlib
import subprocess
import google.generativeai as genai


def main():
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-pro")
    history = []
    round = 0
    while True:
        round += 1
        chat = model.start_chat(history=history)

        prompt_gen_code = """
        Imagine that you are an experienced programmer writing test code for the Secure Processing Unit of Secretflow.
        
        You mainly need to provide specific functions of privacy computation in 'func', and provide data of alice and bob in 'get_alice_data' and 'get_bob_data' respectively.
        
        If there are chat history, please generate a more complicated code then previously generated code.
        
        If previous generated code is fail to run, you need to generate runable code this round. If output of previous generated code complains that some operation in code is unsupported, sentence is like 'This is because the SPU compiler does not support the `xxx` operation.', you need to avoid use them in this round. If output complains ray error, may be we should retry.

        Avoid to use any random functions when generating data or perform computation, since it cause us diffcult to reproduce the problem.

        The generated code need to import necessary libraries in case the code fails to run.

        'secretflow' library only provide functionality like SPU, not others. 
        
        Template as below:

        import secretflow as sf
        # import libraries if needed

        # Initialize SPU and participants
        sf.init(['alice', 'bob'], address='local')
        alice, bob = sf.PYU('alice'), sf.PYU('bob')
        spu_node_device = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

        # Specific functions that need to be generated
        def func(data_alice, data_bob):
            # Use data_alice and data_bob to complete privacy computation and return the calculation results

        def get_alice_data():
            # Return alice's data

        def get_bob_data():
            # Return bob's data

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

        What's more, you should generate another python code, which perform same computation but in normal mode (non-SPU mode), this means code in 'func', 'get_alice_data', and 'get_bob_data' should almost at the same.

        In SPU code, prefer using jax.numpy instead of numpy, while in normal code prefer using numpy instead of jax.numpy.

        When generating SPU code, in `func`, `spu_node_device.xxx` is not allowed, since `spu_node_device` is an SPU node device, it shouldn't appear in private computation.

        Response's format is like that:

        SPU Code:
        ```python
        [Generated Python Code for SPU]
        ```

        Normal Code:
        ```python
        [Generated Python Code that perform same computation but in plaintext mode]
        ```
        """

        print(f"[Round {round}] ========== Request to generate code ==========>>>>>>>>>>")

        for _ in range(0, 5):
            try:
                response = chat.send_message(prompt_gen_code)
            except:
                time.sleep(1)
                continue
        print(f"[Round {round}] <<<<<<<<<<========== Get response ==========")

        match = re.search(
            "SPU Code:.*?```python(.*)```.*Normal Code:.*?```python(.*)```",
            response.text,
            re.DOTALL,
        )

        spu_code = match.group(1)
        spu_code_hash = hashlib.sha256(spu_code.encode()).hexdigest()
        spu_code_file_name = str(spu_code_hash) + "_spu_code.py"
        os.makedirs("seeds", exist_ok=True)
        with open(os.path.join("seeds", spu_code_file_name), "w") as f:
            f.write(spu_code)

        normal_code = match.group(2)
        # normal_code_hash = hashlib.sha256(normal_code.encode()).hexdigest()
        normal_code_file_name = str(spu_code_hash) + "_normal_code.py"
        with open(os.path.join("seeds", normal_code_file_name), "w") as f:
            f.write(normal_code)

        spu_result = subprocess.run(
            [
                "/home/leone/anaconda3/envs/secretflow/bin/python3.8",
                os.path.join("seeds", spu_code_file_name),
            ],
            capture_output=True,
        )

        normal_result = subprocess.run(
            [
                "/home/leone/anaconda3/envs/secretflow/bin/python3.8",
                os.path.join("seeds", normal_code_file_name),
            ],
            capture_output=True,
        )

        prompt_cmp_output = (
            "SPU code execution stdout:\n"
            + spu_result.stdout.decode()
            + "SPU code execution stderr:\n"
            + spu_result.stderr.decode()
            + "Normal code execution stdout:\n"
            + normal_result.stdout.decode()
            + "Normal code execution stderr:\n"
            + normal_result.stderr.decode()
            + """
            Please compare their results followed by 'Result:'. What you need to compare is the calculation results rather than the calculation process.
            
            Finally, answer 'Same', 'Different' or 'Fail' in the first line, and then explain the reason from the second line.

            'Fail' means SPU code run to error or normal code run to error, therefore we cannot obtain the calculation results of both. It may due to generated code is incomplete or other runtime error. Usually when an error occurs, we will find that the output contains error information and even call stack. In SPU code, if output complains that some operation in code is unsupported, sentence is like 'This is because the SPU compiler does not support the `xxx` operation.', you need to avoid use them in next round. If output complains ray error, may be we should retry in next round.
            
            'Different' means we sucessfully obtain calculation results of both, but they are different.

            'Same' means we sucessfully obtain calculation results of both, and they are same.

            SPU code can give wrong result compared with normal code, you need to be careful.

            When comparing outputs, ignore complaints like (since device may not have GPU/TPU, so falling back to CPU is OK):
            ```shell
                INFO:jax._src.xla_bridge:Unable to initialize backend 'cuda': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
                INFO:jax._src.xla_bridge:Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
                INFO:jax._src.xla_bridge:Unable to initialize backend 'tpu': INVALID_ARGUMENT: TpuPlatform is not available.
                INFO:jax._src.xla_bridge:Unable to initialize backend 'plugin': xla_extension has no attributes named get_plugin_device_client. Compile TensorFlow with //tensorflow/compiler/xla/python:enable_plugin_device set to true (defaults to false) to enable this.
                WARNING:jax._src.xla_bridge:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
            ```

            When comparing outputs, ignore complaints like below from SPURuntime (it's irrelevant to the result of computation):
            ```shell
                (SPURuntime pid=1816702) 2023-12-24 21:48:50.413 [info] [default_brpc_retry_policy.cc:DoRetry:52] socket error, sleep=1000000us and retry
                (SPURuntime pid=1816702) 2023-12-24 21:48:51.413 [info] [default_brpc_retry_policy.cc:LogHttpDetail:29] cntl ErrorCode '112', http status code '200', response header '', error msg '[E111]Fail to connect Socket{id=0 addr=127.0.0.1:38701} (0x0x48e4a80): Connection refused [R1][E112]Not connected to 127.0.0.1:38701 yet, server_id=0'
                (SPURuntime pid=1816702) 2023-12-24 21:48:51.414 [info] [default_brpc_retry_policy.cc:DoRetry:75] aggressive retry, sleep=1000000us and retry
                (SPURuntime pid=1816702) 2023-12-24 21:48:52.414 [info] [default_brpc_retry_policy.cc:LogHttpDetail:29] cntl ErrorCode '112', http status code '200', response header '', error msg '[E111]Fail to connect Socket{id=0 addr=127.0.0.1:38701} (0x0x48e4a80): Connection refused [R1][E112]Not connected to 127.0.0.1:38701 yet, server_id=0 [R2][E112]Not connected to 127.0.0.1:38701 yet, server_id=0'
                (SPURuntime pid=1816702) 2023-12-24 21:48:52.414 [info] [default_brpc_retry_policy.cc:DoRetry:75] aggressive retry, sleep=1000000us and retry
                (SPURuntime pid=1816700) 2023-12-24 21:48:52.532 [info] [default_brpc_retry_policy.cc:DoRetry:69] not retry for reached rcp timeout, ErrorCode '1008', error msg '[E1008]Reached timeout=2000ms @127.0.0.1:32921'
            ```

            Notice again that you should answer 'Same', 'Different' or 'Fail' in the first line, and what needs to be compared is the calculation results rather than the calculation process.
            """
        )
        print(f"[Round {round}] ========== Request to compare output ==========>>>>>>>>>>")
        for _ in range(0, 5):
            try:
                response = chat.send_message(prompt_cmp_output)
            except:
                time.sleep(1)
                continue
        print(f"[Round {round}] <<<<<<<<<<========== Get response ==========")

        history += chat.history[-4:]
        if len(history) > 12:
            history = history[4:]
        print("len(history)=%d" % len(history))

        match = re.match(r"(.*)?\n", response.text)
        if match:
            first_line = match.group(1)
            result = first_line.lower()
        else:
            result = response.text

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
                "Compare output:\n"
                + prompt_cmp_output
                + "\nResponse:\n"
                + response.text
            )


if __name__ == "__main__":
    main()
