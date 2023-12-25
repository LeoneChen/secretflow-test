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

        prompt_gen_spu_code = """
        Imagine that you are an experienced programmer writing test code for the Secure Processing Unit (SPU) of Secretflow.
        
        You mainly need to provide specific functions of privacy computation in 'func', and provide data of alice and bob in 'get_alice_data' and 'get_bob_data' respectively, string data should transfer to list data or vector data. In `func`, 'get_alice_data', and 'get_bob_data', the code must have nothing to do with secretflow and spu. Avoid to use any random functions when generating data or perform computation, since it cause us diffcult to reproduce the problem. In SPU code, prefer using jax.numpy instead of numpy, while in normal code prefer using numpy instead of jax.numpy. Note the subtle usage differences between numpy and jax.numpy.
        
        If there are chat history, please generate a more complicated code then previously generated code. If previous generated code is failed to run, you need to avoid same mistake this round. If output of previous generated code complains that some operation in code is unsupported, sentence is like 'This is because the SPU compiler does not support the `xxx` operation.', you need to avoid use them in this round. If output complains ray error, may be we should retry.
        
        Template of SPU code as below (Uncommented code is necessary):
        ```python
        import secretflow as sf # 'secretflow' library only provide functionality like SPU, not others.
        # import libraries if needed

        # Initialize SPU and participants
        sf.init(['alice', 'bob'], address='local')
        alice, bob = sf.PYU('alice'), sf.PYU('bob')
        spu_node_device = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

        # [This is need to be generated]
        def func(data_alice, data_bob):
            # Use data_alice and data_bob to complete privacy computation and return the calculation results

        # [This is need to be generated]
        def get_alice_data():
            # Return alice's data

        # [This is need to be generated]
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
        ```

        Response's format is like that:

        SPU Code:
        ```python
        [Generated Python Code for SPU]
        ```
        """

        print(
            f"[Round {round}] ========== Request to generate SPU code ==========>>>>>>>>>>"
        )
        for _ in range(0, 5):
            try:
                response = chat.send_message(prompt_gen_spu_code)
            except:
                time.sleep(1)
                continue
            break
        print(f"[Round {round}] <<<<<<<<<<========== Get SPU code ==========")

        match_spu_code = re.search(
            "```python(.*?)```",
            response.text,
            re.DOTALL,
        )

        if match_spu_code:
            spu_code = match_spu_code.group(1)
            spu_code_hash = hashlib.sha256(spu_code.encode()).hexdigest()
            spu_code_file_name = str(spu_code_hash) + "_spu_code.py"
            os.makedirs("seeds", exist_ok=True)
            with open(os.path.join("seeds", spu_code_file_name), "w") as f:
                f.write(spu_code)

            prompt_gen_normal_code = """
            What's more, you should generate another normal python code, which perform same computation as SPU code but in normal mode (non-SPU mode), this means code in 'func', 'get_alice_data', and 'get_bob_data' should almost at the same. But in SPU code, it prefers using jax.numpy instead of numpy, while in normal code, it prefers using numpy instead of jax.numpy. Note the subtle usage differences between numpy and jax.numpy.

            Template of normal code as below (Uncommented code is necessary):
            ```python
            # import libraries if needed

            # [This is need to be generated]
            def func(data_alice, data_bob):
                # Use data_alice and data_bob to complete normal computation and return the calculation results (almost same as SPU code)

            # [This is need to be generated]
            def get_alice_data():
                # Return alice's data (almost same as SPU code)

            # [This is need to be generated]
            def get_bob_data():
                # Return bob's data (almost same as SPU code)

            # Pass data to PYU
            data_alice = get_alice_data()
            data_bob = get_bob_data()

            # Perform normal computation
            result = func(data_alice, data_bob)

            # Print revealed result
            print(f"Result: {revealed_result}")
            ```

            Response's format is like that:

            Normal Code:
            ```python
            [Generated Python Code that perform same computation but in plaintext mode]
            ```
            """
            print(
                f"[Round {round}] ========== Request to generate normal code ==========>>>>>>>>>>"
            )
            for _ in range(0, 5):
                try:
                    response = chat.send_message(prompt_gen_normal_code)
                except:
                    time.sleep(1)
                    continue
                break
            print(f"[Round {round}] <<<<<<<<<<========== Get normal code ==========")

            match_normal_code = re.search(
                "```python(.*?)```",
                response.text,
                re.DOTALL,
            )

            if match_normal_code:
                round_chat = model.start_chat(history=chat.history[-4:])
                normal_code = match_normal_code.group(1)
                # normal_code_hash = hashlib.sha256(normal_code.encode()).hexdigest()
                normal_code_file_name = str(spu_code_hash) + "_normal_code.py"
                with open(os.path.join("seeds", normal_code_file_name), "w") as f:
                    f.write(normal_code)

                spu_result = subprocess.run(
                    [
                        "/home/leone/anaconda3/envs/secretflow/bin/python3.8",
                        os.path.join("seeds", spu_code_file_name),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )

                normal_result = subprocess.run(
                    [
                        "/home/leone/anaconda3/envs/secretflow/bin/python3.8",
                        os.path.join("seeds", normal_code_file_name),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )

                whole_output = (
                    "SPU code output:\n"
                    + spu_result.stdout.decode()
                    + "Normal code output:\n"
                    + normal_result.stdout.decode()
                )
                prompt_cmp_output = (
                    whole_output
                    + """
                    Please compare their results followed by 'Result:'. What you need to compare is the calculation results rather than the calculation process, SPU code can give wrong result compared with normal code, you need to be careful. Finally, answer 'Same', 'Different' or 'Fail' in the first line, and then explain the reason from the second line.
                    - 'Fail' means SPU code run to error or normal code run to error, therefore we cannot obtain the calculation results of both. It may due to generated code is incomplete or other runtime error. Usually when an error occurs, we will find that the output contains error information and even call stack. In SPU code, if output complains that some operation in code is unsupported, sentence is like 'This is because the SPU compiler does not support the `xxx` operation.', you need to avoid use them in next round. If output complains ray error, may be we should retry in next round.
                    - 'Different' means we sucessfully obtain calculation results of both, but they are different.
                    - 'Same' means we sucessfully obtain calculation results of both, and they are same.

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
                    """
                )

                print(
                    f"[Round {round}] ========== Request to compare output ==========>>>>>>>>>>"
                )
                for _ in range(0, 5):
                    try:
                        response = round_chat.send_message(prompt_cmp_output)
                    except:
                        time.sleep(1)
                        continue
                    break
                print(f"[Round {round}] <<<<<<<<<<========== Get response ==========")

                match = re.match(r"(.*?)\n", response.text)
                if match:
                    first_line = match.group(1)
                    result = first_line.lower()
                else:
                    result = response.text.lower()

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
                        + whole_output
                        + "\nResponse:\n"
                        + response.text
                    )

                history += round_chat.history
                if len(history) > 12:
                    history = history[6:]


if __name__ == "__main__":
    main()
