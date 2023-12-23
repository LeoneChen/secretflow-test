import os
import re
import hashlib
import subprocess
import google.generativeai as genai


def main():
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-pro")
    history = []
    while True:
        chat = model.start_chat(history=history)

        prompt = """
        Imagine that you are an experienced programmer writing test code for the Secure Processing Unit of Secretflow.
        
        You mainly need to provide specific functions of privacy computation in 'func', and provide data of alice and bob in 'get_alice_data' and 'get_bob_data' respectively.
        
        If there are chat history, please generate a more complicated code then previously generated code.
        
        If previous generated code is fail to run, you need to generate runable code this round. If output of previous generated code complains that some operation in code is unsupported by SPU runtime or JAX and so on, you need to avoid use them in this round. If output complains ray error, may be we should retry.

        Avoid to use any random functions when generating data or perform computation, since it cause us diffcult to reproduce the problem.

        The generated code need to import necessary libraries in case the code fails to run.
        
        Template as below:

        import secretflow as sf
        # import libraries if needed

        # Initialize SPU and participants
        sf.init(['alice', 'bob'], address='local')
        alice, bob = sf.PYU('alice'), sf.PYU('bob')
        spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

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
        result = spu(func)(data_alice, data_bob)

        # Reveal results
        revealed_result = sf.reveal(result)

        # Print revealed result
        print(f"Result: {revealed_result}")

        # Clean envionment
        sf.shutdown()

        What's more, you should generate another python code, which perform same computation but in normal mode (non-SPU mode).

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

        response = chat.send_message(prompt)

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
                "/home/chenliheng/anaconda3/envs/secretflow/bin/python3.8",
                os.path.join("seeds", spu_code_file_name),
            ],
            capture_output=True,
        )

        normal_result = subprocess.run(
            [
                "/home/chenliheng/anaconda3/envs/secretflow/bin/python3.8",
                os.path.join("seeds", normal_code_file_name),
            ],
            capture_output=True,
        )

        prompt = (
            "SPU code execution stdout:\n"
            + spu_result.stdout.decode()
            + "SPU code execution stderr:\n"
            + spu_result.stderr.decode()
            + "Normal code execution stdout:\n"
            + normal_result.stdout.decode()
            + "Normal code execution stderr:\n"
            + normal_result.stderr.decode()
            + """
            Please compare their results followed by 'Result:' and ignore information irrelevant to the results.
            
            Finally, answer 'Same', 'Different' or 'Fail' in the first line, and then explain the reason from the second line.

            'Fail' means SPU code run to error or normal code run to error, therefore we cannot obtain the calculation results of both. It may due to generated code is incomplete or other runtime error. Usually when an error occurs, we will find that the output contains error information and even call stack. In SPU code, if output complains that some operation in code is unsupported by SPU runtime or JAX and so on, you need to avoid use them in next round. If output complains ray error, may be we should retry in next round.
            
            'Different' means we sucessfully obtain calculation results of both, but they are different.

            'Same' means we sucessfully obtain calculation results of both, and they are same.

            SPU code can give wrong result compared with normal code, you need to be careful.
            """
        )
        response = chat.send_message(prompt)

        history += chat.history[-4:]
        if len(history) > 40:
            history = history[4:]

        match = re.match(r"(.*)?\n", response.text)
        first_line = match.group(1)
        if first_line.lower() == "same":
            continue
        elif first_line.lower() == "different":
            os.makedirs("different", exist_ok=True)
            os.rename(
                os.path.join("seeds", spu_code_file_name),
                os.path.join("different", spu_code_file_name),
            )
            os.rename(
                os.path.join("seeds", normal_code_file_name),
                os.path.join("different", normal_code_file_name),
            )
            print(os.path.join("different", spu_code_file_name))
            print(response.text)
        elif first_line.lower() == "fail":
            os.makedirs("fail", exist_ok=True)
            os.rename(
                os.path.join("seeds", spu_code_file_name),
                os.path.join("fail", spu_code_file_name),
            )
            os.rename(
                os.path.join("seeds", normal_code_file_name),
                os.path.join("fail", normal_code_file_name),
            )
            print(os.path.join("fail", spu_code_file_name))
            print(response.text)
        else:
            raise Exception("Unknown response")


if __name__ == "__main__":
    main()
