# pip install -q transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigcode/starcoder"
device = "cuda"  # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(
    checkpoint, token="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"
)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    token="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",
    resume_download=True,
    low_cpu_mem_usage=True,
).to(device)

inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
