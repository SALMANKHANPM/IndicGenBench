from runner import vLLMInterface

model = vLLMInterface(
    "google/gemma-3-4b-it",
    dtype="bfloat16",
    max_gpu_memory_utilization=0.8,
)

while True:
    prompt = input("Enter a prompt: ")
    if prompt == "exit":
        break
    result = model.generate(prompt)
    print(result)
    print("-" * 100)
