from runner import vLLMInterface

model = vLLMInterface(
    "google/gemma-3-4b-it",
    dtype="bfloat16",
    min_kv_cache_blocks=128,
)
result = model.generate("Hello, world!")
print(result)
