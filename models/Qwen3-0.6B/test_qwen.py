
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Specify the local model directory
model_path = "/home/ubuntu/models/Qwen3-0.6B"

# 2. Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 3. Load the model using CPU
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="cpu"
)

# 4. Give the model a question
prompt = "Explain artificial intelligence in simple words."

# 5. Convert text into tokens
inputs = tokenizer(prompt, return_tensors="pt")

# 6. Generate a response
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

# 7. Convert generated tokens back into text
response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)

print("\nModel response:")
print(response)
