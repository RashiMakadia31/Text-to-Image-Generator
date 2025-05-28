from llama_cpp import Llama
import os

llm_path = os.path.join("models", "llama", "mistral-7b-instruct-v0.1.Q4_K_M.gguf")

llm = Llama(
    model_path=llm_path,
    n_ctx=2048,
    n_threads=8  # Tune per your CPU
)

def enhance_prompt(user_input):
    prompt = f"Improve the following image generation prompt to be more descriptive: '{user_input}'"
    output = llm(prompt, max_tokens=128, stop=["\n", "User:", "Assistant:"])
    return output["choices"][0]["text"].strip()
