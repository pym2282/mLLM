from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SAVE_PATH = "./models/TinyLlama"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

print("Download complete!")