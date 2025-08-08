import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Path to your trained model + tokenizer
model_path = "./fine-tuned-model"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Explicitly set special token IDs
# For CodeT5, these are usually handled by the model config itself
# but we keep them here for clarity if needed.
if model.config.decoder_start_token_id is None:
    model.config.decoder_start_token_id = tokenizer.bos_token_id
if model.config.eos_token_id is None:
    model.config.eos_token_id = tokenizer.eos_token_id
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id

# Match max_length with training
MAX_LEN = 256

# Choose device (GPU / MPS / CPU)
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
model.to(device)

# Example Java input
input_text = "public int centerX() {return x + w / 2;}"

# Tokenize and move to device
inputs = tokenizer(
    input_text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN
)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate translation
output = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=MAX_LEN,
    num_beams=5,          # Beam search for better quality
    early_stopping=True,
    no_repeat_ngram_size=2,
    length_penalty=1.0,
    num_return_sequences=1,
    max_new_tokens=MAX_LEN,
)

# Print results
print("Generated token IDs:")
print(output[0].tolist())

result = tokenizer.decode(output[0], skip_special_tokens=True).strip()

print("\nGenerated output (C#):")
print(result)