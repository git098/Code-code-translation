

import argparse
from transformers import RobertaTokenizer, EncoderDecoderModel

def translate_java_to_cs(java_code, model, tokenizer):
    """Translates a Java code snippet to C# using the fine-tuned model."""
    inputs = tokenizer(java_code, return_tensors="pt", padding=True, truncation=True).to(model.device)
    translated_outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(translated_outputs[0], skip_special_tokens=True)

def main(args):
    tokenizer = RobertaTokenizer.from_pretrained(args.model_dir)
    model = EncoderDecoderModel.from_pretrained(args.model_dir)

    if torch.cuda.is_available():
        model.to('cuda')

    with open(args.input_file, 'r') as f:
        java_code = f.read()

    translated_code = translate_java_to_cs(java_code, model, tokenizer)

    print("--- Original Java Code ---")
    print(java_code)
    print("\n--- Translated C# Code ---")
    print(translated_code)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Directory where the fine-tuned model is saved.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the Java code file to translate.")
    args = parser.parse_args()
    main(args)

