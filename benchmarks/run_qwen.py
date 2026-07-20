import argparse
import os
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file

def main():
    parser = argparse.ArgumentParser(description="Run Qwen 0.6B using PyTorch and safetensors.")
    parser.add_argument("model_name", type=str, help="Hugging Face model identifier or local path to configuration/tokenizer")
    parser.add_argument("tensor", type=str, help="Path to the model.safetensors file")
    parser.add_argument("prompt", type=str, help="Text prompt to run the model on")
    
    args = parser.parse_args()
    
    # Resolve paths (expand environment variables and user home directory)
    model_name_path = os.path.expandvars(os.path.expanduser(args.model_name))
    tensor_path = os.path.expandvars(os.path.expanduser(args.tensor))
    
    print(f"Loading config and tokenizer from: {model_name_path}")
    # Load config and tokenizer. We set trust_remote_code=True in case custom architectures are needed.
    config = AutoConfig.from_pretrained(model_name_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_path, trust_remote_code=True)
    
    print("Initializing model from config...")
    # Initialize the model structure with random weights
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    print(f"Loading tensor weights from: {tensor_path}")
    state_dict = load_file(tensor_path)
    
    print("Loading state dict into PyTorch model...")
    # Load state dict into the model. Qwen models might tie embeddings, so we use strict=False or load normally.
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Note: Missing keys in state dict (this can be normal for shared/tied weights): {len(missing_keys)} keys missing.")
    if unexpected_keys:
        print(f"Note: Unexpected keys in state dict: {len(unexpected_keys)} keys unexpected.")
    
    # Check for device allocation (use GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Moving model to device: {device}")
    model = model.to(device)
    model.eval()
    
    print("Tokenizing prompt...")
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    
    print("Running inference...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    print("Decoding output...")
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Model Output ===")
    print(decoded_output)

if __name__ == "__main__":
    main()
