import argparse
import os
import time
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file


class TimingStreamer:
    def __init__(self, synchronize):
        self.synchronize = synchronize
        self.skip_prompt = True
        self.first_token_time = None

    def put(self, value):
        if self.skip_prompt:
            self.skip_prompt = False
            return
        if self.first_token_time is None:
            self.synchronize()
            self.first_token_time = time.perf_counter()

    def end(self):
        pass


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Run Qwen 0.6B using PyTorch and safetensors.")
    parser.add_argument("model_name", type=str, help="Hugging Face model identifier or local path to configuration/tokenizer")
    parser.add_argument("tensor", type=str, help="Path to the model.safetensors file")
    parser.add_argument("prompt", type=str, help="Text prompt to run the model on")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution even when CUDA is available")
    parser.add_argument("--no-cache", action="store_true", help="Disable the KV cache during generation")
    parser.add_argument("--greedy", action="store_true", help="Select the highest-probability token instead of sampling")
    parser.add_argument("--max-token", type=positive_int, default=50, help="Maximum number of new tokens to generate (default: 50)")
    
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
    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    print(f"Attention implementation: {model.config._attn_implementation}")
    
    print(f"Loading tensor weights from: {tensor_path}")
    state_dict = load_file(tensor_path)
    
    print("Loading state dict into PyTorch model...")
    # Load state dict into the model. Qwen models might tie embeddings, so we use strict=False or load normally.
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Note: Missing keys in state dict (this can be normal for shared/tied weights): {len(missing_keys)} keys missing.")
    if unexpected_keys:
        print(f"Note: Unexpected keys in state dict: {len(unexpected_keys)} keys unexpected.")
    
    # Use CUDA when available unless CPU execution was explicitly requested.
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Moving model to device: {device}")
    model = model.to(device)
    model.eval()
    
    print("Tokenizing chat prompt with thinking enabled...")
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)
    
    print("Running inference...")
    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    timing_streamer = TimingStreamer(synchronize)
    synchronize()
    generation_start = time.perf_counter()
    generation_options = {
        "max_new_tokens": args.max_token,
        "do_sample": not args.greedy,
        "use_cache": not args.no_cache,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": timing_streamer,
    }
    if not args.greedy:
        generation_options.update(temperature=0.7, top_p=0.9)
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_options)
    synchronize()
    generation_end = time.perf_counter()

    prompt_tokens = inputs["input_ids"].shape[-1]
    generated_tokens = outputs.shape[-1] - prompt_tokens
    total_seconds = generation_end - generation_start
    first_token_time = timing_streamer.first_token_time or generation_end
    ttft_seconds = first_token_time - generation_start
    post_first_seconds = generation_end - first_token_time
    end_to_end_throughput = generated_tokens / total_seconds if total_seconds > 0 else 0.0
    post_first_throughput = (
        (generated_tokens - 1) / post_first_seconds
        if generated_tokens > 1 and post_first_seconds > 0
        else 0.0
    )
    device_description = (
        f"{device} ({torch.cuda.get_device_name(device)})"
        if device.type == "cuda"
        else str(device)
    )
    
    print("Decoding output...")
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Model Output ===")
    print(decoded_output)
    print("\n=== Performance ===")
    print(f"Device: {device_description}")
    print("Thinking: enabled")
    print(f"Decoding: {'greedy' if args.greedy else 'sampling'}")
    print(f"KV cache: {'disabled' if args.no_cache else 'enabled'}")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"TTFT: {ttft_seconds * 1000:.3f} ms")
    print(f"End-to-end throughput: {end_to_end_throughput:.3f} tokens/s")
    print(f"Post-first-token throughput: {post_first_throughput:.3f} tokens/s")
    print(f"Total generation time: {total_seconds * 1000:.3f} ms")

if __name__ == "__main__":
    main()
