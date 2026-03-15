"""Merge a DCP FSDP2 checkpoint into HF safetensors format."""
import argparse
import shutil
from pathlib import Path
import torch
from torch.distributed.checkpoint import load as dcp_load
from torch.distributed.checkpoint.state_dict import StateDictOptions
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True, help="Path to checkpoint dir (e.g. outputs/X/checkpoints/step_600)")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B", help="Base model name for config/tokenizer")
    parser.add_argument("--output-dir", required=True, help="Output directory for merged HF model")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_dir) / "trainer"
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    state_dict = model.state_dict()

    print(f"Loading DCP checkpoint from {ckpt_path}...")
    dcp_load(state_dict, checkpoint_id=str(ckpt_path))

    model.load_state_dict(state_dict)
    print(f"Saving merged model to {output_path}...")
    model.save_pretrained(output_path, max_shard_size="5GB")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(output_path)
    print("Done!")

if __name__ == "__main__":
    main()
