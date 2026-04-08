#!/usr/bin/env python3
"""Launch short debug runs for speed benchmarking.

Generates configs for the requested compaction modes and launches 10-step
jobs via EAI. Does NOT touch existing configs or jobs — all output goes
to a separate debug directory.

Usage:
    python launch/launch_debug.py                     # generate + launch all 3
    python launch/launch_debug.py --generate          # configs only
    python launch/launch_debug.py --dry-run           # show commands
    python launch/launch_debug.py --filter am         # only attention_matching
    python launch/launch_debug.py --max-steps 20      # override step count
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("EAI_PROFILE", "yul201")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
CONFIGS_DIR = REPO_DIR / "configs" / "compaction" / "debug_runs"
JOBS_DIR = SCRIPT_DIR / "jobs"

# ── Base parameters (small batch, few steps) ─────────────────────────────────

BASE = dict(
    max_steps=10,
    seq_len=6000,
    output_dir_prefix="/mnt/adea/data_rw/finetuning/emilianopp/experiments/debug-speed",
    wandb_project="balrog-rl-speed-debug",
    ckpt_interval=100,  # effectively disabled
    resume_step=-1,
    num_train_gpus=4,
    num_infer_gpus=4,
    gpu_memory_utilization=0.95,
    max_model_len=16384,
    enforce_eager=True,
    enable_compaction=True,
    dist_timeout_seconds=3600,
    compact_target_ratio=0.25,
    lr=1e-6,
    weight_decay=0.01,
    betas1=0.9,
    betas2=0.9,
    loss_type="default",
    kl_tau=0.1,
    batch_size=128,
    rollouts_per_example=4,
    max_inflight_rollouts=128,
    temperature=1.0,
    base_url='["http://localhost:8000/v1"]',
    # Env
    gym="balrog-bench",
    max_kv_len=6000,
    n_max_turns=4,
    n_preserved_turns=3,
    max_turns=60,
    environments='["babyai"]',
    max_text_history=16,
    # Eval
    eval_interval=5,
    eval_num_examples=20,
    eval_rollouts_per_example=1,
)

MODEL = dict(
    name="Qwen/Qwen3-4B-Instruct-2507",
    prefix="babyai",
    wandb_prefix="debug-speed",
)

MODES = {
    "markovian": dict(
        slug="mk",
        dir_slug="markovian",
        trainer_mode="markovian",
        env_mode="markovian",
        max_tokens=512,
        trainer_fields="",
        trainer_model_extra="",
        env_extra="",
    ),
    "attention_matching": dict(
        slug="am",
        dir_slug="am",
        trainer_mode="attention_matching",
        env_mode="attention_matching",
        max_tokens=512,
        trainer_fields="use_suffix_queries = true",
        trainer_model_extra="",
        env_extra="use_suffix_queries = true, ",
    ),
    "markovian_pure": dict(
        slug="mkpure",
        dir_slug="markovian-pure",
        trainer_mode="markovian_pure",
        env_mode="markovian_pure",
        max_tokens=512,
        trainer_fields="",
        trainer_model_extra="",
        env_extra="",
    ),
    "kv_markovian_grad": dict(
        slug="mk_grad",
        dir_slug="mk-grad",
        trainer_mode="kv_markovian_grad",
        env_mode="kv_markovian_grad",
        max_tokens=512,
        trainer_fields="",
        trainer_model_extra='attn = "flex_attention"',
        env_extra="",
    ),
    "am_fast": dict(
        slug="am_fast",
        dir_slug="am-fast",
        trainer_mode="kv_markovian_grad",
        env_mode="attention_matching",
        max_tokens=512,
        trainer_fields="",
        trainer_model_extra='attn = "flex_attention"',
        env_extra="use_suffix_queries = true, ",
    ),
}

# ── EAI settings ─────────────────────────────────────────────────────────────

IMAGE = "registry.toolkit-sp.yul201.service-now.com/snow.shared/ui_copilot_playwright:latest"
HOME_DATA = "snow.research.adea.emiliano_home:/home/toolkit"
DATA_MOUNTS = [
    "snow.research.ui_assist.data:/mnt/ui_assist/data:ro",
    "snow.research.ui_assist.data:/mnt/ui_assist/data_rw",
    "snow.research.adea.data:/mnt/adea/data:ro",
    "snow.research.adea.data:/mnt/adea/data_rw",
]
GPU_COUNT = 8
GPU_MEM = 80
CPU = 64
MEM = 256
WORKDIR = "/home/toolkit/attention-matching-rl"

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _bool_toml(v):
    return "true" if v else "false"


def generate_config(mode_key: str, max_steps: int) -> str:
    b = {**BASE, "max_steps": max_steps}
    mode = MODES[mode_key]
    m = MODEL

    slug = mode["slug"]
    wandb_name = f"{m['wandb_prefix']}-{slug}-speed"
    dir_name = f"debug-speed-{mode['dir_slug']}"
    trainer_id = f"debug-speed-{slug}-trainer"
    orch_id = f"debug-speed-{slug}-orch"
    env_name = f"debug-speed-{slug}"
    eval_env_name = f"debug-speed-{slug}-eval"

    tags = f'["debug", "speed", "babyai", "{mode_key}"]'

    trainer_lines = f'compaction_mode = "{mode["trainer_mode"]}"'
    if mode["trainer_fields"]:
        trainer_lines += f"\n{mode['trainer_fields']}"

    env_args = (
        f'{{ gym = "{b["gym"]}", max_kv_len = {b["max_kv_len"]}, '
        f'max_response_tokens = {mode["max_tokens"]}, '
        f'compact_target_ratio = {b["compact_target_ratio"]}, '
        f'n_max_turns = {b["n_max_turns"]}, n_preserved_turns = {b["n_preserved_turns"]}, '
        f'max_turns = {b["max_turns"]}, '
        f'compaction_mode = "{mode["env_mode"]}", '
        f'{mode["env_extra"]}'
        f'environments = {b["environments"]}, '
        f'max_text_history = {b["max_text_history"]} }}'
    )

    return f"""\
# Debug speed test: {mode_key}, {max_steps} steps
max_steps = {b["max_steps"]}
seq_len = {b["seq_len"]}
output_dir = "{b["output_dir_prefix"]}/{dir_name}"

[model]
name = "{m["name"]}"

[wandb]
project = "{b["wandb_project"]}"
tags = {tags}
name = "{wandb_name}"

[trainer.wandb]
tags = {tags}
id = "{trainer_id}"

[orchestrator.wandb]
tags = {tags}
id = "{orch_id}"

[ckpt]
interval = {b["ckpt_interval"]}
resume_step = {b["resume_step"]}

[deployment]
num_train_gpus = {b["num_train_gpus"]}
num_infer_gpus = {b["num_infer_gpus"]}

[inference]
gpu_memory_utilization = {b["gpu_memory_utilization"]}

[inference.model]
enforce_eager = {_bool_toml(b["enforce_eager"])}
max_model_len = {b["max_model_len"]}

[inference.vllm_extra]
enable_compaction = {_bool_toml(b["enable_compaction"])}

[trainer]
dist_timeout_seconds = {b["dist_timeout_seconds"]}
compact_target_ratio = {b["compact_target_ratio"]}
{trainer_lines}

[trainer.model]
impl = "auto"
{(mode["trainer_model_extra"] + chr(10)) if mode["trainer_model_extra"] else ""}optim_cpu_offload = true

[trainer.model.ac]
freq = 1

[trainer.optim]
lr = {b["lr"]}
weight_decay = {b["weight_decay"]}
betas1 = {b["betas1"]}
betas2 = {b["betas2"]}

[trainer.loss]
type = "{b["loss_type"]}"
kl_tau = {b["kl_tau"]}

[orchestrator]
batch_size = {b["batch_size"]}
rollouts_per_example = {b["rollouts_per_example"]}
max_inflight_rollouts = {b["max_inflight_rollouts"]}

[orchestrator.advantage]
type = "default"

[orchestrator.client]
base_url = {b["base_url"]}

[orchestrator.sampling]
max_tokens = {mode["max_tokens"]}
temperature = {b["temperature"]}

[[orchestrator.env]]
id = "turn_compaction_env"
name = "{env_name}"
args = {env_args}

[orchestrator.eval]
interval = {b["eval_interval"]}
num_examples = {b["eval_num_examples"]}
rollouts_per_example = {b["eval_rollouts_per_example"]}

[orchestrator.eval.sampling]
temperature = {b["temperature"]}
max_tokens = {mode["max_tokens"]}

[[orchestrator.eval.env]]
id = "turn_compaction_env"
name = "{eval_env_name}"
args = {env_args}
"""


def get_account():
    result = subprocess.run(
        ["eai", "account", "get", "--no-header", "--field", "fullName"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Failed to get EAI account: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def launch_job(config_file, run_name, account, dry_run=False):
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    job_name = f"{run_name.replace('-', '_')}_{timestamp}"
    config_rel = f"configs/compaction/debug_runs/{config_file}"
    wandb_key = os.environ.get("WANDB_API_KEY", "")

    cmd = [
        "eai", "job", "new",
        "--account", account,
        "--restartable",
        "--name", job_name,
        "--image", IMAGE,
        "--gpu", str(GPU_COUNT),
        "--gpu-mem", str(GPU_MEM),
        "--cpu", str(CPU),
        "--mem", str(MEM),
        "--data", HOME_DATA,
    ]
    for mount in DATA_MOUNTS:
        cmd += ["--data", mount]
    cmd += [
        "--env", "HOME=/home/toolkit",
        "--env", "PATH=/home/toolkit/.local/bin:/usr/local/bin:/usr/bin:/bin",
        "--env", f"WANDB_API_KEY={wandb_key}",
        "--tag", "debug",
        "--tag", "speed-test",
        "--tag", run_name,
        "--workdir", WORKDIR,
        "--field", "id",
        "--no-header",
        "--", "uv", "run", "rl", "@", config_rel,
    ]

    if dry_run:
        print(f"    cmd: eai job new --name {job_name} ... -- uv run rl @ {config_rel}")
        return None

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    {RED}ERROR:{RESET} {result.stderr.strip()}")
        return None

    job_id = result.stdout.strip()
    if not job_id:
        print(f"    {RED}ERROR:{RESET} no job ID returned")
        return None
    return job_id


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--generate", action="store_true", help="Generate configs only")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without launching")
    parser.add_argument("--filter", type=str, help="Filter modes by substring")
    parser.add_argument("--max-steps", type=int, default=10, help="Training steps (default: 10)")
    args = parser.parse_args()

    modes = list(MODES.keys())
    if args.filter:
        modes = [m for m in modes if args.filter in m]
    if not modes:
        print(f"{RED}No modes match filter '{args.filter}'{RESET}")
        sys.exit(1)

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    configs = []
    for mode_key in modes:
        slug = MODES[mode_key]["slug"]
        config_file = f"debug_speed_{slug}.toml"
        run_name = f"debug-speed-{slug}"

        content = generate_config(mode_key, args.max_steps)
        (CONFIGS_DIR / config_file).write_text(content)
        configs.append((config_file, run_name, mode_key))

    print(f"\n{BOLD}Generated {len(configs)} debug configs -> {CONFIGS_DIR}{RESET}")
    for config_file, run_name, mode_key in configs:
        print(f"  {config_file:40s} ({mode_key})")

    if args.generate:
        return

    print()
    account = get_account()
    print(f"{BOLD}Launching debug speed tests — {account}{RESET}\n")

    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    for config_file, run_name, mode_key in configs:
        print(f"  {YELLOW}LAUNCH{RESET}  {run_name} ({mode_key}, {args.max_steps} steps)")
        job_id = launch_job(config_file, run_name, account, dry_run=args.dry_run)
        if job_id:
            info = {
                "job_id": job_id,
                "config": config_file,
                "run_name": run_name,
                "mode": mode_key,
                "max_steps": args.max_steps,
                "launched_at": datetime.now().isoformat(),
            }
            (JOBS_DIR / f"{run_name}.json").write_text(json.dumps(info, indent=2) + "\n")
            print(f"    -> job {job_id}")

    print(f"\n  {BOLD}Done: {len(configs)} jobs launched{RESET}\n")


if __name__ == "__main__":
    main()
