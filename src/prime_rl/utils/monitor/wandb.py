import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import verifiers as vf
import wandb
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.shared import WandbConfig, WandbWithExtrasConfig
from prime_rl.utils.config import BaseConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor.base import Monitor


class WandbMonitor(Monitor):
    """Logs to Weights and Biases."""

    def __init__(
        self,
        config: WandbConfig | WandbWithExtrasConfig | None,
        output_dir: Path | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        run_config: BaseConfig | None = None,
    ):
        self.config = config
        self.logger = get_logger()
        self.history: list[dict[str, Any]] = []
        self.output_dir = output_dir

        rank = int(os.environ.get("RANK", os.environ.get("DP_RANK", "0")))
        self.enabled = self.config is not None
        self.is_master = rank == 0
        if not self.enabled or not self.is_master:
            if not self.is_master:
                self.logger.warning(f"Skipping {self.__class__.__name__} initialization from non-master rank ({rank})")
            return

        assert config is not None
        self.logger.info(f"Initializing {self.__class__.__name__} ({config})")
        self._maybe_overwrite_wandb_command()
        self.wandb = wandb.init(
            project=config.project,
            name=config.name,
            id=config.id,
            tags=config.tags or None,
            dir=output_dir,
            resume="allow",
            config=run_config.model_dump() if run_config else None,
            mode="offline" if config.offline else None,
        )

        # Optionally, initialize sample logging attributes
        if config is not None and isinstance(config, WandbWithExtrasConfig) and config.log_extras:
            if config.log_extras.samples:
                self.last_log_samples_step = -1
                self.samples_cols = ["step", "task", "example_id", "messages", "input_ids", "reward"]
                self.samples_table = wandb.Table(
                    columns=self.samples_cols,
                    log_mode="INCREMENTAL",
                )
                self.tokenizer = tokenizer
                self.samples = []

    def _maybe_overwrite_wandb_command(self) -> None:
        """Overwrites sys.argv with the start command if it is set in the environment variables."""
        wandb_args = os.environ.get("WANDB_ARGS", None)
        if wandb_args:
            self.logger.debug(f"Found WANDB_ARGS in environment variables {wandb_args}")
            sys.argv = json.loads(wandb_args)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.history.append(metrics)
        if not self.is_master:
            return
        if not self.enabled:
            return
        wandb.log(metrics, step=step)

    def log_samples(self, rollouts: list[vf.RolloutOutput], step: int) -> None:
        """Logs rollouts to W&B table."""
        if not self.is_master:
            return
        if (
            not self.config
            or not isinstance(self.config, WandbWithExtrasConfig)
            or not self.config.log_extras
            or not self.config.log_extras.samples
            or step % self.config.log_extras.interval != 0
        ):
            # Do not log samples if not enabled or not log interval step
            return

        assert self.tokenizer is not None, "Tokenizer is required for sample logging"
        assert self.last_log_samples_step <= step, "Step must be greater than last logged step"
        assert self.logger is not None, "Logger is required for sample logging"

        self.logger.info(f"Logging samples to W&B table at step {step}")
        start_time = time.perf_counter()

        for rollout in rollouts:
            trajectory = rollout["trajectory"]
            if not trajectory:
                continue
            # Reconstruct the full episode conversation from trajectory steps.
            # Each TrajectoryStep has `prompt` (Messages) and `completion`
            # (Messages). Build the complete history by walking steps:
            # step 0's prompt has [system, user_obs], then each step adds
            # its completion. For step > 0, the prompt also contains the
            # new env response (user message) appended after the previous
            # completion. After resets (markovian_pure/summary), later
            # prompts are shorter but still end with the new observation.
            prev_prompt_len = 0
            all_msgs: list[dict] = []
            for traj_step in trajectory:
                prompt_msgs = traj_step.get("prompt") or []
                completion_msgs = traj_step.get("completion") or []
                # Add any new messages from the prompt (env responses, obs)
                # that weren't already in our accumulated conversation.
                # For step 0: all prompt messages are new.
                # For step N: new messages are those beyond prev_prompt_len,
                # UNLESS a reset happened (prompt got shorter), in which case
                # only the last message (new observation) is truly new.
                if len(prompt_msgs) > prev_prompt_len:
                    all_msgs.extend(prompt_msgs[prev_prompt_len:])
                elif len(prompt_msgs) <= prev_prompt_len and prev_prompt_len > 0:
                    # Reset happened — prompt is shorter. Insert a compaction
                    # marker so the viz can show where turns were dropped/compressed.
                    dropped = prev_prompt_len - len(prompt_msgs)
                    all_msgs.append({
                        "role": "system",
                        "content": f"[COMPACTION: {dropped} messages dropped from context]",
                    })
                    if prompt_msgs:
                        all_msgs.append(prompt_msgs[-1])
                all_msgs.extend(completion_msgs)
                prev_prompt_len = len(prompt_msgs) + len(completion_msgs)
            # Convert verifiers Message objects to plain dicts for the tokenizer
            msg_dicts = []
            for m in all_msgs:
                if hasattr(m, "model_dump"):
                    d = m.model_dump()
                elif isinstance(m, dict):
                    d = dict(m)
                else:
                    d = {"role": getattr(m, "role", "user"), "content": str(getattr(m, "content", ""))}
                for key in ("reasoning_content", "thinking_blocks"):
                    d.pop(key, None)
                if "tool_calls" in d and not d["tool_calls"]:
                    del d["tool_calls"]
                msg_dicts.append(d)
            try:
                messages_text = self.tokenizer.apply_chat_template(
                    msg_dicts, tokenize=False, add_generation_prompt=False,
                )
            except Exception:
                messages_text = "\n".join(f"[{m.get('role', '?')}] {m.get('content', '')}" for m in msg_dicts)
            full_ids = self.tokenizer.encode(messages_text)
            sample = {
                "step": step,
                "task": rollout.get("task"),
                "example_id": rollout["example_id"],
                "messages": messages_text,
                "input_ids": str(full_ids),
                "reward": rollout["reward"],
            }
            assert list(sample.keys()) == self.samples_cols, (
                "Order of columns in the table must be the same as order of the keys here"
            )
            self.samples_table.add_data(*sample.values())
            self.samples.append(sample)

        wandb.log({"samples": self.samples_table}, step=step)
        self.last_log_samples_step = step
        self.logger.debug(f"Logged samples at step {step} to W&B table in {time.perf_counter() - start_time:.2f}s")

    def log_final_samples(self) -> None:
        """Log final samples to W&B table."""
        if not self.is_master:
            return
        if (
            not self.config
            or not isinstance(self.config, WandbWithExtrasConfig)
            or not self.config.log_extras
            or not self.config.log_extras.samples
        ):
            return

        self.logger.info("Logging final samples to W&B table")
        df = pd.DataFrame(self.samples)
        table = wandb.Table(dataframe=df)
        wandb.log({"final-samples": table})

    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        """Log distributions (no-op for W&B)."""
        pass

    def flush(self, step: int) -> None:
        if not self.is_master or not self.enabled:
            return
        wandb.log({}, step=step, commit=True)

    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        """Save final summary to W&B table."""
        if not self.is_master or not self.enabled:
            return

        self.logger.info("Saving final summary to file")
        assert self.output_dir is not None, "Output directory is required for saving final summary"
        dir_path = self.output_dir / f"run-{self.wandb.id}"
        dir_path.mkdir(parents=True, exist_ok=True)
        with open(dir_path / filename, "w") as f:
            json.dump(wandb.summary._as_dict(), f)
