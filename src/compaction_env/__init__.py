"""
Compaction Environment for prime-rl/verifiers.

Wraps any SingleTurnEnv to route generation through the CompactionEngine,
which generates text with mid-sequence KV cache compaction.

Usage in TOML config:
    [[orchestrator.env]]
    id = "compaction_env"
    args = {gym = "countdown", max_seq_len = 2048, compact_target_ratio = 0.3, n_compacts = 1}
"""
from compaction_env.env import CompactionEnv, TurnCompactionEnv, load_environment, load_turn_compaction_environment

__all__ = ["CompactionEnv", "TurnCompactionEnv", "load_environment", "load_turn_compaction_environment"]
