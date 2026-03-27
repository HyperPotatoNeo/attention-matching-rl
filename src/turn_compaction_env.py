"""Turn-based compaction environment module.

Exposes load_environment() so verifiers can load it via:
    vf.load_environment("turn_compaction_env", ...)

Delegates to TurnCompactionEnv in compaction_env.
"""
from compaction_env.env import load_turn_compaction_environment as load_environment

__all__ = ["load_environment"]
