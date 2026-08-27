"""
Robust PPO checkpoint loading.

SB3's `PPO.load()` / `PPO.save()` auto-manage a ".zip" suffix: on save, if
your path already ends in ".zip" it's used as-is; if not, ".zip" is
appended. On load, it first tries the exact path given, and *only if that
open fails* does it retry once more with ".zip" appended as a fallback.

That fallback is exactly what produces the confusing "best_model.zip.zip"
in error messages: it's not a real double-suffix bug anywhere in this
codebase (every `model.save(...)` call here already produces a
correctly-suffixed single ".zip" file, and grepping the repo confirms nothing
appends ".zip" a second time to an already-suffixed path). The message shows
up purely because the *first* file (e.g. "best_model.zip") doesn't exist yet
-- most commonly because training never reached its first eval_freq
checkpoint, the checkpoint_dir in the config doesn't match the path you're
loading from, or training crashed/hasn't been run yet -- and SB3's retry
path is the last thing printed before the real FileNotFoundError.

`load_ppo_checkpoint` exists purely to fail fast with a clear, actionable
message instead of surfacing that retry artifact.
"""
from __future__ import annotations

import os


def load_ppo_checkpoint(path: str):
    """Load a PPO checkpoint, raising a clear error if the file is missing
    instead of letting SB3's internal .zip-suffix retry produce a confusing
    'best_model.zip.zip' style message."""
    from stable_baselines3 import PPO

    # Normalize: SB3 handles the suffix itself, so we only need to check
    # the path the user actually gave us (with or without ".zip") resolves
    # to a real file -- trying both forms here means we never depend on
    # SB3's own fallback-retry behavior to find the file.
    candidates = [path] if path.endswith(".zip") else [path, path + ".zip"]
    if not any(os.path.isfile(c) for c in candidates):
        raise FileNotFoundError(
            f"No PPO checkpoint found at '{path}' (also checked "
            f"'{path if path.endswith('.zip') else path + '.zip'}'). "
            f"Common causes: training hasn't reached its first eval_freq "
            f"checkpoint yet, the run crashed before saving, or "
            f"checkpoint_dir in the config doesn't match this path. Check "
            f"that the file actually exists before re-running eval."
        )
    return PPO.load(path)
