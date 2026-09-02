# example/test_init.py — smoke test for ygoenv high-level wrapper
import os
import random
import sys

import numpy as np

from ygoenv import GameMode, YGOEnv
from ygoenv.paths import code_list_path, deck_path, get_repo_root

ROOT = get_repo_root()
os.chdir(ROOT)

deck = deck_path("tear")
code_list = code_list_path()

print(f"Repo root: {ROOT}")
print(f"Deck exists: {deck.exists()}")
print(f"Code list exists: {code_list.exists()}")

seed = 2711989
random.seed(seed)
np.random.seed(seed)

for mode in (GameMode.BOARD_SETUP, GameMode.PLAY_VS_OPPONENT):
    print(f"\n--- {mode.value} ---")
    try:
        env = YGOEnv(
            mode=mode,
            deck="tear",
            num_envs=1,
            seed_mode="full_det",
            base_seed=seed,
            verbose=(mode == GameMode.BOARD_SETUP),
        )
        obs = env.reset()
        print(f"Reset OK, cards shape: {obs['cards_'].shape}")
        env.close()
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\nAll modes OK")
