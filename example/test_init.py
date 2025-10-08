# example/test_init.py
import ygoenv
import sys
import os
import random
import numpy as np


# Import the initialization function
from utils import init_ygopro

# Set up paths
env_id = "YGOPro-v1"
lang = "english"
deck = "assets/deck/Voiceless.ydk"
code_list_file = "example/code_list.txt"

print(f"Current directory: {os.getcwd()}")
print(f"Checking paths:")
print(f"  Deck path exists: {os.path.exists(deck)}")
print(f"  Code list exists: {os.path.exists(code_list_file)}")
print(f"  Assets path: {os.path.abspath('assets')}")
print(f"  Cards.cdb exists: {os.path.exists('assets/cards.cdb')}")

# Initialize YGOPro - this is crucial!
try:
    deck_result, deck_names = init_ygopro(
        env_id, lang, deck, code_list_file, return_deck_names=True
    )
    print(f"Init successful! Deck: {deck_result}")
    print(f"Deck names: {deck_names}")
except Exception as e:
    print(f"Init failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Now try to create the environment with proper initialization

seed = 2711989
random.seed(seed)
np.random.seed(seed)

print("\nCreating environment...")
try:
    # Use the initialized deck
    envs = ygoenv.make(
        task_id=env_id,
        env_type="gymnasium",
        num_envs=1,
        num_threads=1,
        seed=seed,
        deck1=deck_result,  # Use the result from init_ygopro
        deck2=deck_result,  # Use the result from init_ygopro
        player=-1,
        max_options=24,
        n_history_actions=32,
        play_mode='self',
        async_reset=False,
        verbose=True,
        record=False,
    )
    print("Environment created successfully!")

    # Try to reset
    obs, infos = envs.reset()
    print("Reset successful!")
    print(f"Observation shape: {obs['cards_'].shape if 'cards_' in obs else 'N/A'}")
    print(f"To play: {infos['to_play']}")

    # Try a single step
    print("\nTrying a step...")
    while True:
        actions = np.array([0])
        a = envs.step(actions)
        # print("env output:\n", a)
        print("Step successful!")
        break

except Exception as e:
    print(f"Environment creation/usage failed: {e}")
    import traceback
    traceback.print_exc()
