# Native env boot benchmark

BOARD_SETUP, deck `tear`, `seed_mode=full_det`. Python 3.10.12.

## Comparison (repo-root process)

| metric | Before | After |
|---|---:|---:|
| init_ygopro time | 0.405 s | 0.361 s |
| init RSS delta | 45.3 MB | 22.2 MB |
| YGOEnv(n=1) construct+reset | 1.935 s / +123 MB | 0.159 s / +41 MB |
| YGOEnv(n=8) construct+reset | 3.315 s / +248 MB | 0.073 s / +19 MB |
| 200 random steps | 119 step/s | 112 step/s |
| peak RSS after steps | 566 MB | 203 MB |
| works from cwd without `./script/` | no (empty lua buffers) | yes (absolute `scripts_path()`) |

Step throughput is similar (duel process dominates). Memory and boot time drop because scripts are lazy-loaded, `init_module` runs once, board_setup uses dummy P2, and deck rewards run in C++.

Skipped switching `YGOEnv` to a single batched `make(num_envs=N)`: after init-once, n=8 construct is no longer ~8× n=1.

---

# Native env boot benchmark — Before

Python 3.10.12, pid 5237

## Before (cwd=/workspace)

  cwd              /workspace
  repo_root        /workspace
  scripts_path     /workspace/third_party/ygopro-scripts
  scripts_exists   True
  ./script exists  True
  cards.cdb        True
  code_list        True
  tear.ydk         True
  rss_start_mb     120.9

### init_ygopro
  ok            True
  deck_name     tear
  time_s        0.4052
  rss_delta_mb  45.3
  rss_after_mb  166.2
  error         -

### YGOEnv(num_envs=1) construct+reset
  ok            True
  time_s        1.9347
  rss_delta_mb  123.1
  rss_after_mb  289.3
  error         -

### YGOEnv(num_envs=8) construct+reset
  ok            True
  time_s        3.3145
  rss_delta_mb  247.5
  rss_after_mb  536.7
  error         -

### step loop (num_envs=1, target=200)
  steps_ok           200
  resets             77
  time_s             1.6808
  steps_per_s        118.99
  engine_reward_sum  0.0000
  rss_after_mb       566.0

### get_reward('tear') Python
  ok           True
  calls        50
  us_per_call  54.7
  last_value   0.0000
  n_cards      102
  error        -

**peak_rss_mb (approx current): 566.0**

## Before (non-repo cwd, same process as above)

Scripts were already cached from the repo-root run, so this does not show a cold `./script` miss.

## Before (fresh process, cwd=/tmp)

  cwd              /tmp
  repo_root        /workspace
  scripts_path     /workspace/third_party/ygopro-scripts
  scripts_exists   True
  ./script exists  False
  rss_start_mb     118.6

### init_ygopro
  ok            True
  time_s        0.3785
  rss_delta_mb  22.7
  note          Succeeds but lua files resolve via CWD `./script/` (missing). Card scripts are stored as empty buffers; duels boot without real card logic.

### YGOEnv(num_envs=1) construct+reset
  ok            True
  time_s        0.4530
  rss_delta_mb  40.2

### step loop (num_envs=1, target=5)
  steps_ok      5
  steps_per_s   191.42

**peak_rss_mb: 206.4**

# Native env boot benchmark — After

Python 3.10.12, pid 6753

## After (cwd=/workspace)

  cwd              /workspace
  repo_root        /workspace
  scripts_path     /workspace/third_party/ygopro-scripts
  scripts_exists   True
  ./script exists  True
  cards.cdb        True
  code_list        True
  tear.ydk         True
  rss_start_mb     118.5

### init_ygopro
  ok            True
  deck_name     tear
  time_s        0.3608
  rss_delta_mb  22.2
  rss_after_mb  140.7
  error         -

### YGOEnv(num_envs=1) construct+reset
  ok            True
  time_s        0.1591
  rss_delta_mb  41.4
  rss_after_mb  182.0
  error         -

### YGOEnv(num_envs=8) construct+reset
  ok            True
  time_s        0.0728
  rss_delta_mb  18.7
  rss_after_mb  200.6
  error         -

### step loop (num_envs=1, target=200)
  steps_ok           200
  resets             77
  time_s             1.7873
  steps_per_s        111.90
  engine_reward_sum  0.0000
  rss_after_mb       202.7

### get_reward('tear') Python
  ok           True
  calls        50
  us_per_call  68.2
  last_value   0.0000
  n_cards      95
  error        -

**peak_rss_mb (approx current): 202.7**

# Native env boot benchmark — After-fresh-nonrepo-cwd

Python 3.10.12, pid 6975

## After-fresh-nonrepo-cwd (cwd=/tmp)

  cwd              /tmp
  repo_root        /workspace
  scripts_path     /workspace/third_party/ygopro-scripts
  scripts_exists   True
  ./script exists  False
  cards.cdb        True
  code_list        True
  tear.ydk         True
  rss_start_mb     118.6

### init_ygopro
  ok            True
  deck_name     tear
  time_s        0.3518
  rss_delta_mb  22.2
  rss_after_mb  140.8
  error         -

### YGOEnv(num_envs=1) construct+reset
  ok            True
  time_s        0.0882
  rss_delta_mb  41.3
  rss_after_mb  182.0
  error         -

### YGOEnv(num_envs=8) construct+reset
  ok            True
  time_s        0.0254
  rss_delta_mb  18.7
  rss_after_mb  200.7
  error         -

### step loop (num_envs=1, target=200)
  steps_ok           200
  resets             77
  time_s             1.6668
  steps_per_s        119.99
  engine_reward_sum  0.0000
  rss_after_mb       202.8

### get_reward('tear') Python
  ok           True
  calls        50
  us_per_call  52.0
  last_value   0.0000
  n_cards      95
  error        -

**peak_rss_mb (approx current): 202.8**

# Compact obs encoding benchmark

BOARD_SETUP, deck `tear`, `seed_mode=full_det`, CPU (no CUDA). Stub embeddings (no `embeddings.json` in this environment).

| metric | Before (Python encode) | After (native vectorized) |
|---|---:|---:|
| encode_all_batch_fast | 51.9 ms/call (19.3/s) | n/a (skipped) |
| numpy → torch (compact) | 0.003 ms | 0.003 ms |
| YGOEnv.step num_envs=1 | 18.74 step/s (raw numpy) | 20.22 step/s (vectorized numpy) |
| EnvWrapper.step num_envs=1 | 6.91 step/s | 11.97 step/s |
| EnvWrapper peak RSS | 410.6 MB | 386.7 MB |

Engine `YGOEnv.step` is unchanged in cost (duel dominates). `EnvWrapper.step` is faster because it no longer re-parses uint8 obs in `encode_all_batch_fast` (~52 ms). Remaining wrap cost is decode + numpy→torch (negligible on CPU) + reward shaping.

---

# Encode before

Python 3.10.12, pid 95755

## Encode before

  cwd                 /workspace
  embeddings          /workspace/embeddings.json
  embeddings_exists   False
  embeddings_stubbed  True
  device              cpu
  rss_start_mb        313.9

### encode_all_batch_fast (numpy encode + torch transfer)
  ok           True
  calls        200
  ms_per_call  51.890
  it_s         19.3

### numpy → PyTorch (compact arrays only)
  device                     cpu
  from_numpy_to_device_ms    0.003
  from_numpy_to_device_it_s  300495.7
  from_numpy_only_ms         0.002
  to_device_ms               0.001
  to_device_it_s             921998.9

### YGOEnv.step raw numpy (num_envs=1)
  steps_ok      200
  resets        77
  time_s        10.6720
  steps_per_s   18.74
  rss_delta_mb  0.6
  rss_after_mb  386.3

### YGOEnv.step raw numpy (num_envs=8)
  steps_ok      50
  resets        49
  time_s        6.7204
  steps_per_s   7.44
  rss_delta_mb  3.5
  rss_after_mb  407.4

### EnvWrapper.step (num_envs=1, obs_format=raw)
  steps_ok      200
  time_s        28.9339
  steps_per_s   6.91
  rss_delta_mb  0.5
  rss_after_mb  410.6

### EnvWrapper.step (num_envs=8, obs_format=raw)
  steps_ok      50
  time_s        15.7973
  steps_per_s   3.17
  rss_delta_mb  7.0
  rss_after_mb  435.4

**peak_rss_mb (approx current): 435.4**

# Encode after

Python 3.10.12, pid 97787

## Encode after

  embeddings_stubbed  True
  public_ml_keys      ml_card_dynamic_, ml_card_emb_idx_, ml_card_static_, ml_hist_emb_idx_, ml_history_info_, ml_n_me_, ml_prompt_
  public_has_cards_   False
  arrays_are_numpy    True
  device              cpu

### YGOEnv.step vectorized numpy (num_envs=1)
  steps_ok      200
  ms_per_step   49.452
  steps_per_s   20.22
  rss_after_mb  378.1

### numpy → PyTorch (native ml_* arrays)
  device                     cpu
  from_numpy_to_device_ms    0.003
  from_numpy_to_device_it_s  315215.3
  from_numpy_only_ms         0.002
  to_device_ms               0.001
  to_device_it_s             884001.3

### EnvWrapper.step (num_envs=1, obs_format=vectorized)
  steps_ok      200
  time_s        16.7065
  steps_per_s   11.97
  rss_delta_mb  1.0
  rss_after_mb  386.7

**peak_rss_mb (approx current): 386.7**

