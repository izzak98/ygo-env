#!/usr/bin/env python3
"""Benchmark YGOEnv board_setup boot, step throughput, rewards, and RSS.

Usage:
    python example/bench_native.py [--label LABEL] [--out PATH] [--cwd-test]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def rss_mb() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _n_legal(obs: dict, env_i: int = 0) -> int:
    acts = obs["actions_"][env_i]
    return int(np.any(acts != 0, axis=1).sum())


def _fmt(rows: list[tuple[str, str]]) -> str:
    w = max(len(k) for k, _ in rows)
    return "\n".join(f"  {k.ljust(w)}  {v}" for k, v in rows)


def run_bench(label: str, num_step_envs: int = 1, n_steps: int = 200) -> str:
    from ygoenv import GameMode, YGOEnv
    from ygoenv.init import init_ygopro
    from ygoenv.paths import cards_db, code_list_path, deck_path, get_repo_root, scripts_path
    from ygoenv.rewards import get_reward

    lines: list[str] = []
    cwd = Path.cwd()
    root = get_repo_root()
    scripts = scripts_path()
    script_cwd = cwd / "script"
    lines.append(f"## {label}")
    lines.append("")
    lines.append(_fmt([
        ("cwd", str(cwd)),
        ("repo_root", str(root)),
        ("scripts_path", str(scripts)),
        ("scripts_exists", str(scripts.is_dir())),
        ("./script exists", str(script_cwd.exists() or script_cwd.is_symlink())),
        ("cards.cdb", str(cards_db().exists())),
        ("code_list", str(code_list_path().exists())),
        ("tear.ydk", str(deck_path("tear").exists())),
        ("rss_start_mb", f"{rss_mb():.1f}"),
    ]))
    lines.append("")

    rss0 = rss_mb()
    t0 = time.perf_counter()
    try:
        name = init_ygopro("tear", opponent_deck="garnet")
        init_ok = True
        init_err = ""
    except Exception as e:
        name = None
        init_ok = False
        init_err = f"{type(e).__name__}: {e}"
    init_s = time.perf_counter() - t0
    rss1 = rss_mb()
    lines.append("### init_ygopro")
    lines.append(_fmt([
        ("ok", str(init_ok)),
        ("deck_name", str(name)),
        ("time_s", f"{init_s:.4f}"),
        ("rss_delta_mb", f"{rss1 - rss0:.1f}"),
        ("rss_after_mb", f"{rss1:.1f}"),
        ("error", init_err or "-"),
    ]))
    lines.append("")
    if not init_ok:
        lines.append("init_ygopro failed; skipping env benches.")
        lines.append("")
        return "\n".join(lines) + "\n"

    def _construct(n: int) -> tuple[YGOEnv | None, float, float, str]:
        r0 = rss_mb()
        t = time.perf_counter()
        err = ""
        env = None
        try:
            env = YGOEnv(
                mode=GameMode.BOARD_SETUP,
                deck="tear",
                num_envs=n,
                seed_mode="full_det",
                base_seed=42,
            )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        dt = time.perf_counter() - t
        return env, dt, rss_mb() - r0, err

    for n in (1, 8):
        env, dt, drss, err = _construct(n)
        lines.append(f"### YGOEnv(num_envs={n}) construct+reset")
        lines.append(_fmt([
            ("ok", str(env is not None)),
            ("time_s", f"{dt:.4f}"),
            ("rss_delta_mb", f"{drss:.1f}"),
            ("rss_after_mb", f"{rss_mb():.1f}"),
            ("error", err or "-"),
        ]))
        lines.append("")
        if env is not None:
            env.close()

    env, dt, drss, err = _construct(num_step_envs)
    if env is None:
        lines.append("Could not construct env for step loop.")
        lines.append("")
        return "\n".join(lines) + "\n"

    rng = np.random.default_rng(0)
    obs = env.obs
    n_ok = 0
    n_reset = 0
    t_step = time.perf_counter()
    engine_rew_sum = 0.0
    try:
        for _ in range(n_steps):
            active = env.num_envs
            acts = np.zeros(active, dtype=np.int32)
            for i in range(active):
                n_opt = _n_legal(obs, i)
                acts[i] = int(rng.integers(0, n_opt)) if n_opt > 0 else 0
            obs, rews, dones, done_idx, _ = env.step(acts)
            engine_rew_sum += float(np.sum(rews))
            n_ok += 1
            if len(done_idx) > 0:
                obs = env.reset(env_indices=done_idx.tolist())
                n_reset += 1
    except Exception:
        traceback.print_exc()
    step_s = time.perf_counter() - t_step
    sps = n_ok / step_s if step_s > 0 else 0.0

    lines.append(f"### step loop (num_envs={num_step_envs}, target={n_steps})")
    lines.append(_fmt([
        ("steps_ok", str(n_ok)),
        ("resets", str(n_reset)),
        ("time_s", f"{step_s:.4f}"),
        ("steps_per_s", f"{sps:.2f}"),
        ("engine_reward_sum", f"{engine_rew_sum:.4f}"),
        ("rss_after_mb", f"{rss_mb():.1f}"),
    ]))
    lines.append("")

    # Python deck-reward eval on current decoded obs
    t_r = time.perf_counter()
    n_rew = 50
    last = 0.0
    try:
        cards = list(env.decoded_cards[0])
        for _ in range(n_rew):
            last = float(get_reward("tear", cards))
        rew_ok = True
        rew_err = ""
    except Exception as e:
        rew_ok = False
        rew_err = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    rew_s = time.perf_counter() - t_r
    us = (rew_s / n_rew) * 1e6 if rew_ok else float("nan")
    lines.append("### get_reward('tear') Python")
    lines.append(_fmt([
        ("ok", str(rew_ok)),
        ("calls", str(n_rew)),
        ("us_per_call", f"{us:.1f}"),
        ("last_value", f"{last:.4f}"),
        ("n_cards", str(len(list(env.decoded_cards[0])) if rew_ok else "-")),
        ("error", rew_err or "-"),
    ]))
    lines.append("")
    lines.append(f"**peak_rss_mb (approx current): {rss_mb():.1f}**")
    lines.append("")
    env.close()
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="Before")
    p.add_argument("--out", default="")
    p.add_argument("--cwd-test", action="store_true", help="Also record a non-repo CWD run")
    p.add_argument("--steps", type=int, default=200)
    args = p.parse_args()

    body = f"# Native env boot benchmark — {args.label}\n\n"
    body += f"Python {sys.version.split()[0]}, pid {os.getpid()}\n\n"
    body += run_bench(f"{args.label} (cwd={Path.cwd()})", n_steps=args.steps)

    if args.cwd_test:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="ygo-bench-") as tmp:
            old = Path.cwd()
            os.chdir(tmp)
            try:
                body += run_bench(
                    f"{args.label} (non-repo cwd={tmp})",
                    n_steps=min(args.steps, 20),
                )
            finally:
                os.chdir(old)

    print(body)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            existing = out.read_text(encoding="utf-8")
            out.write_text(existing.rstrip() + "\n\n" + body, encoding="utf-8")
        else:
            out.write_text(body, encoding="utf-8")
        print(f"Wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
