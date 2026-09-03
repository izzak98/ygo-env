#!/usr/bin/env python3
"""Benchmark compact obs encoding and numpy→PyTorch conversion.

Usage:
    PYTHONPATH=/workspace/ygoenv python example/bench_encode.py [--label LABEL] [--out PATH]
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


def _stub_embeddings() -> dict:
    from ygoenv.decoding import get_code_list

    codes = get_code_list()
    z = [0.0] * 8
    return {c: {"embedding": z, "name_embedding": z} for c in codes}


def _load_embeddings_for_bench():
    from ygoenv.env_wrapping.interface import load_embeddings
    from ygoenv.paths import embeddings_path

    if embeddings_path().is_file():
        return load_embeddings(), False
    return _stub_embeddings(), True


def _n_legal(obs: dict, env_i: int = 0) -> int:
    acts = obs["actions_"][env_i]
    return int(np.any(acts != 0, axis=1).sum())


def _fmt(rows: list[tuple[str, str]]) -> str:
    w = max(len(k) for k, _ in rows)
    return "\n".join(f"  {k.ljust(w)}  {v}" for k, v in rows)


def _time_loop(fn, n: int, warmup: int = 3) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = time.perf_counter() - t0
    return dt, (n / dt if dt > 0 else 0.0)


def _numpy_compact_from_encoded(encoded, prompt) -> dict[str, np.ndarray]:
    return {
        "card_emb_idx": encoded.card_emb_idx.detach().cpu().numpy(),
        "hist_emb_idx": encoded.hist_emb_idx.detach().cpu().numpy(),
        "card_static_board": encoded.card_static_board.detach().cpu().numpy(),
        "card_dynamic": encoded.card_dynamic.detach().cpu().numpy(),
        "history_info": encoded.history_info.detach().cpu().numpy(),
        "prompt_one_hot": prompt.detach().cpu().numpy(),
    }


def _time_numpy_to_torch(arrays: dict[str, np.ndarray], device, n: int = 200) -> dict[str, str]:
    import torch

    def _convert() -> None:
        for v in arrays.values():
            torch.from_numpy(np.ascontiguousarray(v)).to(device)

    # isolate from_numpy (CPU) vs .to(device)
    def _from_numpy_only() -> None:
        for v in arrays.values():
            torch.from_numpy(np.ascontiguousarray(v))

    def _to_device() -> None:
        for v in cpu_tensors:
            v.to(device)

    cpu_tensors = [
        torch.from_numpy(np.ascontiguousarray(v)) for v in arrays.values()
    ]
    dt_all, ips_all = _time_loop(_convert, n)
    dt_fn, ips_fn = _time_loop(_from_numpy_only, n)
    dt_dev, ips_dev = _time_loop(_to_device, n)
    return {
        "device": str(device),
        "from_numpy_to_device_ms": f"{(dt_all / n) * 1e3:.3f}",
        "from_numpy_to_device_it_s": f"{ips_all:.1f}",
        "from_numpy_only_ms": f"{(dt_fn / n) * 1e3:.3f}",
        "to_device_ms": f"{(dt_dev / n) * 1e3:.3f}",
        "to_device_it_s": f"{ips_dev:.1f}",
    }


def _step_ygo(env, n_steps: int, rng: np.random.Generator) -> tuple[int, int, float]:
    obs = env.obs
    n_ok = 0
    n_reset = 0
    t0 = time.perf_counter()
    for _ in range(n_steps):
        raw = env._obs
        acts = np.zeros(env.num_envs, dtype=np.int32)
        for i in range(env.num_envs):
            n_opt = _n_legal(raw, i)
            acts[i] = int(rng.integers(0, n_opt)) if n_opt > 0 else 0
        obs, _, _, done_idx, _ = env.step(acts)
        n_ok += 1
        if len(done_idx) > 0:
            obs = env.reset(env_indices=done_idx.tolist())
            n_reset += 1
    return n_ok, n_reset, time.perf_counter() - t0


def run_bench(label: str, n_steps: int = 200, encode_loops: int = 200) -> str:
    from ygoenv import EnvWrapper, GameMode, YGOEnv
    from ygoenv.constants import DEVICE
    from ygoenv.env_wrapping.interface import encode_all_batch_fast
    from ygoenv.paths import embeddings_path
    import ygoenv.env_wrapping.env_wrapper as ew_mod
    import ygoenv.env_wrapping.interface as iface_mod

    lines: list[str] = []
    lines.append(f"## {label}")
    lines.append("")
    embeddings, stubbed = _load_embeddings_for_bench()
    iface_mod.load_embeddings = lambda path=None: embeddings
    ew_mod.load_embeddings = lambda path=None: embeddings
    emb_path = embeddings_path()
    lines.append(_fmt([
        ("cwd", str(Path.cwd())),
        ("embeddings", str(emb_path)),
        ("embeddings_exists", str(emb_path.is_file())),
        ("embeddings_stubbed", str(stubbed)),
        ("device", str(DEVICE)),
        ("rss_start_mb", f"{rss_mb():.1f}"),
    ]))
    lines.append("")
    rng = np.random.default_rng(0)

    def _make_ygo(n: int, obs_format: str = "raw") -> YGOEnv:
        kw = dict(
            mode=GameMode.BOARD_SETUP,
            deck="tear",
            num_envs=n,
            seed_mode="full_det",
            base_seed=42,
        )
        try:
            return YGOEnv(**kw, obs_format=obs_format)
        except TypeError:
            return YGOEnv(**kw)

    env = _make_ygo(1)
    raw_obs = {k: np.array(v) for k, v in env._obs.items()}

    # warmup JIT
    try:
        encode_all_batch_fast(raw_obs, embeddings, use_preallocated_gpu=False)
    except TypeError:
        encode_all_batch_fast(raw_obs, embeddings, use_preallocated_gpu=False)

    def _enc():
        return encode_all_batch_fast(raw_obs, embeddings, use_preallocated_gpu=False)

    dt, ips = _time_loop(_enc, encode_loops)
    lines.append("### encode_all_batch_fast (numpy encode + torch transfer)")
    lines.append(_fmt([
        ("ok", "True"),
        ("calls", str(encode_loops)),
        ("ms_per_call", f"{(dt / encode_loops) * 1e3:.3f}"),
        ("it_s", f"{ips:.1f}"),
    ]))
    lines.append("")

    encoded, prompt = _enc()
    np_compact = _numpy_compact_from_encoded(encoded, prompt)
    conv = _time_numpy_to_torch(np_compact, DEVICE, n=encode_loops)
    lines.append("### numpy → PyTorch (compact arrays only)")
    lines.append(_fmt([(k, conv[k]) for k in conv]))
    lines.append("")

    for n_envs in (1, 8):
        e = _make_ygo(n_envs)
        r0 = rss_mb()
        n_ok, n_reset, step_s = _step_ygo(e, n_steps if n_envs == 1 else min(n_steps, 50), rng)
        sps = n_ok / step_s if step_s > 0 else 0.0
        lines.append(f"### YGOEnv.step raw numpy (num_envs={n_envs})")
        lines.append(_fmt([
            ("steps_ok", str(n_ok)),
            ("resets", str(n_reset)),
            ("time_s", f"{step_s:.4f}"),
            ("steps_per_s", f"{sps:.2f}"),
            ("rss_delta_mb", f"{rss_mb() - r0:.1f}"),
            ("rss_after_mb", f"{rss_mb():.1f}"),
        ]))
        lines.append("")
        e.close()

    env.close()

    for n_envs in (1, 8):
        try:
            w = EnvWrapper(
                deck="tear",
                num_envs=n_envs,
                max_episode_steps=250,
                base_seed=42,
                mode="full_det",
                auto_reset=True,
                game_mode="board_setup",
                obs_format="raw",
            )
        except TypeError:
            w = EnvWrapper(
                deck="tear",
                num_envs=n_envs,
                max_episode_steps=250,
                base_seed=42,
                mode="full_det",
                auto_reset=True,
                game_mode="board_setup",
            )
        r0 = rss_mb()
        n_target = n_steps if n_envs == 1 else min(n_steps, 50)
        n_ok = 0
        t0 = time.perf_counter()
        try:
            for _ in range(n_target):
                n_opt = int(np.any(w._obs["actions_"][0] != 0, axis=1).sum())
                act = np.zeros(n_envs, dtype=np.int32)
                for i in range(n_envs):
                    n_i = int(np.any(w._obs["actions_"][i] != 0, axis=1).sum())
                    act[i] = int(rng.integers(0, n_i)) if n_i > 0 else 0
                w.step(act)
                n_ok += 1
        except Exception:
            traceback.print_exc()
        step_s = time.perf_counter() - t0
        sps = n_ok / step_s if step_s > 0 else 0.0
        lines.append(f"### EnvWrapper.step (num_envs={n_envs}, obs_format=raw)")
        lines.append(_fmt([
            ("steps_ok", str(n_ok)),
            ("time_s", f"{step_s:.4f}"),
            ("steps_per_s", f"{sps:.2f}"),
            ("rss_delta_mb", f"{rss_mb() - r0:.1f}"),
            ("rss_after_mb", f"{rss_mb():.1f}"),
        ]))
        lines.append("")
        w.close()

    lines.append(f"**peak_rss_mb (approx current): {rss_mb():.1f}**")
    lines.append("")
    return "\n".join(lines) + "\n"


def run_bench_vectorized(label: str, n_steps: int = 200, encode_loops: int = 200) -> str:
    """Post-change benches: native vectorized numpy + numpy→torch + EnvWrapper."""
    from ygoenv import EnvWrapper, GameMode, YGOEnv
    from ygoenv.constants import DEVICE
    import ygoenv.env_wrapping.env_wrapper as ew_mod
    import ygoenv.env_wrapping.interface as iface_mod

    lines: list[str] = []
    lines.append(f"## {label}")
    lines.append("")
    embeddings, stubbed = _load_embeddings_for_bench()
    iface_mod.load_embeddings = lambda path=None: embeddings
    ew_mod.load_embeddings = lambda path=None: embeddings
    rng = np.random.default_rng(0)
    try:
        env = YGOEnv(
            mode=GameMode.BOARD_SETUP,
            deck="tear",
            num_envs=1,
            seed_mode="full_det",
            base_seed=42,
            obs_format="vectorized",
        )
    except (TypeError, ValueError) as e:
        lines.append(f"vectorized YGOEnv unavailable: {e}")
        lines.append("")
        return "\n".join(lines) + "\n"

    pub = env.obs
    ml_keys = [k for k in pub if k.startswith("ml_")]
    lines.append(_fmt([
        ("embeddings_stubbed", str(stubbed)),
        ("public_ml_keys", ", ".join(sorted(ml_keys)) or "-"),
        ("public_has_cards_", str("cards_" in pub)),
        ("arrays_are_numpy", str(all(isinstance(pub[k], np.ndarray) for k in ml_keys))),
        ("device", str(DEVICE)),
    ]))
    lines.append("")

    def _step():
        raw = env._obs
        n_opt = _n_legal(raw, 0)
        act = np.array([int(rng.integers(0, n_opt)) if n_opt > 0 else 0], dtype=np.int32)
        env.step(act)

    dt, ips = _time_loop(_step, n_steps, warmup=5)
    lines.append("### YGOEnv.step vectorized numpy (num_envs=1)")
    lines.append(_fmt([
        ("steps_ok", str(n_steps)),
        ("ms_per_step", f"{(dt / n_steps) * 1e3:.3f}"),
        ("steps_per_s", f"{ips:.2f}"),
        ("rss_after_mb", f"{rss_mb():.1f}"),
    ]))
    lines.append("")

    ml = {k: np.ascontiguousarray(env._obs[k]) for k in env._obs if k.startswith("ml_")}
    if ml:
        conv = _time_numpy_to_torch(ml, DEVICE, n=encode_loops)
        lines.append("### numpy → PyTorch (native ml_* arrays)")
        lines.append(_fmt([(k, conv[k]) for k in conv]))
        lines.append("")
    env.close()

    w = EnvWrapper(
        deck="tear",
        num_envs=1,
        max_episode_steps=250,
        base_seed=42,
        mode="full_det",
        auto_reset=True,
        game_mode="board_setup",
        obs_format="vectorized",
    )
    r0 = rss_mb()
    n_ok = 0
    t0 = time.perf_counter()
    try:
        for _ in range(n_steps):
            n_opt = int(np.any(w._obs["actions_"][0] != 0, axis=1).sum())
            act = np.array([int(rng.integers(0, n_opt)) if n_opt > 0 else 0], dtype=np.int32)
            w.step(act)
            n_ok += 1
    except Exception:
        traceback.print_exc()
    step_s = time.perf_counter() - t0
    sps = n_ok / step_s if step_s > 0 else 0.0
    lines.append("### EnvWrapper.step (num_envs=1, obs_format=vectorized)")
    lines.append(_fmt([
        ("steps_ok", str(n_ok)),
        ("time_s", f"{step_s:.4f}"),
        ("steps_per_s", f"{sps:.2f}"),
        ("rss_delta_mb", f"{rss_mb() - r0:.1f}"),
        ("rss_after_mb", f"{rss_mb():.1f}"),
    ]))
    lines.append("")
    w.close()
    lines.append(f"**peak_rss_mb (approx current): {rss_mb():.1f}**")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="Encode before")
    p.add_argument("--out", default="")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--vectorized", action="store_true")
    args = p.parse_args()

    body = f"# {args.label}\n\n"
    body += f"Python {sys.version.split()[0]}, pid {os.getpid()}\n\n"
    if args.vectorized:
        body += run_bench_vectorized(args.label, n_steps=args.steps)
    else:
        body += run_bench(args.label, n_steps=args.steps)

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
