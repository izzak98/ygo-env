"""Setuptools hooks to compile the ygoenv C++ extension via xmake."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from setuptools.command.build_py import build_py
from setuptools.command.editable_wheel import editable_wheel


def ygo_env_root() -> Path:
    """Return the ``ygo-env`` git submodule root."""
    return Path(__file__).resolve().parent.parent


def workspace_root() -> Path:
    """Return the top-level training repo root (parent of ``ygo-env``)."""
    return ygo_env_root().parent


def native_extension_path() -> Path:
    return ygo_env_root() / "ygoenv" / "ygoenv" / "ygopro"


def find_built_extension() -> list[Path]:
    return sorted(native_extension_path().glob("ygopro_ygoenv*.so"))


def build_extension(*, force: bool = False) -> None:
    """Compile ``ygopro_ygoenv`` with xmake (no-op if already built unless *force*)."""
    if os.environ.get("YGO_SKIP_NATIVE_BUILD", "").lower() in ("1", "true", "yes"):
        print("YGO_SKIP_NATIVE_BUILD set — skipping native xmake build", file=sys.stderr)
        return

    if not force and find_built_extension():
        print(f"Native extension already present: {find_built_extension()[0]}")
        return

    root = ygo_env_root()
    print(f"Building ygopro_ygoenv via make build_ext in {root} …", file=sys.stderr)
    subprocess.check_call(["make", "build_ext"], cwd=root)

    built = find_built_extension()
    if not built:
        raise RuntimeError(
            "build finished but ygopro_ygoenv*.so was not found under "
            f"{native_extension_path()}"
        )
    print(f"Built native extension: {built[0]}", file=sys.stderr)


def ensure_script_symlink() -> None:
    """Ensure ``/workspace/script`` → ygo-env Lua scripts (engine loads ``./script/``)."""
    ws = workspace_root()
    link = ws / "script"
    if link.is_symlink() or link.exists():
        return

    scripts_dir = ygo_env_root() / "third_party" / "ygopro-scripts"
    if not scripts_dir.is_dir():
        print("Fetching ygopro-scripts via make -C ygo-env scripts …", file=sys.stderr)
        subprocess.check_call(["make", "scripts"], cwd=ygo_env_root())

    target = Path("ygo-env/third_party/ygopro-scripts")
    link.symlink_to(target, target_is_directory=True)
    print(f"Created script symlink: {link} -> {target}", file=sys.stderr)


def prepare_runtime_assets() -> None:
    """Best-effort fetch of card DB + scripts if missing."""
    ygo = ygo_env_root()
    cards_cdb = ygo / "assets" / "locale" / "en" / "cards.cdb"
    scripts = ygo / "third_party" / "ygopro-scripts"
    if cards_cdb.exists() and scripts.is_dir():
        return
    print("Fetching ygo-env assets/scripts via make …", file=sys.stderr)
    subprocess.check_call(["make", "assets", "scripts"], cwd=ygo)


class BuildPyWithNative(build_py):
    """Run xmake before packaging Python sources."""

    def run(self) -> None:
        prepare_runtime_assets()
        build_extension()
        ensure_script_symlink()
        super().run()


class EditableWheelWithNative(editable_wheel):
    """Run xmake for editable installs (`pip install -e .`)."""

    def run(self) -> None:
        prepare_runtime_assets()
        build_extension()
        ensure_script_symlink()
        super().run()
