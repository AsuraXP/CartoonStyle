"""Crash-safe output handling: manifests, incremental zips, Kaggle Dataset pushes.

Nothing waits for the run to finish: every image is written the moment it is
generated, a rolling zip accumulates new files, and (optionally) the output dir
is pushed to a private Kaggle Dataset on a timer so a killed session never
loses work.
"""

from __future__ import annotations

import json
import os
import shutil
import re
import subprocess
import threading
import time
import zipfile


def purge_bad_torchao() -> None:
    """peft raises at LoRA load merely if torchao < 0.16 is installed (we never
    use torchao). Kaggle's image ships 0.10.0 in EVERY session - purge it."""
    try:
        from importlib.metadata import version as _pkg_version

        v = _pkg_version("torchao")
    except Exception:
        return
    try:
        from packaging.version import Version

        if Version(v) >= Version("0.16.0"):
            return
    except Exception:
        pass
    import subprocess
    import sys

    print(f"[main] uninstalling incompatible torchao {v}", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], capture_output=True)


def read_done(manifest_paths: list[str], images_dir: str = "") -> dict[str, dict]:
    done: dict[str, dict] = {}
    for p in manifest_paths:
        if not p or not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a torn last line after a crash
                k = rec.get("key") or rec.get("id")  # "id" = legacy generator field
                if k:
                    done[k] = rec
    # belt and suspenders: a surviving images/ folder counts as done even if the
    # manifest was lost in a session wipe (filename carries the key)
    if images_dir and os.path.isdir(images_dir):
        for fn in os.listdir(images_dir):
            m = re.match(r"^(beat\d+m\d+_v\d+)_s[0-9a-fA-F]+\.png$", fn)
            if m:
                done.setdefault(m.group(1), {"key": m.group(1), "from_image": fn})
    return done


def append_manifest(path: str, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def merge_manifests(out_dir: str, shard_files: list[str], merged_name: str = "manifest.jsonl") -> str:
    done = read_done(shard_files)
    merged = os.path.join(out_dir, merged_name)
    with open(merged, "w", encoding="utf-8") as f:
        for rec in sorted(done.values(), key=lambda r: r.get("index", 0)):
            f.write(json.dumps(rec) + "\n")
    return merged


def _zip_index_path(out_dir: str) -> str:
    return os.path.join(out_dir, "snapshots", "zip_index.json")


def update_zip(out_dir: str, rel_files: list[str]) -> str | None:
    """Append newly generated files to snapshots/images.zip (no duplicates)."""
    new = [f for f in rel_files if os.path.exists(os.path.join(out_dir, f))]
    if not new:
        return None
    snap_dir = os.path.join(out_dir, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    zpath = os.path.join(snap_dir, "images.zip")
    idx_path = _zip_index_path(out_dir)
    seen: set[str] = set()
    if os.path.exists(idx_path):
        with open(idx_path, encoding="utf-8") as f:
            seen = set(json.load(f))
    todo = [f for f in new if f not in seen]
    if not todo:
        return zpath if os.path.exists(zpath) else None
    with zipfile.ZipFile(zpath, "a", zipfile.ZIP_STORED) as z:
        for f in todo:
            z.write(os.path.join(out_dir, f), f)
    seen.update(todo)
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(sorted(seen), f)
    return zpath


def list_new_images(out_dir: str) -> list[str]:
    img_dir = os.path.join(out_dir, "images")
    if not os.path.isdir(img_dir):
        return []
    return sorted(
        os.path.join("images", n) for n in os.listdir(img_dir) if n.lower().endswith(".png")
    )


def resolve_kaggle_user() -> str | None:
    """Kaggle username from any of the places the CLI might keep it."""
    import json as _json
    u = os.environ.get("KAGGLE_USERNAME")
    if u:
        return u
    for p in (os.path.expanduser("~/.kaggle/kaggle.json"), "/root/.kaggle/kaggle.json"):
        try:
            with open(p, encoding="utf-8") as f:
                u = (_json.load(f) or {}).get("username")
            if u:
                return u
        except Exception:
            pass
    try:  # token-auth VMs keep config in ~/.config/kaggle
        r = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True, timeout=30)
        for line in (r.stdout or "").splitlines():
            if "username" in line.lower() and ":" in line:
                u = line.split(":", 1)[1].strip() or None
                if u:
                    return u
    except Exception:
        pass
    return None


def kaggle_push(out_dir: str, message: str = "auto-save") -> bool:
    """Best-effort create-or-version a Kaggle Dataset from `out_dir`.

    Works on both Kaggle CLI generations: 1.x wants `--private`; 2.x removed it
    (private is the default) and requires `-m` on `datasets version`. Both
    generations default `--dir-mode` differently - we always pass `zip` so
    nested dirs (checkpoints/, final/peft/) are uploaded instead of skipped.
    Returns True when the push succeeded.
    """
    # Kaggle CLI 2.x refuses to push without a dataset-metadata.json telling it
    # where the dataset lives. Write one if the user hasn't.
    meta_path = os.path.join(out_dir, "dataset-metadata.json")
    if not os.path.exists(meta_path):
        import json as _json

        slug = os.path.basename(os.path.normpath(out_dir)) or "cartoonstyle-out"
        meta = {"title": slug, "licenses": [{"name": "Apache-2.0"}]}
        username = resolve_kaggle_user()
        if not username:
            print("[pusher] no Kaggle credentials found - staying local-only "
                  "(no cloud backup this run)", flush=True)
        if username:
            meta["id"] = f"{username}/{slug}"
        with open(meta_path, "w", encoding="utf-8") as f:
            _json.dump(meta, f, indent=2)

    try:
        help_txt = subprocess.run(["kaggle", "datasets", "create", "-h"],
                                  capture_output=True, text=True, timeout=60).stdout
    except Exception:
        help_txt = ""

    common = ["-p", out_dir]
    if "--dir-mode" in help_txt:
        common += ["--dir-mode", "zip"]

    create_cmd = ["kaggle", "datasets", "create"] + common
    if "--private" in help_txt:  # 1.x only; 2.x would reject the flag
        create_cmd.append("--private")
    version_cmd = ["kaggle", "datasets", "version"] + common + ["-m", message]

    for cmd in (version_cmd, create_cmd):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        except Exception as e:  # noqa: BLE001
            print(f"[kaggle] {cmd[1]} error: {e}", flush=True)
            return False
        if r.returncode == 0:
            return True
        err = (r.stderr or r.stdout or "").strip().replace("\n", " ")[:300]
        print(f"[kaggle] {cmd[1]} failed: {err}", flush=True)
    return False


class Pusher:
    """Background thread: zip new images and push to Kaggle on a schedule."""

    def __init__(
        self,
        out_dir: str,
        every_images: int = 50,
        every_seconds: int = 600,
        enable_kaggle: bool = True,
    ):
        self.out_dir = out_dir
        self.every_images = max(1, int(every_images))
        self.every_seconds = max(60, int(every_seconds))
        self.enable_kaggle = enable_kaggle
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pushed: set[str] = set()
        self._last_push = 0.0

    def start(self) -> "Pusher":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _flush(self, message: str) -> None:
        files = list_new_images(self.out_dir)
        new = [f for f in files if f not in self._pushed]
        if not new:
            return
        update_zip(self.out_dir, new)
        # keep the ledgers beside the image backup so a wiped session can resume
        for name in ("manifest.jsonl", "qc.jsonl"):
            s = os.path.join(self.out_dir, name)
            if os.path.exists(s):
                shutil.copy(s, os.path.join(self.out_dir, "snapshots", name))
        self._pushed.update(new)
        self._last_push = time.time()
        print(f"[pusher] {len(new)} new images -> snapshots/images.zip", flush=True)
        if self.enable_kaggle:
            ok = kaggle_push(self.out_dir, message)
            print(f"[pusher] kaggle dataset push {'ok' if ok else 'skipped/failed'}", flush=True)

    def _loop(self) -> None:
        while not self._stop.wait(30):
            files = list_new_images(self.out_dir)
            new = [f for f in files if f not in self._pushed]
            due = (
                len(new) >= self.every_images
                or (new and time.time() - self._last_push >= self.every_seconds)
            )
            if due:
                self._flush("auto-save")

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=60)
        self._flush("final-save")
