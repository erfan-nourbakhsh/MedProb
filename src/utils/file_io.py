import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

def load_json(file_path: str) -> list[dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: list[dict[str, Any]], file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_jsonl(file_path: str) -> list[dict[str, Any]]:
    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results

def append_jsonl(file_path: str, obj: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def ensure_header(file_path: str, header: list[str], sep: str = "\t") -> None:
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(sep.join(header) + "\n")

def append_row(file_path: str, row: list[str], sep: str = "\t") -> None:
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(sep.join(row) + "\n")

def _save_npz_atomic(path: str, arrays: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.stem}.",
        suffix=".tmp",
    )
    os.close(fd)
    try:
        np.savez(tmp_path, **arrays)
        saved_tmp_path = f"{tmp_path}.npz"
        with open(saved_tmp_path, "rb") as tmp_file:
            os.fsync(tmp_file.fileno())
        os.replace(saved_tmp_path, path)
    except Exception:
        for candidate in (tmp_path, f"{tmp_path}.npz"):
            if os.path.exists(candidate):
                os.remove(candidate)
        raise

def load_npz_with_integrity_check(path: str) -> np.lib.npyio.NpzFile:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        with zipfile.ZipFile(path) as zf:
            bad_member = zf.testzip()
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"Corrupted NPZ archive: {path}. The file is not a valid ZIP/NPZ archive. "
            f"Regenerate or replace this embeddings file."
        ) from exc
    if bad_member is not None:
        raise RuntimeError(
            f"Corrupted NPZ archive: {path}. CRC check failed for member '{bad_member}'. "
            "This usually means the file was partially written or copied. "
            "Regenerate or replace this embeddings file."
        )
    try:
        return np.load(path)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"Corrupted NPZ archive: {path}. NumPy could not read the archive contents. "
            "Regenerate or replace this embeddings file."
        ) from exc

def save_embeddings(train_embed: dict, test_embed: dict, save_path: str) -> None:
    os.makedirs(save_path, exist_ok=True)
    _save_npz_atomic(os.path.join(save_path, "train_embeddings.npz"), train_embed)
    _save_npz_atomic(os.path.join(save_path, "test_embeddings.npz"), test_embed)
