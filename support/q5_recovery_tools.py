"""Filesystem-only recovery helpers. Never overwrite the original Drive run."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def child_path(root, relative):
    root = Path(root).resolve()
    relative = PurePosixPath(relative)
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
        raise ValueError(f"Unsafe relative path: {relative}")
    result = (root / str(relative)).resolve()
    if result != root and root not in result.parents:
        raise ValueError(f"Path escapes recovery root: {relative}")
    return result


def restore_source(original_run, work):
    original_run, work = Path(original_run), Path(work)
    source = original_run / "source"
    manifest = json.loads((original_run / "source_sha256.json").read_text())
    if not manifest:
        raise ValueError("Empty source manifest.")
    for name, expected in manifest.items():
        src = child_path(source, name)
        dst = child_path(work, name)
        if not src.is_file() or sha256_file(src) != expected:
            raise ValueError(f"Original source hash mismatch: {name}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("xb") as handle, src.open("rb") as incoming:
            shutil.copyfileobj(incoming, handle)
        if sha256_file(dst) != expected:
            raise ValueError(f"Restored source hash mismatch: {name}")
    # The original backup deliberately stored this pickle at the source root.
    records = work / "records"
    records.mkdir(exist_ok=True)
    with (records / "ground_truth_dev.pkl").open("xb") as handle:
        handle.write((work / "ground_truth_dev.pkl").read_bytes())
    for name in ["load_data.py", "t5_utils.py", "train_t5.py", "utils.py", "requirements.txt",
                 "data/train.nl", "data/train.sql", "data/dev.nl", "data/dev.sql",
                 "data/flight_database.db", "records/ground_truth_dev.pkl"]:
        if not (work / name).is_file():
            raise FileNotFoundError(f"Incomplete restored source: {name}")
    return {"verified_files": len(manifest), "source_manifest": str(original_run / "source_sha256.json")}


def restore_checkpoint(archive_path, local_archive, checkpoint_root, name):
    if name not in {"best", "last"}:
        raise ValueError(name)
    archive_path, local_archive = Path(archive_path), Path(local_archive)
    local_archive.parent.mkdir(parents=True, exist_ok=True)
    with local_archive.open("xb") as target, archive_path.open("rb") as source:
        shutil.copyfileobj(source, target)
    if local_archive.stat().st_size != archive_path.stat().st_size:
        raise ValueError("Checkpoint archive copy is incomplete.")
    checkpoint_root = Path(checkpoint_root)
    if (checkpoint_root / name).exists():
        raise FileExistsError(f"Refusing to overwrite restored checkpoint: {name}")
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(local_archive, "r:") as archive:
        members = archive.getmembers()
        seen = set()
        for member in members:
            path = PurePosixPath(member.name)
            child_path(checkpoint_root, member.name)
            if not path.parts or path.parts[0] != name:
                raise ValueError(f"Unexpected archive root: {member.name}")
            if not (member.isdir() or member.isfile()) or member.name in seen:
                raise ValueError(f"Unsafe/duplicate tar member: {member.name}")
            seen.add(member.name)
        required = {f"{name}/{p}" for p in
                    ["config.json", "model.safetensors", "tokenizer.json", "checkpoint_meta.json"]}
        if not required.issubset(seen):
            raise ValueError("Archive lacks required model/tokenizer/metadata files.")
        for member in members:
            target = child_path(checkpoint_root, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            if target.stat().st_size != member.size:
                raise ValueError(f"Incomplete extracted file: {member.name}")
    checkpoint = checkpoint_root / name
    metadata = json.loads((checkpoint / "checkpoint_meta.json").read_text())
    if not isinstance(metadata.get("epoch"), int) or metadata["epoch"] < 1:
        raise ValueError("Checkpoint epoch metadata is invalid.")
    return {"name": name, "epoch": metadata["epoch"], "metadata": metadata,
            "original_archive": str(archive_path), "local_directory": str(checkpoint),
            "archive_bytes": local_archive.stat().st_size,
            "local_archive_sha256": sha256_file(local_archive)}
