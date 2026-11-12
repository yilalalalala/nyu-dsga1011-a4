# Operational notebook preparation. The notebook embeds RUNNER_SOURCE above this.
from pathlib import Path
from datetime import datetime, timezone
from google.colab import drive
import hashlib
import json
import shutil
import subprocess

PART2 = Path("/content/part2")
PYTHON2 = "/content/hw4-part2-env/bin/python"
assert Path(PYTHON2).is_file(), "Use the same Part 2 Colab runtime/environment."
for name in ["load_data.py", "t5_utils.py", "train_t5.py", "utils.py", "requirements.txt"]:
    assert (PART2 / name).is_file(), f"Upload part2 first: missing {name}"
subprocess.run([PYTHON2, "-c", "import torch; assert torch.cuda.is_available(), 'Connect to a GPU'; print('GPU:', torch.cuda.get_device_name(0))"], check=True)
assert shutil.disk_usage("/content").free > 2 * 1024**3, "Need at least 2 GiB free local disk."
drive.mount("/content/drive")
base = Path("/content/drive/MyDrive/NYU_NLP_HW4/part2")

# Require the exact source files that passed the evaluation check.
passed = []
for candidate in sorted((base / "q5_eval_checks").glob("*"), reverse=True):
    status_file = candidate / "status.json"
    if status_file.is_file() and json.loads(status_file.read_text()).get("state") == "COMPLETED":
        passed.append(candidate)
assert passed, "First finish Section 9 in HW4_Part2_Colab.ipynb."
checked = passed[0]
for name in ["load_data.py", "t5_utils.py", "train_t5.py", "utils.py"]:
    assert (PART2 / name).read_bytes() == (checked / name).read_bytes(), (
        f"{name} differs from the source that passed Section 9. Re-run that check first."
    )

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
T5_WORK = Path("/content") / f"hw4_t5_train_{stamp}"
T5_BACKUP = base / "training_runs" / stamp
T5_WORK.mkdir(exist_ok=False)
T5_BACKUP.mkdir(parents=True, exist_ok=False)
source_backup = T5_BACKUP / "source"
source_backup.mkdir()
for source in PART2.glob("*.py"):
    shutil.copy2(source, T5_WORK / source.name)
    shutil.copy2(source, source_backup / source.name)
shutil.copy2(PART2 / "requirements.txt", source_backup / "requirements.txt")
shutil.copytree(PART2 / "data", T5_WORK / "data")
shutil.copytree(PART2 / "data", source_backup / "data")
(T5_WORK / "records").mkdir()
shutil.copy2(PART2 / "records/ground_truth_dev.pkl", T5_WORK / "records/ground_truth_dev.pkl")
shutil.copy2(PART2 / "records/ground_truth_dev.pkl", source_backup / "ground_truth_dev.pkl")
for name in ["environment.json", "pip_freeze.txt"]:
    artifact = Path("/content/hw4-part2-artifacts") / name
    if artifact.is_file():
        shutil.copy2(artifact, T5_BACKUP / name)

T5_SETTINGS = {
    "finetune": True, "experiment_name": "experiment", "seed": 42,
    "optimizer_type": "AdamW", "learning_rate": 3e-4, "weight_decay": 0.0,
    "scheduler_type": "linear", "num_warmup_epochs": 1,
    "max_n_epochs": 20, "patience_epochs": 4,
    "batch_size": 8, "test_batch_size": 8, "use_wandb": False,
    "max_new_tokens": 512, "num_beams": 1,
}
(T5_BACKUP / "configuration.json").write_text(json.dumps(T5_SETTINGS, indent=2))
manifest = {
    str(p.relative_to(source_backup)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in source_backup.rglob("*") if p.is_file()
}
(T5_BACKUP / "source_sha256.json").write_text(json.dumps(manifest, indent=2))
(T5_BACKUP / "passed_evaluation_check.txt").write_text(str(checked))
T5_RUNNER = T5_WORK / "run_with_drive_backup.py"
T5_RUNNER.write_text(RUNNER_SOURCE)
shutil.copy2(T5_RUNNER, T5_BACKUP / T5_RUNNER.name)
(T5_BACKUP / "status.json").write_text(json.dumps({
    "state": "PREPARED_NOT_STARTED", "backup_folder": str(T5_BACKUP),
    "working_folder": str(T5_WORK), "configuration": T5_SETTINGS,
}, indent=2))
print("Preparation complete. TRAINING HAS NOT STARTED.")
print("Working folder:", T5_WORK)
print("SAVE THIS DRIVE LOCATION:", T5_BACKUP)
print(json.dumps(T5_SETTINGS, indent=2))
print("Run the next cell ONCE to start full training. Keep at least ~1 GB available in Drive.")
