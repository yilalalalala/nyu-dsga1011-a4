# Notebook embeds CONTINUATION_SOURCE and BOUNDED_SQL_SOURCE before this code.
from pathlib import Path
from datetime import datetime, timezone
from google.colab import drive
import importlib.util
import json
import shutil
import subprocess
import sys

assert sys.platform == "linux" and Path("/content").is_dir(), "Use your existing L4 Colab runtime."
drive.mount("/content/drive")
RECOVERY = Path("/content/drive/MyDrive/NYU_NLP_HW4/part2/recovery_runs/20261110T132529577782Z")
report = json.loads((RECOVERY / "recovery_report.json").read_text())
context = json.loads((RECOVERY / "recovery_context.json").read_text())
assert report["state"] == "COMPLETED" and report["latest_training_epoch"] == 4
recovered_work = Path(context["work_directory"])
original = Path(context["original_run"])
PYTHON2 = "/content/hw4-part2-env/bin/python"
assert Path(PYTHON2).is_file() and recovered_work.is_dir(), (
    "The recovered local environment is gone. Stop here; do not start from pretrained weights."
)
subprocess.run([PYTHON2, "-c", "import torch; assert torch.cuda.is_available(); print('GPU:',torch.cuda.get_device_name(0))"], check=True)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
T5_WORK = Path("/content") / f"hw4_t5_continue_{stamp}"
T5_BACKUP = original.parent.parent / "continuation_runs" / stamp
T5_WORK.mkdir(exist_ok=False)
T5_BACKUP.mkdir(parents=True, exist_ok=False)

spec = importlib.util.spec_from_file_location("continuation_restore_tools", recovered_work / "recovery_tools.py")
tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)
tools.restore_source(original, T5_WORK)
settings = dict(context["configuration"])
settings.update(
    max_n_epochs=16, completed_epochs_before_restart=4,
    baseline_dev_sql=report["checkpoints"]["last"]["sql_path"],
    baseline_dev_loss=report["checkpoints"]["last"]["dev_loss"],
    baseline_recovery_f1=report["checkpoints"]["last"]["record_f1"],
    optimizer_restarted=True, exact_resume=False,
    recovery_report=str(RECOVERY / "recovery_report.json"),
)
source_checkpoint = Path(context["checkpoints"]["last"]["local_directory"])
assert json.loads((source_checkpoint / "checkpoint_meta.json").read_text())["epoch"] == 4
checkpoint_root = T5_WORK / "checkpoints/ft_experiments" / settings["experiment_name"]
for name in ["best", "last"]:
    shutil.copytree(source_checkpoint, checkpoint_root / name)
    assert tools.sha256_file(checkpoint_root / name / "model.safetensors") == tools.sha256_file(source_checkpoint / "model.safetensors")
shutil.copytree(original / "source", T5_BACKUP / "source")
shutil.copy2(original / "source_sha256.json", T5_BACKUP / "source_sha256.json")
for name in ["environment.json", "pip_freeze.txt"]:
    if (RECOVERY / name).is_file():
        shutil.copy2(RECOVERY / name, T5_BACKUP / name)
(T5_BACKUP / "configuration.json").write_text(json.dumps(settings, indent=2))
(T5_BACKUP / "recovered_baseline_report.json").write_text(json.dumps(report, indent=2))
T5_RUNNER = T5_WORK / "continue_training.py"
T5_RUNNER.write_text(CONTINUATION_SOURCE)
sql_runtime = T5_WORK / "bounded_sql_runtime.py"
sql_runtime.write_text(BOUNDED_SQL_SOURCE)
for path in [T5_RUNNER, sql_runtime]:
    shutil.copy2(path, T5_BACKUP / path.name)

# A 1-second runaway-query check verifies cleanup without training or modifying the DB.
spec = importlib.util.spec_from_file_location("sql_cleanup_smoke", sql_runtime)
sql_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sql_module)
rows, errors = sql_module.compute_records_bounded([
    "SELECT 1",
    "WITH RECURSIVE t(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM t) SELECT sum(x) FROM t",
], T5_WORK / "data/flight_database.db", 1)
assert rows[0] == [(1,)] and not errors[0] and errors[1] == "Query timed out"
(T5_BACKUP / "status.json").write_text(json.dumps({
    "state": "PREPARED_NOT_STARTED", "starting_epoch": 4,
    "next_epoch": 5, "backup_folder": str(T5_BACKUP),
    "optimizer_restarted": True, "exact_resume": False,
}, indent=2))
print("SQL TIMEOUT CLEANUP CHECK PASSED")
print("READY TO CONTINUE FROM EPOCH 4. Training has NOT started.")
print("Original/recovery Drive backups unchanged.")
print("NEW CONTINUATION BACKUP:", T5_BACKUP)
print("Run Section 2 ONCE. It rechecks saved SQL (up to 120 seconds), then starts EPOCH 5/20.")
