# Notebook embeds FINAL_EXPORT_SOURCE and FINAL_SQL_SOURCE before this code.
from pathlib import Path
from datetime import datetime, timezone
from google.colab import drive
import json
import shutil
import subprocess
import sys

assert sys.platform == "linux" and Path("/content").is_dir(), "Use the existing L4 Colab runtime."
drive.mount("/content/drive")
TRAIN_RUN = Path("/content/drive/MyDrive/NYU_NLP_HW4/part2/continuation_runs/20261110T140531341373Z")
TRAIN_REPORT = json.loads((TRAIN_RUN / "final_report.json").read_text())
assert TRAIN_REPORT["state"] == "COMPLETED" and TRAIN_REPORT["best_epoch"] == 17
FINAL_WORK = Path("/content") / f"hw4_t5_continue_{TRAIN_RUN.name}"
PYTHON2 = "/content/hw4-part2-env/bin/python"
assert FINAL_WORK.is_dir() and Path(PYTHON2).is_file(), (
    "This local runtime was replaced. Stop here and ask to restore the epoch 17 checkpoint first."
)
settings = json.loads((TRAIN_RUN / "configuration.json").read_text())
checkpoint = FINAL_WORK / "checkpoints/ft_experiments" / settings["experiment_name"] / "best"
assert json.loads((checkpoint / "checkpoint_meta.json").read_text())["epoch"] == 17
assert (checkpoint / "model.safetensors").is_file()
for existing in (TRAIN_RUN.parent.parent / "final_exports").glob("*/export_report.json"):
    previous = json.loads(existing.read_text())
    if previous.get("state") == "COMPLETED" and previous.get("selected_checkpoint") == str(checkpoint):
        raise RuntimeError(f"Final test files already exist at {existing.parent}. Inspect those files; do not regenerate.")
subprocess.run([PYTHON2, "-c", "import torch; assert torch.cuda.is_available(); print('GPU:',torch.cuda.get_device_name(0))"], check=True)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
FINAL_EXPORT = TRAIN_RUN.parent.parent / "final_exports" / stamp
FINAL_EXPORT.mkdir(parents=True, exist_ok=False)
FINAL_HELPERS = FINAL_WORK / "final_export_helpers" / stamp
FINAL_HELPERS.mkdir(parents=True, exist_ok=False)
FINAL_SCRIPT = FINAL_HELPERS / "export_dev_evidence_and_test.py"
FINAL_SCRIPT.write_text(FINAL_EXPORT_SOURCE)
sql_helper = FINAL_HELPERS / "final_bounded_sql.py"
sql_helper.write_text(FINAL_SQL_SOURCE)
for path in [FINAL_SCRIPT, sql_helper]:
    shutil.copy2(path, FINAL_EXPORT / path.name)
FINAL_CONTEXT = FINAL_EXPORT / "export_context.json"
FINAL_CONTEXT.write_text(json.dumps({
    "work_directory": str(FINAL_WORK), "export_directory": str(FINAL_EXPORT),
    "training_run": str(TRAIN_RUN), "training_report": TRAIN_REPORT,
    "configuration": settings, "best_checkpoint_directory": str(checkpoint),
}, indent=2))
(FINAL_EXPORT / "status.json").write_text(json.dumps({"state":"PREPARED_NOT_STARTED",
    "best_epoch":17,"export_directory":str(FINAL_EXPORT)}, indent=2))
print("FIXED MODEL: epoch 17; dev Record F1 87.25%. No model changes.")
print("EXPORT FOLDER:", FINAL_EXPORT)
print("Preparation complete. Test generation has NOT started.")
print("Run Section 2 once to export Q6 evidence and generate the two Q7 files.")
