# The notebook embeds RECOVERY_TOOLS_SOURCE before this preparation code.
from pathlib import Path
from datetime import datetime, timezone
from google.colab import drive
import importlib.util
import json
import shutil
import subprocess
import sys
import traceback

assert sys.platform == "linux" and Path("/content").exists(), "Select Colab, not Mac Python."
drive.mount("/content/drive")
ORIGINAL_RUN = Path("/content/drive/MyDrive/NYU_NLP_HW4/part2/training_runs/20261110T032045677993Z")
for name in ["source_sha256.json", "configuration.json", "checkpoints/best.tar", "checkpoints/last.tar"]:
    assert (ORIGINAL_RUN / name).is_file(), f"Missing original backup: {name}. Stop and ask for help."
assert shutil.disk_usage("/content").free > 6 * 1024**3, "Need at least 6 GiB local space for environment/models."
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
RECOVERY_WORK = Path("/content") / f"hw4_t5_recovery_{stamp}"
RECOVERY_BACKUP = ORIGINAL_RUN.parent.parent / "recovery_runs" / stamp
RECOVERY_WORK.mkdir(exist_ok=False)
RECOVERY_BACKUP.mkdir(parents=True, exist_ok=False)
ENV2 = Path("/content/hw4-part2-env")
PYTHON2 = str(ENV2 / "bin/python")

def recovery_status(state, stage, **extra):
    path = RECOVERY_BACKUP / "status.json"
    status = json.loads(path.read_text()) if path.is_file() else {}
    status.update(state=state, stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(),
                  original_run=str(ORIGINAL_RUN), recovery_work=str(RECOVERY_WORK),
                  recovery_backup=str(RECOVERY_BACKUP), training_updates=0, held_out_test_used=False,
                  **extra)
    path.write_text(json.dumps(status, indent=2))

def run_recovery_command(command, log_name):
    print("\nRunning:", " ".join(map(str, command)), flush=True)
    with (RECOVERY_BACKUP / log_name).open("a", buffering=1) as log:
        process = subprocess.Popen(list(map(str, command)), cwd=RECOVERY_WORK,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)
            raise
        finally:
            process.stdout.close()
        if code:
            raise RuntimeError(f"Command exited with code {code}. See {log_name} in the recovery folder.")

recovery_status("RUNNING", "restoring_source_and_checkpoints")
try:
    tools_file = RECOVERY_WORK / "recovery_tools.py"
    tools_file.write_text(RECOVERY_TOOLS_SOURCE)
    shutil.copy2(tools_file, RECOVERY_BACKUP / tools_file.name)
    spec = importlib.util.spec_from_file_location("hw4_recovery_tools", tools_file)
    recovery_tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recovery_tools)
    source_info = recovery_tools.restore_source(ORIGINAL_RUN, RECOVERY_WORK)
    print("Restored code/data; SHA-256 verified files:", source_info["verified_files"], flush=True)
    settings = json.loads((ORIGINAL_RUN / "configuration.json").read_text())
    assert settings["finetune"] is True
    experiment = settings["experiment_name"]
    assert experiment and Path(experiment).name == experiment and experiment not in {".", ".."}
    root = RECOVERY_WORK / "checkpoints/ft_experiments" / experiment
    checkpoints = {}
    for name in ["best", "last"]:
        print(f"Copying and extracting {name}.tar from Drive...", flush=True)
        checkpoints[name] = recovery_tools.restore_checkpoint(
            ORIGINAL_RUN / "checkpoints" / f"{name}.tar",
            RECOVERY_WORK / "archives" / f"{name}.tar", root, name,
        )
        print(f"RESTORED {name}: epoch {checkpoints[name]['epoch']}", flush=True)
    context = {"original_run": str(ORIGINAL_RUN), "work_directory": str(RECOVERY_WORK),
               "recovery_backup": str(RECOVERY_BACKUP), "configuration": settings,
               "source_verification": source_info, "checkpoints": checkpoints,
               "training_updates": 0, "held_out_test_used": False}
    CONTEXT_PATH = RECOVERY_BACKUP / "recovery_context.json"
    CONTEXT_PATH.write_text(json.dumps(context, indent=2))
    recovery_status("RESTORED_NOT_YET_MODEL_CHECKED", "source_and_archives_restored")
except BaseException:
    recovery_status("FAILED", "source_or_archive_restore_failed", error=traceback.format_exc())
    raise
print("\nSOURCE AND CHECKPOINT ARCHIVES RESTORED. No training has started.")
print("Local working folder:", RECOVERY_WORK)
print("NEW RECOVERY LOG/REPORT FOLDER:", RECOVERY_BACKUP)
print("Original Drive backup has NOT been changed. Next run Section 2 to install the environment.")
