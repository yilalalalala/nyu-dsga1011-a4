# Run once. This is the long-running cell that actually trains the model.
import json
import subprocess
from datetime import datetime, timezone

assert "T5_WORK" in globals() and T5_RUNNER.is_file(), "Run preparation first."
assert json.loads((T5_BACKUP / "status.json").read_text())["state"] == "PREPARED_NOT_STARTED", (
    "This run was already started. Do not accidentally train twice."
)
with (T5_WORK / ".training_started").open("x") as marker:
    marker.write(datetime.now(timezone.utc).isoformat())

print("STARTING FULL TRAINING. Drive backup:", T5_BACKUP, flush=True)
print("The first Drive model checkpoint appears after the first training epoch.", flush=True)
process = None
try:
    with (T5_BACKUP / "training.log").open("w", buffering=1) as log:
        process = subprocess.Popen(
            [PYTHON2, "-u", str(T5_RUNNER), str(T5_BACKUP)],
            cwd=T5_WORK, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"Training process exited with code {return_code}. Inspect training.log.")
except BaseException:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)
    status_path = T5_BACKUP / "status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    status.update(state="FAILED_OR_INTERRUPTED", updated_utc=datetime.now(timezone.utc).isoformat())
    status_path.write_text(json.dumps(status, indent=2))
    print("Stopped or failed. Completed Drive checkpoints/logs remain at:", T5_BACKUP)
    raise
else:
    assert json.loads((T5_BACKUP / "status.json").read_text())["state"] == "COMPLETED"
    print("\nTRAINING COMPLETE. Everything needed for review is in:", T5_BACKUP)
    print("Save this notebook. Do not rerun training; ask for review of the dev results.")
