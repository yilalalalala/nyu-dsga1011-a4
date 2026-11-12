# Read-only: safe to run after reconnecting, without re-running training.
from pathlib import Path
from google.colab import drive
import json

drive.mount("/content/drive")
if "T5_BACKUP" not in globals():
    runs = sorted(Path("/content/drive/MyDrive/NYU_NLP_HW4/part2/training_runs").glob("*"))
    assert runs, "No saved Part 2 training runs found."
    T5_BACKUP = runs[-1]
print("Inspecting:", T5_BACKUP)
for name in ["status.json", "best_metrics.json", "final_report.json"]:
    path = T5_BACKUP / name
    if path.is_file():
        print("\n" + name + ":\n" + path.read_text())
for name in ["best.tar", "last.tar"]:
    path = T5_BACKUP / "checkpoints" / name
    print(name, f"{path.stat().st_size / 1024**2:.1f} MiB" if path.is_file() else "not yet saved")
log = T5_BACKUP / "training.log"
if log.is_file():
    print("\nLast log lines:\n" + "\n".join(log.read_text(errors="replace").splitlines()[-15:]))
print("A stale RUNNING status after runtime loss does not prove the process is still alive.")
