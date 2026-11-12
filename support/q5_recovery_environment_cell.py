# Rebuild the same Python 3.11 / Part 2 package versions, separate from kernel Python.
assert "RECOVERY_WORK" in globals() and (RECOVERY_WORK / "requirements.txt").is_file(), "Run Section 1 first."
recovery_status("RUNNING", "installing_part2_environment")
try:
    run_recovery_command([sys.executable, "-m", "pip", "install", "uv"], "environment_install.log")
    uv = [sys.executable, "-m", "uv"]
    run_recovery_command(uv + ["python", "install", "3.11"], "environment_install.log")
    if not Path(PYTHON2).is_file():
        run_recovery_command(uv + ["venv", "--python", "3.11", "--seed", str(ENV2)], "environment_install.log")
    run_recovery_command(uv + ["pip", "install", "--python", PYTHON2,
                              "-r", str(RECOVERY_WORK / "requirements.txt")], "environment_install.log")
    run_recovery_command(uv + ["pip", "check", "--python", PYTHON2], "environment_install.log")
    ENVIRONMENT_CHECK = '''import importlib.metadata, json, sys
from pathlib import Path
from packaging.requirements import Requirement
import torch
assert sys.version_info[:2] == (3, 11)
versions = {}
for line in Path(sys.argv[1]).read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"): continue
    requirement = Requirement(line)
    actual = importlib.metadata.version(requirement.name)
    assert requirement.specifier.contains(actual, prereleases=True), (line, actual)
    versions[requirement.name] = actual
assert torch.cuda.is_available(), "Select your Colab GPU runtime."
x = torch.ones((16, 16), device="cuda")
assert (x @ x).sum().item() == 4096
report = {"python": sys.version, "versions": versions, "gpu": torch.cuda.get_device_name(0)}
Path(sys.argv[2]).write_text(json.dumps(report, indent=2))
print("RECOVERY GPU:", report["gpu"])
print("PART 2 RECOVERY ENVIRONMENT READY")
'''
    run_recovery_command([PYTHON2, "-u", "-c", ENVIRONMENT_CHECK,
                          str(RECOVERY_WORK / "requirements.txt"),
                          str(RECOVERY_BACKUP / "environment.json")], "environment_install.log")
    freeze = subprocess.run([PYTHON2, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (RECOVERY_BACKUP / "pip_freeze.txt").write_text(freeze.stdout)
    recovery_status("ENVIRONMENT_READY", "environment_ready_no_training")
except BaseException:
    recovery_status("FAILED", "environment_setup_failed", error=traceback.format_exc())
    raise
print("Next run Section 3 to reload and evaluate the two saved models. No training has started.")
