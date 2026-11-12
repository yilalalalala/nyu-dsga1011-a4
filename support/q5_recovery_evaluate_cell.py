# The notebook embeds RECOVERY_EVALUATE_SOURCE before this code.
assert "RECOVERY_WORK" in globals() and Path(PYTHON2).is_file(), "Run Sections 1 and 2 first."
assert (RECOVERY_BACKUP / "environment.json").is_file(), "Complete the Section 2 GPU/environment check first."
state = json.loads((RECOVERY_BACKUP / "status.json").read_text())["state"]
assert state != "COMPLETED", "This recovery check already completed; inspect its report instead of rerunning."
assert state in {"ENVIRONMENT_READY", "FAILED"}, "Complete environment setup first."
script = RECOVERY_WORK / "check_recovered_models.py"
script.write_text(RECOVERY_EVALUATE_SOURCE)
shutil.copy2(script, RECOVERY_BACKUP / script.name)
try:
    run_recovery_command([PYTHON2, "-u", str(script), str(CONTEXT_PATH)], "recovery_evaluation.log")
except BaseException:
    recovery_status("FAILED", "recovery_evaluation_failed_or_interrupted", error=traceback.format_exc())
    print("Existing recovery logs/results remain at:", RECOVERY_BACKUP)
    raise
else:
    assert json.loads((RECOVERY_BACKUP / "status.json").read_text())["state"] == "COMPLETED"
    print("\nRECOVERY REPORT AND LOGS:", RECOVERY_BACKUP)
    print("Save this notebook and ask for review. No training has been started.")
