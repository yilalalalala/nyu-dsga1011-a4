"""Reload and evaluate saved best/latest checkpoints. NO training or held-out test."""

import gc
import json
import math
from pathlib import Path
import pickle
import sys
import time
import traceback
from datetime import datetime, timezone
from types import SimpleNamespace

import torch
from transformers import T5TokenizerFast


def main(context_path):
    context = json.loads(Path(context_path).read_text())
    work = Path(context["work_directory"]).resolve()
    backup = Path(context["recovery_backup"]).resolve()
    assert Path.cwd().resolve() == work, "Run with cwd set to the restored working folder."
    sys.path.insert(0, str(work))
    from load_data import get_dataloader, preprocess_sql
    from t5_utils import load_model_from_checkpoint
    from train_t5 import eval_epoch
    from utils import set_random_seeds

    args = SimpleNamespace(**context["configuration"])
    assert args.finetune
    assert torch.cuda.is_available(), "Select your L4 (or another CUDA GPU)."
    set_random_seeds(args.seed)
    started = time.monotonic()
    report = {"original_run": context["original_run"], "recovery_backup": str(backup),
              "work_directory": str(work), "gpu": torch.cuda.get_device_name(0),
              "training_updates": 0, "held_out_test_used": False,
              "checkpoints": {}}

    def update(state, stage, **extra):
        path = backup / "status.json"
        status = json.loads(path.read_text()) if path.is_file() else {}
        status.update(state=state, stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(),
                      training_updates=0, held_out_test_used=False, **extra)
        path.write_text(json.dumps(status, indent=2))

    try:
        update("RUNNING", "loading_dev_data")
        # Only the tokenizer may need a small first-time Hugging Face download.
        # Model weights below load from local restored checkpoints only.
        dev_loader = get_dataloader(args.test_batch_size, "dev")
        assert len(dev_loader.dataset) == 466
        raw_nl = (work / "data/dev.nl").read_text().splitlines()
        raw_sql = (work / "data/dev.sql").read_text().splitlines()
        assert len(raw_nl) == len(raw_sql) == 466
        with (work / "records/ground_truth_dev.pkl").open("rb") as handle:
            gold_records, gold_errors = pickle.load(handle)
        assert len(gold_records) == len(gold_errors) == 466

        for name in ["best", "last"]:
            info = context["checkpoints"][name]
            checkpoint = Path(info["local_directory"])
            epoch = info["epoch"]
            update("RUNNING", f"loading_{name}_checkpoint", checkpoint_epoch=epoch)
            print(f"\nLOADING RESTORED {name.upper()} CHECKPOINT: epoch {epoch}", flush=True)
            tokenizer = T5TokenizerFast.from_pretrained(str(checkpoint), local_files_only=True)
            assert tokenizer.get_vocab() == dev_loader.dataset.tokenizer.get_vocab()
            # Ensure saved and freshly cached tokenizer behavior agrees on all dev data.
            assert tokenizer(raw_nl, padding=False, truncation=False)["input_ids"] == dev_loader.dataset.input_ids
            assert tokenizer([preprocess_sql(q) for q in raw_sql], padding=False, truncation=False)["input_ids"] == dev_loader.dataset.target_ids
            model = load_model_from_checkpoint(args, best=(name == "best"))
            assert next(model.parameters()).device.type == "cuda"
            assert model.config.model_type == "t5" and model.config.d_model == 512
            assert model.config.decoder_start_token_id == model.config.pad_token_id == 0
            assert model.config.eos_token_id == 1
            parameters = sum(p.numel() for p in model.parameters())
            assert parameters == 60_506_624
            print(f"Loaded {parameters:,} parameters on {report['gpu']}; tokenizer verified.", flush=True)
            signature = model.shared.weight[:16, :16].detach().clone()
            update("RUNNING", f"evaluating_{name}_on_dev", checkpoint_epoch=epoch)
            sql_path = backup / f"{name}_epoch_{epoch}_dev.sql"
            records_path = backup / f"{name}_epoch_{epoch}_dev.pkl"
            metrics = eval_epoch(
                args, model, dev_loader, str(work / "data/dev.sql"), str(sql_path),
                str(work / "records/ground_truth_dev.pkl"), str(records_path),
            )
            loss, f1, record_em, sql_em, error_rate = map(float, metrics)
            assert all(math.isfinite(v) for v in [loss, f1, record_em, sql_em, error_rate])
            assert all(0 <= v <= 1 for v in [f1, record_em, sql_em, error_rate])
            assert len(sql_path.read_text().splitlines()) == 466
            with records_path.open("rb") as handle:
                records, errors = pickle.load(handle)
            assert len(records) == len(errors) == 466
            assert not model.training and all(p.grad is None for p in model.parameters())
            assert torch.equal(signature, model.shared.weight[:16, :16].detach())
            result = {"epoch": epoch, "parameters": parameters, "dev_examples": 466,
                      "dev_loss": loss, "record_f1": f1, "record_em": record_em,
                      "sql_em": sql_em, "sql_error_rate": error_rate,
                      "sql_path": str(sql_path), "records_path": str(records_path)}
            previous = info["metadata"].get("record_f1")
            if previous is not None:
                result["original_saved_record_f1"] = previous
                result["absolute_f1_difference_after_gpu_change"] = abs(f1 - previous)
                if abs(f1 - previous) > 1e-6:
                    print("NOTE: re-evaluated F1 differs from the old GPU result; inspect before continuing.", flush=True)
            report["checkpoints"][name] = result
            (backup / "recovery_results_partial.json").write_text(json.dumps(report, indent=2))
            print(f"RECOVERED {name.upper()} EPOCH {epoch}: dev loss={loss:.4f}, "
                  f"Record F1={100*f1:.2f}%, Record EM={100*record_em:.2f}%, "
                  f"SQL EM={100*sql_em:.2f}%, SQL errors={100*error_rate:.2f}%", flush=True)
            del model, tokenizer, signature
            gc.collect()
            torch.cuda.empty_cache()

        best_candidate = max(report["checkpoints"], key=lambda name: report["checkpoints"][name]["record_f1"])
        report.update(state="COMPLETED", seconds=time.monotonic() - started,
                      highest_dev_f1_checkpoint=best_candidate,
                      highest_dev_f1_epoch=report["checkpoints"][best_candidate]["epoch"],
                      latest_training_epoch=context["checkpoints"]["last"]["epoch"],
                      optimizer_state_available=False, exact_training_resume_available=False)
        (backup / "recovery_report.json").write_text(json.dumps(report, indent=2))
        update("COMPLETED", "both_checkpoints_reloaded_and_dev_evaluated")
        print("\nPART 2 RECOVERY AND DEV CHECK PASSED", flush=True)
        print(json.dumps(report, indent=2), flush=True)
        print("No training updates. No held-out test inputs used. Original Drive run unchanged.")
        print("Save this notebook and ask for review before continuing training.")
    except BaseException:
        update("FAILED", "model_reload_or_dev_check_failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    main(sys.argv[1])
