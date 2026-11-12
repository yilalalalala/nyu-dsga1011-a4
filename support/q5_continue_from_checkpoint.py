"""Continue recovered epoch 4 weights; isolated SQL workers; no homework-file edits.

Only train/dev are loaded. Checkpoint archives contain weights/config/tokenizer,
not optimizer state; they support later inference, not exact training resumption.
"""

import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
from datetime import datetime, timezone
from types import SimpleNamespace


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def publish_checkpoint(source, destination):
    """Finish a new archive before replacing this run's previous archive."""
    source, destination = Path(source), Path(destination)
    required = {"config.json", "model.safetensors", "tokenizer.json", "checkpoint_meta.json"}
    assert all((source / name).is_file() for name in required)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    with tarfile.open(temporary, "w") as archive:
        archive.add(source, arcname=source.name)
    with tarfile.open(temporary, "r") as archive:
        members = {member.name: member for member in archive.getmembers()}
        for name in required:
            member = members[f"{source.name}/{name}"]
            assert member.size == (source / name).stat().st_size
    # A partial archive never replaces the previous completed archive.
    os.replace(temporary, destination)
    return {"path": str(destination), "bytes": destination.stat().st_size}


def copy_predictions(sql_path, records_path, destination):
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sql_path, destination / "dev.sql")
    shutil.copy2(records_path, destination / "dev.pkl")


def run(backup):
    import torch
    from load_data import get_dataloader
    from t5_utils import (
        initialize_model, initialize_optimizer_and_scheduler,
        save_model, load_model_from_checkpoint,
    )
    from train_t5 import train_epoch, eval_epoch
    from utils import set_random_seeds
    import utils
    from bounded_sql_runtime import install_bounded_sql
    install_bounded_sql(utils)

    backup = Path(backup).resolve()
    settings = json.loads((backup / "configuration.json").read_text())
    args = SimpleNamespace(**settings)
    assert args.finetune and not args.use_wandb
    assert args.max_n_epochs > args.num_warmup_epochs >= 0
    assert args.patience_epochs > 0 and args.learning_rate > 0
    assert torch.cuda.is_available(), "Connect to a GPU before starting."
    set_random_seeds(args.seed)
    started = time.monotonic()
    status = {"state": "RUNNING", "stage": "initializing", "completed_epochs": 0,
              "backup_folder": str(backup), "held_out_test_used": False}

    def update(stage, **extra):
        status.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        write_json(backup / "status.json", status)

    try:
        update("loading_train_and_dev")
        # Deliberately avoid load_t5_data()/main(): no test dataset is loaded.
        train_loader = get_dataloader(args.batch_size, "train")
        dev_loader = get_dataloader(args.test_batch_size, "dev")
        assert len(train_loader.dataset) == 4225 and len(dev_loader.dataset) == 466
        model = load_model_from_checkpoint(args, best=False)
        base_epoch = args.completed_epochs_before_restart
        assert base_epoch == 4, "This continuation starts from the verified epoch 4 model."
        optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))
        checkpoint_dir = Path("checkpoints/ft_experiments") / args.experiment_name
        Path("results").mkdir(exist_ok=True)
        Path("records").mkdir(exist_ok=True)
        sql_path, records_path = "results/current_dev.sql", "records/current_dev.pkl"
        gt_sql, gt_records = "data/dev.sql", "records/ground_truth_dev.pkl"
        assert Path(gt_records).is_file()
        metadata = {
            "settings": settings, "gpu": torch.cuda.get_device_name(0),
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "train_examples": len(train_loader.dataset), "dev_examples": len(dev_loader.dataset),
            "train_batches_per_epoch": len(train_loader),
            "maximum_optimizer_steps": len(train_loader) * args.max_n_epochs,
            "starting_checkpoint_epoch": base_epoch, "optimizer_restarted": True,
            "sql_execution": "read-only isolated subprocess; 120-second batch budget; original metric formulas",
            "warmup_steps": len(train_loader) * args.num_warmup_epochs,
            "precision": "float32", "gradient_accumulation_steps": 1,
            "gradient_clipping": None, "checkpoint_selection": "highest dev record F1; first on ties",
            "pretrained_checkpoint": "google-t5/t5-small",
            "checkpoint_contents": "weights/config/tokenizer, NOT optimizer/scheduler/RNG state",
            "versions": {name: importlib.metadata.version(name) for name in
                         ["torch", "transformers", "tokenizers", "numpy"]},
        }
        write_json(backup / "run_metadata.json", metadata)
        print(json.dumps(metadata, indent=2), flush=True)
        print("Rechecking saved epoch 4 SQL with bounded workers (at most 120s; no generation/training).", flush=True)
        baseline_queries = Path(args.baseline_dev_sql).read_text().splitlines()
        assert len(baseline_queries) == 466
        utils.save_queries_and_records(baseline_queries, "results/baseline_dev.sql", "records/baseline_dev.pkl")
        sql_em, record_em, f1, errors = utils.compute_metrics(
            gt_sql, "results/baseline_dev.sql", gt_records, "records/baseline_dev.pkl",
        )
        best_row = {"epoch": base_epoch, "record_f1": float(f1), "record_em": float(record_em),
                    "sql_em": float(sql_em), "sql_error_rate": sum(bool(e) for e in errors)/len(errors),
                    "dev_loss": args.baseline_dev_loss, "recovered_baseline": True,
                    "previous_recovery_f1": args.baseline_recovery_f1}
        write_json(backup / "baseline_metrics.json", best_row)
        write_json(backup / "best_metrics.json", best_row)
        write_json(checkpoint_dir / "best/checkpoint_meta.json", {
            **best_row, "configuration": settings, "dev_evaluation_completed": True,
        })
        status["best_checkpoint"] = publish_checkpoint(checkpoint_dir / "best", backup / "checkpoints/best.tar")
        copy_predictions("results/baseline_dev.sql", "records/baseline_dev.pkl", backup / "predictions/best")
        print(f"Starting from epoch {base_epoch}: baseline dev Record F1={100*f1:.2f}%; weights backed up.", flush=True)
        print("CONTINUATION TRAINING STARTING. Optimizer/scheduler restarted; no held-out test use.", flush=True)
        history, best_f1, best_epoch, stale = [], float(f1), base_epoch, 0
        update("baseline_saved_ready_to_train", completed_epochs=base_epoch, best_epoch=best_epoch,
               best_dev_record_f1=best_f1)
        for epoch in range(base_epoch + 1, base_epoch + args.max_n_epochs + 1):
            epoch_start = time.monotonic()
            update("training", current_epoch=epoch)
            print(f"\nEPOCH {epoch}/{base_epoch + args.max_n_epochs}: training {len(train_loader)} batches", flush=True)
            train_loss = float(train_epoch(args, model, train_loader, optimizer, scheduler))
            if not math.isfinite(train_loss):
                raise ValueError("Non-finite training loss; stop and inspect.")

            # Protect the completed epoch BEFORE potentially slower dev generation.
            update("backing_up_last_checkpoint", current_epoch=epoch)
            save_model(str(checkpoint_dir), model, best=False)
            write_json(checkpoint_dir / "last/checkpoint_meta.json", {
                "epoch": epoch, "train_loss": train_loss, "dev_evaluation_completed": False,
                "configuration": settings,
            })
            last_info = publish_checkpoint(checkpoint_dir / "last", backup / "checkpoints/last.tar")
            update("evaluating_dev", completed_epochs=epoch, last_checkpoint=last_info)
            print(f"EPOCH {epoch}: TRAINED WEIGHTS BACKED UP TO DRIVE before dev evaluation.", flush=True)

            values = eval_epoch(args, model, dev_loader, gt_sql, sql_path, gt_records, records_path)
            loss, f1, record_em, sql_em, error_rate = map(float, values)
            if not all(math.isfinite(v) for v in [loss, f1, record_em, sql_em, error_rate]):
                raise ValueError("Non-finite dev metrics; stop and inspect.")
            row = dict(epoch=epoch, train_loss=train_loss, dev_loss=loss,
                       record_f1=f1, record_em=record_em, sql_em=sql_em, sql_error_rate=error_rate)
            copy_predictions(sql_path, records_path, backup / "predictions/last")
            improved = f1 > best_f1
            if improved:
                best_f1, best_epoch, stale = f1, epoch, 0
                best_row = row
                update("backing_up_best_checkpoint")
                save_model(str(checkpoint_dir), model, best=True)
                write_json(checkpoint_dir / "best/checkpoint_meta.json", {
                    **row, "dev_evaluation_completed": True, "configuration": settings,
                })
                best_info = publish_checkpoint(checkpoint_dir / "best", backup / "checkpoints/best.tar")
                copy_predictions(sql_path, records_path, backup / "predictions/best")
                status["best_checkpoint"] = best_info
            else:
                stale += 1
            row.update(seconds=time.monotonic() - epoch_start, improved=improved)
            history.append(row)
            write_json(backup / "history.json", history)
            write_json(backup / "best_metrics.json", best_row)
            update("epoch_completed", best_epoch=best_epoch, best_dev_record_f1=best_f1,
                   epochs_without_improvement=stale)
            print(f"EPOCH {epoch} COMPLETE: train loss={train_loss:.4f}, dev loss={loss:.4f}, "
                  f"Record F1={100*f1:.2f}%, Record EM={100*record_em:.2f}%, "
                  f"SQL EM={100*sql_em:.2f}%, SQL errors={100*error_rate:.2f}%", flush=True)
            print(f"Epoch wall time including dev/backups: {row['seconds']/60:.1f} minutes. "
                  f"Best epoch={best_epoch}; saved to Drive.", flush=True)
            if stale >= args.patience_epochs:
                print(f"Early stopping: {stale} epochs without improved dev Record F1.", flush=True)
                break

        update("reloading_best_for_final_dev_check")
        del optimizer, scheduler, model
        gc.collect()
        torch.cuda.empty_cache()
        best_model = load_model_from_checkpoint(args, best=True)
        values = eval_epoch(args, best_model, dev_loader, gt_sql, "results/best_dev.sql",
                            gt_records, "records/best_dev.pkl")
        loss, f1, record_em, sql_em, error_rate = map(float, values)
        assert math.isclose(f1, best_f1, rel_tol=1e-6, abs_tol=1e-8), "Reloaded best dev F1 differs."
        copy_predictions("results/best_dev.sql", "records/best_dev.pkl", backup / "predictions/best")
        final = {"state": "COMPLETED", "experiment_name": args.experiment_name,
                 "epochs_run_after_restart": len(history), "best_epoch": best_epoch,
                 "started_from_epoch": base_epoch, "total_training_epochs": history[-1]["epoch"],
                 "optimizer_restarted": True, "exact_resume": False,
                 "dev_loss": loss, "record_f1": f1, "record_em": record_em,
                 "sql_em": sql_em, "sql_error_rate": error_rate,
                 "seconds": time.monotonic() - started, "held_out_test_used": False,
                 "best_checkpoint_archive": str(backup / "checkpoints/best.tar"),
                 "backup_folder": str(backup)}
        write_json(backup / "final_report.json", final)
        update("completed", state="COMPLETED", final_report=final)
        print("\nPART 2 CONTINUATION AND DEV BACKUP COMPLETE", flush=True)
        print(json.dumps(final, indent=2), flush=True)
        print("Save this notebook and ask for review. Do not submit diagnostic outputs or generate test yet.")
    except BaseException as error:
        import traceback
        update("failed_or_interrupted", state="FAILED_OR_INTERRUPTED", error=traceback.format_exc())
        print("Run did not complete. Existing completed Drive archives/logs remain available:", backup, flush=True)
        raise


if __name__ == "__main__":
    run(sys.argv[1])
