"""Export Q6 dev evidence, then generate Q7 test predictions ONCE from the fixed best model."""
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
import sys
import traceback
from types import SimpleNamespace
import zipfile


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def execution_group(message):
    lower = message.lower()
    if not lower:
        return None
    if "timed out" in lower:
        return "SQL execution timeout"
    if any(s in lower for s in ["syntax error", "incomplete input", "unrecognized token"]):
        return "SQL syntax/format error"
    if any(s in lower for s in ["no such column", "no such table", "ambiguous column"]):
        return "SQL identifier/reference error"
    if any(s in lower for s in ["misuse of aggregate", "wrong number of arguments", "no such function"]):
        return "SQL function/aggregation error"
    return "Other SQL execution error"


def export_dev_evidence(context, output, utils):
    work = Path(context["work_directory"])
    run = Path(context["training_run"])
    gold_sql = (work / "data/dev.sql").read_text().splitlines()
    nl = (work / "data/dev.nl").read_text().splitlines()
    predicted_path, predicted_records = run / "predictions/best/dev.sql", run / "predictions/best/dev.pkl"
    predictions = predicted_path.read_text().splitlines()
    with (work / "records/ground_truth_dev.pkl").open("rb") as handle:
        gold_records, gold_errors = pickle.load(handle)
    with predicted_records.open("rb") as handle:
        records, errors = pickle.load(handle)
    assert all(len(x) == 466 for x in [gold_sql, nl, predictions, gold_records, gold_errors, records, errors])
    sql_em, record_em, f1, _ = utils.compute_metrics(
        str(work / "data/dev.sql"), str(predicted_path),
        str(work / "records/ground_truth_dev.pkl"), str(predicted_records),
    )
    expected = context["training_report"]
    for actual, name in [(sql_em, "sql_em"), (record_em, "record_em"), (f1, "record_f1")]:
        assert math.isclose(float(actual), expected[name], abs_tol=1e-8)
    rows, groups = [], defaultdict(list)
    literal = re.compile(r"'(?:''|[^'])*'|\b\d+(?:\.\d+)?\b")
    for index, (question, gold, predicted, gold_rows, pred_rows, error) in enumerate(
        zip(nl, gold_sql, predictions, gold_records, records, errors), 1,
    ):
        exact_records = set(gold_rows) == set(pred_rows)
        group = execution_group(error)
        row = {"example_number_1_based": index, "nl": question, "gold_sql": gold,
               "predicted_sql": predicted, "sql_exact_match": gold.strip() == predicted.strip(),
               "record_exact_match": exact_records,
               "record_f1": float(utils.compute_record_F1([gold_rows], [pred_rows])),
               "sqlite_error": error, "execution_error_group": group,
               "gold_record_count": len(gold_rows), "predicted_record_count": len(pred_rows),
               "literal_sets_differ_review_hint_only": sorted(literal.findall(gold)) != sorted(literal.findall(predicted))}
        rows.append(row)
        if group:
            groups[group].append(row)
        elif not exact_records:
            groups["Valid SQL but incorrect records: manual semantic review needed"].append(row)
    summary = {"model_epoch": expected["best_epoch"], "dev_examples": 466,
               "record_f1": float(f1), "record_em": float(record_em), "sql_em": float(sql_em),
               "sql_execution_errors": sum(bool(e) for e in errors),
               "record_mismatches": sum(not r["record_exact_match"] for r in rows),
               "sql_text_mismatches": sum(not r["sql_exact_match"] for r in rows),
               "review_group_counts_out_of_466": {key: len(value) for key, value in groups.items()},
               "note": "Execution groups are measured from saved error messages. Semantic hints are NOT final error classifications; inspect examples before writing Q6."}
    q6 = output / "q6_dev_analysis"
    q6.mkdir()
    (q6 / "summary.json").write_text(json.dumps(summary, indent=2))
    (q6 / "all_dev_examples.json").write_text(json.dumps(rows, indent=2))
    with (q6 / "all_dev_examples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    samples = []
    for group, examples in groups.items():
        if group.startswith("Valid SQL"):
            differing = [r for r in examples if r["literal_sets_differ_review_hint_only"]]
            same = [r for r in examples if not r["literal_sets_differ_review_hint_only"]]
            chosen = differing[:3] + same[:3]
        else:
            chosen = examples[:2]
        samples.extend({"review_group": group, **r} for r in chosen)
    (q6 / "review_examples.json").write_text(json.dumps(samples, indent=2))
    print("Q6 DEV EVIDENCE EXPORTED", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("Q6 EXAMPLES FOR REVIEW (not a finished written answer):", flush=True)
    print(json.dumps(samples, indent=2), flush=True)
    return summary


def main(context_path):
    context = json.loads(Path(context_path).read_text())
    work, output = Path(context["work_directory"]), Path(context["export_directory"])
    assert Path.cwd().resolve() == work.resolve()
    sys.path.insert(0, str(work))
    import torch
    from transformers import T5TokenizerFast
    import utils
    from load_data import get_dataloader
    from t5_utils import load_model_from_checkpoint
    from train_t5 import test_inference
    from final_bounded_sql import install_bounded_sql
    install_bounded_sql(utils)

    def status(state, stage, **extra):
        (output / "status.json").write_text(json.dumps({
            "state": state, "stage": stage, "updated_utc": datetime.now(timezone.utc).isoformat(),
            "best_epoch": context["training_report"]["best_epoch"],
            "export_directory": str(output), "training_updates": 0, **extra,
        }, indent=2))

    try:
        status("RUNNING", "exporting_q6_dev_evidence")
        dev_summary = export_dev_evidence(context, output, utils)
        args = SimpleNamespace(**context["configuration"])
        assert torch.cuda.is_available() and args.finetune
        assert args.num_beams == 1 and args.max_new_tokens == 512
        utils.set_random_seeds(args.seed)
        checkpoint = Path(context["best_checkpoint_directory"])
        tokenizer = T5TokenizerFast.from_pretrained(str(checkpoint), local_files_only=True)
        status("RUNNING", "loading_fixed_best_model_for_final_test")
        model = load_model_from_checkpoint(args, best=True)
        assert next(model.parameters()).device.type == "cuda"
        signature = model.shared.weight[:16, :16].detach().clone()
        test_loader = get_dataloader(args.test_batch_size, "test")
        raw_test = (work / "data/test.nl").read_text().splitlines()
        assert len(raw_test) == len(test_loader.dataset) == 432
        assert test_loader.dataset.target_ids is None
        assert tokenizer(raw_test, padding=False, truncation=False)["input_ids"] == test_loader.dataset.input_ids
        assert tokenizer.get_vocab() == test_loader.dataset.tokenizer.get_vocab()
        sql_path = output / "results/t5_ft_experiment_test.sql"
        records_path = output / "records/t5_ft_experiment_test.pkl"
        print(f"\nFINAL TEST GENERATION: fixed best epoch {context['training_report']['best_epoch']}, 432 examples in original order.", flush=True)
        print("No training or model selection on test. Test F1 is available only from Gradescope.", flush=True)
        status("RUNNING", "generating_test_sql_and_records", test_examples=432)
        test_inference(args, model, test_loader, str(sql_path), str(records_path))
        queries = sql_path.read_text().splitlines()
        with records_path.open("rb") as handle:
            records, errors = pickle.load(handle)
        assert len(queries) == len(records) == len(errors) == 432
        assert all(q.strip() for q in queries), "Blank prediction: preserve files and ask for review."
        assert all("less_than" not in q for q in queries), "Unrestored SQL marker."
        assert all(isinstance(e, str) for e in errors)
        assert not model.training and all(p.grad is None for p in model.parameters())
        assert torch.equal(signature, model.shared.weight[:16, :16].detach())
        bundle = output / "Q7_submission_files.zip"
        with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(sql_path, arcname="results/" + sql_path.name)
            archive.write(records_path, arcname="records/" + records_path.name)
        with zipfile.ZipFile(bundle) as archive:
            assert archive.testzip() is None
            assert len(archive.namelist()) == 2
        report = {"state": "COMPLETED", "best_epoch": context["training_report"]["best_epoch"],
                  "training_updates": 0, "test_examples": 432, "test_order_verified": True,
                  "dev_metrics": dev_summary, "test_f1": None,
                  "test_f1_note": "Obtain from Gradescope; no test labels are provided locally.",
                  "test_sql_execution_errors": sum(bool(e) for e in errors),
                  "sql_file": str(sql_path), "records_file": str(records_path),
                  "submission_zip": str(bundle), "q6_evidence": str(output / "q6_dev_analysis"),
                  "selected_checkpoint": str(checkpoint),
                  "checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
                  "test_input_sha256": sha256(work / "data/test.nl"),
                  "file_sha256": {str(p.relative_to(output)): sha256(p) for p in [sql_path, records_path, bundle]}}
        (output / "export_report.json").write_text(json.dumps(report, indent=2))
        status("COMPLETED", "q6_evidence_and_q7_files_exported", test_examples=432)
        print("\nQ6 EVIDENCE AND Q7 TEST FILES READY", flush=True)
        print(json.dumps(report, indent=2), flush=True)
        print("Save this notebook and ask for file checks before submitting to Gradescope.")
        print("Do not rerun generation to tune against the test set. No Gradescope submission was made automatically.")
    except BaseException:
        status("FAILED", "export_failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    main(sys.argv[1])
