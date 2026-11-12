"""Check evaluation on FOUR DEV examples; never train or use held-out test inputs."""

import json
import math
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset, Subset

part2 = Path.cwd().resolve()
sys.path.insert(0, str(part2))
from load_data import T5Dataset, normal_collate_fn, test_collate_fn, preprocess_sql
from t5_utils import initialize_model
import train_t5 as runner
from utils import save_queries_and_records, set_random_seeds

assert torch.cuda.is_available(), "Connect to a Colab GPU."
assert runner.DEVICE.type == "cuda"
set_random_seeds(42)
report_path = Path(sys.argv[1]).resolve()
out = report_path.parent / "diagnostic_only"
out.mkdir(parents=True, exist_ok=False)
args = SimpleNamespace(finetune=True, max_new_tokens=512, num_beams=1)
model = initialize_model(args)
dataset = T5Dataset(str(part2 / "data"), "dev")
subset = Subset(dataset, [0, 1, 2, 3])
loader = DataLoader(subset, batch_size=2, shuffle=False, collate_fn=normal_collate_fn)
tokenizer = dataset.tokenizer

gold = (part2 / "data/dev.sql").read_text().splitlines()[:4]
gold_path, gold_records = out / "four_dev_gold.sql", out / "four_dev_gold.pkl"
save_queries_and_records(gold, str(gold_path), str(gold_records))
before = model.shared.weight[:16, :16].detach().clone()

print("Checking real greedy generation on FOUR DEV examples, not the test split.", flush=True)
metrics = runner.eval_epoch(
    args, model, loader, str(gold_path), str(out / "four_dev_predictions.sql"),
    str(gold_records), str(out / "four_dev_predictions.pkl"),
)
assert len(metrics) == 5 and all(math.isfinite(float(x)) for x in metrics)
assert metrics[0] > 0 and all(0 <= x <= 1 for x in metrics[1:])
assert len((out / "four_dev_predictions.sql").read_text().splitlines()) == 4
assert not model.training
assert torch.equal(before, model.shared.weight[:16, :16].detach())
print("Real forward loss + greedy generation + SQL/record files: PASSED", flush=True)

# Independently recompute token-weighted, PAD-excluding loss, without generation.
total_loss, total_tokens = 0.0, 0
with torch.no_grad():
    for enc, mask, dec, targets, _ in loader:
        enc, mask, dec, targets = [x.to(runner.DEVICE) for x in (enc, mask, dec, targets)]
        logits = model(input_ids=enc, attention_mask=mask, decoder_input_ids=dec).logits
        keep = targets != 0
        total_loss += torch.nn.functional.cross_entropy(
            logits[keep], targets[keep], reduction="sum",
        ).item()
        total_tokens += keep.sum().item()
        del logits
assert math.isclose(metrics[0], total_loss / total_tokens, rel_tol=1e-5)
print("Token-weighted evaluation loss and PAD handling: PASSED", flush=True)

# Controlled SQL fixtures test metric order, < / <= restoration and SQL errors.
# They are diagnostic data, not predictions or results for the assignment.
fixture_gold = ["SELECT 1 WHERE 1 < 2", "SELECT 2 WHERE 2 <= 2", "SELECT 3", "SELECT 4"]
fixture_predictions = fixture_gold[:3] + ["NOT SQL"]
fixture_path, fixture_records = out / "fixture_gold.sql", out / "fixture_gold.pkl"
save_queries_and_records(fixture_gold, str(fixture_path), str(fixture_records))

def fake_generation(**kwargs):
    assert not torch.is_grad_enabled() and not model.training
    assert kwargs["decoder_input_ids"].shape[1] == 1, "Gold tokens leaked into generation."
    assert torch.all(kwargs["decoder_input_ids"] == 0)
    assert not kwargs["do_sample"] and kwargs["num_beams"] == 1
    assert kwargs["max_new_tokens"] == 512
    batch_size = kwargs["input_ids"].shape[0]
    start = fake_generation.offset
    queries = fixture_predictions[start:start + batch_size]
    fake_generation.offset += batch_size
    encoded = tokenizer([preprocess_sql(q) for q in queries], padding=True, return_tensors="pt")["input_ids"]
    return torch.cat([torch.zeros((batch_size, 1), dtype=torch.long), encoded], dim=1).to(runner.DEVICE)

fake_generation.offset = 0
with patch.object(model, "generate", side_effect=fake_generation):
    fixture_metrics = runner.eval_epoch(
        args, model, loader, str(fixture_path), str(out / "fixture_predictions.sql"),
        str(fixture_records), str(out / "fixture_predictions.pkl"),
    )
assert math.isclose(fixture_metrics[1], 0.75, abs_tol=1e-7), fixture_metrics
assert fixture_metrics[2:] == (0.75, 0.75, 0.25), fixture_metrics
assert (out / "fixture_predictions.sql").read_text().splitlines() == fixture_predictions
with (out / "fixture_predictions.pkl").open("rb") as handle:
    records, errors = pickle.load(handle)
assert len(records) == len(errors) == 4 and sum(bool(e) for e in errors) == 1
print("SQL marker restoration, metric return order, error rate, file ordering: PASSED", flush=True)

class DevInputsOnly(Dataset):
    tokenizer = dataset.tokenizer
    def __len__(self):
        return len(subset)
    def __getitem__(self, index):
        return subset[index][0]

proxy_loader = DataLoader(DevInputsOnly(), batch_size=2, shuffle=False, collate_fn=test_collate_fn)
fake_generation.offset = 0
with patch.object(model, "generate", side_effect=fake_generation):
    runner.test_inference(
        args, model, proxy_loader, str(out / "dev_inputs_only.sql"),
        str(out / "dev_inputs_only.pkl"),
    )
assert (out / "dev_inputs_only.sql").read_text().splitlines() == fixture_predictions
with (out / "dev_inputs_only.pkl").open("rb") as handle:
    records, errors = pickle.load(handle)
assert len(records) == len(errors) == 4
assert torch.equal(before, model.shared.weight[:16, :16].detach())
assert all(p.grad is None for p in model.parameters())
print("NL-only inference interface on dev inputs (NOT held-out test): PASSED", flush=True)

# Refuse shuffled evaluation before any generation or output writes.
shuffled = DataLoader(subset, batch_size=2, shuffle=True, collate_fn=normal_collate_fn)
try:
    runner.eval_epoch(args, model, shuffled, str(gold_path), str(out / "unused.sql"),
                      str(gold_records), str(out / "unused.pkl"))
except ValueError:
    pass
else:
    raise AssertionError("Evaluation incorrectly accepted shuffled data.")

report = {
    "stage": "evaluation_diagnostic_only_not_final_results",
    "dev_examples": 4, "held_out_test_used": False, "training_updates": 0,
    "real_untrained_four_dev_metrics_not_reportable": [float(x) for x in metrics],
    "controlled_fixture_expected_f1_em_sql_em_error": [0.75, 0.75, 0.75, 0.25],
    "checks_passed": ["real_generation", "weighted_loss", "sql_marker_restoration",
                      "metric_order", "sql_error_rate", "output_order_and_format",
                      "nl_only_interface_on_dev", "shuffle_guard", "no_training"],
}
report_path.write_text(json.dumps(report, indent=2))
print("Q5 EVALUATION CHECK PASSED", flush=True)
print("No training updates; no held-out test inputs used. Diagnostic scores are NOT report results.")
