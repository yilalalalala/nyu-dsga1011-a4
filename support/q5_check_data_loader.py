"""Read-only Colab check of the student's uploaded part2/load_data.py."""

import importlib.metadata
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import RandomSampler, SequentialSampler

assert Path("load_data.py").is_file(), "Run with cwd=/content/part2."
sys.path.insert(0, str(Path.cwd()))
from load_data import PAD_IDX, load_t5_data, preprocess_sql, restore_sql


train_loader, dev_loader, test_loader = load_t5_data(8, 8)
loaders = {"train": train_loader, "dev": dev_loader, "test": test_loader}
expected_sizes = {"train": 4225, "dev": 466, "test": 432}
assert isinstance(train_loader.sampler, RandomSampler)
assert isinstance(dev_loader.sampler, SequentialSampler)
assert isinstance(test_loader.sampler, SequentialSampler)
report = {
    "checkpoint": "google-t5/t5-small",
    "batch_size_for_check": 8,
    "versions": {name: importlib.metadata.version(name)
                 for name in ["torch", "transformers", "tokenizers"]},
    "splits": {},
}

for split, loader in loaders.items():
    dataset = loader.dataset
    assert len(dataset) == expected_sizes[split], f"Wrong {split} example count"
    tokenizer = dataset.tokenizer
    assert PAD_IDX == tokenizer.pad_token_id == 0
    assert all(ids[-1] == tokenizer.eos_token_id for ids in dataset.input_ids)
    batch = next(iter(loader))
    expected_columns = 3 if split == "test" else 5
    assert len(batch) == expected_columns, f"Wrong {split} batch structure"
    encoder_ids, encoder_mask = batch[:2]
    assert encoder_ids.dtype == encoder_mask.dtype == torch.long
    assert encoder_ids.shape == encoder_mask.shape
    assert encoder_ids.shape[0] == 8
    assert torch.equal(encoder_mask, (encoder_ids != PAD_IDX).long())

    info = {
        "examples": len(dataset),
        "batches": len(loader),
        "max_input_tokens_with_EOS": max(map(len, dataset.input_ids)),
        "batch_shapes": [list(t.shape) for t in batch],
    }
    if split == "test":
        assert dataset.target_ids is None
        initial = batch[2]
        assert initial.shape == (8, 1) and torch.all(initial == PAD_IDX)
        raw_test = Path("data/test.nl").read_text().splitlines()
        expected_ids = tokenizer(
            [s.strip() for s in raw_test], add_special_tokens=True,
            padding=False, truncation=False, return_attention_mask=False,
        )["input_ids"]
        assert dataset.input_ids == expected_ids, "Test inputs were reordered or changed."
    else:
        decoder_inputs, targets, initial = batch[2:]
        assert decoder_inputs.shape == targets.shape
        assert targets.dtype == decoder_inputs.dtype == initial.dtype == torch.long
        assert torch.all(targets >= 0), "Starter loss expects PAD=0, not -100 labels."
        assert initial.shape == (8, 1)
        assert torch.equal(initial, decoder_inputs[:, :1])
        assert torch.all(initial == PAD_IDX)
        assert torch.equal(decoder_inputs[:, 1:], targets[:, :-1]), (
            "Incorrect teacher-forcing shift."
        )
        assert all(ids[-1] == tokenizer.eos_token_id for ids in dataset.target_ids)
        assert all(tokenizer.unk_token_id not in ids for ids in dataset.target_ids)
        info["max_target_tokens_with_EOS"] = max(map(len, dataset.target_ids))
        expected_max = {"train": (60, 511), "dev": (44, 503)}[split]
        assert (info["max_input_tokens_with_EOS"],
                info["max_target_tokens_with_EOS"]) == expected_max, (
            "Loader token lengths differ from Q4 + EOS; inspect preprocessing/truncation."
        )
    report["splits"][split] = info
    print(f"{split}: {len(dataset)} examples; {len(loader)} batches", flush=True)
    print("  batch shapes:", info["batch_shapes"], flush=True)

tokenizer = train_loader.dataset.tokenizer
for raw in ["x < 1200", "x <= 1200", "x > 1200", "x >= 1200"]:
    encoded = preprocess_sql(raw)
    ids = tokenizer(encoded, add_special_tokens=True)["input_ids"]
    assert tokenizer.unk_token_id not in ids
    decoded = tokenizer.decode(ids, skip_special_tokens=True,
                               clean_up_tokenization_spaces=False)
    assert restore_sql(decoded) == raw

Path(sys.argv[1]).write_text(json.dumps(report, indent=2))
print("Q5 DATA LOADER CHECK PASSED", flush=True)
print("EOS, decoder shift, padding, token coverage, and split ordering verified.")
print("No model training has started.")
