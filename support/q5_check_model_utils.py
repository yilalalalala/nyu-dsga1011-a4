"""Check uploaded model helpers using a temporary, non-final model experiment."""

import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast

part2 = Path.cwd().resolve()
assert (part2 / "t5_utils.py").is_file(), "Run with cwd=/content/part2."
sys.path.insert(0, str(part2))
from load_data import T5Dataset, normal_collate_fn
from t5_utils import (
    DEVICE, initialize_model, initialize_optimizer_and_scheduler,
    save_model, load_model_from_checkpoint,
)
from utils import set_random_seeds

assert torch.cuda.is_available(), "Connect to your Colab GPU before this model check."
assert DEVICE.type == "cuda"
set_random_seeds(42)
report_path = Path(sys.argv[1]).resolve()

# These hyperparameters are for ONE diagnostic update, not the final experiment.
args = SimpleNamespace(
    finetune=True, experiment_name="smoke_only",
    optimizer_type="AdamW", learning_rate=3e-4, weight_decay=0.0,
    scheduler_type="linear", num_warmup_epochs=0, max_n_epochs=2,
)
print("Loading pretrained T5-small; the first weights download may take a few minutes.", flush=True)
model = initialize_model(args)
assert isinstance(model, T5ForConditionalGeneration), "initialize_model returned the wrong model."
assert model.config.model_type == "t5"
assert model.config.d_model == 512 and model.config.num_layers == 6
assert model.config.decoder_start_token_id == model.config.pad_token_id == 0
assert model.config.eos_token_id == 1
assert next(model.parameters()).device.type == "cuda"
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert total == trainable
print(f"Parameters: {total:,}; trainable: {trainable:,}", flush=True)

dataset = T5Dataset(str(part2 / "data"), "train")
batch = normal_collate_fn([dataset[0], dataset[1]])
encoder_ids, encoder_mask, decoder_inputs, targets, _ = [t.to(DEVICE) for t in batch]
inputs = dict(input_ids=encoder_ids, attention_mask=encoder_mask,
              decoder_input_ids=decoder_inputs)

def probe(current_model):
    current_model.eval()
    with torch.no_grad():
        # Small numerical signature: enough to check exact save/load behavior.
        return current_model(**inputs).logits[:, :2, :64].detach().cpu().clone()

before = probe(model)
assert torch.isfinite(before).all()

# Isolate smoke checkpoints so no real training checkpoint is overwritten.
work = Path(tempfile.mkdtemp(prefix="hw4_q5_model_check_", dir="/content"))
os.chdir(work)
checkpoint_dir = Path("checkpoints/ft_experiments") / args.experiment_name
print("Temporary diagnostic checkpoint directory:", work / checkpoint_dir, flush=True)
save_model(str(checkpoint_dir), model, best=True)

best_dir = checkpoint_dir / "best"
assert (best_dir / "config.json").is_file()
assert (best_dir / "model.safetensors").is_file()
saved_tokenizer = T5TokenizerFast.from_pretrained(str(best_dir), local_files_only=True)
assert saved_tokenizer.get_vocab() == dataset.tokenizer.get_vocab()
restored_best = load_model_from_checkpoint(args, best=True)
torch.testing.assert_close(probe(restored_best), before, rtol=1e-5, atol=1e-6)
del restored_best
gc.collect()
torch.cuda.empty_cache()
print("BEST checkpoint and tokenizer reload: PASSED", flush=True)

optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, epoch_length=1)
assert scheduler is not None
initial_weights = model.shared.weight[:64, :64].detach().clone()
model.train()
optimizer.zero_grad()
logits = model(**inputs).logits
non_pad = targets != 0
loss = torch.nn.functional.cross_entropy(logits[non_pad], targets[non_pad])
assert torch.isfinite(loss)
loss.backward()
assert model.shared.weight.grad is not None
assert torch.isfinite(model.shared.weight.grad).all()
optimizer.step()
scheduler.step()
optimizer.zero_grad(set_to_none=True)
assert not torch.equal(initial_weights, model.shared.weight[:64, :64].detach())
after = probe(model)
assert not torch.equal(before, after)
print(f"ONE diagnostic optimizer step: PASSED; loss={loss.item():.6f}", flush=True)
del logits, initial_weights

save_model(str(checkpoint_dir), model, best=False)
assert (checkpoint_dir / "last" / "model.safetensors").is_file()
restored_last = load_model_from_checkpoint(args, best=False)
torch.testing.assert_close(probe(restored_last), after, rtol=1e-5, atol=1e-6)
del restored_last
gc.collect()
torch.cuda.empty_cache()

# Saving last must not change the best checkpoint saved before the update.
restored_best = load_model_from_checkpoint(args, best=True)
torch.testing.assert_close(probe(restored_best), before, rtol=1e-5, atol=1e-6)
print("LAST checkpoint reload and BEST/LAST separation: PASSED", flush=True)

report = {
    "stage": "diagnostic_model_check_not_final_training",
    "pretrained_checkpoint": "google-t5/t5-small",
    "parameters": total, "trainable_parameters": trainable,
    "gpu": torch.cuda.get_device_name(0),
    "one_step_loss": loss.item(),
    "best_reload_matches": True, "last_reload_matches": True,
    "best_and_last_are_distinct": True, "tokenizer_reload_matches": True,
    "temporary_checkpoint_root": str(work / checkpoint_dir),
    "diagnostic_args_not_final_hyperparameters": vars(args),
    "versions": {name: importlib.metadata.version(name)
                 for name in ["torch", "transformers", "tokenizers"]},
}
report_path.write_text(json.dumps(report, indent=2))
print("Q5 MODEL AND CHECKPOINT CHECK PASSED", flush=True)
print("Only one diagnostic update was run. Full training has NOT started.")
print("Diagnostic weights stay in /content; the report/log are saved to Drive.")
