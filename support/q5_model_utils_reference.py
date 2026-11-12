# REFERENCE FUNCTIONS FOR YOU TO PASTE INTO part2/t5_utils.py.
# Replace ONLY initialize_model(), save_model(), and load_model_from_checkpoint()
# with the corresponding functions below. Keep all existing imports, DEVICE,
# mkdir(), setup_wandb(), optimizer helpers, and scheduler helpers.
# This reference file is not a standalone script.


def initialize_model(args):
    checkpoint = "google-t5/t5-small"
    if args.finetune:
        model = T5ForConditionalGeneration.from_pretrained(
            checkpoint, use_safetensors=True,
        )
    else:
        # This branch is for the optional from-scratch setting, not Q5 fine-tuning.
        config = T5Config.from_pretrained(checkpoint)
        model = T5ForConditionalGeneration(config)

    # Fine-tune the entire model; no layers are frozen.
    model.requires_grad_(True)
    return model.to(DEVICE)


def save_model(checkpoint_dir, model, best):
    name = "best" if best else "last"
    save_dir = os.path.join(checkpoint_dir, name)
    mkdir(save_dir)
    model.save_pretrained(save_dir, safe_serialization=True)

    # The tokenizer vocabulary is unchanged by the Q4 text-marker preprocessing.
    tokenizer = transformers.T5TokenizerFast.from_pretrained("google-t5/t5-small")
    tokenizer.save_pretrained(save_dir)
    print(f"Saved {name} checkpoint: {save_dir}", flush=True)


def load_model_from_checkpoint(args, best):
    model_type = "ft" if args.finetune else "scr"
    name = "best" if best else "last"
    load_dir = os.path.join(
        "checkpoints", f"{model_type}_experiments", args.experiment_name, name,
    )
    if not os.path.isfile(os.path.join(load_dir, "config.json")):
        raise FileNotFoundError(f"No saved model found at {load_dir}")
    model = T5ForConditionalGeneration.from_pretrained(
        load_dir, local_files_only=True, use_safetensors=True,
    )
    return model.to(DEVICE)
