# REFERENCE BLOCK: paste into part2/train_t5.py in place of the existing
# eval_epoch() and test_inference() block, ending immediately BEFORE def main().
# Include ALL helper functions below. Keep the rest of train_t5.py unchanged.
# This reference is not a standalone script and does not start training.

def _evaluation_tokenizer(loader):
    # Also support Subset wrappers used by the diagnostic check.
    dataset = loader.dataset
    while not hasattr(dataset, "tokenizer") and hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset.tokenizer


def _check_evaluation_order(loader):
    from torch.utils.data import SequentialSampler
    if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
        raise ValueError("Evaluation requires original order and drop_last=False.")


def _generate_sql(args, model, tokenizer, encoder_input, encoder_mask, initial):
    from load_data import restore_sql
    # Greedy decoding by default. Never use gold SQL tokens for generation.
    generated = model.generate(
        input_ids=encoder_input,
        attention_mask=encoder_mask,
        decoder_input_ids=initial.to(DEVICE),
        max_new_tokens=getattr(args, "max_new_tokens", 512),
        num_beams=getattr(args, "num_beams", 1),
        do_sample=False,
        use_cache=True,
    )
    decoded = tokenizer.batch_decode(
        generated.detach().cpu(), skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    # Restore < / <= BEFORE execution; retain one output line per example.
    return [restore_sql(sql).replace("\r", " ").replace("\n", " ").strip()
            for sql in decoded]


def _save_sql_predictions(queries, sql_path, record_path):
    for path in (sql_path, record_path):
        os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    save_queries_and_records(queries, sql_path, record_path)


def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path,
               gt_record_path, model_record_path):
    _check_evaluation_order(dev_loader)
    tokenizer = _evaluation_tokenizer(dev_loader)
    with open(gt_sql_pth) as handle:
        gold_count = len(handle.read().splitlines())
    if gold_count != len(dev_loader.dataset):
        raise ValueError("Gold SQL count does not match the evaluation dataset.")

    model.eval()
    total_loss, total_tokens = 0.0, 0
    predictions = []
    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, targets, initial in tqdm(
            dev_loader, desc="Dev loss + generation",
        ):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            targets = targets.to(DEVICE)
            logits = model(
                input_ids=encoder_input, attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )["logits"]
            non_pad = targets != PAD_IDX
            # Token-weighted mean over the whole split, ignoring PAD labels.
            total_loss += nn.functional.cross_entropy(
                logits[non_pad], targets[non_pad], reduction="sum",
            ).item()
            total_tokens += non_pad.sum().item()
            del logits
            predictions.extend(_generate_sql(
                args, model, tokenizer, encoder_input, encoder_mask, initial,
            ))

    if total_tokens == 0 or len(predictions) != gold_count:
        raise ValueError("Empty targets or missing/extra generated predictions.")
    _save_sql_predictions(predictions, model_sql_path, model_record_path)
    # The provided helper's order differs from the training loop's return order.
    sql_em, record_em, record_f1, errors = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path,
    )
    if len(errors) != len(predictions):
        raise ValueError("SQL error count does not match predictions.")
    error_rate = sum(bool(error) for error in errors) / len(errors)
    return total_loss / total_tokens, record_f1, record_em, sql_em, error_rate


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    _check_evaluation_order(test_loader)
    tokenizer = _evaluation_tokenizer(test_loader)
    model.eval()
    predictions = []
    with torch.no_grad():
        for encoder_input, encoder_mask, initial in tqdm(
            test_loader, desc="SQL inference",
        ):
            predictions.extend(_generate_sql(
                args, model, tokenizer, encoder_input.to(DEVICE),
                encoder_mask.to(DEVICE), initial,
            ))
    if len(predictions) != len(test_loader.dataset):
        raise ValueError("Missing/extra inference predictions.")
    _save_sql_predictions(predictions, model_sql_path, model_record_path)
    print(f"Saved {len(predictions)} SQL predictions: {model_sql_path}", flush=True)
