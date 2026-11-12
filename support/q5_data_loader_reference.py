# REFERENCE BLOCK FOR YOU TO PASTE INTO part2/load_data.py.
# Replace everything from `class T5Dataset(Dataset):` up to, but not including,
# `def get_dataloader(batch_size, split):` with this entire block.
# Keep the existing imports, PAD_IDX = 0, get_dataloader(), load_t5_data(),
# load_lines(), and the rest of the file. This file is not a standalone script.

SQL_MARKER = "less_than"


def preprocess_sql(sql):
    sql = sql.strip()
    if SQL_MARKER in sql:
        raise ValueError("The reserved SQL marker already occurs in an original target.")
    return sql.replace("<", SQL_MARKER)


def restore_sql(sql):
    # Call after tokenizer decoding and BEFORE SQL execution or evaluation.
    return sql.replace(SQL_MARKER, "<").strip()


class T5Dataset(Dataset):
    def __init__(self, data_folder, split):
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        assert self.tokenizer.pad_token_id == PAD_IDX
        self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        sources = load_lines(os.path.join(data_folder, f"{split}.nl"))
        assert sources and all(sources), "Empty input or blank example."
        self.input_ids = tokenizer(
            sources,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        assert all(ids[-1] == tokenizer.eos_token_id for ids in self.input_ids)

        # The held-out test split contains NL inputs only.
        self.target_ids = None
        if split != "test":
            raw_sql = load_lines(os.path.join(data_folder, f"{split}.sql"))
            assert len(raw_sql) == len(sources) and all(raw_sql)
            processed_sql = [preprocess_sql(sql) for sql in raw_sql]
            assert all(restore_sql(p) == s for p, s in zip(processed_sql, raw_sql))
            self.target_ids = tokenizer(
                processed_sql,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )["input_ids"]
            assert all(ids[-1] == tokenizer.eos_token_id for ids in self.target_ids)
            assert all(tokenizer.unk_token_id not in ids for ids in self.target_ids), (
                "Unknown SQL tokens remain after preprocessing."
            )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        encoder_ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
        if self.split == "test":
            return encoder_ids
        targets = torch.tensor(self.target_ids[idx], dtype=torch.long)
        return encoder_ids, targets


def normal_collate_fn(batch):
    encoder_sequences, target_sequences = zip(*batch)
    encoder_ids = pad_sequence(
        encoder_sequences, batch_first=True, padding_value=PAD_IDX,
    )
    encoder_mask = (encoder_ids != PAD_IDX).long()
    decoder_targets = pad_sequence(
        target_sequences, batch_first=True, padding_value=PAD_IDX,
    )

    # Teacher forcing: prepend T5's PAD start token and shift targets right.
    decoder_inputs = torch.full_like(decoder_targets, PAD_IDX)
    decoder_inputs[:, 1:] = decoder_targets[:, :-1]
    initial_decoder_inputs = decoder_inputs[:, :1].clone()

    return (
        encoder_ids, encoder_mask, decoder_inputs,
        decoder_targets, initial_decoder_inputs,
    )


def test_collate_fn(batch):
    encoder_ids = pad_sequence(batch, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.full(
        (len(batch), 1), PAD_IDX, dtype=torch.long,
    )
    return encoder_ids, encoder_mask, initial_decoder_inputs
