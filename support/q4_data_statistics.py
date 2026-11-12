"""Q4 measurement helper. Reads data only; does not import homework skeletons."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
from statistics import mean


MODEL_NAME = "google-t5/t5-small"
ROWS = [
    ("Number of examples", "examples"),
    ("Mean sentence length (T5 tokens)", "mean_sentence_length"),
    ("Mean SQL query length (T5 tokens)", "mean_sql_length"),
    ("Vocabulary size (natural language)", "nl_vocabulary_size"),
    ("Vocabulary size (SQL)", "sql_vocabulary_size"),
    ("Maximum sentence length (T5 tokens)", "max_sentence_length"),
    ("Maximum SQL query length (T5 tokens)", "max_sql_length"),
    ("SQL unknown-token occurrences", "sql_unknown_tokens"),
]


def read_split(data_dir, split):
    # Keep every example, including duplicate examples, in its original order.
    inputs = (data_dir / f"{split}.nl").read_text(encoding="utf-8").splitlines()
    targets = (data_dir / f"{split}.sql").read_text(encoding="utf-8").splitlines()
    assert inputs and len(inputs) == len(targets), f"Unaligned {split} files"
    assert all(x.strip() for x in inputs + targets), f"Blank example in {split}"
    return inputs, targets


def preprocess(text):
    # Baseline: match load_lines()'s outer-whitespace stripping only.
    # Preserve case, internal whitespace, SQL literals, punctuation, and order.
    # If the training preprocessing changes later, update this and rerun Q4.
    return text.strip()


def preprocess_sql(text, marker=None):
    text = text.strip()
    if marker is None:
        return text
    assert marker not in text, "SQL marker collision: choose a different reserved marker."
    return text.replace("<", marker)


def restore_sql(text, marker):
    return text.replace(marker, "<")


def validate_marker(tokenizer, marker):
    probes = ["<", "<=", ">", ">=", "x < 1200", "x <= 1200"]
    results = []
    for text in probes:
        encoded_text = preprocess_sql(text, marker)
        ids = tokenizer(encoded_text, add_special_tokens=False)["input_ids"]
        assert tokenizer.unk_token_id not in ids, f"Marker still yields UNK: {text}"
        decoded = tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        restored = restore_sql(decoded, marker).strip()
        assert restored == text, f"Tokenization round-trip failed: {text!r} -> {restored!r}"
        print(f"Operator check: {text!r} -> {encoded_text!r} -> {restored!r}", flush=True)
        results.append({"original": text, "encoded": encoded_text, "restored": restored})
    return results


def text_statistics(tokenizer, texts, batch_size=256):
    lengths = []
    vocabulary = set()
    unknown_count = 0
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start:start + batch_size],
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )["input_ids"]
        assert len(encoded) == len(texts[start:start + batch_size])
        for ids in encoded:
            lengths.append(len(ids))
            vocabulary.update(ids)
            unknown_count += ids.count(tokenizer.unk_token_id)
    assert len(lengths) == len(texts)
    return {
        "mean_length": mean(lengths),
        "vocabulary_size": len(vocabulary),
        "minimum_length": min(lengths),
        "maximum_length": max(lengths),
        "examples_over_512_tokens": sum(n > 512 for n in lengths),
        "unknown_tokens": unknown_count,
        "total_tokens": sum(lengths),
    }


def paired_statistics(tokenizer, inputs, targets):
    nl = text_statistics(tokenizer, inputs)
    sql = text_statistics(tokenizer, targets)
    return {
        "examples": len(inputs),
        "mean_sentence_length": nl["mean_length"],
        "mean_sql_length": sql["mean_length"],
        "nl_vocabulary_size": nl["vocabulary_size"],
        "sql_vocabulary_size": sql["vocabulary_size"],
        "max_sentence_length": nl["maximum_length"],
        "max_sql_length": sql["maximum_length"],
        "sql_unknown_tokens": sql["unknown_tokens"],
        "input_audit": nl,
        "target_audit": sql,
    }


def collect_statistics(tokenizer, data_dir, sql_marker=None):
    before, after, changes = {}, {}, {}
    for split in ["train", "dev"]:
        inputs, targets = read_split(data_dir, split)
        print(f"Measuring {split}: {len(inputs)} aligned pairs", flush=True)
        processed_inputs = [preprocess(s) for s in inputs]
        processed_targets = [preprocess_sql(s, sql_marker) for s in targets]
        if sql_marker:
            assert all(restore_sql(encoded, sql_marker) == raw.strip()
                       for raw, encoded in zip(targets, processed_targets))
        before[split] = paired_statistics(tokenizer, inputs, targets)
        after[split] = paired_statistics(tokenizer, processed_inputs, processed_targets)
        if sql_marker:
            assert after[split]["sql_unknown_tokens"] == 0, (
                f"Unknown SQL tokens remain in {split}; do not finalize preprocessing."
            )
            print(f"{split}: all {len(targets)} SQL text conversions are reversible; "
                  "processed SQL has zero unknown tokens.", flush=True)
        changes[split] = {
            "changed_nl_lines": sum(a != b for a, b in zip(inputs, processed_inputs)),
            "changed_sql_lines": sum(a != b for a, b in zip(targets, processed_targets)),
        }
    return before, after, changes


def formatted(value):
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def table_rows(splits, include_examples):
    return [
        [label, formatted(splits["train"][key]), formatted(splits["dev"][key])]
        for label, key in ROWS if include_examples or key != "examples"
    ]


def render_latex_table(rows, caption, label):
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\begin{tabularx}{\linewidth}{Yrr}", r"\toprule",
        r"Statistic & Train & Dev \\", r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([
        r"\bottomrule", r"\end{tabularx}",
        "\\caption{" + caption + "}", "\\label{" + label + "}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sql-marker", default=None,
                        help="Encode '<' using this collision-checked ordinary text marker.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import T5TokenizerFast

    print("Loading tokenizer only; no model weights or training.", flush=True)
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    probes = validate_marker(tokenizer, args.sql_marker) if args.sql_marker else []
    before, after, changes = collect_statistics(tokenizer, args.data_dir, args.sql_marker)
    expected = {"train": 4225, "dev": 466}
    for split, size in expected.items():
        assert before[split]["examples"] == size, f"Unexpected {split} example count"

    report = {
        "tokenizer_checkpoint": MODEL_NAME,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_size_for_reference_not_observed_vocabulary": len(tokenizer),
        "versions": {name: importlib.metadata.version(name)
                     for name in ["transformers", "tokenizers", "sentencepiece"]},
        "conventions": {
            "mean_lengths": "Mean emitted subword-token count per example; before truncation.",
            "special_tokens": "No added EOS/BOS or padding; add_special_tokens=False.",
            "vocabulary": "Distinct emitted token IDs, separately per split and modality; includes UNK if emitted.",
            "preprocessing": (
                "Outer-whitespace stripping only, matching load_lines(); no lowercasing, task prefix, or example removal."
                if not args.sql_marker else
                f"NL: outer-whitespace stripping only. SQL: strip outer whitespace, replace '<' by {args.sql_marker!r}; restore the marker to '<' after decoding and before SQL execution/evaluation. No tokenizer vocabulary changes, lowercasing, task prefix, or example removal."
            ),
            "splits": "Train and dev only; no test-set analysis.",
        },
        "data_sha256": {
            name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
            for name in ["train.nl", "train.sql", "dev.nl", "dev.sql"]
        },
        "changed_lines": changes,
        "sql_marker": args.sql_marker,
        "operator_round_trip_checks": probes,
        "before": before,
        "after": after,
    }
    (args.output_dir / "q4_statistics.json").write_text(json.dumps(report, indent=2))

    tex = []
    for phase, splits, include_examples in [
        ("before", before, True), ("after", after, False),
    ]:
        rows = table_rows(splits, include_examples)
        with (args.output_dir / f"q4_{phase}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Statistic", "Train", "Dev"])
            writer.writerows(rows)
        print(f"\nQ4 TABLE: {phase.upper()} PREPROCESSING")
        print("Statistic | Train | Dev")
        for row in rows:
            print(" | ".join(row))
        tex.append(render_latex_table(
            rows, f"Data statistics {phase} preprocessing.",
            f"tab:{phase}-preprocessing",
        ))
    (args.output_dir / "q4_tables.tex").write_text("\n\n".join(tex) + "\n")

    print("\nLength audit (untruncated T5 tokens):")
    for split in ["train", "dev"]:
        for modality, key in [("NL", "input_audit"), ("SQL", "target_audit")]:
            audit = after[split][key]
            print(f"{split} {modality}: max={audit['maximum_length']}; "
                  f"over 512={audit['examples_over_512_tokens']}; "
                  f"UNK tokens={audit['unknown_tokens']}")
    print("Changed lines:", json.dumps(changes))
    print("\nQ4 STATISTICS SAVED:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
