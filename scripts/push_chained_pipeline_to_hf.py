import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, upload_file


def load_examples(path: Path):
    return json.loads(path.read_text())


def split(examples, train_ratio=0.8, val_ratio=0.1):
    n = len(examples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = examples[:n_train]
    val = examples[n_train:n_train + n_val]
    test = examples[n_train + n_val:]
    return train, val, test


def build_dataset_card(examples, repo_id: str):
    n = len(examples)
    lengths = Counter(e.get('chain_length', len(e.get('transform_chain', []))) for e in examples)
    transform_freq = Counter()
    co_occurrence = defaultdict(Counter)
    for e in examples:
        chain = e.get('transform_chain', [])
        transform_freq.update(chain)
        # simple pairwise co-occurrence (unordered)
        unique = sorted(set(chain))
        for i, a in enumerate(unique):
            for b in unique[i+1:]:
                co_occurrence[a][b] += 1
                co_occurrence[b][a] += 1

    lines = []
    lines.append(f"# {repo_id}\n")
    lines.append("Automated chained DSL transformation dataset including all current generator functions (mirror, rotate, symmetry completion, flood fill, move shape, counting/marking, create shape, connect, extract pattern variants, repeat pattern variants).\n")
    lines.append("## Summary\n")
    lines.append(f"Total examples: **{n}**  ")
    lines.append("Chain length distribution:")
    for L in sorted(lengths):
        lines.append(f"- length {L}: {lengths[L]}")
    lines.append("\nUnique transforms: **" + str(len(transform_freq)) + "**\n")
    lines.append("### Transform Frequency\n")
    for name, cnt in transform_freq.most_common():
        lines.append(f"- {name}: {cnt}")
    lines.append("\n### Features\n")
    lines.append("Each row has: \n- train_input1 / train_output1 \n- train_input2 / train_output2 \n- test_input / test_output \n- solution (Python lines executing the chain) \n- transform_chain (list of transform spec names) \n- chain_length (int)\n")
    lines.append("### Intended Use\n")
    lines.append("For training and evaluating models on multi-step spatial reasoning over a small DSL of grid transformations. Evaluate stratified by chain_length.\n")
    lines.append("### License\n")
    lines.append("CC-BY-4.0 (adjust if needed).\n")
    lines.append("### Generation Timestamp\n")
    lines.append(datetime.utcnow().isoformat() + "Z\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default='JSON_training/ChainedPipeline.json')
    ap.add_argument('--repo', default='middles/dsl-chained-pipeline-v0.1.0')
    ap.add_argument('--private', action='store_true')
    ap.add_argument('--push', action='store_true')
    ap.add_argument('--token', default=None, help='HF token (or set HF_TOKEN / HUGGINGFACEHUB_API_TOKEN env)')
    ap.add_argument('--no-card', action='store_true', help='Skip README card upload')
    args = ap.parse_args()

    examples = load_examples(Path(args.json))
    train, val, test = split(examples)

    def to_ds(data):
        return Dataset.from_list(data)

    dset = DatasetDict({
        'train': to_ds(train),
        'validation': to_ds(val),
        'test': to_ds(test)
    })

    print(dset)
    if not args.push:
        print("(Dry run) Add --push to upload. Repo:", args.repo)
        return

    token = args.token or os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACEHUB_API_TOKEN')
    if not token:
        raise SystemExit("No token provided. Use --token or set HF_TOKEN env var.")

    api = HfApi(token=token)
    # Create repo if missing
    api.create_repo(repo_id=args.repo, repo_type='dataset', private=args.private, exist_ok=True)
    # Push dataset splits
    dset.push_to_hub(args.repo, private=args.private, token=token)
    print(f"Pushed dataset splits to {args.repo}")

    if not args.no_card:
        card_text = build_dataset_card(examples, args.repo)
        tmp_card = Path('TMP_DATASET_CARD.md')
        tmp_card.write_text(card_text)
        upload_file(
            path_or_fileobj=str(tmp_card),
            path_in_repo='README.md',
            repo_id=args.repo,
            repo_type='dataset',
            token=token
        )
        print("Uploaded dataset card README.md")
        tmp_card.unlink(missing_ok=True)

    print("Done.")


if __name__ == '__main__':
    main()
