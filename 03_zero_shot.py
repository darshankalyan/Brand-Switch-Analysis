"""
Step 3: zero-shot classification of switch candidates.

Best on a free GPU (Colab or Kaggle). It also runs on a laptop CPU, just slowly:
set LIMIT below so it only does the records most likely to give flows.

  pip install "transformers[sentencepiece]" accelerate protobuf pandas torch
  python 03_zero_shot.py

Progress is saved every chunk to classified_partial.csv, so a restart resumes.
Ctrl+C is safe: it stops and still writes classified.csv from what is done.
Records with 2+ brands go first (those are the ones that can give a direction).
"""

import os
import time

import pandas as pd
import torch
from transformers import pipeline

MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"  # alt: "facebook/bart-large-mnli"
INPUT = "candidates.csv"
PARTIAL = "classified_partial.csv"
OUTPUT = "classified.csv"
LIMIT = None          # CPU laptop: set to 7543 (the two-brand records). GPU: leave None.
CHUNK = 200
BATCH = 32
MAX_CHARS = 1200
CONFIRMED_THRESHOLD = 0.5

LABELS = {
    "confirmed": "someone who already switched or moved from one credit card or bank to another",
    "intent": "someone considering switching to a different credit card or bank",
    "comparison": "comparing credit cards or banks",
    "other": "a question, complaint, or other discussion about credit cards",
}
TEMPLATE = "This text is about {}"  # the wording this model family was trained on


def build_text(row):
    text = str(row["text"])[:MAX_CHARS]
    if row["type"] == "comment" and isinstance(row["parent_title"], str) and row["parent_title"]:
        return f"Reply to the post '{row['parent_title'][:200]}': {text}"
    return text


def write_output():
    if not os.path.exists(PARTIAL):
        print("Nothing classified yet.")
        return
    res = pd.read_csv(PARTIAL, dtype={"id": str}).drop_duplicates(subset=["type", "id"])
    res["is_confirmed"] = (res["zs_label"] == "confirmed") & (res["zs_score"] >= CONFIRMED_THRESHOLD)
    res.to_csv(OUTPUT, index=False)

    print(f"\nClassified {len(res):,} records")
    print(res["zs_label"].value_counts().to_string())
    n_conf = int(res["is_confirmed"].sum())
    print(f"Confirmed switches (score >= {CONFIRMED_THRESHOLD}): {n_conf:,}")
    if n_conf:
        print("\nEyeball these 12 before trusting anything:")
        sample = res[res["is_confirmed"]].sample(min(12, n_conf), random_state=1)
        for _, r in sample.iterrows():
            print(f"  [{r['zs_score']:.2f}] {str(r['text'])[:160]!r}")


def main():
    df = pd.read_csv(INPUT, dtype={"id": str})
    df = df.sort_values(["n_brands", "date"], ascending=[False, False]).reset_index(drop=True)
    if LIMIT:
        df = df.head(LIMIT)

    done = set()
    if os.path.exists(PARTIAL):
        prev = pd.read_csv(PARTIAL, dtype={"id": str})
        done = set(zip(prev["type"], prev["id"]))
        print(f"Resuming: {len(done):,} already classified")
    todo = df[[(t, i) not in done for t, i in zip(df["type"], df["id"])]]
    print(f"{len(todo):,} records to classify out of {len(df):,}")

    device = 0 if torch.cuda.is_available() else -1
    if device == -1:
        print("No GPU found, running on CPU. Expect a few records per second at best.")
    clf = pipeline("zero-shot-classification", model=MODEL, device=device)

    names = list(LABELS)
    phrases = [LABELS[n] for n in names]
    phrase_to_name = {LABELS[n]: n for n in names}
    start = time.time()

    try:
        for s in range(0, len(todo), CHUNK):
            chunk = todo.iloc[s:s + CHUNK].copy()
            texts = [build_text(r) for _, r in chunk.iterrows()]
            out = clf(texts, candidate_labels=phrases, hypothesis_template=TEMPLATE,
                      multi_label=False, batch_size=BATCH, truncation=True)
            if isinstance(out, dict):
                out = [out]

            scores = {n: [] for n in names}
            top, top_score = [], []
            for o in out:
                by_name = {phrase_to_name[l]: sc for l, sc in zip(o["labels"], o["scores"])}
                for n in names:
                    scores[n].append(by_name[n])
                best = max(by_name, key=by_name.get)
                top.append(best)
                top_score.append(by_name[best])

            chunk["zs_label"] = top
            chunk["zs_score"] = top_score
            for n in names:
                chunk[f"zs_{n}"] = scores[n]
            chunk.to_csv(PARTIAL, mode="a", header=not os.path.exists(PARTIAL), index=False)

            n_done = min(s + CHUNK, len(todo))
            rate = n_done / max(time.time() - start, 1)
            eta = (len(todo) - n_done) / max(rate, 0.01) / 60
            print(f"{n_done:,}/{len(todo):,}  {rate:.1f} rec/s  ~{eta:.0f} min left", flush=True)
    except KeyboardInterrupt:
        print("\nStopped by you. Writing results from what is done so far.")

    write_output()


if __name__ == "__main__":
    main()
