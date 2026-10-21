"""
Step 5: BERTopic on the switch stories behind your brand flows, to find the reasons people give.

Run on Colab/Kaggle in the same session as 03 (or locally if you have the patience).
  pip install -q bertopic sentence-transformers umap-learn hdbscan scikit-learn pandas

Input : classified.csv and flows.csv (flows.csv picks which records to model)
Output: topics_summary.csv    one row per topic: size, top words, example posts
        doc_topics.csv        which topic each confirmed switch landed in
        topics_by_origin.csv  topic counts by the brand people left (needs flows.csv)

BERTopic only gives keyword clusters. You name the topics yourself after
reading topics_summary.csv ("annual fee devaluation", "lounge access", ...).
"""

import os

import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from umap import UMAP

SEED = 42
BRAND_WORDS = {"hdfc", "regalia", "infinia", "icici", "amazon", "pay", "sbi", "axis", "magnus", "amex",
               "american", "express", "mrcc", "kotak", "idfc", "yes", "bank", "indusind", "hsbc", "rbl"}
EXTRA_STOP = {"card", "cards", "credit", "banks", "like", "just", "also", "one", "get", "got", "use",
              "using", "know", "would", "could", "https", "http", "www", "com", "reddit", "thanks",
              "hi", "hello", "really", "going", "think", "want", "need"}
STOP = sorted(set(ENGLISH_STOP_WORDS) | BRAND_WORDS | EXTRA_STOP)


def main():
    print("05_bertopic")
    df = pd.read_csv("classified.csv", dtype={"id": str})
    if os.path.exists("flows.csv"):
        # best input: the posts and comments that produced a directed flow
        fl = pd.read_csv("flows.csv", dtype={"id": str})
        keys = set(zip(fl["type"], fl["id"]))
        conf = df[[(t, i) in keys for t, i in zip(df["type"], df["id"])]].copy()
        print(f"Using {len(conf):,} records that produced a brand flow (from flows.csv)")
    else:
        conf = df[df["is_confirmed"]].copy()
        print(f"No flows.csv found, using {len(conf):,} zero-shot confirmed switches")
    conf["doc"] = conf["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    conf = conf[conf["doc"].str.len() >= 80].reset_index(drop=True)  # very short text gives junk topics
    docs = conf["doc"].tolist()
    if len(docs) < 100:
        raise SystemExit(f"Only {len(docs)} usable confirmed switches, too few for topic modeling.")
    min_topic = max(8, min(25, len(docs) // 100))
    print(f"{len(docs):,} documents, min topic size {min_topic}")

    topic_model = BERTopic(
        embedding_model=SentenceTransformer("all-MiniLM-L6-v2"),
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED),
        hdbscan_model=HDBSCAN(min_cluster_size=min_topic, metric="euclidean",
                              cluster_selection_method="eom", prediction_data=True),
        vectorizer_model=CountVectorizer(stop_words=STOP, ngram_range=(1, 2), min_df=3),
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(docs)
    conf["topic"] = topics

    rows = []
    for t in sorted(set(topics)):
        if t == -1:
            continue
        words = ", ".join(w for w, _ in topic_model.get_topic(t)[:8])
        members = conf[conf["topic"] == t]
        ex = members["doc"].sample(min(2, len(members)), random_state=SEED).str.slice(0, 220).tolist()
        rows.append({"topic": t, "count": len(members), "top_words": words,
                     "example_1": ex[0], "example_2": ex[1] if len(ex) > 1 else ""})
    summary = pd.DataFrame(rows).sort_values("count", ascending=False)
    summary.to_csv("topics_summary.csv", index=False)
    conf[["id", "type", "date", "topic", "doc"]].to_csv("doc_topics.csv", index=False)

    outliers = (conf["topic"] == -1).mean()
    print(f"\n{len(summary)} topics found, {outliers:.0%} of documents not assigned to any topic (normal for short text)")
    print(summary[["topic", "count", "top_words"]].head(15).to_string(index=False))

    if os.path.exists("flows.csv"):
        fl = pd.read_csv("flows.csv", dtype={"id": str})[["id", "type", "origin"]]
        j = conf[conf["topic"] != -1][["id", "type", "topic"]].merge(fl, on=["id", "type"])
        if len(j):
            j.groupby(["origin", "topic"]).size().unstack(fill_value=0).to_csv("topics_by_origin.csv")
            print("Wrote topics_by_origin.csv")
    print("Wrote topics_summary.csv, doc_topics.csv")


if __name__ == "__main__":
    main()
