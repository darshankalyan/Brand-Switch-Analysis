"""
Step 2: clean the recent pull, match brands properly, and find switch candidates.

Inputs (from 01b_collect_recent.py, same folder):
  posts_recent_raw.csv, comments_recent_raw.csv

Outputs:
  clean_all.csv           every cleaned record with brand + candidate flags
  candidates.csv          records that look like switch narratives
  validation_sample.csv   records for you to hand-label (flags hidden)
  validation_key.csv      the hidden flags, do not open until you finish labeling

pip install pandas
"""

import re
import warnings

import pandas as pd

warnings.filterwarnings("ignore", message="This pattern is interpreted")

SEED = 42

# True reproduces the run behind the published numbers (short alias list, upgrade/convert
# words count as switch words). False uses the wider alias list and drops upgrade/convert noise.
LEGACY_FILTER = True

# canonical brand -> aliases, matched on word boundaries (so "axis" won't hit "taxis")
ALIASES_V1 = {
    "HDFC": ["hdfc", "regalia", "infinia"],
    "ICICI": ["icici", "amazon pay icici", "amazon pay credit card"],
    "SBI": ["sbi", "sbi card", "cashback sbi"],
    "Axis": ["axis", "magnus"],
    "Amex": ["amex", "american express", "mrcc"],
    "Kotak": ["kotak"],
    "IDFC First": ["idfc"],
    "Yes Bank": ["yes bank"],
    "IndusInd": ["indusind"],
    "HSBC": ["hsbc"],
    "RBL": ["rbl"],
}
ALIASES_V2 = {
    "HDFC": ["hdfc", "regalia", "infinia", "millennia", "moneyback", "dcbm", "diners club black",
             "swiggy hdfc", "tata neu"],
    "ICICI": ["icici", "amazon pay icici", "amazon pay credit card", "coral", "sapphiro", "emeralde", "rubyx"],
    "SBI": ["sbi", "sbi card", "cashback sbi", "simplyclick", "simplysave"],
    "Axis": ["axis", "magnus", "atlas", "flipkart axis", "airtel axis"],
    "Amex": ["amex", "american express", "mrcc", "platinum travel"],
    "Kotak": ["kotak", "myntra kotak"],
    "IDFC First": ["idfc"],
    "Yes Bank": ["yes bank"],
    "IndusInd": ["indusind", "iconia"],
    "HSBC": ["hsbc", "travelone"],
    "RBL": ["rbl"],
    "BoB": ["bobcard", "bank of baroda", "eterna"],
    "Federal": ["federal bank", "scapia"],
    "AU": ["au bank", "au small finance"],
    "StanChart": ["standard chartered", "stanchart"],
    "Citi": ["citi", "citibank", "citi bank"],
    "OneCard": ["onecard"],
}
BRAND_ALIASES = ALIASES_V1 if LEGACY_FILTER else ALIASES_V2
BRAND_RX = {
    b: re.compile(r"\b(" + "|".join(re.escape(a) for a in al) + r")\b", re.I)
    for b, al in BRAND_ALIASES.items()
}

# broad on purpose: high recall, the classifier decides precision later
SWITCH_V1 = (
    r"\b(switch(ed|ing|es)?|mov(e|ed|ing) (from|to|over)|migrat(e|ed|ing)|"
    r"clos(e|ed|ing) (my|the|it)|cancel(l)?(ed|ing)?|surrender(ed)?|"
    r"downgrad(e|ed|ing)|upgrad(e|ed|ing)|replac(e|ed|ing)|ditch(ed|ing)?|"
    r"instead of|convert(ed)? to|product change|"
    r"(left|leaving|quit|dropped|dumped) (my |the )?(\w+ ){0,2}(bank|card))\b"
)
SWITCH_V2 = (
    r"\b(switch(ed|ing|es)?|mov(e|ed|ing) (from|to|over)|migrat(e|ed|ing)|"
    r"clos(e|ed|ing) (my|the|it)|cancel(l)?(ed|ing)?|surrender(ed)?|"
    r"replac(e|ed|ing)|ditch(ed|ing)?|instead of|"
    r"(left|leaving|quit|dropped|dumped) (my |the )?(\w+ ){0,2}(bank|card))\b"
)
SWITCH_RX = re.compile(SWITCH_V1 if LEGACY_FILTER else SWITCH_V2, re.I)
FROM_TO_RX = re.compile(r"\bfrom\b.{0,60}\bto\b", re.I | re.S)

BOT_AUTHORS = {"automoderator", "[deleted]", "reddit"}


def load_posts():
    p = pd.read_csv("posts_recent_raw.csv", dtype={"id": str}).drop_duplicates(subset="id")
    p["text"] = p["title"].fillna("") + "\n" + p["selftext"].fillna("")
    p["type"] = "post"
    p["post_id"] = p["id"]
    p["parent_title"] = ""
    return p[["id", "type", "post_id", "parent_title", "author", "created_utc", "score", "text"]]


def load_comments(posts):
    c = pd.read_csv("comments_recent_raw.csv", dtype={"id": str, "link_id": str}).drop_duplicates(subset="id")
    c["text"] = c["body"].fillna("")
    c["type"] = "comment"
    c["post_id"] = c["link_id"].str.replace("t3_", "", regex=False)
    titles = posts.set_index("id")["text"].str.split("\n").str[0]
    c["parent_title"] = c["post_id"].map(titles).fillna("")
    return c[["id", "type", "post_id", "parent_title", "author", "created_utc", "score", "text"]]


posts = load_posts()
comments = load_comments(posts)
df = pd.concat([posts, comments], ignore_index=True)
n0 = len(df)

# ---- clean ----
df = df.drop_duplicates(subset=["type", "id"])
low = df["text"].str.strip().str.lower()
df = df[~low.isin(["[removed]", "[deleted]", "[removed]\n", "[deleted]\n"])]
df = df[~df["author"].fillna("").str.lower().isin(BOT_AUTHORS)]
df = df[df["text"].str.strip().str.len() >= 25]
df = df.reset_index(drop=True)
df["date"] = pd.to_datetime(df["created_utc"], unit="s")
n1 = len(df)

# ---- brands (look at parent title too for comments, since replies often omit the brand) ----
scan = df["text"] + " " + df["parent_title"]
for b, rx in BRAND_RX.items():
    df[f"b_{b}"] = scan.str.contains(rx)
brand_cols = [f"b_{b}" for b in BRAND_ALIASES]
df["n_brands"] = df[brand_cols].sum(axis=1)
df["brands"] = df[brand_cols].apply(lambda r: "|".join(c[2:] for c in brand_cols if r[c]), axis=1)
df["has_brand"] = df["n_brands"] > 0

# ---- switch candidates ----
df["switch_kw"] = df["text"].str.contains(SWITCH_RX)
df["from_to"] = (df["n_brands"] >= 2) & df["text"].str.contains(FROM_TO_RX)
df["is_candidate"] = df["has_brand"] & (df["switch_kw"] | df["from_to"])

out_cols = ["id", "type", "post_id", "date", "author", "score", "brands", "n_brands",
            "switch_kw", "from_to", "is_candidate", "parent_title", "text"]
df[out_cols].to_csv("clean_all.csv", index=False)
cand = df[df["is_candidate"]]
cand[out_cols].to_csv("candidates.csv", index=False)

# ---- validation sample: 200 candidates + 100 non-candidates, flags hidden ----
branded = df[df["has_brand"]]
s_c = branded[branded["is_candidate"]].sample(min(200, int(branded["is_candidate"].sum())), random_state=SEED)
s_n = branded[~branded["is_candidate"]].sample(min(100, int((~branded["is_candidate"]).sum())), random_state=SEED)
samp = pd.concat([s_c, s_n]).sample(frac=1, random_state=SEED).reset_index(drop=True)
samp["text_short"] = samp["text"].str.slice(0, 1500)
samp["label"] = ""        # switch_confirmed / switch_intent / comparison / complaint_no_switch / other
samp["origin"] = ""       # only for switch_confirmed: brand they left
samp["destination"] = ""  # only for switch_confirmed: brand they moved to
samp[["id", "type", "parent_title", "text_short", "label", "origin", "destination"]].to_csv(
    "validation_sample.csv", index=False)
samp[["id", "type", "is_candidate", "brands"]].to_csv("validation_key.csv", index=False)

# ---- summary ----
print(f"Loaded {n0:,} -> after cleaning {n1:,} ({n0 - n1:,} dropped: removed/deleted/bots/very short/dupes)")
print(f"Mention a brand (word-boundary match): {int(df['has_brand'].sum()):,}")
print(f"Switch candidates: {len(cand):,}  "
      f"(posts {int((cand['type'] == 'post').sum()):,}, comments {int((cand['type'] == 'comment').sum()):,})")
print(f"  with 2+ brands (direction is extractable): {int((cand['n_brands'] >= 2).sum()):,}")
print("\nCandidates per month:")
print(cand.groupby(cand["date"].dt.to_period("M")).size().to_string())
print("\nCandidates mentioning each brand:")
print(pd.Series({b: int(cand[f"b_{b}"].sum()) for b in BRAND_ALIASES}).sort_values(ascending=False).to_string())
print("\nWrote clean_all.csv, candidates.csv, validation_sample.csv, validation_key.csv")
