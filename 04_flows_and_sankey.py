"""
Step 4: turn switch stories into directed brand flows, a Sankey table,
and simple network metrics. Runs locally, no GPU, no model downloads.

Input : classified.csv (from 03_zero_shot.py)
Output: flows.csv                row-level flows, ready for Power BI
        sankey_data.csv          aggregated From -> To counts for the Sankey visual
        brand_metrics.csv        gained / lost / net / PageRank / community per brand
        flows_spotcheck.csv      60 random between-issuer flows for you to verify by hand

The direction rules are strict (they need explicit wording like "from X to Y"),
so they do most of the filtering. Zero-shot is used as a veto on top.

pip install pandas networkx
"""

import re

import pandas as pd

GATE = "veto"      # "veto": drop only clear intent/comparison records; "strict": zero-shot confirmed only
VETO_SCORE = 0.6   # how sure zero-shot must be before it vetoes a record
MIN_FLOW = 2       # hide Sankey links with fewer flows than this
SEED = 42

# keep in sync with 02_clean_and_candidates.py
BRAND_ALIASES = {
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
ALIAS_TO_BRAND = {a: b for b, al in BRAND_ALIASES.items() for a in al}
_alts = "|".join(re.escape(a) for a in sorted(ALIAS_TO_BRAND, key=len, reverse=True))
B = rf"\b(?:{_alts})\b"
GAP = r"[^.!?\n]{0,40}?"

DROP_VERBS = (r"(?:replac(?:e|ed|ing)|ditch(?:ed|ing)?|dump(?:ed|ing)?|dropp?(?:ed|ing)?|left|leav(?:e|ing)|"
              r"quit|clos(?:e|ed|ing)|cancel+(?:ed|ing)?|surrender(?:ed|ing)?|abandon(?:ed|ing)?)")
MOVE_VERBS = (r"(?:switch(?:ed|ing)?|convert(?:ed|ing)?|upgrad(?:e|ed|ing)|downgrad(?:e|ed|ing)|"
              r"migrat(?:e|ed|ing)|mov(?:e|ed|ing)|shift(?:ed|ing)?|chang(?:e|ed|ing))")
CONNECT = (r"(?:\bfor\b|\bwith\b|\bin favou?r of\b|"
           r"\band (?:then )?(?:moved|switched|went|shifted|took|got|opted|chose|going|moving|switching)\b(?: to| for| with)?|"
           r"\bthen (?:moved|switched|went)\b(?: to)?|->|→|=>)")

RULES = [
    ("from_to",   re.compile(rf"\bfrom\b{GAP}(?P<o>{B}){GAP}\b(?:to|into)\b{GAP}(?P<d>{B})", re.I)),
    ("to_from",   re.compile(rf"\b(?:to|into)\b{GAP}(?P<d>{B}){GAP}\bfrom\b{GAP}(?P<o>{B})", re.I)),
    ("drop_for",  re.compile(rf"\b{DROP_VERBS}\b{GAP}(?P<o>{B}){GAP}{CONNECT}{GAP}(?P<d>{B})", re.I)),
    ("move_x_to", re.compile(rf"\b{MOVE_VERBS}\b{GAP}(?P<o>{B}){GAP}\b(?:to|into)\b{GAP}(?P<d>{B})", re.I)),
    ("instead",   re.compile(rf"(?P<d>{B}){GAP}\binstead of\b{GAP}(?P<o>{B})", re.I)),
    ("arrow",     re.compile(rf"(?P<o>{B}){GAP}(?:->|-->|→|=>)\s*{GAP}(?P<d>{B})", re.I)),
]
BRAND_RX = re.compile(B, re.I)
# from/to rules need a real move or drop verb in the sentence, otherwise
# "from my experience ... compared to ..." reads as a switch
VERB_ANY = re.compile(rf"\b(?:{MOVE_VERBS}|{DROP_VERBS}|went|opted|chose)\b", re.I)
# questions, plans and negations are not switches that happened
HYPO_RX = re.compile(r"\?|\b(?:should i|shall i|thinking (?:of|about)|planning (?:to|on)|plan to|considering|"
                     r"wondering|want to|wanna|wish i|can i|how (?:to|do|can)|is it (?:worth|possible|better)|"
                     r"worth (?:it|switching|moving)|if i)\b", re.I)
NEG_RX = re.compile(rf"\b(?:not|never|didn['’]?t|don['’]?t|won['’]?t|wouldn['’]?t|haven['’]?t|without)\b"
                    rf"\W+(?:\w+\W+){{0,2}}(?:{MOVE_VERBS}|{DROP_VERBS})", re.I)
DROP_CUE = re.compile(rf"\b{DROP_VERBS}\b", re.I)
DEST_CUE = re.compile(r"\b(?:going (?:with|for)|go(?:ing)? with|moved to|switched to|went (?:with|to|for)|"
                      r"now (?:using|have|with)|opted for|chose|took|applied for)\b", re.I)


def canon(s):
    return ALIAS_TO_BRAND[s.lower()]


def pair_fallback(text):
    """Weak rule: one brand follows a drop cue, another follows a destination cue."""
    ments = [(m.start(), canon(m.group())) for m in BRAND_RX.finditer(text)]
    if len({b for _, b in ments}) != 2:
        return []
    origin, dest = set(), set()
    for pos, b in ments:
        before = text[max(0, pos - 45):pos]
        d = [m.end() for m in DROP_CUE.finditer(before)]
        v = [m.end() for m in DEST_CUE.finditer(before)]
        if d and (not v or max(d) > max(v)):
            origin.add(b)
        elif v:
            dest.add(b)
    if len(origin) == 1 and len(dest) == 1 and origin != dest:
        return [(origin.pop(), dest.pop(), "pair_cues")]
    return []


def extract(text):
    found = []
    for sent in re.split(r"(?<=[.!?])\s+|\n+", text):
        if HYPO_RX.search(sent) or NEG_RX.search(sent):
            continue
        for name, rx in RULES:
            if name in ("from_to", "to_from") and not VERB_ANY.search(sent):
                continue
            m = rx.search(sent)
            if m:
                found.append((canon(m.group("o")), canon(m.group("d")), name))
                break
    if not found and not (HYPO_RX.search(text) or NEG_RX.search(text)):
        found = pair_fallback(text)
    seen, out = set(), []
    for o, d, r in found:
        if (o, d) not in seen:
            seen.add((o, d))
            out.append((o, d, r))
    return out


def main():
    print("04_flows_and_sankey")
    df = pd.read_csv("classified.csv", dtype={"id": str})
    print(f"Extracting directions from all {len(df):,} classified records...")
    df["flows"] = df["text"].astype(str).apply(extract)
    df["has_flow"] = df["flows"].str.len() > 0

    df["bucket"] = df["zs_label"]
    conf = df["zs_label"] == "confirmed"
    df.loc[conf & (df["zs_score"] >= 0.5), "bucket"] = "confirmed >=0.5"
    df.loc[conf & (df["zs_score"] < 0.5), "bucket"] = "confirmed <0.5"

    print("\nWhere directed flows come from, by zero-shot bucket:")
    funnel = df.groupby("bucket").agg(records=("id", "size"), with_flow=("has_flow", "sum"))
    print(funnel.to_string())

    if GATE == "strict":
        keep = df["bucket"] == "confirmed >=0.5"
    else:
        keep = ~(df["zs_label"].isin(["intent", "comparison"]) & (df["zs_score"] >= VETO_SCORE))
    print(f"\nGate '{GATE}' keeps {int((keep & df['has_flow']).sum()):,} records with a direction")

    rows = []
    for _, r in df[keep & df["has_flow"]].iterrows():
        for o, d, rule in r["flows"]:
            rows.append({"id": r["id"], "type": r["type"], "date": r["date"], "origin": o,
                         "destination": d, "rule": rule, "intra_issuer": o == d,
                         "bucket": r["bucket"], "zs_score": round(float(r["zs_score"]), 3),
                         "text": str(r["text"])[:500]})
    fl = pd.DataFrame(rows)
    if fl.empty:
        raise SystemExit("No directed flows found.")

    fl["source"] = "From: " + fl["origin"]
    fl["target"] = "To: " + fl["destination"]
    fl.drop(columns="text").to_csv("flows.csv", index=False)

    inter = fl[~fl["intra_issuer"]]
    intra = fl[fl["intra_issuer"]]
    print(f"Flows: {len(fl):,}   between different issuers: {len(inter):,}   "
          f"within one issuer: {len(intra):,}")
    print(f"Posts: {int((inter['type'] == 'post').sum()):,}   comments: {int((inter['type'] == 'comment').sum()):,} (between issuers)")
    print("\nBy rule (weaker rules deserve a closer look in the spot-check):")
    print(fl["rule"].value_counts().to_string())
    print("\nBy zero-shot bucket:")
    print(fl["bucket"].value_counts().to_string())

    agg = (inter.groupby(["source", "target"]).size().reset_index(name="value")
           .sort_values("value", ascending=False))
    agg[agg["value"] >= MIN_FLOW].to_csv("sankey_data.csv", index=False)
    print("\nTop flows (From -> To):")
    print(agg.head(15).to_string(index=False))
    print(f"Sankey links kept (>= {MIN_FLOW}): {int((agg['value'] >= MIN_FLOW).sum())} of {len(agg)}")
    if len(intra):
        print("\nWithin-issuer product changes:")
        print(intra["origin"].value_counts().head(5).to_string())

    try:
        import networkx as nx
    except ImportError:
        nx = None
        print("\nnetworkx is not installed, skipping brand metrics. Run: python -m pip install networkx")
    if nx is not None:
        g = nx.DiGraph()
        for (o, d), n in inter.groupby(["origin", "destination"]).size().items():
            g.add_edge(o, d, weight=int(n))
        try:
            pr = nx.pagerank(g, weight="weight")
        except ImportError as e:  # networkx 3.x needs scipy for PageRank
            pr = {}
            print(f"\nPageRank skipped ({e}). To get it: python -m pip install scipy")
        comms = nx.community.louvain_communities(g.to_undirected(), weight="weight", seed=SEED)
        comm_of = {b: i for i, c in enumerate(comms) for b in c}
        m = pd.DataFrame({
            "brand": list(g.nodes),
            "lost_to_others": [g.out_degree(b, weight="weight") for b in g.nodes],
            "gained_from_others": [g.in_degree(b, weight="weight") for b in g.nodes],
        })
        m["net"] = m["gained_from_others"] - m["lost_to_others"]
        m["pagerank"] = m["brand"].map(pr).round(3)
        m["community"] = m["brand"].map(comm_of)
        m = m.sort_values("net", ascending=False)
        m.to_csv("brand_metrics.csv", index=False)
        print("\nBrand metrics (net = gained - lost):")
        print(m.to_string(index=False))

    inter.sample(min(60, len(inter)), random_state=SEED)[
        ["id", "origin", "destination", "rule", "bucket", "text"]].assign(
        is_real_switch="", direction_correct="").to_csv("flows_spotcheck.csv", index=False)
    print("\nWrote flows.csv, sankey_data.csv, brand_metrics.csv, flows_spotcheck.csv")


if __name__ == "__main__":
    main()
