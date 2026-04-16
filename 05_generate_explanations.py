
import pandas as pd
import json
import re
from collections import Counter

# -----------------------------
# CONFIG
# -----------------------------
INPUT_DATA = "hybrid_results_enriched_csv"
OUTPUT_JSON = "explanation_output.json"
TOP_K = 10
QUERY = "wireless headphones for studying"


import os
import glob
import pandas as pd


def load_input_data(path: str) -> pd.DataFrame:
    """
    Supports:
    1. single local CSV file
    2. Spark CSV output folder containing part-* files
    3. Spark parquet output folder containing part-* files
    4. single local parquet file
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found locally: {path}")

    # Case 1: single file
    if os.path.isfile(path):
        if path.endswith(".csv"):
            return pd.read_csv(path, quotechar='"', escapechar='\\', engine='python')
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported file type: {path}")

    # Case 2: directory
    if os.path.isdir(path):
        part_files = sorted(
            f for f in glob.glob(os.path.join(path, "part-*"))
            if not os.path.basename(f).startswith(".")
        )

        if not part_files:
            raise FileNotFoundError(f"No part-* files found in directory: {path}")

        # Detect parquet vs csv by extension/name
        parquet_like = any(
            f.endswith(".parquet") or ".snappy" in f or ".parq" in f
            for f in part_files
        )

        if parquet_like:
            dfs = [pd.read_parquet(f) for f in part_files]
            return pd.concat(dfs, ignore_index=True)

        # Otherwise assume CSV-style Spark output
        dfs = [pd.read_csv(f, quotechar='"', escapechar='\\', engine='python') for f in part_files]
        return pd.concat(dfs, ignore_index=True)

    raise ValueError(f"Unsupported input path: {path}")

# -----------------------------
# FEATURE PATTERNS
# -----------------------------
FEATURE_PATTERNS = {
    "wireless": r"wireless|bluetooth",
    "microphone": r"microphone|\bmic\b",
    "over-ear": r"over[- ]ear",
    "noise cancelling": r"noise cancell|anc",
    "battery life": r"battery|playtime|hours|\b60h\b|\b100h\b",
    "school-friendly": r"school|student|classroom|study",
    "tv/home use": r"\btv\b|transmitter|optical|rca|home",
    "curved display": r"curved",
    "high refresh rate": r"144hz|165hz|180hz|200hz|240hz|refresh",
    "fast response": r"1ms|2ms|response",
    "high resolution": r"qhd|uhd|4k|1440p|1080p|fhd|wqhd",
    "ergonomic": r"ergonomic|vertical|wrist|carpal",
    "portable": r"portable|foldable|lightweight",
    "comfort": r"comfort|comfortable|soft|padded",
    "built-in speakers": r"built[- ]in speakers|speakers",
    "low latency": r"low latency|no delay",
}

# -----------------------------
# LOAD CSV
# -----------------------------
def load_csv(path):
    return pd.read_csv(
        path,
        quotechar='"',
        escapechar='\\',
        engine='python'
    )

# -----------------------------
# HELPERS
# -----------------------------
def safe_get(row, col, default=0):
    val = row.get(col, default)
    return default if pd.isna(val) else val

def clean_title(title):
    if title is None:
        return ""
    title = str(title)
    short_title = re.split(r"[,\-\|\(\)\[\]]", title)[0].strip()
    words = short_title.split()
    if len(words) > 6:
        short_title = " ".join(words[:6]) + "..."
    return short_title

def extract_features(title, product_text=""):
    text = f"{str(title)} {str(product_text)}".lower()
    features = []

    for label, pattern in FEATURE_PATTERNS.items():
        if re.search(pattern, text):
            features.append(label)

    return features

def infer_product_family(query, titles):
    text = f"{query} " + " ".join([str(t) for t in titles])
    text = text.lower()

    if re.search(r"headphone|earbud|headset|earphone", text):
        return "audio"
    if re.search(r"monitor|display|screen", text):
        return "monitor"
    if re.search(r"\bmouse\b", text):
        return "mouse"
    if re.search(r"laptop|notebook|ultrabook", text):
        return "laptop"
    if re.search(r"keyboard", text):
        return "keyboard"
    return "electronics"

def get_trust_label(avg_rating, review_count, verified_ratio):
    if avg_rating >= 4.3 and review_count >= 40:
        return "highly trusted"
    if avg_rating >= 4.0 and review_count >= 40:
        return "well established"
    if avg_rating >= 4.0 and review_count >= 20:
        return "well reviewed"
    if avg_rating >= 3.5 and review_count >= 10:
        return "reasonably rated"
    return "more weakly supported by buyer feedback"

def get_price_bucket(price):
    if price is None or pd.isna(price):
        return "unknown"
    if price < 50:
        return "budget"
    if price < 150:
        return "mid-range"
    return "premium"

# -----------------------------
# SCORING
# -----------------------------
def choose_best_overall(df):
    temp = df.copy()

    rmin, rmax = temp["avg_rating"].min(), temp["avg_rating"].max()
    temp["rating_norm"] = (temp["avg_rating"] - rmin) / ((rmax - rmin) + 1e-9)

    cmin, cmax = temp["review_count"].min(), temp["review_count"].max()
    temp["review_norm"] = (temp["review_count"] - cmin) / ((cmax - cmin) + 1e-9)

    hmin, hmax = temp["total_helpful_votes"].min(), temp["total_helpful_votes"].max()
    temp["helpful_norm"] = (temp["total_helpful_votes"] - hmin) / ((hmax - hmin) + 1e-9)

    temp["verified_norm"] = temp["verified_purchase_ratio"].fillna(0.0)

    temp["explanation_rank"] = (
        0.50 * temp["final_score"] +
        0.20 * temp["rating_norm"] +
        0.15 * temp["review_norm"] +
        0.10 * temp["verified_norm"] +
        0.05 * temp["helpful_norm"]
    )

    return temp.sort_values("explanation_rank", ascending=False).iloc[0].to_dict()

def choose_best_value(df, best_overall_parent_asin=None):
    temp = df.copy()

    if best_overall_parent_asin is not None:
        temp = temp[temp["parent_asin"] != best_overall_parent_asin]

    temp["value_bonus"] = temp["price_bucket"].map({
        "budget": 0.15,
        "mid-range": 0.08,
        "unknown": 0.03,
        "premium": 0.0
    }).fillna(0.0)

    rmin, rmax = temp["avg_rating"].min(), temp["avg_rating"].max()
    temp["rating_norm"] = (temp["avg_rating"] - rmin) / ((rmax - rmin) + 1e-9)

    cmin, cmax = temp["review_count"].min(), temp["review_count"].max()
    temp["review_norm"] = (temp["review_count"] - cmin) / ((cmax - cmin) + 1e-9)

    temp["value_rank"] = (
        0.65 * temp["final_score"] +
        0.20 * temp["value_bonus"] +
        0.10 * temp["rating_norm"] +
        0.05 * temp["review_norm"]
    )

    return temp.sort_values("value_rank", ascending=False).iloc[0].to_dict()

def choose_specialized_pick(df, feature_name, exclude_asins=None):
    exclude_asins = exclude_asins or set()
    matches = df[
        (df["key_features"].apply(lambda xs: feature_name in xs)) &
        (~df["parent_asin"].isin(exclude_asins))
    ]
    if len(matches) == 0:
        return None
    return matches.sort_values("final_score", ascending=False).iloc[0].to_dict()

# -----------------------------
# PAYLOAD BUILDER
# -----------------------------
def build_payload(query, df):
    df = df.copy().reset_index(drop=True)

    if "product_text" not in df.columns:
        df["product_text"] = ""

    if "price_bucket" not in df.columns:
        if "price" in df.columns:
            df["price_bucket"] = df["price"].apply(get_price_bucket)
        else:
            df["price_bucket"] = "unknown"

    # new title update
    df["short_title"] = df["title"].apply(clean_title)

    enriched_rows = []
    feature_counter = Counter()

    for _, row in df.iterrows():
        title = safe_get(row, "title", "")
        product_text = safe_get(row, "product_text", "")
        features = extract_features(title, product_text)

        for f in features:
            feature_counter[f] += 1

        item = row.to_dict()
        item["display_title"] = row.get("short_title", row.get("title", ""))
        item["key_features"] = features
        item["avg_rating"] = safe_get(row, "avg_rating", 0.0)
        item["review_count"] = int(safe_get(row, "review_count", 0))
        item["total_helpful_votes"] = int(safe_get(row, "total_helpful_votes", 0))
        item["verified_purchase_ratio"] = float(safe_get(row, "verified_purchase_ratio", 0.0))
        item["price_bucket"] = safe_get(row, "price_bucket", "unknown")
        item["trust_label"] = get_trust_label(
            item["avg_rating"],
            item["review_count"],
            item["verified_purchase_ratio"]
        )
        enriched_rows.append(item)

    enriched_df = pd.DataFrame(enriched_rows)

    top_features = [f for f, _ in feature_counter.most_common(5)]
    best_overall = choose_best_overall(enriched_df)
    best_value = choose_best_value(enriched_df, best_overall_parent_asin=best_overall["parent_asin"])

    used_asins = {best_overall["parent_asin"], best_value["parent_asin"]}

    specialized_picks = {}
    for feature_name, label in [
        ("school-friendly", "best_for_school"),
        ("tv/home use", "best_for_home_use"),
        ("comfort", "best_for_comfort"),
        ("battery life", "best_for_long_sessions"),
        ("ergonomic", "best_for_ergonomics"),
        ("high refresh rate", "best_for_performance"),
    ]:
        chosen = choose_specialized_pick(enriched_df, feature_name, exclude_asins=used_asins)
        if chosen is not None:
            specialized_picks[label] = chosen
            used_asins.add(chosen["parent_asin"])

    return {
        "query": query,
        "product_family": infer_product_family(query, enriched_df["title"].tolist()),
        "top_common_attributes": top_features,
        "best_overall": best_overall,
        "best_value": best_value,
        "specialized_picks": specialized_picks,
        "top_results": enriched_df.to_dict(orient="records")
    }

# -----------------------------
# TEXT GENERATION
# -----------------------------
def generate_overview(payload):
    query = payload["query"]
    attrs = payload["top_common_attributes"]
    best = payload["best_overall"]
    best_value = payload["best_value"]

    rating = best.get("avg_rating", 0)
    reviews = int(best.get("review_count", 0))
    trust_label = best.get("trust_label", "well reviewed")

    intro = (
        f"For '{query}', the strongest results emphasize {', '.join(attrs[:4])}."
        if attrs else
        f"For '{query}', the strongest results balance relevance, quality, and user fit."
    )

    body = (
        "The top-ranked products combine query relevance with personalized ranking signals, "
        "while buyer feedback helps distinguish strong all-round options from more niche alternatives."
    )

    confidence = (
        f"The leading option is {trust_label}, with an average rating of {rating:.1f} from {reviews:,} reviews."
        if rating > 0 and reviews > 0 else
        "The leading option stands out mainly because of its ranking and feature fit."
    )

    quick_picks = [f"Best overall: {best['display_title']}"]
    if best_value:
        quick_picks.append(f"Best value: {best_value['display_title']}")

    for label, item in payload["specialized_picks"].items():
        quick_picks.append(f"{label.replace('_', ' ').title()}: {item['display_title']}")

    quick_text = "Quick picks:\n- " + "\n- ".join(quick_picks[:5])

    return f"{intro}\n\n{body}\n\n{confidence}\n\n{quick_text}"

def generate_why_best(payload):
    best = payload["best_overall"]
    features = best.get("key_features", [])
    rating = best.get("avg_rating", 0)
    reviews = int(best.get("review_count", 0))
    verified = best.get("verified_purchase_ratio", 0.0)
    price_bucket = best.get("price_bucket", "unknown")

    feature_text = ", ".join(features[:3]) if features else "strong relevance"

    text = (
        f"{best['display_title']} ranks first because it combines {feature_text} "
        f"with the strongest overall blend of relevance, personalized ranking, and buyer confidence signals."
    )

    if rating > 0 and reviews > 0:
        text += f" It is rated {rating:.1f} from {reviews:,} reviews."

    if verified > 0:
        text += " A large share of its reviews also come from verified purchases."

    if price_bucket != "unknown":
        text += f" It sits in the {price_bucket} segment."

    return text

# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Loading enriched hybrid data...")
    df = load_input_data(INPUT_DATA)

    print("Preview:")
    print(df.head())

    top_df = df.sort_values("final_score", ascending=False).head(TOP_K).copy()

    print("Building payload...")
    payload = build_payload(QUERY, top_df)

    print("Generating explanation text...")
    overview = generate_overview(payload)
    why_best = generate_why_best(payload)

    output = {
        "query": QUERY,
        "overview": overview,
        "why_best_overall": why_best,
        "best_overall": payload["best_overall"],
        "best_value": payload["best_value"],
        "specialized_picks": payload["specialized_picks"],
        "top_results": payload["top_results"]
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print("\n--- OVERVIEW ---\n")
    print(overview)
    print("\n--- WHY BEST OVERALL ---\n")
    print(why_best)
    print(f"\nSaved {OUTPUT_JSON}")

if __name__ == "__main__":
    main()