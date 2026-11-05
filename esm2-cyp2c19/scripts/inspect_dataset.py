from pathlib import Path

import pandas as pd
import requests
import re

urn = "urn:mavedb:00001199-a-1"

url = f"https://api.mavedb.org/api/v1/score-sets/{urn}/scores"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
data_dir = PROJECT_ROOT / "data" / "raw"
data_dir.mkdir(parents=True, exist_ok=True)

response = requests.get(url, timeout=60)
response.raise_for_status()

csv_path = data_dir / "cyp2c19_scores.csv"
csv_path.write_bytes(response.content)

df = pd.read_csv(csv_path)

print("\n--- Experimental Data ---")
print("Rows and columns:", df.shape)
print("Column names:", df.columns.tolist())
print("\nFirst 3 records:")
print(df.head(3).to_string(index=False))

print("\n--- First Variant Inspection ---")

variant = df.iloc[0]

print("Python type:", type(variant))
print("Mutation:", variant["hgvs_pro"])
print("Score:", variant["score"])

print("Replicate 1:", variant["score_rep1"])
print("Replicate 2:", variant["score_rep2"])
print("Replicate 3:", variant["score_rep3"])

print("\n--- Data Completeness ---")

print("Missing protein annotations:",
      df["hgvs_pro"].isna().sum())

print("Missing experimental scores:",
      df["score"].isna().sum())

mutation = df.iloc[0]["hgvs_pro"]

pattern = r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})"

match = re.fullmatch(pattern, mutation)

if match is None:
    raise ValueError(f"Unsupported mutation: {mutation}")

print("\n--- Mutation Parsing ---")
print("Original:", mutation)
print("Parsed parts:", match.groups())


def parse_variant(mutation: str) -> tuple[str, int, str]:
    pattern = r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})"

    match = re.fullmatch(pattern, mutation)

    if match is None:
        raise ValueError(f"Unsupported mutation: {mutation}")

    original, position_text, replacement = match.groups()

    position = int(position_text)
    return original, position, replacement

mutation = df.iloc[0]["hgvs_pro"]

result = parse_variant(mutation)

print("\n--- Parsed Variant ---")
print("Result:", result)
print("Position type:", type(result[1]))
