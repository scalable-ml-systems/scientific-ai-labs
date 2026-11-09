import pandas as pd

review_df = pd.read_csv(
    "data/processed/cyp2c19_variants_for_review.csv"
)

unsupported_df = review_df[
    review_df["validation_reason"]
    == "unsupported_variant_notation"
]

print(unsupported_df.to_string(index=False))
