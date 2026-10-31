import pandas as pd

labels = pd.read_parquet("data/labels/labels.parquet")
labels["flagged"] = labels["label"].isin(["jamming", "spoofing", "both"])
judged = labels[labels["label"] != "unknown"]

print("Flagged share of judged cells, per region and day:")
print((judged.groupby(["region", "date"])["flagged"].mean() * 100)
      .round(1).unstack("date").to_string())

print("\nTop 5 cells per region: on how many of the days each was flagged")
day = judged.groupby(["region", "cell_lat", "cell_lon", "date"])["flagged"].mean()
days_hot = (day >= 0.25).groupby(["region", "cell_lat", "cell_lon"]).sum()
for region, s in days_hot.groupby(level="region"):
    print(region, s.droplevel("region").sort_values(ascending=False).head(5).to_dict())
