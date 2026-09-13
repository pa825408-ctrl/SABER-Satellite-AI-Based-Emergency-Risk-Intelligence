import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "../data/processed/cyclone_tracks.csv"
)

df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"])

name = "AMPHAN"

storm = df[
    df["NAME"].str.upper() == name
].copy()

storm = storm.sort_values("ISO_TIME")

plt.figure(figsize=(10, 8))

plt.plot(
    storm["LON"],
    storm["LAT"],
    marker="o"
)

for _, row in storm.iterrows():

    plt.annotate(
        row["ISO_TIME"].strftime("%m-%d"),
        (row["LON"], row["LAT"]),
        fontsize=7
    )

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"{name} Historical Track")

plt.grid(True)
plt.tight_layout()

plt.show()
print(df.isna().sum())