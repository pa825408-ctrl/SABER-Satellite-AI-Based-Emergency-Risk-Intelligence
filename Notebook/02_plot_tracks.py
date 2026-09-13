import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "../data/processed/cyclone_tracks.csv"
)

df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"])

plt.figure(figsize=(10, 8))

for cyclone_id, group in df.groupby("SID"):

    group = group.sort_values("ISO_TIME")

    plt.plot(
        group["LON"],
        group["LAT"],
        marker="o",
        linewidth=1,
        markersize=3
    )

    # Label cyclone at first point
    first = group.iloc[0]

    plt.text(
        first["LON"],
        first["LAT"],
        str(first["NAME"]),
        fontsize=8
    )

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(
    "Historical North Indian Ocean Cyclone Tracks"
)

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "../data/processed/cyclone_tracks.png",
    dpi=200
)

plt.show()