import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("dataset/flood.csv")

# -----------------------------
# 1. Correlation Heatmap
# -----------------------------
plt.figure(figsize=(12, 8))
plt.imshow(data.corr(), cmap="Blues", aspect="auto")
plt.colorbar()
plt.xticks(range(len(data.columns)), data.columns, rotation=90)
plt.yticks(range(len(data.columns)), data.columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# -----------------------------
# 2. Scatter Plot
# -----------------------------
plt.figure(figsize=(7, 5))
plt.scatter(
    data["MonsoonIntensity"],
    data["FloodProbability"],
    alpha=0.5
)
plt.title("Monsoon Intensity vs Flood Probability")
plt.xlabel("Monsoon Intensity")
plt.ylabel("Flood Probability")
plt.grid(True)
plt.show()

# -----------------------------
# 3. Histogram
# -----------------------------
plt.figure(figsize=(7, 5))
plt.hist(data["FloodProbability"], bins=20)
plt.title("Flood Probability Distribution")
plt.xlabel("Flood Probability")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

print("All visualizations generated successfully!")