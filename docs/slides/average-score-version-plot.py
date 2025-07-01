import matplotlib.pyplot as plt

# Data extracted from the user's image
versions = ["v1.0.0", "v1.1.0", "v1.2.0", "v1.3.0", "v1.3.5"]
normal_scores = [269.375, 272.336, 263.583, 278.793, 286.815]
very_hard_scores = [290.562, 330.348, 292.506, 339.330, 348.456]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(versions, normal_scores, marker='o', label='Normal Mode', linewidth=2)
plt.plot(versions, very_hard_scores, marker='s', label='Very Hard Mode', linewidth=2)

plt.title("Average Score Over Versions", fontsize=16)
plt.xlabel("Version", fontsize=12)
plt.ylabel("Average Score", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

# Save to file for export if needed
plt.savefig("./average-score-version-plot.png", dpi=300)
plt.show()
