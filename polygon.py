import numpy as np
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)

# Generate noisy data
n = 160
x = np.linspace(0, 10, n)

def f(x):
    return np.sin(0.75 * x) + 0.25 * np.sin(2.2 * x)

noise = np.random.normal(0, 0.22, size=n)
y = f(x) + noise

# Partition boundaries
bounds = np.array([0, 2, 4, 6, 8, 10])

# Compute mean value in each segment
centers = []
segment_means = []

for i in range(len(bounds) - 1):
    a, b = bounds[i], bounds[i + 1]

    if i < len(bounds) - 2:
        mask = (x >= a) & (x < b)
    else:
        mask = (x >= a) & (x <= b)

    centers.append((a + b) / 2)
    segment_means.append(np.mean(y[mask]))

centers = np.array(centers)
segment_means = np.array(segment_means)

# Plot
plt.figure(figsize=(10, 6))

plt.scatter(x, y, s=25, alpha=0.8, label="Noisy data")

plt.plot(
    centers,
    segment_means,
    linewidth=2,
    marker="o",
    color="red",
    markersize=14,
    label="Mean in each segment"
)

for b in bounds:
    plt.axvline(b, linestyle="--", linewidth=1.5, alpha=0.7, color="grey")

plt.xlabel("x")
plt.ylabel("y")
#plt.title("Partitioned Means of Noisy Data")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()