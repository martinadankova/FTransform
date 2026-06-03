import numpy as np
import matplotlib.pyplot as plt

from FTransform.fuzzy_transform import FuzzyTransform1D


# --------------------------------------------------
# Fuzzy partition used in the F-transform
# --------------------------------------------------

segment_bounds = np.array([0, 2, 4, 6, 8, 10])
segment_centers = (segment_bounds[:-1] + segment_bounds[1:]) / 2

n_sets = len(segment_centers)

ft = FuzzyTransform1D(
    n_sets=n_sets,
    time_col="x",
    group_col="group",
    time_min=segment_centers[0],
    time_max=segment_centers[-1]
)

# initialize partition
ft.nodes_ = segment_centers
ft.membership_cols_ = [f"A{k+1}" for k in range(n_sets)]

# grid for plotting
x_grid = np.linspace(0, 10, 1000)
A_grid = ft.membership_matrix(x_grid)


# --------------------------------------------------
# Plot fuzzy sets
# --------------------------------------------------

plt.figure(figsize=(10, 4.8))

for k in range(n_sets):
    plt.plot(
        x_grid,
        A_grid[:, k],
        linewidth=2.5,
        label=fr"$A_{k+1}$"
    )

# crisp segment boundaries
for b in segment_bounds:
    plt.axvline(
        b,
        linestyle="--",
        linewidth=1,
        alpha=0.35
    )

# fuzzy partition nodes = segment midpoints
for c in segment_centers:
    plt.axvline(
        c,
        linestyle=":",
        linewidth=1.5,
        alpha=0.75
    )

plt.scatter(
    segment_centers,
    np.ones_like(segment_centers),
    s=70,
    zorder=5,
    label="Fuzzy partition nodes"
)

plt.xlabel("x")
plt.ylabel("Membership degree")
plt.title("Fuzzy sets used for the F-transform")
plt.ylim(-0.05, 1.05)
plt.grid(alpha=0.25)
plt.legend(ncol=3)
plt.tight_layout()
plt.show()