import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from FTransform.fuzzy_transform import FuzzyTransform1D


# --------------------------------------------------
# 1. Generate noisy data
# --------------------------------------------------

np.random.seed(42)

n = 160
x = np.linspace(0, 10, n)

def true_function(x):
    return np.sin(0.75 * x) + 0.25 * np.sin(2.2 * x)

y = true_function(x) + np.random.normal(0, 0.22, size=n)

data = pd.DataFrame({
    "x": x,
    "y": y,
    "group": "data"
})


# --------------------------------------------------
# 2. Define crisp segments and their midpoints
# --------------------------------------------------

segment_bounds = np.array([0, 2, 4, 6, 8, 10])
segment_centers = (segment_bounds[:-1] + segment_bounds[1:]) / 2

n_sets = len(segment_centers)


# --------------------------------------------------
# 3. Fit fuzzy transform using segment midpoints as nodes
# --------------------------------------------------

ft = FuzzyTransform1D(
    n_sets=n_sets,
    time_col="x",
    group_col="group",
    time_min=segment_centers[0],
    time_max=segment_centers[-1]
)

ft.fit_partition(data)

# overwrite automatically generated nodes by segment midpoints
ft.nodes_ = segment_centers
ft.membership_cols_ = [f"A{k+1}" for k in range(n_sets)]


components = ft.compute_components(
    data=data,
    value_cols="y",
    groups=["data"]
)

inverse = ft.inverse_transform(
    component_table=components,
    day_grid=np.linspace(x.min(), x.max(), 500)
)


# --------------------------------------------------
# 4. Crisp segment means
# --------------------------------------------------

segment_means = []

for k in range(len(segment_bounds) - 1):
    a, b = segment_bounds[k], segment_bounds[k + 1]

    if k < len(segment_bounds) - 2:
        mask = (x >= a) & (x < b)
    else:
        mask = (x >= a) & (x <= b)

    segment_means.append(np.mean(y[mask]))

segment_means = np.asarray(segment_means)


# --------------------------------------------------
# 5. Plot everything in one graph
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    data["x"],
    data["y"],
    s=25,
    alpha=0.65,
    label="Noisy observations"
)

plt.plot(
    segment_centers,
    segment_means,
    marker="o",
    markersize=11,
    linewidth=2,
    color="red",
    label="Segment means"
)

plt.scatter(
    components["node"],
    components["value"],
    s=90,
    color="black",
    zorder=5,
    label="F-transform components"
)

plt.plot(
    inverse["x"],
    inverse["fuzzy_value"],
    linewidth=3,
    color="orange",
    label="Inverse F-transform"
)

# crisp segment boundaries
for b in segment_bounds:
    plt.axvline(
        b,
        linestyle="--",
        linewidth=1,
        alpha=0.4
    )

# fuzzy partition nodes = segment midpoints
for c in segment_centers:
    plt.axvline(
        c,
        linestyle=":",
        linewidth=1.5,
        alpha=0.7
    )

plt.xlabel("x")
plt.ylabel("y")
#plt.title("Segment Means and Inverse F-transform")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()