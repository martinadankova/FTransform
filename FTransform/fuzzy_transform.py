import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from scipy import stats

class FuzzyTransformND:
    def __init__(
        self,
        input_cols,
        n_sets,
        group_col="file",
        group_order=None,
        input_mins=None,
        input_maxs=None,
    ):
        self.input_cols = input_cols
        self.group_col = group_col
        self.group_order = group_order

        if isinstance(n_sets, int):
            self.n_sets = {col: n_sets for col in input_cols}
        else:
            self.n_sets = n_sets

        self.input_mins = input_mins or {}
        self.input_maxs = input_maxs or {}

        self.nodes_ = {}
        self.membership_cols_ = {}
        self.component_table_ = None

    # --------------------------------------------------
    # Fit partitions
    # --------------------------------------------------

    def fit_partition(self, data):
        for col in self.input_cols:
            x_min = self.input_mins.get(col, data[col].min())
            x_max = self.input_maxs.get(col, data[col].max())

            self.input_mins[col] = x_min
            self.input_maxs[col] = x_max

            self.nodes_[col] = np.linspace(
                x_min,
                x_max,
                self.n_sets[col]
            )

            self.membership_cols_[col] = [
                f"{col}_A{k+1}" for k in range(self.n_sets[col])
            ]

        return self

    # --------------------------------------------------
    # 1D membership matrix for one input variable
    # --------------------------------------------------

    def _membership_matrix_1d(self, values, nodes):
        values = np.asarray(values)
        n_sets = len(nodes)

        A = np.zeros((len(values), n_sets))

        for k in range(n_sets):
            if k == 0:
                left = nodes[0]
                right = nodes[1]

                A[:, k] = np.where(
                    values <= left,
                    1,
                    np.where(
                        values >= right,
                        0,
                        (right - values) / (right - left)
                    )
                )

            elif k == n_sets - 1:
                left = nodes[-2]
                right = nodes[-1]

                A[:, k] = np.where(
                    values <= left,
                    0,
                    np.where(
                        values >= right,
                        1,
                        (values - left) / (right - left)
                    )
                )

            else:
                left = nodes[k - 1]
                center = nodes[k]
                right = nodes[k + 1]

                A[:, k] = np.where(
                    values <= left,
                    0,
                    np.where(
                        values <= center,
                        (values - left) / (center - left),
                        np.where(
                            values < right,
                            (right - values) / (right - center),
                            0
                        )
                    )
                )

        return A

    # --------------------------------------------------
    # Add 1D memberships and tensor-product memberships
    # --------------------------------------------------

    def add_memberships(self, data):
        data = data.copy()

        if not self.nodes_:
            self.fit_partition(data)

        for col in self.input_cols:
            A = self._membership_matrix_1d(
                data[col].values,
                self.nodes_[col]
            )

            for j, A_col in enumerate(self.membership_cols_[col]):
                data[A_col] = A[:, j]

        component_specs = []

        index_ranges = [
            range(self.n_sets[col]) for col in self.input_cols
        ]

        for multi_index in itertools.product(*index_ranges):
            component_name = "C_" + "_".join(
                f"{col}A{k+1}"
                for col, k in zip(self.input_cols, multi_index)
            )

            weight = np.ones(len(data))

            for col, k in zip(self.input_cols, multi_index):
                weight *= data[self.membership_cols_[col][k]]

            data[component_name] = weight

            component_specs.append({
                "component": component_name,
                "multi_index": multi_index,
                **{
                    f"{col}_component": k + 1
                    for col, k in zip(self.input_cols, multi_index)
                },
                **{
                    f"{col}_node": self.nodes_[col][k]
                    for col, k in zip(self.input_cols, multi_index)
                }
            })

        self.component_table_ = pd.DataFrame(component_specs)

        component_cols = self.component_table_["component"].tolist()
        data["A_sum"] = data[component_cols].sum(axis=1)

        return data

    # --------------------------------------------------
    # Compute direct F-transform components
    # --------------------------------------------------

    def compute_components(self, data, value_cols, groups=None):
        data = self.add_memberships(data)

        if isinstance(value_cols, str):
            value_cols = [value_cols]

        if groups is not None:
            groups_to_use = groups
        elif self.group_order is None:
            groups_to_use = sorted(data[self.group_col].dropna().unique())
        else:
            groups_to_use = self.group_order

        component_cols = self.component_table_["component"].tolist()

        rows = []

        for grp in groups_to_use:
            sub = data.loc[data[self.group_col] == grp].copy()

            for value_col in value_cols:
                for _, comp in self.component_table_.iterrows():
                    component = comp["component"]
                    denominator = sub[component].sum()

                    if denominator == 0:
                        value = np.nan
                    else:
                        value = np.sum(sub[component] * sub[value_col]) / denominator

                    row = {
                        "group": grp,
                        "variable": value_col,
                        "component": component,
                        "value": value,
                        "denominator": denominator,
                    }

                    for col in self.input_cols:
                        row[f"{col}_component"] = comp[f"{col}_component"]
                        row[f"{col}_node"] = comp[f"{col}_node"]

                    rows.append(row)

        return pd.DataFrame(rows)

    # --------------------------------------------------
    # Inverse F-transform on a grid
    # --------------------------------------------------

    def inverse_transform(self, component_table, grid=None, n_grid=100):
        if grid is None:
            grid_dict = {}

            for col in self.input_cols:
                grid_dict[col] = np.linspace(
                    self.input_mins[col],
                    self.input_maxs[col],
                    n_grid
                )

            mesh = np.meshgrid(
                *[grid_dict[col] for col in self.input_cols],
                indexing="ij"
            )

            grid = pd.DataFrame({
                col: mesh[j].ravel()
                for j, col in enumerate(self.input_cols)
            })
        else:
            grid = grid.copy()

        grid_with_A = self.add_memberships(grid)
        component_cols = self.component_table_["component"].tolist()

        rows = []

        for (grp, variable), sub in component_table.groupby(["group", "variable"]):
            sub = sub.set_index("component")

            values = np.zeros(len(grid_with_A))

            for component in component_cols:
                F_value = sub.loc[component, "value"]

                values += F_value * grid_with_A[component].values

            tmp = grid.copy()
            tmp["group"] = grp
            tmp["variable"] = variable
            tmp["fuzzy_value"] = values

            rows.append(tmp)

        return pd.concat(rows, ignore_index=True)

    # --------------------------------------------------
    # Slopes along selected dimension
    # --------------------------------------------------

    def compute_slopes_along(self, component_table, along_col):
        if along_col not in self.input_cols:
            raise ValueError(f"{along_col} is not in input_cols.")

        rows = []

        other_cols = [
            col for col in self.input_cols if col != along_col
        ]

        index_cols = [
            f"{col}_component" for col in other_cols
        ]

        along_component_col = f"{along_col}_component"
        along_node_col = f"{along_col}_node"

        group_cols = ["group", "variable"] + index_cols

        for keys, sub in component_table.groupby(group_cols, observed=True):
            sub = sub.sort_values(along_component_col)

            F = sub["value"].values
            x = sub[along_node_col].values

            if not isinstance(keys, tuple):
                keys = (keys,)

            key_dict = dict(zip(group_cols, keys))

            for k in range(len(F) - 1):
                slope = (F[k + 1] - F[k]) / (x[k + 1] - x[k])

                row = {
                    **key_dict,
                    "along": along_col,
                    "segment": f"A{k+1}_to_A{k+2}",
                    "from_node": x[k],
                    "to_node": x[k + 1],
                    "from_value": F[k],
                    "to_value": F[k + 1],
                    "slope": slope
                }

                rows.append(row)

        return pd.DataFrame(rows)

    # --------------------------------------------------
    # Participant-level subsampling stability
    # --------------------------------------------------

    def subsampling_slope_stability(
        self,
        data,
        value_cols,
        along_col,
        subject_col="name",
        subset_fraction=0.8,
        n_iter=1000,
        seed=123,
    ):
        rng = np.random.default_rng(seed)

        if isinstance(value_cols, str):
            value_cols = [value_cols]

        if not 0 < subset_fraction <= 1:
            raise ValueError("subset_fraction must be in (0, 1].")

        if not self.nodes_:
            self.fit_partition(data)

        if self.group_order is None:
            groups = sorted(data[self.group_col].dropna().unique())
        else:
            groups = self.group_order

        full_components = self.compute_components(
            data=data,
            value_cols=value_cols,
            groups=groups
        )

        full_slopes = self.compute_slopes_along(
            full_components,
            along_col=along_col
        )

        all_rows = []

        for grp in groups:
            sub_group = data.loc[data[self.group_col] == grp].copy()
            persons = sub_group[subject_col].dropna().unique()

            n_persons = len(persons)
            n_sample = int(np.floor(subset_fraction * n_persons))

            if n_sample < 2:
                raise ValueError(
                    f"Too few participants in group {grp}."
                )

            for variable in value_cols:
                slope_storage = {}

                for _ in range(n_iter):
                    sampled_persons = rng.choice(
                        persons,
                        size=n_sample,
                        replace=False
                    )

                    sub_iter = sub_group.loc[
                        sub_group[subject_col].isin(sampled_persons)
                    ].copy()

                    comp_iter = self.compute_components(
                        data=sub_iter,
                        value_cols=[variable],
                        groups=[grp]
                    )

                    slopes_iter = self.compute_slopes_along(
                        comp_iter,
                        along_col=along_col
                    )

                    for _, row in slopes_iter.iterrows():
                        key = tuple(
                            row[col]
                            for col in slopes_iter.columns
                            if col not in [
                                "from_node",
                                "to_node",
                                "from_value",
                                "to_value",
                                "slope"
                            ]
                        )

                        slope_storage.setdefault(key, []).append(row["slope"])

                for key, slopes in slope_storage.items():
                    slopes = np.asarray(slopes, dtype=float)
                    slopes = slopes[~np.isnan(slopes)]

                    key_cols = [
                        col for col in slopes_iter.columns
                        if col not in [
                            "from_node",
                            "to_node",
                            "from_value",
                            "to_value",
                            "slope"
                        ]
                    ]

                    key_dict = dict(zip(key_cols, key))

                    full_match = full_slopes.copy()

                    for col, val in key_dict.items():
                        full_match = full_match.loc[
                            full_match[col] == val
                        ]

                    if len(full_match) == 0:
                        full_data_slope = np.nan
                    else:
                        full_data_slope = full_match["slope"].iloc[0]

                    all_rows.append({
                        **key_dict,
                        "full_data_slope": full_data_slope,
                        "median_subsample_slope": np.median(slopes),
                        "mean_subsample_slope": np.mean(slopes),
                        "sd_subsample_slope": np.std(slopes, ddof=1)
                        if len(slopes) > 1 else np.nan,
                        "ci_lower_2_5": np.percentile(slopes, 2.5),
                        "ci_upper_97_5": np.percentile(slopes, 97.5),
                        "prop_positive": np.mean(slopes > 0),
                        "prop_negative": np.mean(slopes < 0),
                        "n_valid_slopes": len(slopes),
                        "n_iter": n_iter,
                        "subset_fraction": subset_fraction,
                        "n_persons_total": n_persons,
                        "n_persons_sampled": n_sample
                    })

        return pd.DataFrame(all_rows)

    # --------------------------------------------------
    # Plot membership functions for selected dimension
    # --------------------------------------------------

    def plot_memberships(self, input_col, filename=None):
        if input_col not in self.input_cols:
            raise ValueError(f"{input_col} is not in input_cols.")

        if not self.nodes_:
            raise ValueError("Run fit_partition(data) first.")

        x_grid = np.linspace(
            self.input_mins[input_col],
            self.input_maxs[input_col],
            500
        )

        A_grid = self._membership_matrix_1d(
            x_grid,
            self.nodes_[input_col]
        )

        plt.figure(figsize=(9, 5))

        for k in range(self.n_sets[input_col]):
            plt.plot(
                x_grid,
                A_grid[:, k],
                linewidth=2,
                label=f"{input_col}_A{k+1}"
            )

        plt.xlabel(input_col)
        plt.ylabel("Membership degree")
        plt.title(f"Fuzzy membership functions for {input_col}")
        plt.legend()
        plt.tight_layout()

        if filename is not None:
            plt.savefig(filename, dpi=200, bbox_inches="tight")

        plt.show()

    # --------------------------------------------------
    # Plot inverse transform for 1D case
    # --------------------------------------------------

    def plot_inverse_1d(
        self,
        inverse_table,
        variable,
        x_col=None,
        group_colors=None,
        filename=None
    ):
        if x_col is None:
            if len(self.input_cols) != 1:
                raise ValueError("x_col must be specified for nD data.")
            x_col = self.input_cols[0]

        sub_data = inverse_table.loc[
            inverse_table["variable"] == variable
        ]

        plt.figure(figsize=(9, 6))

        for grp in sub_data["group"].unique():
            sub = sub_data.loc[sub_data["group"] == grp]

            kwargs = {}
            if group_colors is not None:
                kwargs["color"] = group_colors.get(grp, None)

            plt.plot(
                sub[x_col],
                sub["fuzzy_value"],
                linewidth=2.5,
                label=grp,
                **kwargs
            )

        plt.xlabel(x_col)
        plt.ylabel(f"Fuzzy transform of {variable}")
        plt.title(f"Inverse fuzzy transform: {variable}")
        plt.legend()
        plt.tight_layout()

        if filename is not None:
            plt.savefig(filename, dpi=200, bbox_inches="tight")

        plt.show()

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    def save_results(
        self,
        component_table,
        inverse_table,
        slope_table,
        prefix="fuzzy_transform"
    ):
        component_table.to_csv(f"{prefix}_components.csv", index=False)
        inverse_table.to_csv(f"{prefix}_inverse.csv", index=False)
        slope_table.to_csv(f"{prefix}_slopes.csv", index=False)

    def segment_inference(
    self,
    data,
    value_cols,
    along_col,
    subject_col="name",
    groups=None,
    ):

        if isinstance(value_cols, str):
            value_cols = [value_cols]

        if not self.nodes_:
            self.fit_partition(data)

        data_A = self.add_memberships(data)

        if groups is not None:
            groups_to_use = groups
        elif self.group_order is None:
            groups_to_use = sorted(data_A[self.group_col].dropna().unique())
        else:
            groups_to_use = self.group_order

        components = self.compute_components(
            data=data,
            value_cols=value_cols,
            groups=groups_to_use
        )

        slopes = self.compute_slopes_along(
            components,
            along_col=along_col
        )

        rows = []

        for grp in groups_to_use:
            sub_g = data_A.loc[data_A[self.group_col] == grp].copy()

            persons = sub_g[subject_col].dropna().unique()
            G = len(persons)

            for value_col in value_cols:
                for _, slope_row in slopes.loc[
                    (slopes["group"] == grp)
                    & (slopes["variable"] == value_col)
                ].iterrows():

                    segment = slope_row["segment"]
                    slope = slope_row["slope"]

                    from_comp = int(segment.split("_to_")[0].replace("A", ""))
                    to_comp = int(segment.split("_to_")[1].replace("A", ""))

                    A_from_col = self.membership_cols_[along_col][from_comp - 1]
                    A_to_col = self.membership_cols_[along_col][to_comp - 1]

                    x0 = slope_row["from_node"]
                    x1 = slope_row["to_node"]
                    h = x1 - x0

                    denom_from = sub_g[A_from_col].sum()
                    denom_to = sub_g[A_to_col].sum()

                    if denom_from == 0 or denom_to == 0 or G < 2:
                        se = np.nan
                        t_value = np.nan
                        p_value = np.nan
                        ci_lower = np.nan
                        ci_upper = np.nan
                    else:
                        # linear-contrast weights
                        sub_g["_c"] = (
                            sub_g[A_to_col] / denom_to
                            - sub_g[A_from_col] / denom_from
                        ) / h

                        # participant-level contributions
                        contrib = (
                            sub_g
                            .assign(_cy=sub_g["_c"] * sub_g[value_col])
                            .groupby(subject_col, observed=True)["_cy"]
                            .sum()
                        )
                        # cluster variance of linear contrast
                        centered = contrib - contrib.mean()

                        var = (
                            G / (G - 1)
                            * np.sum(centered ** 2)
                        )

                        se = np.sqrt(var)

                        t_value = slope / se if se > 0 else np.nan

                        df = G - 1

                        p_value = (
                            2 * stats.t.sf(np.abs(t_value), df=df)
                            if np.isfinite(t_value)
                            else np.nan
                        )

                        t_crit = stats.t.ppf(0.975, df=df)

                        ci_lower = slope - t_crit * se
                        ci_upper = slope + t_crit * se

                    rows.append({
                        "group": grp,
                        "variable": value_col,
                        "along": along_col,
                        "segment": segment,
                        "from_node": slope_row["from_node"],
                        "to_node": slope_row["to_node"],
                        "slope": slope,
                        "SE": se,
                        "t_value": t_value,
                        "df": G - 1,
                        "p_value": p_value,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "n_persons": G,
                        "n_obs": len(sub_g)
                    })

        return pd.DataFrame(rows)

class FuzzyTransform1D:
    def __init__(
        self,
        n_sets,
        time_col="day",
        group_col="file",
        group_order=None,
        time_min=None,
        time_max=None,
    ):
        self.n_sets = n_sets
        self.time_col = time_col
        self.group_col = group_col
        self.group_order = group_order
        self.time_min = time_min
        self.time_max = time_max
        self.nodes_ = None
        self.membership_cols_ = None

    def fit_partition(self, data):
        if self.time_min is None:
            self.time_min = data[self.time_col].min()

        if self.time_max is None:
            self.time_max = data[self.time_col].max()

        self.nodes_ = np.linspace(self.time_min, self.time_max, self.n_sets)
        self.membership_cols_ = [f"A{k+1}" for k in range(self.n_sets)]

        return self

    def membership_matrix(self, time_values):
        if self.nodes_ is None:
            raise ValueError("Run fit_partition(data) first.")

        t = np.asarray(time_values)
        A = np.zeros((len(t), self.n_sets))

        for k in range(self.n_sets):
            if k == 0:
                left = self.nodes_[0]
                right = self.nodes_[1]

                A[:, k] = np.where(
                    t <= left,
                    1,
                    np.where(t >= right, 0, (right - t) / (right - left))
                )

            elif k == self.n_sets - 1:
                left = self.nodes_[-2]
                right = self.nodes_[-1]

                A[:, k] = np.where(
                    t <= left,
                    0,
                    np.where(t >= right, 1, (t - left) / (right - left))
                )

            else:
                left = self.nodes_[k - 1]
                center = self.nodes_[k]
                right = self.nodes_[k + 1]

                A[:, k] = np.where(
                    t <= left,
                    0,
                    np.where(
                        t <= center,
                        (t - left) / (center - left),
                        np.where(t < right, (right - t) / (right - center), 0)
                    )
                )

        return A

    def add_memberships(self, data):
        data = data.copy()

        if self.nodes_ is None:
            self.fit_partition(data)

        A = self.membership_matrix(data[self.time_col].values)

        for j, col in enumerate(self.membership_cols_):
            data[col] = A[:, j]

        data["A_sum"] = data[self.membership_cols_].sum(axis=1)

        return data

    def compute_components(self, data, value_cols, groups=None):
        data = self.add_memberships(data)

        if isinstance(value_cols, str):
            value_cols = [value_cols]

        if groups is not None:
            groups_to_use = groups
        elif self.group_order is None:
            groups_to_use = sorted(data[self.group_col].dropna().unique())
        else:
            groups_to_use = self.group_order

        rows = []

        for grp in groups_to_use:
            sub = data.loc[data[self.group_col] == grp].copy()

            for value_col in value_cols:
                for k, A_col in enumerate(self.membership_cols_):
                    denominator = sub[A_col].sum()

                    if denominator == 0:
                        F_k = np.nan
                    else:
                        F_k = np.sum(sub[A_col] * sub[value_col]) / denominator

                    rows.append({
                        "group": grp,
                        "variable": value_col,
                        "component": A_col,
                        "component_index": k + 1,
                        "node": self.nodes_[k],
                        "value": F_k,
                        "denominator": denominator
                    })

        return pd.DataFrame(rows)

    def inverse_transform(self, component_table, day_grid=None):
        if day_grid is None:
            day_grid = np.linspace(self.time_min, self.time_max, 300)

        A_grid = self.membership_matrix(day_grid)

        rows = []

        for (grp, variable), sub in component_table.groupby(["group", "variable"], observed=True):
            sub = sub.sort_values("component_index")
            F = sub["value"].values

            fuzzy_values = A_grid @ F

            rows.append(pd.DataFrame({
                self.time_col: day_grid,
                "group": grp,
                "variable": variable,
                "fuzzy_value": fuzzy_values
            }))

        return pd.concat(rows, ignore_index=True)

    def compute_slopes(self, component_table):
        rows = []

        for (grp, variable), sub in component_table.groupby(["group", "variable"]):
            sub = sub.sort_values("component_index")

            F = sub["value"].values
            x = sub["node"].values

            for k in range(len(F) - 1):
                slope = (F[k + 1] - F[k]) / (x[k + 1] - x[k])

                rows.append({
                    "group": grp,
                    "variable": variable,
                    "segment": f"A{k+1}_to_A{k+2}",
                    "from_component": f"A{k+1}",
                    "to_component": f"A{k+2}",
                    "from_node": x[k],
                    "to_node": x[k + 1],
                    "from_value": F[k],
                    "to_value": F[k + 1],
                    "slope": slope
                })

        return pd.DataFrame(rows)

    def plot_memberships(self, filename=None):
        if self.nodes_ is None:
            raise ValueError("Run fit_partition(data) first.")

        day_grid = np.linspace(self.time_min, self.time_max, 500)
        A_grid = self.membership_matrix(day_grid)

        plt.figure(figsize=(9, 5))

        for k, col in enumerate(self.membership_cols_):
            plt.plot(day_grid, A_grid[:, k], linewidth=2, label=col)

        plt.xlabel(self.time_col)
        plt.ylabel("Membership degree")
        plt.title(f"{self.n_sets} equidistant fuzzy membership functions")
        plt.legend()
        plt.tight_layout()

        if filename is not None:
            plt.savefig(filename, dpi=200, bbox_inches="tight")

        plt.show()

    def plot_inverse(self, inverse_table, variable, group_colors=None, filename=None):
        sub_data = inverse_table.loc[inverse_table["variable"] == variable]

        plt.figure(figsize=(9, 6))

        for grp in sub_data["group"].unique():
            sub = sub_data.loc[sub_data["group"] == grp]

            kwargs = {}
            if group_colors is not None:
                kwargs["color"] = group_colors.get(grp, None)

            plt.plot(
                sub[self.time_col],
                sub["fuzzy_value"],
                linewidth=2.5,
                label=grp,
                **kwargs
            )

        plt.xlabel(self.time_col)
        plt.ylabel(f"Fuzzy transform of {variable}")
        plt.title(f"Inverse fuzzy transform: {variable}")
        plt.legend()
        plt.tight_layout()

        if filename is not None:
            plt.savefig(filename, dpi=200, bbox_inches="tight")

        plt.show()

    def save_results(
        self,
        component_table,
        inverse_table,
        slope_table,
        prefix="fuzzy_transform"
    ):
        component_table.to_csv(f"{prefix}_components.csv", index=False)
        inverse_table.to_csv(f"{prefix}_inverse.csv", index=False)
        slope_table.to_csv(f"{prefix}_slopes.csv", index=False)

    def subsampling_slope_stability(
        self,
        data,
        value_cols,
        subject_col="name",
        subset_fraction=0.8,
        n_iter=1000,
        seed=123,
    ):
        rng = np.random.default_rng(seed)

        if isinstance(value_cols, str):
            value_cols = [value_cols]

        if not 0 < subset_fraction <= 1:
            raise ValueError("subset_fraction must be in (0, 1].")

        if self.nodes_ is None:
            self.fit_partition(data)

        if self.group_order is None:
            groups = sorted(data[self.group_col].dropna().unique())
        else:
            groups = self.group_order

        full_components = self.compute_components(
            data=data,
            value_cols=value_cols,
            groups=groups
        )

        full_slopes = self.compute_slopes(full_components)

        all_rows = []

        for grp in groups:
            sub_group = data.loc[data[self.group_col] == grp].copy()
            persons = sub_group[subject_col].dropna().unique()

            n_persons = len(persons)
            n_sample = int(np.floor(subset_fraction * n_persons))

            if n_sample < 2:
                raise ValueError(
                    f"Too few participants in group {grp} for subsampling."
                )

            for variable in value_cols:
                slope_storage = {
                    f"A{k+1}_to_A{k+2}": []
                    for k in range(self.n_sets - 1)
                }

                for _ in range(n_iter):
                    sampled_persons = rng.choice(
                        persons,
                        size=n_sample,
                        replace=False
                    )

                    sub_iter = sub_group.loc[
                        sub_group[subject_col].isin(sampled_persons)
                    ].copy()

                    comp_iter = self.compute_components(
                        data=sub_iter,
                        value_cols=[variable],
                        groups=[grp]
                    )

                    slopes_iter = self.compute_slopes(comp_iter)

                    for _, row in slopes_iter.iterrows():
                        if pd.notna(row["slope"]):
                            slope_storage[row["segment"]].append(row["slope"])

                for segment, slopes in slope_storage.items():
                    slopes = np.asarray(slopes, dtype=float)
                    slopes = slopes[~np.isnan(slopes)]

                    full_slope = full_slopes.loc[
                        (full_slopes["group"] == grp)
                        & (full_slopes["variable"] == variable)
                        & (full_slopes["segment"] == segment),
                        "slope"
                    ].iloc[0]

                    if len(slopes) == 0:
                        median_slope = np.nan
                        mean_slope = np.nan
                        sd_slope = np.nan
                        ci_lower = np.nan
                        ci_upper = np.nan
                        prop_positive = np.nan
                        prop_negative = np.nan
                    else:
                        median_slope = np.median(slopes)
                        mean_slope = np.mean(slopes)
                        sd_slope = np.std(slopes, ddof=1) if len(slopes) > 1 else np.nan
                        ci_lower = np.percentile(slopes, 2.5)
                        ci_upper = np.percentile(slopes, 97.5)
                        prop_positive = np.mean(slopes > 0)
                        prop_negative = np.mean(slopes < 0)

                    all_rows.append({
                        "group": grp,
                        "variable": variable,
                        "segment": segment,
                        "full_data_slope": full_slope,
                        "median_subsample_slope": median_slope,
                        "mean_subsample_slope": mean_slope,
                        "sd_subsample_slope": sd_slope,
                        "ci_lower_2_5": ci_lower,
                        "ci_upper_97_5": ci_upper,
                        "prop_positive": prop_positive,
                        "prop_negative": prop_negative,
                        "n_valid_slopes": len(slopes),
                        "n_iter": n_iter,
                        "subset_fraction": subset_fraction,
                        "n_persons_total": n_persons,
                        "n_persons_sampled": n_sample
                    })

        return pd.DataFrame(all_rows)