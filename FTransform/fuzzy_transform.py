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

    def compute_statistical_characteristics(
        self,
        data,
        value_cols,
        groups=None,
        alpha=0.05,
        null_value=0.0,
        variance_df="smoother_residual_trace",
        include_anova_heuristic=False,
    ):
        """
        Compute statistical characteristics of the multidimensional
        fuzzy transform.

        The method treats the tensor-product fuzzy memberships as the
        basis functions of the ND F-transform.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data.

        value_cols : str or list of str
            Response variable(s).

        groups : list, optional
            Groups to analyze. If None, use all groups.

        alpha : float, default=0.05
            Significance level for confidence intervals and tests.

        null_value : float, default=0.0
            Null value used in component and fitted-value t-tests.

        variance_df : str, default="smoother_residual_trace"
            Method used for residual variance and t degrees of freedom.

            Options:

            "component_count"
                df = m - M,
                where M is the number of supported tensor components.

            "effective"
                df = m - tr(S).

            "smoother_residual_trace"
                df = tr((I-S)^T(I-S))
                = m - 2 tr(S) + tr(S^T S).

            The last option is most directly consistent with the
            residual-SSE identity for a general linear smoother.

        include_anova_heuristic : bool, default=False
            If True, also compute the ANOVA-like F statistic from the
            manuscript. This should be interpreted only heuristically,
            because a general FT smoother is not an orthogonal projection.

        Returns
        -------
        dict with keys

            "components"
                Component estimates, variances, SEs, CIs, t-tests.

            "fitted"
                Observation-level inverse FT fitted values, residuals,
                variances, SEs, CIs and t-tests.

            "model"
                Global characteristics:
                SSE, SST, SSR, pseudo-R2, local R2,
                tr(S), residual degrees of freedom,
                residual variance estimates, etc.

            "smoothing_matrices"
                Dictionary of smoothing matrices indexed by
                (group, variable).
        """

        # --------------------------------------------------
        # Checks
        # --------------------------------------------------

        if isinstance(value_cols, str):
            value_cols = [value_cols]

        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1).")

        allowed_df_methods = {
            "component_count",
            "effective",
            "smoother_residual_trace",
        }

        if variance_df not in allowed_df_methods:
            raise ValueError(
                "variance_df must be one of "
                "'component_count', 'effective', "
                "'smoother_residual_trace'."
            )

        for value_col in value_cols:
            if value_col not in data.columns:
                raise ValueError(
                    f"{value_col} is not present in data."
                )

        # --------------------------------------------------
        # Fit partition and create memberships
        # --------------------------------------------------

        if not self.nodes_:
            self.fit_partition(data)

        data_A = self.add_memberships(data)

        component_cols = (
            self.component_table_["component"].tolist()
        )

        component_meta = (
            self.component_table_
            .set_index("component")
            .to_dict(orient="index")
        )

        # --------------------------------------------------
        # Groups
        # --------------------------------------------------

        if groups is not None:
            groups_to_use = groups

        elif self.group_order is None:
            groups_to_use = sorted(
                data_A[self.group_col]
                .dropna()
                .unique()
            )

        else:
            groups_to_use = self.group_order

        # --------------------------------------------------
        # Output storage
        # --------------------------------------------------

        component_rows = []
        fitted_rows = []
        model_rows = []

        smoothing_matrices = {}

        # --------------------------------------------------
        # Group loop
        # --------------------------------------------------

        for grp in groups_to_use:

            sub_group = data_A.loc[
                data_A[self.group_col] == grp
            ].copy()

            # --------------------------------------------------
            # Response-variable loop
            # --------------------------------------------------

            for value_col in value_cols:

                # Use complete observations for this response
                valid_rows = (
                    sub_group[value_col].notna()
                    &
                    sub_group[component_cols]
                    .notna()
                    .all(axis=1)
                )

                sub = (
                    sub_group
                    .loc[valid_rows]
                    .copy()
                )

                m = len(sub)

                if m == 0:
                    continue

                # --------------------------------------------------
                # Tensor-product membership matrix B
                #
                # B[r,k] = B_k(x_r)
                # --------------------------------------------------

                B = sub[
                    component_cols
                ].to_numpy(dtype=float)

                y = sub[
                    value_col
                ].to_numpy(dtype=float)

                # --------------------------------------------------
                # Check partition of unity
                # --------------------------------------------------

                membership_sum = B.sum(axis=1)

                partition_error = np.max(
                    np.abs(membership_sum - 1.0)
                )

                # --------------------------------------------------
                # Component denominators
                #
                # D_k = sum_r B_k(x_r)
                # --------------------------------------------------

                denominators = B.sum(axis=0)

                supported = denominators > 0

                M_supported = int(
                    supported.sum()
                )

                Bv = B[:, supported]

                Dv = denominators[supported]

                supported_components = [
                    component_cols[k]
                    for k in np.where(supported)[0]
                ]

                # --------------------------------------------------
                # Direct F-transform
                #
                # F_k =
                # sum B_k(x_r)y_r / sum B_k(x_r)
                # --------------------------------------------------

                Fv = (
                    Bv.T @ y
                ) / Dv

                # Full vector including unsupported components
                F = np.full(
                    len(component_cols),
                    np.nan
                )

                F[supported] = Fv

                # --------------------------------------------------
                # Smoothing matrix
                #
                # S = B D^{-1} B^T
                # --------------------------------------------------

                S = (
                    Bv / Dv
                ) @ Bv.T

                smoothing_matrices[
                    (grp, value_col)
                ] = S

                # --------------------------------------------------
                # Inverse F-transform on observed data
                #
                # y_tilde = S y
                # --------------------------------------------------

                fitted = S @ y

                residuals = y - fitted

                # --------------------------------------------------
                # Effective degrees of freedom
                # --------------------------------------------------

                trace_S = np.trace(S)

                trace_STS = np.sum(S ** 2)

                # df_eff = tr(S)
                df_eff = trace_S

                # Earlier manuscript formula: m - number components
                df_component_count = (
                    m - M_supported
                )

                # Linear-smoother approximation
                df_effective = (
                    m - trace_S
                )

                # Residual trace implied by
                # tr((I-S)^T(I-S))
                df_residual_trace = (
                    m
                    - 2 * trace_S
                    + trace_STS
                )

                # --------------------------------------------------
                # SSE
                # --------------------------------------------------

                SSE = np.sum(
                    residuals ** 2
                )

                # --------------------------------------------------
                # Three residual variance estimators
                # --------------------------------------------------

                sigma2_component_count = (
                    SSE / df_component_count
                    if df_component_count > 0
                    else np.nan
                )

                sigma2_effective = (
                    SSE / df_effective
                    if df_effective > 0
                    else np.nan
                )

                sigma2_residual_trace = (
                    SSE / df_residual_trace
                    if df_residual_trace > 0
                    else np.nan
                )

                # --------------------------------------------------
                # Select variance estimator
                # --------------------------------------------------

                if variance_df == "component_count":

                    df_used = df_component_count
                    sigma2 = sigma2_component_count

                elif variance_df == "effective":

                    df_used = df_effective
                    sigma2 = sigma2_effective

                else:

                    df_used = df_residual_trace
                    sigma2 = sigma2_residual_trace

                # --------------------------------------------------
                # t critical value
                # --------------------------------------------------

                if (
                    np.isfinite(df_used)
                    and df_used > 0
                ):

                    t_crit = stats.t.ppf(
                        1 - alpha / 2,
                        df=df_used
                    )

                else:

                    t_crit = np.nan

                # ==================================================
                # COMPONENT STATISTICS
                # ==================================================

                # Theoretical variance factor:
                #
                # Var(F_k) =
                # sigma^2 *
                # sum_r B_k(x_r)^2 / D_k^2
                # --------------------------------------------------

                variance_factor = np.full(
                    len(component_cols),
                    np.nan
                )

                variance_factor[supported] = (
                    np.sum(
                        Bv ** 2,
                        axis=0
                    )
                    /
                    (Dv ** 2)
                )

                component_variance = (
                    sigma2
                    * variance_factor
                )

                component_se = np.sqrt(
                    component_variance
                )

                component_t = (
                    (F - null_value)
                    /
                    component_se
                )

                if (
                    np.isfinite(df_used)
                    and df_used > 0
                ):

                    component_p = (
                        2
                        * stats.t.sf(
                            np.abs(component_t),
                            df=df_used
                        )
                    )

                else:

                    component_p = np.full(
                        len(component_cols),
                        np.nan
                    )

                component_ci_lower = (
                    F
                    - t_crit
                    * component_se
                )

                component_ci_upper = (
                    F
                    + t_crit
                    * component_se
                )

                # --------------------------------------------------
                # Store component results
                # --------------------------------------------------

                for k, component in enumerate(
                    component_cols
                ):

                    result = {
                        "group":
                            grp,

                        "variable":
                            value_col,

                        "component":
                            component,

                        "value":
                            F[k],

                        "denominator":
                            denominators[k],

                        "variance_factor":
                            variance_factor[k],

                        "variance":
                            component_variance[k],

                        "SE":
                            component_se[k],

                        "null_value":
                            null_value,

                        "t_value":
                            component_t[k],

                        "df":
                            df_used,

                        "p_value":
                            component_p[k],

                        "ci_lower":
                            component_ci_lower[k],

                        "ci_upper":
                            component_ci_upper[k],

                        "supported":
                            bool(supported[k]),
                    }

                    # Add node/component information
                    result.update(
                        component_meta[component]
                    )

                    component_rows.append(
                        result
                    )

                # ==================================================
                # FITTED-VALUE STATISTICS
                # ==================================================

                # Var(y_tilde)
                #
                # = sigma^2 S S^T
                #
                # diagonal elements:
                # sigma^2 * sum_k S_jk^2
                # --------------------------------------------------

                fitted_variance_factor = (
                    np.sum(
                        S ** 2,
                        axis=1
                    )
                )

                fitted_variance = (
                    sigma2
                    * fitted_variance_factor
                )

                fitted_se = np.sqrt(
                    fitted_variance
                )

                fitted_t = (
                    (fitted - null_value)
                    /
                    fitted_se
                )

                if (
                    np.isfinite(df_used)
                    and df_used > 0
                ):

                    fitted_p = (
                        2
                        * stats.t.sf(
                            np.abs(fitted_t),
                            df=df_used
                        )
                    )

                else:

                    fitted_p = np.full(
                        m,
                        np.nan
                    )

                fitted_ci_lower = (
                    fitted
                    - t_crit
                    * fitted_se
                )

                fitted_ci_upper = (
                    fitted
                    + t_crit
                    * fitted_se
                )

                # --------------------------------------------------
                # Store fitted-value results
                # --------------------------------------------------

                for r in range(m):

                    fitted_rows.append({
                        "group":
                            grp,

                        "variable":
                            value_col,

                        "original_index":
                            sub.index[r],

                        "observed":
                            y[r],

                        "fitted":
                            fitted[r],

                        "residual":
                            residuals[r],

                        "variance_factor":
                            fitted_variance_factor[r],

                        "variance":
                            fitted_variance[r],

                        "SE":
                            fitted_se[r],

                        "null_value":
                            null_value,

                        "t_value":
                            fitted_t[r],

                        "df":
                            df_used,

                        "p_value":
                            fitted_p[r],

                        "ci_lower":
                            fitted_ci_lower[r],

                        "ci_upper":
                            fitted_ci_upper[r],
                    })

                # ==================================================
                # GLOBAL MODEL CHARACTERISTICS
                # ==================================================

                y_bar = np.mean(y)

                # Total variability
                SST = np.sum(
                    (y - y_bar) ** 2
                )

                # Squared variability of fitted values
                SSR_naive = np.sum(
                    (fitted - y_bar) ** 2
                )

                # --------------------------------------------------
                # Cross term
                #
                # For a general smoother:
                #
                # SST =
                # SSE + SSR + cross_term
                #
                # and cross_term need not be zero.
                # --------------------------------------------------

                cross_term = (
                    2
                    * np.sum(
                        residuals
                        * (fitted - y_bar)
                    )
                )

                # Numerical check
                decomposition_error = (
                    SST
                    - SSE
                    - SSR_naive
                    - cross_term
                )

                # --------------------------------------------------
                # Pseudo-R2
                # --------------------------------------------------

                pseudo_R2 = (
                    1 - SSE / SST
                    if SST > 0
                    else np.nan
                )

                # --------------------------------------------------
                # Normalized local reconstruction index
                #
                # R_local^2 =
                #
                # 1 - SSE /
                # sum_k sum_r
                # B_k(x_r)(y_r-F_k)^2
                # --------------------------------------------------

                local_dispersion = np.sum(
                    Bv
                    * (
                        y[:, None]
                        - Fv[None, :]
                    ) ** 2
                )

                local_R2 = (
                    1
                    - SSE
                    / local_dispersion
                    if local_dispersion > 0
                    else np.nan
                )

                # --------------------------------------------------
                # Optional ANOVA-like statistic
                #
                # IMPORTANT:
                # This is only the heuristic formula appearing
                # in the manuscript appendix.
                # --------------------------------------------------

                anova_F = np.nan
                anova_p = np.nan

                if include_anova_heuristic:

                    df_model_anova = (
                        M_supported - 1
                    )

                    df_resid_anova = (
                        m - M_supported
                    )

                    if (
                        df_model_anova > 0
                        and df_resid_anova > 0
                        and SSE > 0
                    ):

                        anova_F = (
                            SSR_naive
                            / df_model_anova
                        ) / (
                            SSE
                            / df_resid_anova
                        )

                        anova_p = stats.f.sf(
                            anova_F,
                            df_model_anova,
                            df_resid_anova
                        )

                # --------------------------------------------------
                # Model output
                # --------------------------------------------------

                model_rows.append({
                    "group":
                        grp,

                    "variable":
                        value_col,

                    "n_obs":
                        m,

                    "n_components_total":
                        len(component_cols),

                    "n_components_supported":
                        M_supported,

                    "partition_max_abs_error":
                        partition_error,

                    "SSE":
                        SSE,

                    "SST":
                        SST,

                    "SSR_naive":
                        SSR_naive,

                    "cross_term":
                        cross_term,

                    "decomposition_error":
                        decomposition_error,

                    "pseudo_R2":
                        pseudo_R2,

                    "local_dispersion":
                        local_dispersion,

                    "local_R2":
                        local_R2,

                    "df_eff":
                        df_eff,

                    "trace_STS":
                        trace_STS,

                    "df_component_count":
                        df_component_count,

                    "df_effective":
                        df_effective,

                    "df_residual_trace":
                        df_residual_trace,

                    "variance_df_method":
                        variance_df,

                    "df_used":
                        df_used,

                    "sigma2":
                        sigma2,

                    "sigma2_component_count":
                        sigma2_component_count,

                    "sigma2_effective":
                        sigma2_effective,

                    "sigma2_residual_trace":
                        sigma2_residual_trace,

                    "anova_F_heuristic":
                        anova_F,

                    "anova_p_heuristic":
                        anova_p,
                })

        # --------------------------------------------------
        # Return everything
        # --------------------------------------------------

        return {
            "components":
                pd.DataFrame(component_rows),

            "fitted":
                pd.DataFrame(fitted_rows),

            "model":
                pd.DataFrame(model_rows),

            "smoothing_matrices":
                smoothing_matrices,
        }
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
        """
        Statistical inference for local slopes of multidimensional
        F-transform components.

        For each slope along `along_col`, the standard error is calculated
        from participant-level contributions to the exact tensor-product
        linear contrast defining that slope.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data.

        value_cols : str or list of str
            Response variable(s).

        along_col : str
            Input dimension along which local slopes are computed.

        subject_col : str, default="name"
            Column identifying participants / clusters.

        groups : list, optional
            Groups to analyse. If None, all groups are used.

        Returns
        -------
        pandas.DataFrame
            Table containing slopes, standard errors, t statistics,
            p-values, confidence intervals, and component information.
        """

        # --------------------------------------------------
        # Basic checks
        # --------------------------------------------------

        if isinstance(value_cols, str):
            value_cols = [value_cols]

        if along_col not in self.input_cols:
            raise ValueError(
                f"{along_col} is not in input_cols."
            )

        if subject_col not in data.columns:
            raise ValueError(
                f"{subject_col} is not present in data."
            )

        for value_col in value_cols:
            if value_col not in data.columns:
                raise ValueError(
                    f"{value_col} is not present in data."
                )

        # --------------------------------------------------
        # Fit fuzzy partition if necessary
        # --------------------------------------------------

        if not self.nodes_:
            self.fit_partition(data)

        # Add both:
        #   1. marginal memberships A_{j,k}
        #   2. tensor-product memberships B_k
        #
        # The tensor-product components have names such as
        # C_x1A1_x2A3
        # --------------------------------------------------

        data_A = self.add_memberships(data)

        # --------------------------------------------------
        # Groups
        # --------------------------------------------------

        if groups is not None:
            groups_to_use = groups

        elif self.group_order is None:
            groups_to_use = sorted(
                data_A[self.group_col]
                .dropna()
                .unique()
            )

        else:
            groups_to_use = self.group_order

        # --------------------------------------------------
        # Direct F-transform components
        # --------------------------------------------------

        components = self.compute_components(
            data=data,
            value_cols=value_cols,
            groups=groups_to_use
        )

        # --------------------------------------------------
        # Local slopes of neighbouring F-transform components
        # --------------------------------------------------

        slopes = self.compute_slopes_along(
            components,
            along_col=along_col
        )

        rows = []

        # Component column of the selected dimension
        along_component_col = f"{along_col}_component"

        # Dimensions held fixed when slope is taken
        other_cols = [
            col
            for col in self.input_cols
            if col != along_col
        ]

        # --------------------------------------------------
        # Loop over groups
        # --------------------------------------------------

        for grp in groups_to_use:

            sub_g = data_A.loc[
                data_A[self.group_col] == grp
            ].copy()

            persons = (
                sub_g[subject_col]
                .dropna()
                .unique()
            )

            G = len(persons)

            # --------------------------------------------------
            # Loop over response variables
            # --------------------------------------------------

            for value_col in value_cols:

                slopes_sub = slopes.loc[
                    (slopes["group"] == grp)
                    & (slopes["variable"] == value_col)
                ]

                # --------------------------------------------------
                # Loop over individual local slopes
                # --------------------------------------------------

                for _, slope_row in slopes_sub.iterrows():

                    segment = slope_row["segment"]
                    slope = slope_row["slope"]

                    # Example:
                    # A2_to_A3  -->  from_comp = 2, to_comp = 3

                    from_comp = int(
                        segment
                        .split("_to_")[0]
                        .replace("A", "")
                    )

                    to_comp = int(
                        segment
                        .split("_to_")[1]
                        .replace("A", "")
                    )

                    x0 = slope_row["from_node"]
                    x1 = slope_row["to_node"]

                    h = x1 - x0

                    # --------------------------------------------------
                    # Construct the two FULL tensor-product components
                    #
                    # Example:
                    #
                    # along_col = x1
                    # fixed x2 component = 3
                    #
                    # from:
                    #   C_x1A2_x2A3
                    #
                    # to:
                    #   C_x1A3_x2A3
                    #
                    # --------------------------------------------------

                    from_indices = {}
                    to_indices = {}

                    for col in self.input_cols:

                        if col == along_col:

                            from_indices[col] = from_comp
                            to_indices[col] = to_comp

                        else:

                            fixed_index = int(
                                slope_row[
                                    f"{col}_component"
                                ]
                            )

                            from_indices[col] = fixed_index
                            to_indices[col] = fixed_index

                    from_component = (
                        "C_"
                        + "_".join(
                            f"{col}A{from_indices[col]}"
                            for col in self.input_cols
                        )
                    )

                    to_component = (
                        "C_"
                        + "_".join(
                            f"{col}A{to_indices[col]}"
                            for col in self.input_cols
                        )
                    )

                    # --------------------------------------------------
                    # Check that tensor-product components exist
                    # --------------------------------------------------

                    if from_component not in sub_g.columns:
                        raise ValueError(
                            f"Component {from_component} "
                            "not found in membership data."
                        )

                    if to_component not in sub_g.columns:
                        raise ValueError(
                            f"Component {to_component} "
                            "not found in membership data."
                        )

                    # --------------------------------------------------
                    # Tensor-product denominators
                    #
                    # D_- = sum B_-(x_r)
                    # D_+ = sum B_+(x_r)
                    # --------------------------------------------------

                    denom_from = sub_g[from_component].sum()
                    denom_to = sub_g[to_component].sum()

                    # --------------------------------------------------
                    # Default degrees of freedom
                    # --------------------------------------------------

                    df = G - 1

                    # --------------------------------------------------
                    # Invalid cases
                    # --------------------------------------------------

                    if (
                        denom_from <= 0
                        or denom_to <= 0
                        or G < 2
                        or h == 0
                        or not np.isfinite(slope)
                    ):

                        se = np.nan
                        t_value = np.nan
                        p_value = np.nan
                        ci_lower = np.nan
                        ci_upper = np.nan

                    else:

                        # --------------------------------------------------
                        # Exact linear-contrast weights
                        #
                        # c_r =
                        #
                        # 1/h [
                        #     B_+(x_r) / D_+
                        #     -
                        #     B_-(x_r) / D_-
                        # ]
                        #
                        # Hence
                        #
                        # S = sum_r c_r y_r
                        #
                        # --------------------------------------------------

                        contrast_weights = (
                            sub_g[to_component].to_numpy(
                                dtype=float
                            )
                            / denom_to
                            -
                            sub_g[from_component].to_numpy(
                                dtype=float
                            )
                            / denom_from
                        ) / h

                        # --------------------------------------------------
                        # Observation-level contributions
                        #
                        # c_r y_r
                        # --------------------------------------------------

                        response = sub_g[
                            value_col
                        ].to_numpy(dtype=float)

                        obs_contribution = (
                            contrast_weights * response
                        )

                        # --------------------------------------------------
                        # Participant-level contributions
                        #
                        # U_p = sum_{r in I_p} c_r y_r
                        # --------------------------------------------------

                        contribution_df = pd.DataFrame({
                            "_subject":
                                sub_g[subject_col].to_numpy(),
                            "_contribution":
                                obs_contribution
                        })

                        contribution_df = (
                            contribution_df
                            .dropna(subset=["_subject"])
                        )

                        contrib = (
                            contribution_df
                            .groupby(
                                "_subject",
                                observed=True
                            )["_contribution"]
                            .sum()
                        )

                        # --------------------------------------------------
                        # Make sure all participants are represented.
                        #
                        # A participant with zero contribution must still
                        # enter the cluster variance as zero.
                        # --------------------------------------------------

                        contrib = contrib.reindex(
                            persons,
                            fill_value=0.0
                        )

                        # --------------------------------------------------
                        # Participant-level variance estimator
                        #
                        # Var(S) =
                        #
                        # G/(G-1)
                        # sum_p (U_p - U_bar)^2
                        #
                        # --------------------------------------------------

                        centered = (
                            contrib
                            - contrib.mean()
                        )

                        var = (
                            G / (G - 1)
                            * np.sum(
                                centered.to_numpy() ** 2
                            )
                        )

                        # Protect against tiny negative values caused by
                        # floating-point arithmetic
                        var = max(float(var), 0.0)

                        se = np.sqrt(var)

                        # --------------------------------------------------
                        # t statistic
                        # --------------------------------------------------

                        if se > 0 and np.isfinite(se):

                            t_value = slope / se

                            # Two-sided p-value
                            p_value = (
                                2
                                * stats.t.sf(
                                    np.abs(t_value),
                                    df=df
                                )
                            )

                            # 95% confidence interval
                            t_crit = stats.t.ppf(
                                0.975,
                                df=df
                            )

                            ci_lower = (
                                slope
                                - t_crit * se
                            )

                            ci_upper = (
                                slope
                                + t_crit * se
                            )

                        else:

                            t_value = np.nan
                            p_value = np.nan
                            ci_lower = np.nan
                            ci_upper = np.nan

                    # --------------------------------------------------
                    # Output row
                    # --------------------------------------------------

                    result_row = {
                        "group": grp,
                        "variable": value_col,
                        "along": along_col,

                        "segment": segment,

                        "from_component":
                            from_component,

                        "to_component":
                            to_component,

                        "from_node":
                            x0,

                        "to_node":
                            x1,

                        "slope":
                            slope,

                        "SE":
                            se,

                        "t_value":
                            t_value,

                        "df":
                            df,

                        "p_value":
                            p_value,

                        "ci_lower":
                            ci_lower,

                        "ci_upper":
                            ci_upper,

                        "n_persons":
                            G,

                        "n_obs":
                            len(sub_g)
                    }

                    # --------------------------------------------------
                    # Preserve information about dimensions held fixed
                    #
                    # Example:
                    # x2_component = 3
                    # x2_node = ...
                    # --------------------------------------------------

                    for col in other_cols:

                        result_row[
                            f"{col}_component"
                        ] = int(
                            slope_row[
                                f"{col}_component"
                            ]
                        )

                        if f"{col}_node" in slope_row.index:
                            result_row[
                                f"{col}_node"
                            ] = slope_row[
                                f"{col}_node"
                            ]

                    rows.append(result_row)

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