### functions for cluster analysis
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, normalize, StandardScaler
from statsmodels.api import OLS, add_constant
from scipy.stats.mstats import winsorize
from rich.progress import (
    track,
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)


def create_cluster_dummy(adata, cluster_col, cluster):
    """
    Creates a dummy variable where a cell is 1 if it belongs to
    a cluster and 0 otherwise.
    """
    x = np.zeros([adata.X.shape[0], 1])

    for cell in adata.obs.index:

        if adata.obs[cluster_col][int(cell)] == cluster:
            x[int(cell)] = 1

    return x


def get_feature_matrix(adata, scale_values=False, cofactor=1):
    """
    Scales and concatenates all the features arrays: 
    expression, co-localization and morphology.
    """

    correlations = adata.obsm["correlations"]

    if scale_values:
        morphology = adata.obsm["morphology"]
        for i in range(adata.obsm["morphology"].shape[1]):
            morphology[:, i] = winsorize(
                adata.obsm["morphology"][:, i], limits=[0, 0.01]
            )

        expression = np.arcsinh(adata.X / cofactor)
        for j in range(adata.X.shape[1]):
            expression[:, j] = winsorize(expression[:, j], limits=[0, 0.01])
    else:
        morphology = adata.obsm["morphology"]
        expression = adata.X

    y = StandardScaler().fit_transform(
        np.concatenate([expression, correlations, morphology], axis=1)
    )

    var_names = np.concatenate(
        [
            adata.var_names,
            adata.uns["names_correlations"],
            adata.uns["names_morphology"],
        ]
    )

    return y, var_names


def rank_features_in_groups(adata, group_col, scale_values=False, cofactor=1):
    """
    For each cluster, uses the dummy variable created for each cluster and
    the full feature matrix, to run an Ordinary Least Squares to find and rank
    which features are driving the membership of each cluster.
    """
    ranked_features_in_groups = {}
    dfs = []
    # create the feature matrix for entire adata
    y, var_names = get_feature_matrix(
        adata, scale_values=scale_values, cofactor=cofactor
    )
    y = add_constant(y)  # add intercept

    for group in adata.obs[group_col].unique():
        ranked_features_in_groups[group] = {}
        x = create_cluster_dummy(adata, group_col, group)
        mod = OLS(x, y)
        res = mod.fit()

        df_values = pd.DataFrame(
            res.tvalues[1:],  # remove the intercept value
            index=var_names,
            columns=[f"{group}"],
        ).sort_values(by=f"{group}", ascending=False)

        ranked_features_in_groups[group]["names"] = df_values.index.to_list()
        ranked_features_in_groups[group]["t_values"] = df_values[
            f"{group}"
        ].to_list()

        dfs.append(df_values)

    fc_df = pd.concat(
        dfs, axis=1
    ).sort_index()  # index is sorted as alphabetical! (order with original var_names is NOT maintained!)

    return ranked_features_in_groups, fc_df
