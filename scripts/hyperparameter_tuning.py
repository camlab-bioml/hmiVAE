## run with hmivae
from anndata import AnnData
import scanpy as sc
import pandas as pd
import numpy as np
import hmivae
from hmivae._hmivae_model import hmivaeModel
from hmivae.ScModeDataloader import ScModeDataloader
import argparse
import wandb
import os
import squidpy as sq
import time
from statsmodels.api import OLS, add_constant
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
import time
import phenograph
import torch
from collections import OrderedDict
from rich.progress import (
    track,
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
import seaborn as sns
import matplotlib.pyplot as plt
import yaml

def arrange_features(vars_lst, adata):

    arranged_features = {'E':[], 'C': [], 'M': [], 'S': []}
    orig_list = vars_lst

    for i in orig_list:

        if i in adata.var_names:
            arranged_features['E'].append(i)
        elif i in adata.uns['names_morphology']:
            arranged_features['M'].append(i)
        elif i in adata.uns['names_correlations']:
            arranged_features['C'].append(i)
        else:
            arranged_features['S'].append(i)
    arr_list = [*np.sort(arranged_features['E']).tolist(),*np.sort(arranged_features['C']).tolist(),*np.sort(arranged_features['M']).tolist(),*np.sort(arranged_features['S']).tolist()]
    return arranged_features, arr_list


def create_cluster_dummy(adata, cluster_col, cluster):
    # n_clusters = len(adata.obs[cluster_col].unique().tolist())
    x = np.zeros([adata.X.shape[0], 1])

    for cell in adata.obs.index:
        # cell_cluster = int(adata.obs[cluster_col][cell])
        # print(type(cell), type(cluster))

        if adata.obs[cluster_col][int(cell)] == cluster:
            x[int(cell)] = 1

    return x


def get_feature_matrix(adata, scale_values=False, cofactor=1, weights=True):

    correlations = adata.obsm["correlations"]
    if weights:
        correlations = np.multiply(correlations, adata.obsm['weights']) #multiply weights with correlations

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
    
    progress = Progress(
        TextColumn(f"[progress.description]Ranking features in {group_col} groups"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )
    ranked_features_in_groups = {}
    dfs = []
    # create the feature matrix for entire adata
    y, var_names = get_feature_matrix(
        adata, scale_values=scale_values, cofactor=cofactor
    )
    y = add_constant(y)  # add intercept

    with progress:

        for group in progress.track(adata.obs[group_col].unique()):
            ranked_features_in_groups[str(group)] = {}
            x = create_cluster_dummy(adata, group_col, group)
            mod = OLS(x, y)
            res = mod.fit()

            df_values = pd.DataFrame(
                res.tvalues[1:],  # remove the intercept value
                index=var_names,
                columns=[f"t_value_{group}"],
            ).sort_values(by=f"t_value_{group}", ascending=False)

            ranked_features_in_groups[str(group)]["names"] = df_values.index.to_list()
            ranked_features_in_groups[str(group)]["t_values"] = df_values[
                f"t_value_{group}"
            ].to_list()

            #print('df index:', df_values.index.tolist())

            dfs.append(df_values)

    fc_df = pd.concat(
        dfs, axis=1
    ).sort_index()  # index is sorted as alphabetical! (order with original var_names is NOT maintained!)

    fc_df.index = fc_df.index.map(str)
    fc_df.columns = fc_df.columns.map(str)

    adata.uns[f"{group_col}_ranked_features_in_groups"] = ranked_features_in_groups
    adata.uns[f"{group_col}_feature_scores"] = fc_df

    #return adata

def top_common_features(df, top_n_features=10):

    sets_list = []

    for i in df.columns:
        abs_sorted_col = df[i].map(abs).sort_values(ascending=False)
        for j in abs_sorted_col.index.to_list()[0:top_n_features]:
            sets_list.append(j)

    common_features = list(set(sets_list))

    common_feat_df = df.loc[common_features]

    return common_feat_df


parser = argparse.ArgumentParser(description="Run hmiVAE")

parser.add_argument(
    "--adata", type=str, required=True, help="AnnData file with all the inputs"
)

parser.add_argument(
    "--include_all_views",
    type=int,
    help="Run model using all views",
    default=1,
    choices=[0, 1],
)

parser.add_argument(
    "--remove_view",
    type=str,
    help="Name of view to leave out. One of ['expression', 'correlation', 'morphology', 'spatial']. Must be given when `include_all_views` is False",
    default=None,
    choices=["expression", "correlation", "morphology", "spatial"],
)

parser.add_argument(
    "--use_covs",
    type=bool,
    help="True/False for using background covariates",
    default=True,
)

parser.add_argument(
    "--use_weights",
    type=bool,
    help="True/False for using correlation weights",
    default=True,
)

parser.add_argument(
    "--batch_correct",
    type=bool,
    help="True/False for using one-hot encoding for batch correction",
    default=True,
)

parser.add_argument(
    "--batch_size",
    type=int,
    help="Batch size for train/test data, default=1234",
    default=1234,
)

parser.add_argument(
    "--hidden_dim_size",
    type=int,
    help='Size for view-specific hidden layers',
    default=32,
)

parser.add_argument(
    "--latent_dim",
    type=int,
    help='Size for the final latent representation layer',
    default=10,
)

parser.add_argument(
    "--n_hidden",
    type=int,
    help='Number of hidden layers',
    default=1,
)

parser.add_argument(
    "--beta_scheme",
    type=str,
    help='Scheme to use for beta vae',
    default='warmup',
    choices=['constant', 'warmup'],
)

parser.add_argument(
    "--cofactor", type=float, help="Cofactor for arcsinh transformation", default=1.0
)

parser.add_argument(
    "--random_seed", type=int, help='Random seed for weights initialization', default=1234
)

parser.add_argument("--cohort", type=str, help="Cohort name", default="cohort")

parser.add_argument(
    "--output_dir", type=str, help="Directory to store the outputs", default="."
)

args = parser.parse_args()

orig_adata = sc.read_h5ad(args.adata)

if args.cohort == 'Jackson-BC': # METABRIC used all the proteins listed -- panels are 'different'
    DROP_STAINS = ['Betacatenin','DNA1','DNA2','EpCAM','pERK12'] #stains that didn't work or want to be dropped

else:
    DROP_STAINS = ['DNA1', 'DNA2']

STAIN_IDS = np.in1d(orig_adata.var_names, DROP_STAINS) #boolean for dropped stains

hyperparams_dict = {
    'use_covs': args.use_covs,
    'use_weights': args.use_weights,
    'n_hidden': args.n_hidden,
    'hidden_dim_size': args.hidden_dim_size,
    'batch_size': args.batch_size,
    'latent_dim': args.latent_dim,
    'random_seed': args.random_seed,
    'beta_scheme': args.beta_scheme,
    'cofactor': args.cofactor,
}

DROP_CORRELATIONS = []

DROP_WEIGHTS = []

for i in DROP_STAINS:
    for j in orig_adata.uns['names_correlations'].tolist():
        if i in j:
            DROP_CORRELATIONS.append(orig_adata.uns['names_correlations'].tolist().index(j))
    for k in orig_adata.uns['names_weights'].tolist():
        if i in k:
            DROP_WEIGHTS.append(orig_adata.uns['names_weights'].tolist().index(k))


raw_adata = orig_adata.copy()[:, ~STAIN_IDS]

raw_adata.obsm['correlations'] = np.delete(raw_adata.obsm['correlations'], DROP_CORRELATIONS, axis=1) # drop the associated correlations
raw_adata.uns['names_correlations'] = np.delete(raw_adata.uns['names_correlations'], DROP_CORRELATIONS) # also drop the names

raw_adata.obsm['weights'] = np.delete(raw_adata.obsm['weights'], DROP_WEIGHTS, axis=1) # drop the associated weight correlations
raw_adata.uns['names_weights'] = np.delete(raw_adata.uns['names_weights'], DROP_CORRELATIONS) # also drop the names

print('raw', raw_adata.X.shape[1])

print('original and dropped', orig_adata.X.shape[1] , len(DROP_STAINS))

assert raw_adata.X.shape[1] == orig_adata.X.shape[1] - len(DROP_STAINS)
assert raw_adata.obsm['correlations'].shape[1] == orig_adata.obsm['correlations'].shape[1] - len(DROP_CORRELATIONS)

raw_adata1 = raw_adata.copy() #create a copy of the adata that was input

n_total_features = (
    raw_adata.X.shape[1]
    + raw_adata.obsm["correlations"].shape[1]
    + raw_adata.obsm["morphology"].shape[1]
)

hyperparams_dict.update({
    #  'raw_adata_exp_max': np.max(raw_adata.X),
    #  'raw_adata_exp_min': np.min(raw_adata.X),
    #  'raw_adata_corr_max': np.max(raw_adata.obsm['correlations']),
    #  'raw_adata_corr_min': np.min(raw_adata.obsm['correlations']),
    #  'raw_adata_morph_max': np.max(raw_adata.obsm['morphology']),
    #  'raw_adata_morph_min': np.min(raw_adata.obsm['morphology']), # wasn't really saving what I thought it was
     'N_features': n_total_features,
     'N_cells': raw_adata.X.shape[0]
})

# print(
#      'raw_adata_exp_max', np.max(raw_adata.X),
#      'raw_adata_exp_min', np.min(raw_adata.X),
#      'raw_adata_corr_max', np.max(raw_adata.obsm['correlations']),
#      'raw_adata_corr_min', np.min(raw_adata.obsm['correlations']),
#      'raw_adata_morph_max', np.max(raw_adata.obsm['morphology']),
#      'raw_adata_morph_min', np.min(raw_adata.obsm['morphology']),
#      'N_features', n_total_features,
#      'N_cells', raw_adata.X.shape[0]
# )


print("Set up the model")

start = time.time()


E_me, E_cr, E_mr, E_sc = [args.hidden_dim_size, args.hidden_dim_size, args.hidden_dim_size, args.hidden_dim_size]
input_exp_dim, input_corr_dim, input_morph_dim, input_spcont_dim = [
    raw_adata.shape[1],
    raw_adata.obsm["correlations"].shape[1],
    raw_adata.obsm["morphology"].shape[1],
    n_total_features,
]

print(f"input dims:{input_exp_dim}, {input_corr_dim}, {input_morph_dim}, {input_spcont_dim}")
keys = []
if args.use_covs:
    cat_list = []
    
    for key in raw_adata.obsm.keys():
        # print(key)
        if key not in ["correlations", "morphology", "spatial", "xy"]:
            keys.append(key)
    for cat_key in keys:
        # print(cat_key)
        # print(f"{cat_key} shape:", adata.obsm[cat_key].shape)
        category = raw_adata.obsm[cat_key]
        cat_list.append(category)
    cat_list = np.concatenate(cat_list, 1)
    n_covariates = cat_list.shape[1]
    E_cov = args.hidden_dim_size
else:
    n_covariates = 0
    E_cov = 0

model = hmivaeModel(
    adata=raw_adata,
    input_exp_dim=input_exp_dim,
    input_corr_dim=input_corr_dim,
    input_morph_dim=input_morph_dim,
    input_spcont_dim=input_spcont_dim,
    E_me=E_me,
    E_cr=E_cr,
    E_mr=E_mr,
    E_sc=E_sc,
    E_cov=E_cov,
    latent_dim=args.latent_dim,
    cofactor=args.cofactor,
    use_covs=args.use_covs,
    cohort=args.cohort,
    use_weights=args.use_weights,
    beta_scheme=args.beta_scheme,
    n_covariates=n_covariates,
    batch_correct=args.batch_correct,
    batch_size=args.batch_size,
    random_seed=args.random_seed,
    n_hidden=args.n_hidden,
    leave_out_view=args.remove_view,
    output_dir=args.output_dir,
)


print("Start training")


model.train()

wandb.finish()

model_checkpoint = [i for i in os.listdir(args.output_dir) if ".ckpt" in i] #should only be 1 -- saved best model

hyperparams_dict['best_model'] = model_checkpoint[0]

hyperparams_dict['recon_lik_test'] = float(model_checkpoint[0].split('=')[-1].split('.ckpt')[0]) # save the best reconstruction likelihood of the run

stop = time.time()

hyperparams_dict['training_time_mins'] = (stop-start)/60

hp_file=open(os.path.join(args.output_dir, "hyperparameters.yaml"),"w")

yaml.dump(hyperparams_dict,hp_file)

hp_file.close()
