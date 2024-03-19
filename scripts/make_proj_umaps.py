# make umaps for all runs
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

def top_common_features(df, top_n_features=5):

    sets_list = []

    for i in df.columns:
        abs_sorted_col = df[i].map(abs).sort_values(ascending=False)
        for j in abs_sorted_col.index.to_list()[0:top_n_features]:
            sets_list.append(j)

    common_features = list(set(sets_list))

    common_feat_df = df.loc[common_features]

    return common_feat_df



parser = argparse.ArgumentParser(description="Cluster output from best runs and make the umaps")

parser.add_argument("--init_adata", type=str, required=True, help='adata to read in the model')

parser.add_argument("--proj_adata", type=str, required=True, help='adata that will be projected')

parser.add_argument("--cohort", type=str, required=True, help="cohort name")

parser.add_argument("--batch_size", type=int, help='batch size for cohort')

parser.add_argument("--use_covs", type=bool, help='True or False for using covs', default=True)

parser.add_argument("--bkg_key_check", type=str, help='Key in background that might be different b/w base and proj adata', default=None)

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
    "--cofactor", type=float, help="Cofactor for arcsinh transformation", default=1.0
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
    "--random_seed",
    type=int,
    help='Random seed to set initialization',
    default=123,
)


parser.add_argument('--checkpoint_dir', type=str, required=True, help='directory with checkpoint to load')

parser.add_argument(
    "--output_dir", type=str, help="Directory to store the outputs", default="."
)

args = parser.parse_args()

init_adata = sc.read_h5ad(args.init_adata)

proj_adata = sc.read_h5ad(args.proj_adata)

if args.bkg_key_check is None:

    print('Nothing to check')

else:

    background_stains = ['RutheniumTetroxide', 'None', 'ArAr80', 'Hg202']

    background_stains_ind = []

    if proj_adata.obsm[args.bkg_key_check].shape[1] > init_adata.obsm[args.bkg_key_check].shape[1]:
        for i in proj_adata.uns[f"{args.bkg_key_check}_names"]:
            if len(set(i.split('_')) & set(background_stains)) > 0: # select the stains we want to match
                background_stains_ind.append(list(proj_adata.uns[f"{args.bkg_key_check}_names"]).index(i))

        proj_adata.obsm[args.bkg_key_check] = proj_adata.obsm[args.bkg_key_check][:,background_stains_ind] # subset to init size of key
    else:
        n_add = init_adata.obsm[args.bkg_key_check].shape[1] - proj_adata.obsm[args.bkg_key_check].shape[1] # diff b/w init and proj key features

        zeros_n_add = np.zeros((proj_adata.X.shape[0], n_add)) # create an array of 0's to append to end of key in proj

        proj_adata.obsm[args.bkg_key_check] = np.concatenate([proj_adata.obsm[args.bkg_key_check], zeros_n_add], axis=1) # add zeros for remaining features

    assert proj_adata.obsm[args.bkg_key_check].shape[1] == init_adata.obsm[args.bkg_key_check].shape[1] # check if same

#if args.cohort == 'Jackson-BC': # METABRIC used all the proteins listed -- panels are 'different'
#DROP_STAINS = ['Betacatenin','DNA1','DNA2','EpCAM','pERK12'] #stains that didn't work or want to be dropped

# else:
DROP_STAINS = ['DNA1', 'DNA2']

STAIN_IDS_init = np.in1d(init_adata.var_names, DROP_STAINS) #boolean for dropped stains

STAIN_IDS_proj = np.in1d(proj_adata.var_names, DROP_STAINS)

DROP_CORRELATIONS_init = []

DROP_CORRELATIONS_proj = []

DROP_WEIGHTS_init = []

DROP_WEIGHTS_proj = []

for i in DROP_STAINS:
    for j in init_adata.uns['names_correlations'].tolist():
        if i in j:
            DROP_CORRELATIONS_init.append(init_adata.uns['names_correlations'].tolist().index(j))
    for k in init_adata.uns['names_weights'].tolist():
        if i in k:
            DROP_WEIGHTS_init.append(init_adata.uns['names_weights'].tolist().index(k))

    for j in proj_adata.uns['names_correlations'].tolist():
        if i in j:
            DROP_CORRELATIONS_proj.append(proj_adata.uns['names_correlations'].tolist().index(j))
    for k in proj_adata.uns['names_weights'].tolist():
        if i in k:
            DROP_WEIGHTS_proj.append(proj_adata.uns['names_weights'].tolist().index(k))


raw_adata_init = init_adata.copy()[:, ~STAIN_IDS_init]

raw_adata_proj = proj_adata.copy()[:, ~STAIN_IDS_proj]

raw_adata_init.obsm['correlations'] = np.delete(raw_adata_init.obsm['correlations'], DROP_CORRELATIONS_init, axis=1) # drop the associated correlations
raw_adata_init.uns['names_correlations'] = np.delete(raw_adata_init.uns['names_correlations'], DROP_CORRELATIONS_init) # also drop the names

raw_adata_proj.obsm['correlations'] = np.delete(raw_adata_proj.obsm['correlations'], DROP_CORRELATIONS_proj, axis=1) # drop the associated correlations
raw_adata_proj.uns['names_correlations'] = np.delete(raw_adata_proj.uns['names_correlations'], DROP_CORRELATIONS_proj) # also drop the names

raw_adata_init.obsm['weights'] = np.delete(raw_adata_init.obsm['weights'], DROP_WEIGHTS_init, axis=1) # drop the associated weight correlations
raw_adata_init.uns['names_weights'] = np.delete(raw_adata_init.uns['names_weights'], DROP_WEIGHTS_init) # also drop the names

raw_adata_proj.obsm['weights'] = np.delete(raw_adata_proj.obsm['weights'], DROP_WEIGHTS_proj, axis=1) # drop the associated weight correlations
raw_adata_proj.uns['names_weights'] = np.delete(raw_adata_proj.uns['names_weights'], DROP_WEIGHTS_proj) # also drop the names

print('raw init', raw_adata_init.X.shape[1])
print('raw proj', raw_adata_proj.X.shape[1])

print('original and dropped init', init_adata.X.shape[1] , len(DROP_STAINS))
print('original and dropped init', proj_adata.X.shape[1], len(DROP_STAINS))

assert raw_adata_init.X.shape[1] == init_adata.X.shape[1] - len(DROP_STAINS)
assert raw_adata_init.obsm['correlations'].shape[1] == init_adata.obsm['correlations'].shape[1] - len(DROP_CORRELATIONS_init)
assert raw_adata_proj.X.shape[1] == proj_adata.X.shape[1] - len(DROP_STAINS)
assert raw_adata_proj.obsm['correlations'].shape[1] == proj_adata.obsm['correlations'].shape[1] - len(DROP_CORRELATIONS_proj)
assert raw_adata_init.X.shape[1] == raw_adata_proj.X.shape[1] # have to have the same number of features


#raw_adata1 = raw_adata.copy() #create a copy of the adata that was input

n_total_features = (
    raw_adata_init.X.shape[1]
    + raw_adata_init.obsm["correlations"].shape[1]
    + raw_adata_init.obsm["morphology"].shape[1]
)


print("Set up the model")



E_me, E_cr, E_mr, E_sc = [args.hidden_dim_size, args.hidden_dim_size, args.hidden_dim_size, args.hidden_dim_size]


input_exp_dim, input_corr_dim, input_morph_dim, input_spcont_dim = [
    raw_adata_init.shape[1],
    raw_adata_init.obsm["correlations"].shape[1],
    raw_adata_init.obsm["morphology"].shape[1],
    n_total_features,
]

print(f"input dims:{input_exp_dim}, {input_corr_dim}, {input_morph_dim}, {input_spcont_dim}")
keys = []
if args.use_covs:
    cat_list = []
    
    for key in raw_adata_init.obsm.keys():
        # print(key)
        if key not in ["correlations", "morphology", "spatial", "xy"]:
            keys.append(key)
    for cat_key in keys:
        # print(cat_key)
        # print(f"{cat_key} shape:", adata.obsm[cat_key].shape)
        category = raw_adata_init.obsm[cat_key]
        cat_list.append(category)
    cat_list = np.concatenate(cat_list, 1)
    n_covariates = cat_list.shape[1]
    E_cov = args.hidden_dim_size
else:
    n_covariates = 0
    E_cov = 0



model_checkpoint = [i for i in os.listdir(args.checkpoint_dir) if ".ckpt" in i] #should only be 1 -- saved best model

print("model_checkpoint", model_checkpoint)

load_chkpt = torch.load(os.path.join(args.checkpoint_dir, model_checkpoint[0]))

state_dict = load_chkpt['state_dict']
#print(state_dict)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    #print("key", k)
    if "weight" or "bias" in k:
        #print("changing", k)
        name = "module."+k # add `module.`
        #print("new name", name)
    else:
        #print("staying same", k)
        name = k
    new_state_dict[name] = v
# load params

load_chkpt['state_dict'] = new_state_dict


model = hmivaeModel(
    adata=raw_adata_init,
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

wandb.finish()

model.load_state_dict(new_state_dict, strict=False)


print("Best model loaded from checkpoint")


adata = model.get_latent_representation( #use the best model to get the latent representations
    adata=raw_adata_proj, #put in the copy of the original from before training
    protein_correlations_obsm_key="correlations",
    cell_morphology_obsm_key="morphology",
    continuous_covariate_keys=keys,
    is_trained_model=True,
    batch_correct=args.batch_correct,
    use_covs=True,
)

# print("Doing cluster and neighbourhood enrichment analysis")

# print("===> Clustering using integrated space")

# sc.pp.neighbors(
#     adata, n_neighbors=100, use_rep="VAE", key_added="vae"
# )  # 100 nearest neighbours, will be used in downstream tests -- keep with PG


# sc.tl.leiden(adata, neighbors_key="vae")

# print("===> Clustering using specific views")

# print("Expression")

# sc.pp.neighbors(
#     adata, n_neighbors=100, use_rep="expression_embedding", key_added="expression"
# )  # 100 nearest neighbours, will be used in downstream tests -- keep with PG

# sc.tl.leiden(adata, neighbors_key="expression", key_added='expression_leiden', random_state=args.random_seed, resolution=0.5) # expression wasn't too bad

# print("Correlation")

# sc.pp.neighbors(
#     adata, n_neighbors=100, use_rep="correlation_embedding", key_added="correlation" 
# )

# sc.tl.leiden(adata, neighbors_key="correlation", key_added='correlation_leiden',random_state=args.random_seed) # probably no need to change correlation because there were few anyways

# print("Morphology")

# sc.pp.neighbors(
#     adata, n_neighbors=100, use_rep="morphology_embedding", key_added="morphology"
# )

# sc.tl.leiden(adata, neighbors_key="morphology", key_added='morphology_leiden',random_state=args.random_seed, resolution=0.1) # pull it way down because there were LOTS of clusters

# print("Spatial context")

# sc.pp.neighbors(
#     adata, n_neighbors=100, use_rep="spatial_context_embedding", key_added="spatial_context"
# )

# sc.tl.leiden(adata, neighbors_key="spatial_context", key_added='spatial_context_leiden', random_state=args.random_seed, resolution=0.5)

# print("===> Creating UMAPs")

# print("Integrated space")

# sc.tl.umap(adata, neighbors_key="vae", random_state=args.random_seed)

# adata.obsm['X_umap_int'] = adata.obsm['X_umap'].copy()

# print("Expression")

# sc.tl.umap(adata, neighbors_key="expression", random_state=args.random_seed)

# adata.obsm['X_umap_exp'] = adata.obsm['X_umap'].copy()

# print("Correlations")

# sc.tl.umap(adata, neighbors_key="correlation", random_state=args.random_seed)

# adata.obsm['X_umap_corr'] = adata.obsm['X_umap'].copy()

# print("Morphology")

# sc.tl.umap(adata, neighbors_key="morphology", random_state=args.random_seed)

# adata.obsm['X_umap_morph'] = adata.obsm['X_umap'].copy()

# print("Spatial context")

# sc.tl.umap(adata, neighbors_key="spatial_context", random_state=args.random_seed)

# adata.obsm['X_umap_spct'] = adata.obsm['X_umap'].copy()

# print("Neighbourhood enrichment analysis")

# sq.gr.spatial_neighbors(adata)
# sq.gr.nhood_enrichment(adata, cluster_key="leiden")


# adata.obs.to_csv(os.path.join(args.output_dir, f"{args.cohort}_clusters.tsv"), sep="\t")
adata.write(os.path.join(args.output_dir, f"{args.cohort}_adata_new.h5ad"))



