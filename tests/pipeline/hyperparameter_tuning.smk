## Snakefile for running different hyperparameters per cohort

import pandas as pd 
import numpy as np 
import scanpy as sc
import os 

cohort = config['cohort']

input_dir = config['tidy_output_dir']

hidden_dims = [8,32,64]
latent_dims = [10,20]

random_seeds = [0,1,42,123,1234]

beta_scheme = ['constant', 'warmup']

n_hiddens = [1,2]

N_total_cells = 800000

batch_sizes = [
    int(np.floor(N_total_cells/20)),
    int(np.floor(N_total_cells/50)),
    int(np.floor(N_total_cells/100)),
    int(np.floor(N_total_cells/150)),
    int(np.floor(N_total_cells/200))
]

hyperparams_outputs = {
    'logs': expand(os.path.join(
        config['hyperparams_tuning_dir']+'/'+cohort+'_vae_out_nh{n_hidden}_hd{hidden_dim}_ls{latent_dim}_beta{betascheme}_rs{random_seed}_bs{batchsize}', 
        "hyperparameters.yaml"),
    hidden_dim=hidden_dims,
    latent_dim=latent_dims,
    batchsize=batch_sizes,
    betascheme=beta_scheme,
    random_seed=random_seeds,
    n_hidden=n_hiddens,
    )
}


rule run_hyperparameter_tuning:
    params:
        cofactor = config['cofactor'],
        cohort = config['cohort'],
        output_dir = config['hyperparams_tuning_dir']+'/'+cohort+'_vae_out_nh{n_hidden}_hd{hidden_dim}_ls{latent_dim}_beta{betascheme}_rs{random_seed}_bs{batchsize}'


    input:
        merged_h5ad,

    output:
        os.path.join(
        config['hyperparams_tuning_dir']+'/'+cohort+'_vae_out_nh{n_hidden}_hd{hidden_dim}_ls{latent_dim}_beta{betascheme}_rs{random_seed}_bs{batchsize}', 
        "hyperparameters.yaml"),
    resources:
        mem_mb = 5000,

    shell:
        "python hyperparameter_tuning.py --adata {input} "
        "--cofactor {params.cofactor} --cohort {params.cohort} "
        "--batch_size {wildcards.batchsize} "
        "--beta_scheme {wildcards.betascheme} "
        "--hidden_dim_size {wildcards.hidden_dim} "
        "--latent_dim {wildcards.latent_dim} "
        "--random_seed {wildcards.random_seed} "
        "--n_hidden {wildcards.n_hidden} "
        "--output_dir {params.output_dir} "


