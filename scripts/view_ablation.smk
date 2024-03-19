### Snakefile for ablation tests -- did not do this anymore
import numpy as np 
import pandas as pd
import os


merged_h5ad = config['tidy_output_dir']+'/all_samples_merged_vae.h5ad'

cohorts = ['basel', 'melanoma', 'metabric']

cofactor = {'basel': 5.0, 'metabric': 0.8, 'melanoma': 1.0}

batch_size = {'basel': 2314, 'metabric': 1423, 'melanoma': 4321}

n_hiddens = [8, 32, 64]

latent_dims = [10, 20]

betas = ['constant', 'warmup']

# this snakemake file specific

#views = ['expression', 'correlation', 'morphology', 'spatial']

# ablation_outputs = {
#     "ablation_h5ads": expand(os.path.join(config['vae_run_output_dir'], "{cohort}/{view}/{cohort}_adata_remove_{view}.h5ad"), cohort=cohorts, view=views),
#     "ablation_tsvs": expand(os.path.join(config['vae_run_output_dir'], "{cohort}/{view}/{cohort}_remove_{view}_clusters.tsv"), cohort=cohorts, view=views),
#     }

# survival_outputs = {
#     "cluster_props": expand(os.path.join(config['vae_run_output_dir'], "{cohort}/{view}/{cohort}_patient_cluster_props_remove_{view}.tsv"), cohort=cohorts, view=views),
#     "hi_low": expand(os.path.join(config['vae_run_output_dir'], "{cohort}/{view}/{cohort}_patient_cluster_hi_or_low_remove_{view}.tsv"), cohort=cohorts, view=views),
#     "patient_clusters": expand(os.path.join(config['vae_run_output_dir'], "{cohort}/{view}/{cohort}_patient_cluster_survival_remove_{view}.tsv"), cohort=cohorts, view=views),
# }

tuning_outputs = {
    'logs': expand(os.path.join(config['vae_run_output_dir'], "{cohort}/{cohort}_nhidden{n_hidden}_latentdim{latent_dim}_betascheme{beta}_run_log.txt"),
    cohort=cohorts,
    n_hidden=n_hiddens,
    latent_dim=latent_dims,
    beta=betas,
    )
}

# model_outputs = {
#     "h5ads": expand(os.path.join(config['vae_run_output_dir'], "{cohort}/n_hidden_{n_hidden}/{cohort}_adata_new.h5ad"), cohort=cohorts, n_hidden=n_hiddens),
#     "ablation_tsvs": expand(os.path.join(config['vae_run_output_dir'], "{cohort}/n_hidden_{n_hidden}/{cohort}_clusters.tsv"), cohort=cohorts, n_hidden=n_hiddens),
# }

# survival_outputs = {
#     "cluster_props": expand(os.path.join(config['vae_run_output_dir'], "{cohort}/n_hidden_{n_hidden}/{cohort}_patient_cluster_props.tsv"), cohort=cohorts, n_hidden=n_hiddens),
#     "hi_low": expand(os.path.join(config['vae_run_output_dir'], "{cohort}/n_hidden_{n_hidden}/{cohort}_patient_cluster_hi_or_low.tsv"), cohort=cohorts, n_hidden=n_hiddens),
#     "patient_clusters": expand(os.path.join(config['vae_run_output_dir'], "{cohort}/n_hidden_{n_hidden}/{cohort}_patient_cluster_survival.tsv"), cohort=cohorts, n_hidden=n_hiddens),
# }


rule run_ablations:
    params:
        output_dir = config['vae_run_output_dir']+'/{cohort}',
        cofactor = lambda wildcards: cofactor[wildcards.cohort],
        batch_size = lambda wildcards: batch_size[wildcards.cohort],
        latent_dim = lambda wildcards: latent_dims[wildcards.n_hidden],
        cohort= config['cohort'],
        #include_all_views = 0

    input:
        os.path.join(config['tidy_output_dir'],'{cohort}/{cohort}_new_dna_corrs_weight_prep/all_samples_merged_vae.h5ad'),

    output:
        # os.path.join(config['vae_run_output_dir'], "{cohort}/n_hidden_{n_hidden}/{cohort}_adata_new.h5ad"),
        # os.path.join(config['vae_run_output_dir'], "{cohort}/n_hidden_{n_hidden}/{cohort}_clusters.tsv"),
        os.path.join(config['vae_run_output_dir'], "{cohort}/{cohort}_nhidden{n_hidden}_latentdim{latent_dim}_betascheme{beta}_run_log.txt"),


    shell:
        "python ../../hmivae_runs/run_hmivae.py --adata {input} "
        "--cofactor {params.cofactor} --cohort {wildcards.cohort} "
        "--batch_size {params.batch_size} "
        "--beta_scheme {wildcards.beta} "
        "--hidden_dim_size {wildcards.n_hidden} "
        "--latent_dim {wildcards.latent_dim} "
        #"--include_all_views {params.include_all_views} --remove_view {wildcards.view} "
        "--output_dir {params.output_dir} "


# rule create_albation_hi_low_file:
    # params:
    #     output_dir = config['vae_run_output_dir']+'/{cohort}/n_hidden_{n_hidden}',
    #     cofactor = lambda wildcards: cofactor[wildcards.cohort],
    #     #cohort= config['cohort'],
    #     cluster_col = 'leiden'

    # input:
    #     clusters = os.path.join(config['vae_run_output_dir'], "{cohort}/n_hidden_{n_hidden}/{cohort}_clusters.tsv"),
    #     patient_file = os.path.join('../analysis/survival_analysis','{cohort}/{cohort}_survival_patient_samples.tsv'),

    # output:
    #     os.path.join(config['vae_run_output_dir'], "{cohort}/n_hidden_{n_hidden}/{cohort}_patient_cluster_props.tsv"),
    #     os.path.join(config['vae_run_output_dir'], "{cohort}/n_hidden_{n_hidden}/{cohort}_patient_cluster_hi_or_low.tsv"),
    #     os.path.join(config['vae_run_output_dir'], "{cohort}/n_hidden_{n_hidden}/{cohort}_patient_cluster_survival.tsv")

    # shell:
    #     "python create_patient_hi_low_file.py --cohort {wildcards.cohort} "
    #     #"--remove_view {wildcards.view} "
    #     "--cluster_tsv {input.clusters} --cluster_col {params.cluster_col} "
    #     "--patient_file {input.patient_file} "
    #     "--output_dir {params.output_dir} "
