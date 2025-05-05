# hmiVAE (highly multiplexed imaging Variational AutoEncoder)
For details regarding running the model on the full dataset and reproducing the figures in the manuscript, please see: [hmiVAE_manuscript](https://github.com/camlab-bioml/hmiVAE_manuscript/tree/main).  
The test data for running the hmiVAE model can be found [here](https://zenodo.org/records/15346211).  
Scripts for the running the model and generating embeddings and clusters can be found in `tests/pipeline`. For running hmiVAE, the inputs are first prepped and stored in an AnnData object file using the `vae_data_prep_sample.py` and `vae_data_prep_concat.py`. The `AnnData` is concatenated object over all the sample `AnnData` objects for each sample. It stores the values for each view at the level of single-cells. The next step is to find the best combination of hyperparameters using `hyperparameter_tuning.py`. Finally, embeddings are generated and clusters found using the script in `run_w_best_hparams/make_umaps.py`. The output is also in the form of an `AnnData` object file. It is similar to the input `AnnData` object but with an additional column containing cluster IDs as well as view-specific embedding and latent space values for each cell.
The pipeline can be run by editing the `full_config.yml`. This file has the following parameters:  
```
cohort: <name_of_cohort> # used for naming output files  
cofactor: <float> # cofactor for arcsinh normalization
samples_tsv: <path_to_sample_tsv_file> # tsv file containing paths to tiffs and their associated single-cell masks
tidy_output_dir: <path_to_store_output_of_prep_scripts>
hyperparams_tuning_dir: <path_to_store_model_checkpoints> # stores checkpoints for each hyperparameter combination tested
feature_data: <path_to_panel_file>  
non_proteins: <path_to_file_with_list_of_non-protein_markers> # can be commented out where not present, needed for removing background channels (e.g. ArAr) and computing some background features
```
The pipeline is intended to be run using `snakemake` by: `snakemake --configfile full_config.yml --cores 10`. More details on `snakemake` can be found from their documentation [here](https://snakemake.readthedocs.io/en/stable/).  
To create the embeddings and clusters using the optimal combination of hyperparameters can be run similarly using the `Snakefile` contained in `run_w_best_hparams/`. If you want to run parameters for just a single cohort, you can do so by:
```
python make_umaps.py --adata <input_adata> --cofactor <float_arcsinh_cofactor> --cohort <cohort_name> --batch_size <batch_size_int> --beta_scheme <beta_scheme_str> --hidden_dim_size <hidden_dim_int> --latent_dim <latent_dim_int> --random_seed <random_seed_int> --n_hidden <n_hidden_int> --checkpoint_dir <best_hparams_checkpoint_dir> --output_dir <output_dir>
```


