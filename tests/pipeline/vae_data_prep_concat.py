import scanpy as sc
import numpy as np
import os
import argparse
import pandas as pd
import anndata as ad

parser = argparse.ArgumentParser(description="merge all adata for VAE version 1")

parser.add_argument("--input_dir", type=str, required=True, help="Input directory with all the vae input files to be concatenated")

parser.add_argument("--output_h5ad", type=str, required=True, help="Name of h5ad output file to be created")

args = parser.parse_args()

input_list = os.listdir(args.input_dir)
input_list = [i for i in input_list if "h5ad" in i]

adatas = [sc.read_h5ad(os.path.join(args.input_dir, i)) for i in input_list]

Ns = len(adatas)

max_x = np.array([adata.obs.x.max() for adata in adatas])
safety_transform = 10. * max_x.max()

for i in range(Ns):
    adatas[i].obs['x_new'] = adatas[i].obs['x'] + i * safety_transform

adata = sc.concat(adatas, uns_merge = "first")

adata.obsm['spatial'] = np.column_stack([adata.obs['x_new'], adata.obs['y']])
sc.pp.neighbors(adata, use_rep='spatial', n_neighbors=10)



adata.write(filename=args.output_h5ad)
