### making patient hi/low files

import numpy as np
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser(
    description="Create patient hi/low files for survival analysis"
)

parser.add_argument("--cohort", type=str, help="Name of cohort", default="IMC_cohort")

parser.add_argument(
    "--remove_view",
    type=str,
    help="Name of view to leave out. One of ['expression', 'correlation', 'morphology', 'spatial'].",
    default=None,
    choices=["expression", "correlation", "morphology", "spatial"],
)

parser.add_argument(
    "--cluster_tsv",
    type=str,
    required=True,
    help="tsv with cell_ids and cluster assignment",
)

parser.add_argument(
    "--cluster_col",
    type=str,
    help="name of column that has the cluster assignment",
    default="leiden",
)

parser.add_argument(
    "--patient_file",
    type=str,
    help="File containing patient information. Must have 'Sample_name' column.",
    default=None
)

parser.add_argument(
    "--output_dir", type=str, help="Output directory to save outputs", default="."
)

args = parser.parse_args()


df = pd.read_csv(args.cluster_tsv, sep="\t", index_col=0)

cohort = args.cohort

cluster_col = args.cluster_col

output_dir = args.output_dir

hi_or_low = df[["Sample_name", cluster_col]]

## Proportion of cells belonging to each cluster for each image / patient

hi_or_low = hi_or_low.groupby(["Sample_name", cluster_col]).size().unstack(fill_value=0)


hi_or_low = hi_or_low.div(hi_or_low.sum(axis=1), axis=0).fillna(0)

if args.remove_view is None:
    hi_or_low.to_csv(
        os.path.join(output_dir, f"{cohort}_patient_cluster_props.tsv"), sep="\t"
    )
else:
    hi_or_low.to_csv(
        os.path.join(output_dir, f"{cohort}_patient_cluster_props_remove_{args.remove_view}.tsv"), sep="\t"
    )

## Classify patients as hi or low for each cluster based on proportion of cells belonging to each clusters

medians = {}

for leiden in hi_or_low.columns.to_list():
    # print(leiden)
    med = hi_or_low[[leiden]].median().item()
    # print(med)
    medians[leiden] = med

hi_low_cols = {}


for col in hi_or_low.columns.to_list():
    hi_low_cols[col] = []
    for sample in hi_or_low.index.to_list():
        if hi_or_low.loc[sample, col].item() > medians[col]:
            hi_low_cols[col].append(str(col) + "_hi")
        else:
            hi_low_cols[col].append(str(col) + "_low")

hi_low_df = pd.DataFrame.from_dict(hi_low_cols)

hi_low_df["Sample_name"] = hi_or_low.index.to_list()

newcols = [str(i) + "_level" for i in hi_low_df.columns[:-1]]

hi_low_df.columns = newcols + [hi_low_df.columns.to_list()[-1]]

#cluster_hi_low = pd.merge(df[['Sample_name']], hi_low_df, on="Sample_name")

if args.patient_file is not None:
    patient_file = pd.read_csv(args.patient_file, sep='\t')

    patient_cluster_file = pd.merge(hi_low_df, patient_file, on='Sample_name')


if args.remove_view is None:
    hi_low_df.to_csv(
        os.path.join(output_dir, f"{cohort}_patient_cluster_hi_or_low.tsv"), sep="\t"
    )

    patient_cluster_file.to_csv(
        os.path.join(output_dir, f"{cohort}_patient_cluster_survival.tsv"), sep="\t"
    )
else:
    hi_low_df.to_csv(
        os.path.join(output_dir, f"{cohort}_patient_cluster_hi_or_low_remove_{args.remove_view}.tsv"), sep="\t"
    )
    patient_cluster_file.to_csv(
        os.path.join(output_dir, f"{cohort}_patient_cluster_survival_remove_{args.remove_view}.tsv"), sep="\t"
    )

