import pandas as pd
import numpy as np
import tifffile
import scanpy as sc
import anndata as ad
import argparse
import os
from skimage.measure import regionprops, moments  # for cell morphology features
from einops import rearrange  # for correct rearrangement of arrays
from scipy.spatial import Delaunay  # Delaunay triangulation, for spatial neighbours
from scipy.spatial.distance import pdist, squareform  # for spatial neighbours

#MIN_CELL_PIXELS = 16 -- not doing this anymore


def interaction_id(x1, x2):
    """This takes a pair of proteins (e.g. CD44, EpCam)
    and assigns them a unique ID
    e.g. "CD44_EpCam" that is invariant to the
    protein set, meaning we can filter out duplicates

    """
    x = np.sort([x1, x2])
    return "_".join(x)


def get_correlation_features(x, c_id):
    """Extract a dataframe of correlation features
    from the upper triangle of the correlation matrix
    for a dataframe corresponding to a cell only"""

    ## Subset to specific cell id
    x = x.query("cell_id==@c_id")

    ## Reset the index to avoid multiple indexing
    x = x.reset_index()

    ## Move to a 'long' format
    x = pd.melt(x, id_vars=["level_1", "cell_id"])

    ## Assign the unique ID for the protein pair
    x["pair"] = list(map(interaction_id, x.level_1, x.variable))
    x["is_diagonal"] = x.level_1 == x.variable

    ## Drop the duplicates and correlations of 1 (i.e. select the upper triangle)
    x = (
        x[["cell_id", "pair", "value", "is_diagonal"]]
        .drop_duplicates()
        .query("is_diagonal==False")
    )  # there were values = 1 that were not the diagonal (case: all pixels for two features are 0 except 1 => 1.0 correlation between those two features)
    return x


def convert_expression_to_weight(x, x_max, x_min):
    """
    Converts the per cell mean protein expression to
    a 'weight' which is a value between 0 and 1. 
    This value will be used for weighting protein-protein correlations.
    x: np.array. Nx1 array for per cell expression for selected channel where N is the number of cells.
    x_max: float. The max value across cells for the selected channel.
    x_min: float. The min value across cells for the selected channel.
    """

    max_diff = x_max - x_min
    all_diffs = x - x_min

    return all_diffs / max_diff


parser = argparse.ArgumentParser(
    description="Create protein-by-protein correlation and expression for VAE"
)

# parser.add_argument('--cluster_tsv', type=str, required=True, help='tsv file with cluster info')
parser.add_argument(
    "--feature_data", type=str, required=True, help="file with protein/channel info"
)
parser.add_argument(
    "--non_proteins",
    type=str,
    help="txt file with a list of stains that are to be removed in the final output",
)
parser.add_argument("--cofactor", type=float, help="cofactor for HMI signal dilution")
parser.add_argument("--sample", type=str, required=True, help="Sample name")
parser.add_argument(
    "--tiff", type=str, required=True, help="IMC raw expression tiff file"
)
parser.add_argument(
    "--mask", type=str, required=True, help="IMC Cell segmentation tiff"
)
parser.add_argument("--output_dir", type=str, help="Name of dir for output files.")

args = parser.parse_args()

if args.output_dir is not None:
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        output_dir = "."


mdata = pd.read_csv(args.feature_data, sep="\t")

if "channel" not in mdata.columns:
    mdata.columns = mdata.columns.str.lower()
    #print(mdata.columns)
    if "target" in mdata.columns:
        mdata = mdata.rename(columns={"target": "channel"})
    if "protein" in mdata.columns:
        mdata = mdata.rename(columns={"protein": "channel"})

sample = args.sample
mask_file = args.mask
exp_file = args.tiff

mask = tifffile.imread(mask_file)
exp = tifffile.TiffFile(exp_file)

raw_data = exp.asarray()
C, H, W = raw_data.shape

mask = mask.reshape([H, W])  # this was to get around deepcell's extra dim

N_channels = mdata.shape[0]

sample_id = sample

cell_ids = list(np.unique(mask.reshape(-1)))
cell_ids.remove(0)

N_cells = len(cell_ids)


if args.non_proteins is not None:

    non_proteins = pd.read_csv(args.non_proteins, sep="\t", index_col=0)
    non_prot_lst = non_proteins.index.tolist()
    prot_lst = mdata.loc[~mdata["channel"].isin(non_prot_lst)].channel.tolist()

    background_channel_inds = (
        mdata.loc[mdata["channel"].isin(non_prot_lst)].reset_index().index.tolist()
    )
    background_arr = raw_data[background_channel_inds, ...]

    sample_background_mean = np.arcsinh(np.mean(background_arr) / args.cofactor)
    sample_mean_intensity = np.arcsinh(np.mean(raw_data) / args.cofactor)
    antibody_background_correlation = pd.DataFrame(
        np.corrcoef(np.arcsinh(raw_data.reshape(raw_data.shape[0], -1)/args.cofactor)),
        columns=mdata.channel.tolist(),
        index=mdata.channel.to_list(),
    )
    antibody_background_correlation = antibody_background_correlation.loc[
        non_prot_lst, prot_lst
    ]
    ab_bkg_df = antibody_background_correlation.stack().reset_index()
    ab_bkg_df["pair"] = list(map(interaction_id, ab_bkg_df.level_0, ab_bkg_df.level_1))
    ab_bkg_df = ab_bkg_df[["pair", 0]].drop_duplicates()
    ab_bkg_df = ab_bkg_df.pivot_table(values=0, columns='pair')
else:
    sample_mean_intensity = np.arcsinh(np.mean(raw_data) / args.cofactor)
    sample_ab_bkg_mean = pd.DataFrame(
        np.arcsinh(np.mean(raw_data[..., mask == 0], 1)/args.cofactor), index=mdata.channel.to_list()
    ).transpose()





# We want to get our data into a 'tidy' format data frame with one measurement per row, holding:
# 1. The cell ID of that pixel
# 2. The expression value
# 3. The channel
# 4. A unique pixel ID we'll create
# 5. The x,y co-ordinates:

df_cell_id = []
df_expression = []
df_channel = []
df_pixel_ids = []
xs = []
ys = []

assert all(cell_id >= 0 for cell_id in cell_ids) # "cell_ids should not contain negative values"

for cell_id in cell_ids:
    whereis_cell = np.where(mask == cell_id)
    n_matches = len(whereis_cell[0])
    #print(cell_id)
    if n_matches > 0:  # removing cells with less than 16 pixels -- not doing this anymore because causes issues for regionprops later
        for match in range(n_matches):
            x = whereis_cell[0][match]
            y = whereis_cell[1][match]
            pixel_id = str(cell_id) + "_" + str(match)
            for channel_idx in range(mdata.shape[0]):
                df_cell_id.append(cell_id)
                df_expression.append(raw_data[channel_idx, x, y])
                df_channel.append(mdata.channel[channel_idx])
                df_pixel_ids.append(pixel_id)
                xs.append(x)
                ys.append(y)

# print(len(df_cell_id))
# print(len(df_pixel_ids))
# print(len(df_channel))
# print(len(df_expression))
# print(len(xs))
# print(len(ys))

#print(np.unique(df_cell_id))

df = pd.DataFrame(
    {
        "cell_id": list(map(int,df_cell_id)),
        "pixel_id": df_pixel_ids,
        "channel": df_channel,
        "expression": df_expression,
        "x": xs,
        "y": ys,
    }
)

# for i in df.cell_id.tolist():
#     if i < 0:
#         print(i)
#print(df.cell_id.unique())

## Remove duplicates / channels we don't care about
# df = df.query("channel != 'RutheniumTetroxide'").query("channel != 'None'")
if args.non_proteins is not None:
    df_background = df.loc[df["channel"].isin(non_prot_lst)]
    mean_background = df_background.groupby("cell_id").expression.mean()
    #print(mean_background)
    df = df.loc[~df["channel"].isin(non_prot_lst)]


df.channel = df.channel.str.replace(r" |/|-|\(|\)+.", "", regex=True)

#print(df)


# We can use this to get a handy data frame mapping pixel_ids to cell_ids:
df_pixel_cell = df[["pixel_id", "cell_id"]].drop_duplicates()


# And enumerate the pixels per cell easily!
df_cell_size = df_pixel_cell.groupby("cell_id").count()
df_cell_size.columns = ["cell_size_pixels"]
df_cell_size["x"] = df.groupby("cell_id")["x"].mean()
df_cell_size["y"] = df.groupby("cell_id")["y"].mean()


# We can easily create e.g. a pixel-position data frame:
df_location = df.loc[:, ["pixel_id", "x", "y"]].set_index(
    "pixel_id"
)  # a data frame to store locations of pixel IDs
df_location = df_location.drop_duplicates()


# We can use `df.pivot` to pivot back to a 'wide' format which matches the `Y` input we're expecting - note we still keep the `pixel_id` column:

df2 = df.reset_index(drop=True).pivot_table(
    index="pixel_id", columns="channel", values="expression"
)  # the earlier one was giving a duplicate entries in index error

#print(df2)

# If instead we want to get the mean expression per cell (for our VAE), this is also relatively straightforward:
df_mean_expression = (
    df[["cell_id", "channel", "expression"]]
    .groupby(["cell_id", "channel"])
    .mean()
    .reset_index()
)
y = df_mean_expression.pivot(index="cell_id", columns="channel", values="expression")
#print(cell_ids)
#print(df2)
#print(y.index)

# ## Calculate the weights for each channel

weights_list = []

for col in y.columns.tolist():
    
    y_col_max = np.arcsinh(y[col].max().item() / args.cofactor)
    y_col_min = np.arcsinh(y[col].min().item() / args.cofactor)
    t = y[col].apply(lambda x: convert_expression_to_weight(np.arcsinh(x / args.cofactor), y_col_max, y_col_min))

    assert int(t.max()) == 1
    assert int(t.min()) == 0
    col_df = pd.DataFrame(t)
    
    weights_list.append(t)

df_weights = pd.concat(weights_list, axis=1)
df_weights = df_weights[y.columns.tolist()]

# ## Correlation modelling

# First create a dataframe with the mean DNA stain for each pixel. This will be the mean nuclear stain.
nuc_df = df_pixel_cell.set_index('pixel_id').join(pd.DataFrame(df.loc[df['channel'].str.contains('DNA'),:].groupby(['pixel_id'])['expression'].mean()))
nuc_df = nuc_df.rename(columns = {'expression': 'mean_nuc_stain'})
nuc_df['mean_nuc_stain'] = nuc_df['mean_nuc_stain'].apply(lambda x: x / args.cofactor).apply(np.arcsinh) #make sure to apply arcsinh transform to this too!


# We can group by the cell ID and compute protein-protein correlations:
df2 = df2.apply(lambda x: x / args.cofactor).apply(
    np.arcsinh
)  # normalize before correlations

# df_correlations = (
#     df2.join(df_pixel_cell.set_index("pixel_id")).groupby("cell_id").corr()
# )

df_correlations = (
    df2.join(nuc_df).groupby("cell_id").corr()
) # going to add nuc_df cols at the end, so the relevant cols will be the last col/row


# However, this isn't exactly the format we want for modelling - it double counts the correlations and includes the diagonal. For this we need some custom code:


#cell_ids = list(df_cell_size.index)  # new filtered list -- no longer dropping any cells so no need to create this

dfs = [get_correlation_features(df_correlations, i) for i in cell_ids]


df_cor_features = pd.concat(dfs)

s = df_cor_features.pivot(index="cell_id", columns="pair", values="value")

#print(s)

for col in s.columns.to_list():
    s[col] = np.where(
        s[col] >= 1.0, 0, s[col]
    )  # there are values like 1.000000002, setting them to 0 because know we've removed the diagonal
    s[col] = np.where(
        s[col] <= -1.0, -0.999, s[col] # replacing any values less than -1 
    )

# Finally we need to deal with `NaN` values that come from constant (0) expression of features in cells. We can do this by replacing the values by the median expression of each feature:


s = s.apply(lambda x: x.fillna(x.median()), axis=0) # fill nans with median for cell

protein_nuc_corr_cols = [i for i in s.columns if 'mean_nuc_stain' in i] # select the appropriate columns

s = s.loc[:, protein_nuc_corr_cols] # contains only the correlations of each protein with the mean nuclear stain

assert np.isnan(s.to_numpy()).sum() == 0 # small cells cause correlation to be nan, want to make sure we don't have any nans during training

# print('s sorted:', np.sort(s.columns.tolist()))

# print('s df cols:', s.columns.tolist())


s_cols = []

for i in s.columns.tolist(): # to get the same col names as y
    if 'mean_nuc_stain_' in i:
        s_cols.append(i.replace('mean_nuc_stain_', ''))
    else:
        s_cols.append(i.replace('_mean_nuc_stain', ''))

s.columns = s_cols

#print(s.index)

s = s[y.columns.tolist()] # put in the same order as y 

new_cols = [i+'_mean_nuc_stain' for i in s.columns.tolist()]

s.columns = new_cols

s_cols_y = []
for i in s.columns.tolist(): # for checking the same order with y
    if 'mean_nuc_stain_' in i:
        s_cols_y.append(i.replace('mean_nuc_stain_', ''))
    else:
        s_cols_y.append(i.replace('_mean_nuc_stain', ''))


## Finding cell morphology features


def centroid_dif(prop, **kwargs):
    # from: https://github.com/angelolab/ark-analysis/blob/master/ark/segmentation/regionprops_extraction.py
    """Return the normalized euclidian distance between the centroid of the cell
    and the centroid of the corresponding convex hull
    Args:
        prop (skimage.measure.regionprops):
            The property information for a cell returned by regionprops
        **kwargs:
            Arbitrary keyword arguments
    Returns:
        float:
            The centroid shift for the cell
    """

    cell_image = prop.image
    cell_M = moments(cell_image)
    cell_centroid = np.array([cell_M[1, 0] / cell_M[0, 0], cell_M[0, 1] / cell_M[0, 0]])

    convex_image = prop.convex_image
    convex_M = moments(convex_image)
    convex_centroid = np.array(
        [convex_M[1, 0] / convex_M[0, 0], convex_M[0, 1] / convex_M[0, 0]]
    )

    centroid_dist = np.linalg.norm(cell_centroid - convex_centroid) / np.sqrt(prop.area)

    return centroid_dist


def convex_hull_resid(prop, **kwargs):
    # from: https://github.com/angelolab/ark-analysis/blob/master/ark/segmentation/regionprops_extraction.py
    """Return the ratio of the difference between convex area and area to convex area
    Args:
        prop (skimage.measure.regionprops):
            The property information for a cell returned by regionprops
        **kwargs:
            Arbitrary keyword arguments
    Returns:
        float:
            (convex area - area) / convex area
    """

    return (prop.convex_area - prop.area) / prop.convex_area


rearranged_tiff = rearrange(raw_data, "c w l -> w l c")

props = regionprops(mask, rearranged_tiff)

area = []
perimeter = []
centroid_x = []
centroid_y = []
fill = []  # in place of convex area
asymmetry = []  # in place of equivalent_diameter
eccentricity = []  # in place of major_axis_length

for i, cell in enumerate(cell_ids):
    area.append(props[i].area)  # number of pixels in the region
    perimeter.append(
        props[i].perimeter
    )  # Perimeter of object which approximates the contour as a line through the centers of border pixels using a 4-connectivity.
    fill.append(
        convex_hull_resid(props[i])
    )  # Proportion of region filled by the convex hull (-from deepcell paper) ~0 means, no fill (cell == convex_hull), ~1, a lot of fill (cell has indentations)
    asymmetry.append(
        centroid_dif(props[i])
    )  # The difference between the centroid of the convex hull and the cell. (-from deepcell paper)
    eccentricity.append(
        props[i].eccentricity
    )  # The eccentricity of the ellipse that has the same normalized second central moments as the region.


cell_morph = pd.DataFrame(
    {
        "cell_id": list(map(int,cell_ids)),
        "area": area,
        "perimeter": perimeter,
        "concavity": fill,
        "asymmetry": asymmetry,
        "eccentricity": eccentricity,
    }
).set_index("cell_id")

#print(cell_morph)

s = s.loc[y.index, :]

cell_morph = cell_morph.loc[y.index, :]
df_cell_size = df_cell_size.loc[y.index, :]

print('s_cols:', s.columns.tolist())
print('y_cols:', y.columns.tolist())

assert s_cols_y == y.columns.tolist() # make sure the channels are in the same order in both s and y

assert y.index.to_list() == s.index.to_list()  # cell_ids match between y and s
assert y.index.to_list() == cell_morph.index.to_list()
assert (
    y.index.to_list() == df_cell_size.index.to_list()
)  # checking if the cell_ids between y and neighbour df match


cells_missing_in_corrs = []

for i in y.index.to_list():
    if i not in s.index.to_list():
        cells_missing_in_corrs.append(i)


data_cell_ids = s.index.to_list()

adata = ad.AnnData(y)

adata.obsm["correlations"] = s.values
adata.obsm["morphology"] = cell_morph.values

adata.obsm["weights"] = df_weights.values

adata.uns["names_correlations"] = s.columns.tolist()
adata.uns["names_morphology"] = cell_morph.columns.tolist()

adata.uns["names_weights"] = df_weights.columns.tolist()

if args.non_proteins is not None:
    assert y.index.to_list() == mean_background.index.to_list()
    #mean_background = mean_background.loc(axis=0)[y.index, :]
    adata.obsm["background_mean"] = pd.DataFrame(
        {sample_id: [sample_background_mean] * y.shape[0]}
    ).values
    adata.obsm["cell_background_stain"] = np.arcsinh(mean_background.values / args.cofactor).reshape((-1,1))
    adata.obsm["Ab_background_corrs"] = pd.concat(
        [ab_bkg_df] * y.shape[0], ignore_index=True
    ).values
    adata.uns["Ab_background_corrs_names"] = ab_bkg_df.columns.tolist()
    adata.obsm["sample_mean_intensity"] = pd.DataFrame(
        {sample_id: [sample_mean_intensity] * y.shape[0]}
    ).values

else:
    adata.obsm["sample_mean_intensity"] = pd.DataFrame(
        {sample_id: [sample_mean_intensity] * y.shape[0]}
    ).values
    adata.obsm["sample_Ab_background_mean"] = pd.concat(
        [sample_ab_bkg_mean] * y.shape[0], ignore_index=True
    ).values
    adata.uns["sample_Ab_background_mean_names"] = sample_ab_bkg_mean.columns.tolist()

adata.obs["Sample_name"] = args.sample
adata.obs["x"] = df_cell_size.x.tolist()
adata.obs["y"] = df_cell_size.y.tolist()


adata.write(filename=args.output_dir + "/" + sample + "_vae_input.h5ad")
