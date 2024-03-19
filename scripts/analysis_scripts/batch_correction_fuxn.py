import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from collections import Counter
from sklearn.metrics import silhouette_score
from collections import Counter
from rich.progress import track, Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

def negative_entropy(p_dict, q_dict):
    """
    Takes a `p_dict` which is the frequency of a 
    sample ID in the nearest neighbour of a cell;
    and a `q_dict` which has the global frequencies
    of all the sample IDs in the dataset. Calculates
    the negative entropy as defined in the totalVI paper.
    """
    ne = 0
    for key in p_dict.keys():
        p = p_dict[key]
        q = q_dict[key]
        
        if p == 0:
            ne += 0 #consistent with totalVI
            
        else:
            neg_ent = p * np.log(p/q)
        
            ne += neg_ent
    
    return ne

def latent_mixing_metric(
    adata,
    image_id_freqs_dict:dict, #global frequency of images in the dataset i.e. num of cells in image / total num of cells in dataset
    distances:list = None,
    obsp_key:str = 'vae_distances',
    s_iter:int = 50,
    n_neighbours:int = 100,
    n_random_cells:int=100,
    
):
    """Going to change the name, but keeping it the same as
    totalVI for now. To see how well-distributed the batches are 
    in the latent space."""
    
    
    progress = Progress(
        TextColumn("[progress.description]Running latent mixing metric"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ) 
    
    N_cells = adata.obs.shape[0]
    sample_ids = adata.obs.Sample_name.unique().tolist()
    
    vae_nonzero_distances = list(zip(adata.obsp[obsp_key].nonzero()[0],
                                     adata.obsp[obsp_key].nonzero()[1]))
    
    df = pd.DataFrame(vae_nonzero_distances)
    
    neg_entropy_arr = np.zeros(s_iter)
    
    with progress:
    
        for i in progress.track(range(s_iter)):
            random_nn_cells_idx = np.random.choice(N_cells, n_random_cells)
            neg_entropy = np.zeros(n_random_cells)
            cell_counter = 0
            for c in random_nn_cells_idx:
                
                c_nn_freq_dict = {key:0 for key in sample_ids}
                c_nns = df.loc[df[0]==c,[1]].to_numpy().reshape(-1) #takes ~0.0013secs
                
                c_nn_sample_counts = Counter(adata.obs.Sample_name[c_nns].to_list()) #number of cells belonging to an image in the cell's neighbourhood

                c_nn_freq_dict.update({k:v/n_neighbours for k, v in c_nn_sample_counts.items()}) #frequency of an image in a cell's neighbourhood

                neg_entropy[cell_counter] = negative_entropy(c_nn_freq_dict, image_id_freqs_dict)

                cell_counter += 1

            neg_entropy_arr[i] = neg_entropy.mean() #change to mean over cells
        
        
    return neg_entropy_arr.mean()

def clustering_mixing_metric(adata, 
                            sample_name:str,
                            n_neighbours:int = 10,
                            cluster_df = None,
                            cluster_col = None):
    """
    Clustering mixing metric from totalVI. Measures how 
    well-preserved clusters are in the integrated vs non-integrated space.
    Here "integrated vs non-integrated" refers to clusters in an image found 
    based on the latent space of the whole dataset vs clusters found based on 
    the latent space of one image. 
    """
    if cluster_df is None:
        sample_adata = adata.copy()[adata.obs.Sample_name.isin([sample_name]), :]
    else:
        sample_adata = adata.copy()[cluster_df['index'], :]
        
        if cluster_col is None: #cluster col with ground truth labels
            raise Exception("cluster_col cannot be None when cluster_df is given!")
    
        assert sample_adata.X.shape[0] == cluster_df.shape[0]
    
    sc.pp.neighbors(sample_adata, n_neighbors=n_neighbours, key_added='sample', use_rep='VAE') #compute sample-specific neighbours
    
    if cluster_df is None:
    
        sc.tl.leiden(sample_adata, neighbors_key='sample', key_added='sample_leiden') #compute sample-specific clusters
        labels = sample_adata.obs['sample_leiden'].to_numpy()
        
    else:
        labels = cluster_df[cluster_col].to_numpy() #if ground truth present, select correct column
    
    sample_sil = silhouette_score(sample_adata.obsp['sample_distances'],  #compute sample-specific cluster silhouette score
                                  labels=labels)
    
    integrated_sil = silhouette_score(sample_adata.obsp['vae_distances'],
                                     labels=sample_adata.obs['leiden'].to_numpy()) #compute whole-data cluster silhouette score
    
    
    return integrated_sil - sample_sil #difference -- if similar, means that clusters are well-preserved in the integrated space


def run_clustering_mm(adata, 
                      cluster_df= None, 
                      cluster_col:str = None):
    """
    Runs clustering mixing metric over whole dataset
    """
    
    progress = Progress(
        TextColumn("[progress.description]Running clustering mixing metric"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )
    
    if cluster_df is not None:
        if cluster_col is None:
            raise Exception("cluster_col cannot be None when cluster_df is given!")
    
    samples = adata.obs.Sample_name.unique().tolist()
    scores = np.zeros(len(samples)) # calculate scores for every sample
    counter = 0
    with progress:
    
        for s in progress.track(samples):
            
            if cluster_df is None:
                scores[counter] = clustering_mixing_metric(adata, s) #when we don't have ground truth labels
            else:
                # this is for when we have ground truth labels
                sample_cells = pd.merge(adata.obs.query("Sample_name==@s").reset_index(), 
                                         cluster_df.query("Sample_name==@s").reset_index(drop=True), 
                                         on=['Sample_name', 'cell_id'])
                
                scores[counter] = clustering_mixing_metric(adata, s, cluster_df=sample_cells, 
                                                           cluster_col=cluster_col)
            
            counter += 1
            
    return scores.mean() # get average score across whole dataset