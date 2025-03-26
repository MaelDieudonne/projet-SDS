import numpy as np
import pandas as pd

from collections import Counter
from itertools import combinations
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import pairwise_distances


##### For all models #####

# Avoid throwing errors when a CVI is undefined
def sil_score(data, pred_clust):
    try: 
        sil_score = silhouette_score(data, pred_clust, metric='manhattan')
    except ValueError:
        sil_score = np.nan
    return sil_score

def ch_score(data, pred_clust):
    try:
        ch_score = calinski_harabasz_score(data, pred_clust)
    except ValueError:
        ch_score = np.nan
    return ch_score

def db_score(data, pred_clust):
    try:
        db_score = davies_bouldin_score(data, pred_clust)
    except ValueError:
        db_score = np.nan
    return db_score


# Function to compute the Dunn score 43
def dunn_score(data, pred_clust, metric='cityblock'):
    data = np.asarray(data)
    pred_clust = np.asarray(pred_clust)
    clusters = np.unique(pred_clust)
    
    # Compute min centroid distance (for separation)
    min_centroid_dist = None
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            # Get data points for each cluster
            cluster_i = data[pred_clust == clusters[i]]
            cluster_j = data[pred_clust == clusters[j]]
            
            # Compute centroids
            centroid_i = np.mean(cluster_i, axis=0)
            centroid_j = np.mean(cluster_j, axis=0)
            
            # Reshape centroids for cdist (needs 2D arrays)
            centroid_i = centroid_i.reshape(1, -1)
            centroid_j = centroid_j.reshape(1, -1)
            
            # Compute distance between centroids using specified metric
            centroid_dist = cdist(centroid_i, centroid_j, metric=metric)[0, 0]
            
            if min_centroid_dist is None or centroid_dist < min_centroid_dist:
                min_centroid_dist = centroid_dist
    
    # Compute max cluster diameter (for cohesion)
    max_diameter = 0
    for cluster in clusters:
        clust = data[pred_clust == cluster]
        if len(clust) > 0:
            centroid = np.mean(clust, axis=0)
            centroid_2d = centroid.reshape(1, -1)
            # Use same distance metric for consistency
            diam = np.mean(cdist(clust, centroid_2d, metric=metric))
            max_diameter = max(max_diameter, diam)        
    
    # Compute Dunn score
    if max_diameter > 0 and min_centroid_dist is not None:
        dunn_score = min_centroid_dist / max_diameter
    else:
        dunn_score = np.nan
    
    return dunn_score


# Clusters min an max sizes
def clust_size(pred_clust):
    cluster_sizes = Counter(pred_clust)
    min_size = min(cluster_sizes.values())
    max_size = max(cluster_sizes.values())
    
    return min_size, max_size


# Return all model parameters and CVI at once
def get_metrics(model, params, n, data, pred_clust, **additional_metrics):
  
    # Remove noise
    noise = pred_clust == -1
    denoised_data = data[~noise]
    denoised_pred_clust = pred_clust[~noise]

    base_metrics = {
        'model': model,
        'params': params,
        'n_clust': n,
        'min_clust_size': clust_size(pred_clust)[0],
        'max_clust_size': clust_size(pred_clust)[1],
        'silhouette': float(sil_score(data, pred_clust)),
        'calinski_harabasz': float(ch_score(data, pred_clust)),
        'davies_bouldin': float(db_score(data, pred_clust)),
        'dunn': float(dunn_score(data, pred_clust))
    }

    base_metrics.update(additional_metrics)
    return base_metrics



##### For LCA #####

# BVRs
def bvr(data, post_probs, coeffs, var1, var2):
    # Get number of classes and observations
    n_classes = post_probs.shape[1]
    n_obs = len(data)
    
    # Calculate expected frequencies under the model
    expected = np.zeros((2, 2))
    
    # For each class...
    for k in range(n_classes):
        # Get class membership probability
        class_prob = post_probs[:, k].mean()
    
        # Get probabilities for var1 and var 2 in class k
        prob_var1 = coeffs[(coeffs['class_no'] == k) & (coeffs['variable'] == var1)]['value'].values.item()
        prob_var2 = coeffs[(coeffs['class_no'] == k) & (coeffs['variable'] == var2)]['value'].values.item()

        # Calculate expected frequencies for this class
        expected[0, 0] += n_obs * (1 - prob_var1) * (1 - prob_var2) * class_prob
        expected[0, 1] += n_obs * (1 - prob_var1) * prob_var2 * class_prob
        expected[1, 0] += n_obs * prob_var1 * (1 - prob_var2) * class_prob
        expected[1, 1] += n_obs * prob_var1 * prob_var2 * class_prob 
        
    expected = pd.DataFrame(expected)
    try: 
        bvr = chi2_contingency(expected)[0]
    except:
        bvr = 0

    return bvr

# BVRT
def bvrt(data, post_probs, coeffs):
    variables = data.columns
    bvrt = 0
    sig_bvr = 0

    for var1, var2 in combinations(variables, 2):
        if var1[:6] != var2[:6]:
            residual = bvr(data, post_probs, coeffs, var1, var2)
            bvrt += residual
            sig_bvr += 1 if residual >= 3.84 else 0

    V = data.shape[1]
    bvrt_df = (V*(V-1)/2) * (5-1)**2
    bvrt_pval = 1 - chi2.cdf(bvrt_stat, bvrt_df)
    
    return bvrt, bvrt_pval, sig_bvr
