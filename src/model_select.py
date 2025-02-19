import pandas as pd
import numpy as np

from src.model_fit import do_StepMix, do_kmeans, do_AHC, do_hdbscan



# Generate reference data from a uniform distribution
def gen_ref_data(data):
    return np.random.uniform(low=data.min(axis=0), 
                             high=data.max(axis=0), 
                             size=data.shape)


# Create empty df to store results
def create_empty_df(indices):
    cols = ['model', 'params', 'n_clust'] + \
       [f'{index}_gs' for index in indices] + \
       [f'{index}_s' for index in indices]
    
    df = pd.DataFrame(columns=cols)

    float_cols = [col for col in cols if col not in ['model', 'params', 'n_clust']]
    df[float_cols] = df[float_cols].astype('float64')
    
    df['model'] = df['model'].astype('object')
    df['params'] = df['params'].astype('object')
    df['n_clust'] = df['n_clust'].astype('int64')

    return df


# Compute the Gap Statistic
def compute_gap_statistic(data, controls, results, max_clust, indices, iters, model, params):
    gap_values = create_empty_df(indices)

    # Loop over n values
    if model == 'latent': n_min = 1
    else: n_min = 2
    
    for n in range(n_min, max_clust+1):
    
        # Fit the model on random datasets
        rand_scores_all = pd.DataFrame()
        
        for _ in range(iters):
            rand_data = gen_ref_data(data)
            
            if model == 'latent':
                rand_scores = do_StepMix(rand_data, controls, n, **params)

            elif model == 'kmeans':
                rand_scores = do_kmeans(rand_data, n, **params)

            elif model == 'AHC':
                rand_scores = do_AHC(rand_data, n, **params)
            
            rand_scores = pd.DataFrame([rand_scores])
            rand_scores_all = pd.concat([rand_scores_all, rand_scores], ignore_index=True)

        # Retrive scores for the assessed model
        mod_scores = results.loc[(results['model'] == model) & 
                                 (results['params'].apply(eval) == params) & 
                                 (results['n_clust'] == n)]

        # Calculate the Gap statistic and s value for each validity index
        for index in indices:
            rand_ind = rand_scores_all[index]
            mod_ind = mod_scores[index]

            # Rescale the Silhouette index on [0,1] to avoid errors when it is negative
            if index == 'silhouette':
                rand_ind = (rand_ind + 1) / 2
                mod_ind = (mod_ind + 1) / 2
                
            gap = np.log(np.mean(rand_ind)) - np.log(mod_ind)
            s = np.std(np.log(rand_ind)) * np.sqrt(1 + (1 / iters))

            # Store the results
            ## Check if the corresponding row exists in the df
            row_id = ((gap_values['model'] == model) & 
                      (gap_values['params'] == params) & 
                      (gap_values['n_clust'] == n))

            if gap_values[row_id].empty:
            ## If not, create a new one
                new_row = {
                    'model': model,
                    'params': params,
                    'n_clust': n,
                    f'{index}_gs': gap.values[0],
                    f'{index}_s': s
                }
                new_row = pd.DataFrame([new_row])
                gap_values = pd.concat([gap_values, new_row], ignore_index=True)
            
            else:
            # Otherwise, update the existing row
                gap_values.loc[row_id, f'{index}_gs'] = gap.values[0]
                gap_values.loc[row_id, f'{index}_s'] = s

    return gap_values


# Select the optimal number of clusters
def get_best_gap(gap_values, model, params, index):
    # Subset gap_values to the right model and params
    rows_id = ((gap_values['model'] == model) & (gap_values['params'] == params))
    df = gap_values[rows_id].reset_index(drop=True)

    # Extract gap and s values
    gap = df[f'{index}_gs']
    s = df[f'{index}_s']

    # Select rows such that GS(k) >= GS(k+1) - s(k+1)
    # Skipping the last row and adjusting for index-based calculations
    n_min = df['n_clust'].min()
    stats = []
    
    for i in range(0, len(df) - 1):
        stat = gap[i] - gap[i+1] + s[i+1]
        if stat >= 0: 
            stats.append([i+n_min, stat])

    # Return optimal cluster number
    stats = np.array(stats)
    if stats.size == 0:
        best_n = 'none'
    else:
        best_n = int(stats[np.argmin(stats[:, 1]), 0])

    return best_n
