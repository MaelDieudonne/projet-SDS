import pandas as pd
import numpy as np

from src.model_fit import do_StepMix, do_kmeans, do_AHC, do_hdbscan



def bootstrap_gap(data, controls, bvr_data, n, model, params, iter_num):
    # Create random dataset
    rand_data = np.random.uniform(low=data.min(axis=0),
                                  high=data.max(axis=0) + 1,
                                  size=data.shape)
    rand_data = pd.DataFrame(rand_data, columns=data.columns)
    
    # Fit model
    if model == 'latent':
        res = do_StepMix(rand_data,
                         controls if params.get('covar') == 'with' else None,
                         bvr_data if params.get('msrt') == 'categorical' else None,
                         n,
                         **params)
    elif model == 'kmeans':
        res = do_kmeans(rand_data, n, **params)
    elif model == 'AHC':
        res = do_AHC(rand_data, n, **params)
    
    # Add iteration number
    res = pd.DataFrame([res])
    res['bootstrap_iter'] = iter_num + 1
    
    return res


def compute_gap(bootstrap_results, model_results, model, params, indices):
    gap_values = pd.DataFrame()

    grouped = bootstrap_results.groupby('n_clust')
    
    for n_clust, group in grouped:
        # Get corresponding model score
        mod_scores = model_results.loc[
            (model_results['model'] == model) & 
            (model_results['params'] == params) &
            (model_results['n_clust'] == n_clust)
        ]
        
        row_data = {
            'model': model,
            'params': params,
            'n_clust': n_clust
        }
        
        # Calculate gap statistic for each index
        for index in indices:
            rand_ind = group[index]
            mod_ind = mod_scores[index]
            
            # Rescale Silhouette index if needed
            if index == 'silhouette':
                rand_ind = (rand_ind + 1) / 2
                mod_ind = (mod_ind + 1) / 2
            
            # Calculate gap statistic and s value
            gap = np.log(np.mean(rand_ind)) - np.log(mod_ind)
            s = np.std(np.log(rand_ind)) * np.sqrt(1 + (1 / len(group)))
            
            # Add to row data
            row_data[f'{index}_gs'] = gap.values[0]
            row_data[f'{index}_s'] = s
        
        # Append to results
        gap_values = pd.concat([gap_values, pd.DataFrame([row_data])], ignore_index=True)
    
    return gap_values


def get_gap(gap_values, model, params, index):
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
