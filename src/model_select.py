import pandas as pd
import numpy as np
#import warnings

from src.model_eval import local_stats, global_stats
from src.model_fit import build_latent_model, do_StepMix, do_kmeans, do_AHC, do_hdbscan



##### Gap statistic #####

def bootstrap_gap(data, controls, bvr_data, n, model, params, iter_num):
    # Create a random dataset
    rand_data = np.random.uniform(low=data.min(axis=0),
                                  high=data.max(axis=0) + 1,
                                  size=data.shape)
    rand_data = pd.DataFrame(rand_data, columns=data.columns)
    
    # Fit the model
    if model == 'latent':
        res = do_StepMix(
            rand_data,
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



##### Boostrap Chi2 for latent models #####

def baseline_chi2(data, bvr_data, n, covar, controls):
    # Refit model
    latent_mod = build_latent_model(n, 'categorical', covar)
    #warnings.filterwarnings('ignore', module='sklearn.*', category=FutureWarning)
    latent_mod = latent_mod.fit(data, controls)

    # Extract coefficients and posterior probabilities
    coeffs = latent_mod.get_parameters_df()
    coeffs = coeffs.reset_index()
    coeffs = coeffs[['class_no', 'variable', 'value']]
    post_probs = latent_mod.predict_proba(data, controls)

    # Reference l2
    mod_probs = np.max(post_probs, axis=1)
    l2_stat, chi2_stat = global_stats(bvr_data, post_probs, coeffs)
#bvr_datad=data_f_oh 
    # Build the most likely dataset under the model
    ref_data = pd.DataFrame()

    for var in data.columns:
        df = []
    
        for i in range(5):
            var_f = f'{var}_{i}'
            col = np.zeros(data.shape[0])

            for k in range(n):
                coeff_id = (coeffs['class_no'] == k) & (coeffs['variable'] == var_f)
                coeff_value = coeffs.loc[coeff_id, 'value'].values[0]
                col += coeff_value * post_probs[:, k]

            df.append(pd.DataFrame(col, columns=[var_f]))

        # Convert to binary using the most likely value
        temp = pd.concat(df, axis=1)
        temp = temp.apply(lambda row: (row == row.max()).astype(int), axis=1)

        # Convert to label encoding
        temp = temp.idxmax(axis=1).str.extract(r'(\d+)').astype(int).squeeze()
        temp = pd.DataFrame(temp.tolist(), columns=[var])
        ref_data = pd.concat([ref_data, temp], axis=1)
        
    return l2_stat, ref_data


def bootstrap_chi2(ref_data, controls, n, covar, iter_num):
    # Draw random sample with replacement
    btsp_sample = ref_data.sample(len(ref_data), replace=True)

    # One-hot encoding for BVR computation
    columns = []
    for col in btsp_sample.columns:
        for val in btsp_sample[col].unique():
            columns.append((btsp_sample[col] == val).astype(int).rename(f'{col}_{val}'))
    btsp_sample_oh = pd.concat(columns, axis=1)
    ## Drop empty columns to keep dimension coherent with model output
    btsp_sample_oh = btsp_sample_oh.loc[:, (btsp_sample_oh != 0).any()]
    
    # Fit the model
    latent_mod = build_latent_model(n, 'categorical', covar)
    
    # warnings.filterwarnings('ignore', module='sklearn.*', category=FutureWarning)
    latent_mod.fit(
        btsp_sample,
        controls_dum if covar == 'with' else None)

    # Compute l2 stat
    coeffs = latent_mod.get_parameters_df()
    coeffs = coeffs.reset_index()
    coeffs = coeffs[['class_no', 'variable', 'value']]
    post_probs = latent_mod.predict_proba(btsp_sample, controls)  
    l2_stat, chi2_stat = global_stats(btsp_sample_oh, post_probs, coeffs)

    return l2_stat