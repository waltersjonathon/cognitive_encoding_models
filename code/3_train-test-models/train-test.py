import os
import pickle
import glob
import numpy as np
import pandas as pd
import argparse
import json
import sys
from itertools import combinations
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import LeavePGroupsOut
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import LeaveOneGroupOut


################################################################################################################


def GroupKFold_random_splits(n_splits, groups, rng):
    '''
    Implements functionality of sklearn's GroupKFold, but with random assignment of groups to folds (GroupKFold does not do this).
    Requires an rng.
    
    Output: list of (train, test) splits as lists of indices
    '''
    ix = np.arange(len(groups))
    unique = np.unique(groups)
    rng.shuffle(unique)
    splits_ix = []
    for split in np.array_split(unique, n_splits):
        mask = np.isin(groups, split)
        train, test = ix[~mask], ix[mask]
        splits_ix.append((train, test))
        
    return splits_ix



def avg_map_corr(estimator, X, Y):
    '''
    Scoring function. Calculates the average correlation between the predicted and held-out maps.
    '''
    Y_hat = estimator.predict(X)
    corrs = [np.corrcoef(true, pred)[0, 1] for true, pred in zip(Y, Y_hat)]
    return np.mean(corrs)
    

    
def avg_map_r2(estimator, X, Y):
    '''
    Scoring function. Calculates average r-squared between the predicted and held-out maps.
    '''
    Y_hat = estimator.predict(X)
    return r2_score(Y.T, Y_hat.T) # r2 per map, not per parcel



def calculate_metrics(Y_TEST, Y_PRED, test_groups):
    '''
    Calculates 2-way classification accuracy, along with correlation and r-squared for each test item.
    '''
    # Because there are unequal groups (e.g., a held-out task can have 2 or 4 instances), convert these group indices into a binary array
    task_0_indices = np.argwhere(test_groups == np.min(test_groups)).flatten()
    task_1_indices = np.argwhere(test_groups == np.max(test_groups)).flatten()

    # Separate out the true maps for the two tasks
    tests_0 = Y_TEST[task_0_indices]
    tests_1 = Y_TEST[task_1_indices]

    # There will be 2 unique predictions per split, one for each held-out task
    pred_0 = Y_PRED[task_0_indices[0]]
    pred_1 = Y_PRED[task_1_indices[0]]

    # Calculate the 2-way correlations
    corrs_pred0_test0 = np.corrcoef(pred_0, tests_0)[0,1:]
    corrs_pred0_test1 = np.corrcoef(pred_0, tests_1)[0,1:]
    corrs_pred1_test1 = np.corrcoef(pred_1, tests_1)[0,1:]
    corrs_pred1_test0 = np.corrcoef(pred_1, tests_0)[0,1:]
    
    # Aggregate correlation lists (of variable length) 
    corr_vals = [corrs_pred0_test0, corrs_pred1_test1] 
    
    # Calculate 2-way classification accuracy (based on average correlations)
    acc_0, acc_1, accuracy = 0, 0, 0
    if np.mean(corrs_pred0_test0) > np.mean(corrs_pred0_test1):
        acc_0 = 1
    if np.mean(corrs_pred1_test1) > np.mean(corrs_pred1_test0):
        acc_1 = 1
    if acc_0 and acc_1:
        accuracy = 1
    accuracy_tuple = (acc_0, acc_1)
    
    # Calculate r-squared list (of variable length) for each task
    # r-squared by map
    r2_pred0_test0 = np.array([r2_score(true.T, pred_0.T) for true in tests_0])
    r2_pred1_test1 = np.array([r2_score(true.T, pred_1.T) for true in tests_1])
    r2_vals = [r2_pred0_test0, r2_pred1_test1]
    
    # r-squared by region
    r2_pred0_test0_byRegion = np.array([r2_score(true, pred_0) for true in tests_0])
    r2_pred1_test1_byRegion = np.array([r2_score(true, pred_1) for true in tests_1])
    r2_vals_byRegion = [r2_pred0_test0_byRegion, r2_pred1_test1_byRegion]
    
    return accuracy, accuracy_tuple, corr_vals, r2_vals, r2_vals_byRegion



def extract_stats(results, verbose=True, save_coefs=False):
    '''
    Aggregates results across iterations (e.g., null models have multiple iterations of the leave-two-out analysis)
    '''
    keys = ['testpairs', 'test_indices',
            'acc1_vals', 'acc2_vals', 'acc1', 'acc2',
            'corr_vals', 'r2_vals',
            'alphas',
            'coefs', 'intercepts',
            'preds',
            'corrs_predsTrain', 'corr_preds'
           ]
    res = {k:[] for k in keys}

    for r in results:
        res['testpairs'].append(r['testpairs'])
        res['test_indices'].append(r['test_indices'])
        
        res['acc1_vals'].append(r['acc1'])
        res['acc2_vals'].append(r['acc2'])
        res['acc1'].append(np.mean(r['acc1']))
        res['acc2'].append(np.mean(np.array(r['acc2']).flatten()))
        
        res['corr_vals'].append(r['corrs'])
        res['r2_vals'].append(r['r2'])
        
        res['alphas'].append(r['alphas'])
        
        if save_coefs:
            res['coefs'].append(r['coefs'])
            res['intercepts'].append(r['intercepts']) 
            res['preds'].append(r['preds'])
        
        if 'corrs_predsTrain' in r:
            res['corrs_predsTrain'].append(r['corrs_predsTrain'])
        if 'corr_preds' in r:
            res['corr_preds'].append(r['corr_preds'])

    if verbose:
        print('\nacc1: %.4f' % np.mean(res['acc1']))
        print('acc2: %.4f' % np.mean(res['acc2']))

    return(res)


    
def run_leave2out(X, Y, model_name, features=None,
                  scale_x=True, scale_y=False, x_scaler_type='maxabs',
                  corr_Preds_TrainingMean=None, corr_twoPreds=None,
                  verbose=True,):

    # Outer loop params
    splitter_outerLoop = LeavePGroupsOut(n_groups=2)

    # Inner loop params
    k = 10
    alphas = np.arange(2, 11, dtype=float) #[1e-3, 1e-2, 1e-1] + np.arange(1, 11, dtype=float)
    kfold_rng = np.random.default_rng(0)

    # Model params
    groups = np.array(X.task_index)
    scorer = avg_map_corr

    # Initialize dictionary
    res_keys = ['acc1', 'acc2', 'corrs', 'r2',
                'testpairs', 'test_indices',
                'alphas',
                'coefs', 'intercepts',
                'preds',
                'corrs_predsTrain', 'corr_preds']
    res = {v:[] for v in res_keys}

    # Train and test
    i = 0
    for train_idx, test_idx in splitter_outerLoop.split(X, Y, groups=groups):
        if verbose:
            if i%100 == 0:
                print('Split %s/946' % i)
            i += 1

        # Separate train/test data in the outer loop using leave-2-out
        X_TRAIN = X.iloc[train_idx].filter(features).values
        X_TEST  = X.iloc[test_idx].filter(features).values

        Y_TRAIN = Y.iloc[train_idx].filter(regex='region').values
        Y_TEST  = Y.iloc[test_idx].filter(regex='region').values

        # Scale X
        if scale_x:
            if x_scaler_type == 'standard':
                X_train_scaler = StandardScaler()
            if x_scaler_type == 'maxabs':
                X_train_scaler = MaxAbsScaler()
            X_TRAIN = X_train_scaler.fit_transform(X_TRAIN)
            X_TEST = X_train_scaler.transform(X_TEST)

        # Find optimal regularization parameter in the inner loop using K-fold
        estimator = RidgeCV(alphas=alphas,
                            scoring=scorer,
                            cv=GroupKFold_random_splits(n_splits=k, groups=groups[train_idx], rng=kfold_rng))
        estimator.fit(X_TRAIN, Y_TRAIN)

        # Measure model performance using optimal regularization parameter
        test_groups = groups[test_idx]
        Y_PRED = estimator.predict(X_TEST)
        acc1, acc2, corrs, r2, r2_byRegion = calculate_metrics(Y_TEST, Y_PRED, test_groups)

        # Gather metrics
        res['acc1'].append(acc1)
        res['acc2'].append(acc2)
        res['corrs'].append(corrs)
        res['r2'].append(r2)

        res['testpairs'].append(test_groups) # task IDs
        res['test_indices'].append(test_idx) # task indices in the full X and Y

        res['alphas'].append(estimator.alpha_)
        res['coefs'].append(estimator.coef_)
        res['intercepts'].append(estimator.intercept_)

        res['preds'].append(Y_PRED)

        if corr_Preds_TrainingMean:
            AVG_MAP = np.mean(Y_TRAIN, axis=0)
            corrs_predsTrain = np.corrcoef(AVG_MAP, Y_PRED)[0, 1:]
            res['corrs_predsTrain'].append(corrs_predsTrain)
        if corr_twoPreds:
            corr_preds = np.corrcoef(Y_PRED[test_groups.argmin()],
                                     Y_PRED[test_groups.argmax()])[0, 1] 
            res['corr_preds'].append(corr_preds)
        
    return res



    
def run_noCV(X, Y, model_name, features=None,
             scale_x=True, scale_y=False, x_scaler_type='maxabs'):
  
    # RidgeCV params
    groups = np.array(X.task_index)
    scorer = avg_map_corr
    k = 10
    alphas = [1e-3, 1e-2, 1e-1]
    alphas.extend(np.arange(2, 11, dtype=float))
    kfold_rng = np.random.default_rng(0)

    # Initialize dictionary
    res_keys = ['alphas', 'coefs', 'intercepts', 'best_score', 'r2_byMap', 'r2_byRegion']
    res = {v:[] for v in res_keys}

    # Filter the dataframes
    X = X.filter(features).values
    Y = Y.filter(regex='region').values

    # Scale X
    if scale_x:
        if x_scaler_type == 'standard':
            X_train_scaler = StandardScaler()
        if x_scaler_type == 'maxabs':
            X_train_scaler = MaxAbsScaler()
        X = X_train_scaler.fit_transform(X)

    # Find optimal regularization parameter using K-fold
    estimator = RidgeCV(alphas=alphas, scoring=scorer, cv=GroupKFold_random_splits(n_splits=k, groups=groups, rng=kfold_rng))
    estimator.fit(X, Y)

    # Gather metrics
    res['alphas'].append(estimator.alpha_)
    res['coefs'].append(estimator.coef_)
    res['intercepts'].append(estimator.intercept_)
    res['best_score'].append(estimator.best_score_)
    res['r2_byMap'].append(r2_score(Y.T, estimator.predict(X).T, multioutput='raw_values'))
    res['r2_byRegion'].append(r2_score(Y, estimator.predict(X), multioutput='raw_values'))

    return res




################################################################################################
# Arguments
################################################################################################

parser = argparse.ArgumentParser()

parser.add_argument('-firstlevel_label', required=True)
parser.add_argument('-analysis_name', required=True)

parser.add_argument('-num_parcels', type=int, required=True)
parser.add_argument('-feature_set', required=True)
parser.add_argument('-model_name', required=True)
parser.add_argument('-null_model', required=True)
parser.add_argument('-sub', required=True)
parser.add_argument('-seed', default=0)

parser.add_argument('-scale_x', dest='scale_x', action='store_true')
parser.add_argument('-scale_y', dest='scale_y', action='store_true')
parser.set_defaults(scale_x=False)
parser.set_defaults(scale_y=False)
parser.add_argument('-scale_x_type', required=True)

parser.add_argument('-pcr', dest='pcr', action='store_true')
parser.set_defaults(pcr=False)
parser.add_argument('-n_components', default=None)

parser.add_argument('-max_iter', default=5000)
parser.add_argument('-l1_ratio', default=0.05)
parser.add_argument('-alpha', default=0.9)

parser.add_argument('-cv', dest='cv', action='store_true')
parser.set_defaults(cv=False)

parser.add_argument('-save_coefs', dest='save_coefs', action='store_true')
parser.add_argument('-save', dest='save', action='store_true')
parser.set_defaults(save_coefs=False)
parser.set_defaults(save=False)

parser.add_argument('-parc_label', default='schaefer_2018')
parser.add_argument('-space_label', default='MNI152NLin2009cAsym')

parser.add_argument('-corr_Preds_TrainingMean', dest='corr_Preds_TrainingMean', action='store_true')
parser.add_argument('-corr_twoPreds', dest='corr_twoPreds', action='store_true')
parser.set_defaults(corr_Preds_TrainingMean=False)
parser.set_defaults(corr_twoPreds=False)


args = parser.parse_args()

firstlevel_label = args.firstlevel_label
analysis_name = args.analysis_name

num_parcels = args.num_parcels
feature_set = args.feature_set
model_name = args.model_name
null_model = args.null_model
sub = args.sub
seed = int(args.seed)

scale_x = args.scale_x
scale_y = args.scale_y

x_scaler_type = args.scale_x_type

pcr = args.pcr
n_components = args.n_components

max_iter = args.max_iter
l1_ratio = args.l1_ratio
alpha = args.alpha

cv = args.cv

save_coefs = args.save_coefs
save = args.save

parc_label = args.parc_label
space_label = args.space_label

corr_Preds_TrainingMean = args.corr_Preds_TrainingMean
corr_twoPreds = args.corr_twoPreds


####################################################################################################
# Random seed
####################################################################################################

np.random.seed(seed)


####################################################################################################
# Define and create directories
####################################################################################################

proj_dir = os.path.abspath(os.path.join(os.getcwd(), '../..'))
resources_dir = os.path.join(proj_dir, 'resources')
features_dir = os.path.join(proj_dir, 'X_features')
firstlevel_dir = os.path.join(proj_dir, 'Y_data', firstlevel_label)
output_dir = os.path.join(proj_dir, 'model_outputs', firstlevel_label, analysis_name, model_name)

if not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir)
    except:
        pass

if null_model != 'None':
    feature_set_label = f'{feature_set}-{null_model}_seed-{seed}'
else:
    feature_set_label = feature_set
    
outfile = os.path.join(output_dir,
                       f'{firstlevel_label}_{analysis_name}_{num_parcels}-parcels_{model_name}_{feature_set_label}_{sub}.pkl')

if os.path.exists(outfile):
    print('\n', 'File already exists:', '\n', outfile, '\n')
    sys.exit()
    

####################################################################################################
# Y: TASK MAPS
####################################################################################################

# Load all parcellated contrasts

# Y DF: Task name, Task index, Session (a1/a2/b1/b2), Region 1 ... Region 1000
# X DF: Task name, Task index, Session (a1/a1/b1/b2), Feature 1 ... Feature X

## Load original feature matrix
X_original = pd.read_csv(os.path.join(features_dir, f'byTask/X_{feature_set}.csv'), index_col=0)
task_order = X_original.index.tolist()
features = X_original.columns.tolist()
num_tasks = len(task_order)
                              
## Load task2task dict (for more readable naming format), and get task order
with open(os.path.join(resources_dir, 'dicts/task2task.json'), 'rb') as f:
    task2task = json.load(f)
task2task_inv = {v: k for k, v in task2task.items()}

## Load X
X = pd.read_csv(os.path.join(features_dir, f'byTask_bySession/X_{feature_set}.csv'), index_col=0)

## Construct Y
y_rows = []
for i, row in X.iterrows():
    task_name = row['task_name']
    task_name_short = task2task_inv[task_name]
    task_index = row['task_index']
    session = row['session']
    contrast_id = row['contrast_id']
    
    y_row = [task_name, task_index, session, contrast_id]
    image_path = f'{firstlevel_dir}/{session}/{sub}/parcellated/{parc_label}/{space_label}/{sub}_{task_name_short}_zmap.npy'
    y_row.extend(list(np.load(image_path)[0]))
    y_rows.append(y_row)
    
metadata_cols = ['task_name', 'task_index', 'session', 'contrast_id']
y_columns = metadata_cols.copy()
for i in range(num_parcels):
    y_columns.append('region_' + str(i))
Y = pd.DataFrame(y_rows, columns=y_columns)

features = [f for f in X.columns.tolist() if f not in metadata_cols]

        
#######################################################################################################
# Train/test model
#######################################################################################################

print(f'\nNumber of parcels: {num_parcels}')
print(f'Subject: {sub}')
print(f'Feature set: {feature_set}')
print(f'Null model: {null_model}\n')

## Initialize results dict
parcel_results_dict = {num_parcels:{feature_set:{model_name:{sub:{}}}}}


## Train and test models

if null_model != 'None':
    sub_res = []
    
    ## Create null model
    X_null = X.copy()

    # If shuffle-tasks, establish a random re-numbering of tasks
    if null_model == 'shuffle-tasks':
        random_task_mapping = np.arange(0, num_tasks)
        np.random.shuffle(random_task_mapping)

    # For each of the 44 tasks, create the appropriate null models from the taskSessions-by-features df, of shape (112, 36)
    for task_index in range(num_tasks):
        task_name = task_order[task_index]
        # indices of the task's occurences in the (112 by 36) matrix
        task_df_indices = X.index[X['task_name'] == task_name].tolist()

        if null_model == 'shuffle-tasks':
            # Get the task's new randomly assigned feature vector
            remapped_task_idx = random_task_mapping[task_index]
            feature_names = X_original.iloc[remapped_task_idx].index.tolist()
            feature_vals  = X_original.iloc[remapped_task_idx].values

        elif null_model == 'shuffle-features':
            # Randomly shuffle features of each task
            feature_names = X_original.iloc[task_index].index.tolist()
            feature_vals  = X_original.iloc[task_index].values
            np.random.shuffle(feature_vals)

        X_null.loc[task_df_indices, feature_names] = feature_vals

    # Train and test        
    sub_res.append(run_leave2out(X_null, Y, model_name, features=features,
                                 scale_x=scale_x, scale_y=scale_y, x_scaler_type=x_scaler_type,
                                 corr_Preds_TrainingMean=corr_Preds_TrainingMean, corr_twoPreds=corr_twoPreds))
    parcel_results_dict[num_parcels][feature_set][model_name][sub] = extract_stats(sub_res, save_coefs=save_coefs)
                         
elif null_model == 'None':
    if cv:
        sub_res = run_leave2out(X, Y, model_name, features=features,
                                scale_x=scale_x, scale_y=scale_y, x_scaler_type=x_scaler_type,
                                corr_Preds_TrainingMean=corr_Preds_TrainingMean, corr_twoPreds=corr_twoPreds)
        parcel_results_dict[num_parcels][feature_set][model_name][sub] = extract_stats([sub_res], save_coefs=save_coefs)
    else:
        sub_res = run_noCV(X, Y, model_name, features=features,
                           scale_x=scale_x, scale_y=scale_y, x_scaler_type=x_scaler_type)
        parcel_results_dict[num_parcels][feature_set][model_name][sub] = sub_res



## Save results
if save:
    with open(outfile, 'wb') as f:
        pickle.dump(parcel_results_dict, f)
    print(f'\nSaved at:\n{outfile}\n\n')