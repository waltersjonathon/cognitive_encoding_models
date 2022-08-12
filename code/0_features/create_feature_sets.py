import os
import glob
import json
import pandas as pd



# Define paths
proj_dir = '../..'
firstlevel_label = 'first_level_bySession'
features_dir = os.path.join(proj_dir, 'X_features')
resources_dir = os.path.join(proj_dir, 'resources')
data_dir = os.path.join(proj_dir, 'Y_data', firstlevel_label)


# Load task2task dict (for more readable naming format)
with open(os.path.join(resources_dir, 'dicts', 'task2task.json'), 'rb') as f:
    task2task = json.load(f)
task2task_inv = {v: k for k, v in task2task.items()}


# Get lists of 'cognitive' and 'perceptual-motor' features from csv file that labels each cogatlas entity
entity_labels = os.path.join(features_dir, 'cognitive_perceptual-motor_labels.csv')
df = pd.read_csv(entity_labels)
cognitive_features = df[df['Cognitive'] == 1]['Entity'].tolist()
perceptualmotor_features = df[df['Perceptual-motor'] == 1]['Entity'].tolist()


# Load original annotations (X_all) and filter by cognitive and perceptual-motor features
X_all = pd.read_csv(os.path.join(features_dir, 'source_annotation/X_all.csv'), index_col=0)
X_cognitive = X_all.filter(cognitive_features)
X_perceptualmotor = X_all.filter(perceptualmotor_features)


    
# Create feature matrices for session-specific contrasts
tasks = X_all.index.tolist()
features = X_all.columns.tolist()

metadata_cols = ['task_name', 'task_index', 'session', 'contrast_id']
X_columns = metadata_cols + features
X_rows = []

sub = 'sub-02' # an example sub to build the df

count = 0
for task_name in tasks:
    task = task2task_inv[task_name]
    maps = sorted(glob.glob(f'{data_dir}/*/{sub}/parcellated/{sub}_{task}_zmap.npy'))
    for m in maps:
        start = m.find(firstlevel_label) + 22
        session = m[start:start + 2]

        task_index = tasks.index(task2task[task])
        contrast_id = count

        X_row = [task2task[task], task_index, session, count]
        X_row.extend(X_all.loc[task_name].values)
        X_rows.append(X_row)

        count += 1

X_bySession_all = pd.DataFrame(X_rows, columns=X_columns)
X_bySession_cognitive = X_bySession_all.filter(metadata_cols + cognitive_features)
X_bySession_perceptualmotor = X_bySession_all.filter(metadata_cols + perceptualmotor_features)



## Save

# byTask
outdir = os.path.join(features_dir, 'byTask')
X_all.to_csv(os.path.join(outdir, 'X_all.csv'))
X_cognitive.to_csv(os.path.join(outdir, 'X_cognitive.csv'))
X_perceptualmotor.to_csv(os.path.join(outdir, 'X_perceptualmotor.csv'))
print('Features saved to:', outdir)

# yTask and bySession
outdir = os.path.join(features_dir, 'byTask_bySession')
X_bySession_all.to_csv(os.path.join(outdir, 'X_all.csv'))
X_bySession_cognitive.to_csv(os.path.join(outdir, 'X_cognitive.csv'))
X_bySession_perceptualmotor.to_csv(os.path.join(outdir, 'X_perceptualmotor.csv'))
print('Features saved to:', outdir)