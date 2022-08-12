import os
import sys
import argparse
import json
import glob
import numpy as np
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn.masking import intersect_masks
from nilearn.plotting import plot_design_matrix
from nilearn.plotting import plot_stat_map
from nilearn import image as nimg
import matplotlib.pyplot as plt


def preprocess_confounds(confounds_df):
    cols_to_include = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'framewise_displacement']
    regex = ['cosine', 'a_comp_cor']
    for r in regex:
        cols_to_include.extend([col for col in confounds_df.columns.tolist() if r in col])
    confounds_df = confounds_df[cols_to_include]
    
    return confounds_df


def save_contrast_png(zmap, label, zmap_out_dir, thresh=0., vmax=5):
    zmap_out_file = os.path.join(zmap_out_dir, '%s.png' % label)
    fig = plot_stat_map(zmap, display_mode='x', threshold=thresh, cut_coords=range(0, 51, 10), title=label, vmax=vmax)
    fig.savefig(zmap_out_file)
    plt.close()

    
def save_contrast_nii(zmap, label, zstat_out_dir):
    zstat_out_file = os.path.join(zstat_out_dir, '%s.nii.gz' % label)
    zmap.to_filename(zstat_out_file)
    

def assign_durations(resources_dir, events_df):
    
    # Load task2task dict (for more readable naming format), and get task order
    with open(os.path.join(resources_dir, 'dicts/task2task.json'), 'rb') as f:
        task2task = json.load(f)
    task2task['Instruct'] = 'Instructions'
    task2task['Rest'] = 'Rest'
    task2task_inv = {v: k for k, v in task2task.items()}

    task_durations = pd.read_csv(os.path.join(resources_dir, 'condition_durations.csv'))

    event_related_conditions = ['StroopCon', 'StroopIncon',
                                'UnpleasantScenes', 'PleasantScenes',
                                'Go', 'NoGo',
                                'HappyFaces', 'SadFaces',
                                'Prediction', 'PredictScram', 'PredictViol']
    # Assign durations
    df = events_df.copy()
    prev_condition = None
    current_condition = None
    for i, row in df.iterrows():
        current_condition = row['trial_type']
        
        if current_condition in event_related_conditions:
            df.at[i, 'duration'] = task_durations[task_durations['condition']==
                                                  task2task[current_condition]]['duration'].values[0]
        else:
            if current_condition != prev_condition:
                df.at[i, 'duration'] = task_durations[task_durations['condition']==
                                                      task2task[current_condition]]['duration'].values[0]
            else:
                df.at[i, 'duration'] = np.nan

        prev_condition = current_condition
       
    
    # Check that each task block is 30 seconds in duration
    for i, row in df.groupby('taskName').agg('sum').iterrows():
        if 'instruct' not in i:
            assert row['duration'] == 30, f'Duration of {i} does not equal 30 seconds'
    
    # Only keep the first instance of each task condition
    df = df[df['duration'].notnull()]
    
    # Filter columns
    cols_to_keep = ['onset', 'duration', 'trial_type']
    df = df.filter(cols_to_keep)

    return df



############################################################################
# Args
############################################################################
parser = argparse.ArgumentParser(description="First level script for one subject and one session")
parser.add_argument('-sc', default=None)
parser.add_argument('-session', default=None)
parser.add_argument('-sub', default=None)
parser.add_argument('-analysis_name', default=None)
parser.add_argument('-proj_dir', default=None)
parser.add_argument('-bids_dir', default=None)

args = parser.parse_args()
SC = int(args.sc) # 1 or 2
SESSION = int(args.session) # 1 or 2
SUB = int(args.sub) 
SUB = '%02d' % SUB
analysis_name = args.analysis_name
proj_dir = args.proj_dir
bids_dir = args.bids_dir



############################################################################
# Definitions
############################################################################
proj_dir = os.path.abspath(os.path.join(os.getcwd(), '../..'))
resources_dir = os.path.join(proj_dir, 'resources')

fmriprep_dir = os.path.join(proj_dir, f'Y_data/fmriprep_output')
fmriprep_sc_dir = os.path.join(fmriprep_dir, f'sc{SC}/BIDS/derivatives/fmriprep')
fmriprep_sub_dir = os.path.join(fmriprep_sc_dir, f'sub-{SUB}')
fmriprep_sub_ses_dir = os.path.join(fmriprep_sub_dir, f'ses-{SESSION}/func')


if SC == 1:
    TASK = 'a'
if SC == 2:
    TASK = 'b' 
SES = TASK + str(SESSION)
RUNS = np.arange(1,9)

if SC == 1:
    conditions = ['Instruct', 'IntervalTiming','WordRead','VerbGen','Rest','Math','DigitJudgement',
                  'Verbal2Back','FingerSimple','FingerSeq','VisualSearchEasy','VisualSearchMed',
                  'VisualSearchHard','StroopIncon','StroopCon','Objects','HappyFaces','SadFaces',
                  'TheoryOfMind','SpatialImagery','VideoActions','VideoKnots','Object2Back','NoGo',
                  'Go','MotorImagery','PleasantScenes', 'UnpleasantScenes']
elif SC == 2:
    conditions = ['Instruct','VisualSearchEasy','VisualSearchMed','VisualSearchHard','NatureMovie',
                  'Rest','PermutedRules','SpatialMapHard','SpatialMapMed','SpatialMapEasy',
                  'WordRead','VerbGen','SpatialImagery','Object2Back','RespAltEasy','RespAltMed','RespAltHard',
                  'MentalRotMed','MentalRotHard','MentalRotEasy','AnimatedMovie','VideoActions','VideoKnots',
                  'TheoryOfMind','BiologicalMotion','ScrambledMotion','FingerSimple','FingerSeq',
                  'PredictViol','PredictScram','Prediction','LandscapeMovie']

    
print(f'\n\nsub-{SUB}, session {SES}')
output_dir = os.path.join(proj_dir, 'Y_data', analysis_name, SES, f'sub-{SUB}')
if os.path.exists(output_dir):
    print('Results already exist:', '\n', output_dir, '\n')
    sys.exit()



############################################################################
# First level objects: image, confounds, and events
############################################################################


# Get paths of events, specs, confounds, and niis
events_paths = []
for RUN in RUNS:
    path = f'sub-{SUB}/ses-{SES}/func/sub-{SUB}_ses-{SES}_task-{TASK}_run-{RUN}_events.tsv'
    events_paths.append(os.path.join(proj_dir, bids_dir, path))

specs_paths     = sorted(glob.glob(os.path.join(fmriprep_sub_ses_dir,'*preproc*.json')))
confounds_paths = sorted(glob.glob(os.path.join(fmriprep_sub_ses_dir,'*confounds*')))
img_paths       = sorted(glob.glob(os.path.join(fmriprep_sub_ses_dir,'*preproc*.nii*')))


# Get masks from all 32 sessions
sub_masks = sorted(glob.glob(os.path.join(fmriprep_dir, f'sc*/BIDS/derivatives/fmriprep/sub-{SUB}/ses-*/func/*mask*nii*')))
sub_mask  = intersect_masks(sub_masks)


# If subject 21, remove run 2 in session 1 due to incomplete fmriprep preprocessing
if '21' in SUB:
    specs_paths     = [i for i in specs_paths if not ('sc2' in i and 'ses-1' in i and 'task-2' in i)]
    confounds_paths = [i for i in confounds_paths if not ('sc2' in i and 'ses-1' in i and 'task-2' in i)]
    events_paths    = [i for i in events_paths if not ('sc2' in i and 'ses-1' in i and 'task-2' in i)]
    img_paths       = [i for i in img_paths if not ('sc2' in i and 'ses-1' in i and 'task-2' in i)]
num_runs = len(img_paths)

# Get tr and slice time ref
with open(specs_paths[0]) as f:
    spec = json.load(f)
try:
    slice_time_ref = spec['SliceTimingRef']
except:
    slice_time_ref = 0.

    

############################################################################
# Model parameters
############################################################################

sub_label = SUB
t_r = (float(spec['RepetitionTime']))
slice_time_ref = slice_time_ref
hrf_model = 'spm'
mask = sub_mask
min_onset = 0 
standardize = False
signal_scaling = False
noise_model = 'ar1'
drift_model = None
verbose = 0
n_jobs = 1  # -1
memory_level = 1
minimize_memory = True


# Other params
num_scans = 595  # Number of scans, excluding dummies
num_dummy = 6  # Number of dummy scans
dummy_dur = num_dummy * t_r 

start_time = slice_time_ref * t_r
end_time = ((num_scans-1) * t_r) + (slice_time_ref * t_r)
frame_times = np.linspace(start_time, end_time, num_scans)




############################################################################
# Make design matrices
############################################################################

X_list = []
for i in range(num_runs):
    
    # Remove dummy scans from confounds and events
    confounds_df = preprocess_confounds(pd.read_csv(confounds_paths[i], delimiter='\t'))
    confounds_df = confounds_df.iloc[num_dummy:]
    confounds_names = confounds_df.columns.tolist()
    confounds_df.reset_index(drop=True, inplace=True)
    
    # Assign durations of task condition regressors
    events_df = pd.read_csv(events_paths[i], delimiter='\t')
    events_df = assign_durations(resources_dir, events_df)
    events_df.reset_index(drop=True, inplace=True)
    
    # Shift onsets by duration of dummy scans
    events_df['onset'] = events_df['onset'] - dummy_dur

    # Make design matrix
    X = make_first_level_design_matrix(frame_times=frame_times,
                                       events=events_df,
                                       hrf_model=hrf_model,
                                       drift_model=drift_model,
                                       add_regs=confounds_df,
                                       add_reg_names=confounds_names,
                                       min_onset=min_onset)
    X_list.append(X)
    

    
############################################################################
# Load images and remove dummy scans
############################################################################

print('Loading images...')
slice_start = num_dummy
slice_end = num_scans + num_dummy

Y_list = []
for path in img_paths:
    Y = nimg.load_img(path)
    Y = nimg.index_img(Y, slice(slice_start, slice_end))
    Y_list.append(Y)
   

    
############################################################################
# Create model object
############################################################################

model = FirstLevelModel(subject_label=sub_label,
                        t_r=t_r,
                        slice_time_ref=slice_time_ref,
                        hrf_model=hrf_model,
                        mask_img=mask,
                        drift_model=drift_model,
                        min_onset=min_onset,
                        memory_level=memory_level,
                        standardize=standardize,
                        signal_scaling=signal_scaling,
                        noise_model=noise_model,
                        verbose=verbose,
                        n_jobs=n_jobs,
                        minimize_memory=minimize_memory
                       )


############################################################################
# Fit model
############################################################################

print(f'Fitting model...')
model.fit(run_imgs=Y_list, design_matrices=X_list)



############################################################################
# Save figures and niis
############################################################################
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

save_png = True
save_nii = True
thresh = 0.
vmax = 5

print(f'Saving zmap figures and niis...')
zmap_out_dir_figs = os.path.join(output_dir, 'figs')
zmap_out_dir_niis = os.path.join(output_dir, 'niis')
for d in [zmap_out_dir_figs, zmap_out_dir_niis]:
    if not os.path.exists(d):
        os.makedirs(d)

for condition in conditions:
    if condition not in ['Rest', 'Instruct']:
        zmap = model.compute_contrast([f'{condition}-Rest' for i in range(num_runs)])
        label = f'sub-{SUB}_{condition}_zmap'
        if save_png:
            save_contrast_png(zmap, label, zmap_out_dir_figs, thresh=0., vmax=5)
        if save_nii:
            save_contrast_nii(zmap, label, zmap_out_dir_niis)

            

############################################################################       
# Save design matrices for QA
############################################################################

designs_out_dir = os.path.join(proj_dir, 'Y_data', analysis_name, f'designs/sub-{SUB}')
if not os.path.exists(designs_out_dir):
    os.makedirs(designs_out_dir)

for i in range(len(X_list)):
    X = X_list[i]
    plt.figure(figsize=(10,50))
    im = plt.imshow(X, vmin=0, vmax=1)
    plt.savefig(os.path.join(designs_out_dir, f'{SES}_run-{i+1}.png'))
    plt.close()
    
    
print('\n\n')