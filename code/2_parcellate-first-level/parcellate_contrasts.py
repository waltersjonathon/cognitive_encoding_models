import os
import glob
import numpy as np
from nilearn.input_data import NiftiLabelsMasker
from nilearn import datasets
from nilearn.image import load_img
from templateflow.api import get


## Params
analysis_name = 'first_level_bySession'
num_parcels = 1000
standardize = False
parc_label = 'schaefer_2018'
space_label = 'MNI152NLin2009cAsym'


## Dirs
proj_dir = os.path.abspath(os.path.join(os.getcwd(), '../..'))
data_dir = os.path.join(proj_dir, 'Y_data')
firstlevel_dir = os.path.join(data_dir, analysis_name)
parcellations_dir = os.path.join(proj_dir, 'Y_data/parcellation_schemes')

session_a1_dir = os.path.join(firstlevel_dir, 'a1')
session_a2_dir = os.path.join(firstlevel_dir, 'a2')
session_b1_dir = os.path.join(firstlevel_dir, 'b1')
session_b2_dir = os.path.join(firstlevel_dir, 'b2')
                              
session_dirs = [session_a1_dir, session_a2_dir,
                session_b1_dir, session_b2_dir]


## Define atlas

# MNI (note: not using this one)
atlas_mni = datasets.fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=7, resolution_mm=1, data_dir=parcellations_dir)

# Atlas in MNI152NLin2009cAsym space
atlas_path = get(template=space_label,
                 resolution=1,
                 atlas='Schaefer2018',
                 desc=f'{num_parcels}Parcels7Networks')
atlas_filename = atlas_path.name
atlas_img = load_img(str(atlas_path))

atlas_out = os.path.join(parcellations_dir, parc_label, atlas_filename)
if not os.path.exists(atlas_out):
    atlas_img.to_filename(atlas_out)   

    
## Define masker
masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=standardize, resampling_target='data')
masker.fit()

## Parcellate all sessions, subs, tasks
for session_dir in session_dirs: 
    subs = sorted([os.path.basename(i) for i in glob.glob(session_dir + '/*')])
    
    for sub in subs: 
        out_dir = os.path.join(session_dir, sub, 'parcellated', parc_label, space_label)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        nii_files = sorted(glob.glob(os.path.join(session_dir, sub, 'niis/*')))
        
        for f in nii_files:
            task_label = os.path.basename(f).split('.')[0]
            out_file = os.path.join(out_dir, task_label) + '.npy'
            
            if not os.path.exists(out_file):
                print(f'Parcellating: {f}')
                parcellated_img = masker.transform([f])
                with open(out_file, 'wb') as out:
                    np.save(out, parcellated_img)
            else:
                print(f'Already parcellated, skipping: {f}')