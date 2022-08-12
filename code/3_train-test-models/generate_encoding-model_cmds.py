import os

def writeCommand(cmd, firstlevel_label, analysis_name, model_name):
    outfile = f'em_cmds_{firstlevel_label}_{analysis_name}_{model_name}.sh'
    with open(outfile,'a') as f:
        f.write(cmd + '\n')


## Define parameters
firstlevel_label = 'first_level_bySession'
analysis_name = 'analysis-3'
model_names = ['RidgeCV']
parcellations = [1000]
feature_sets = ['all', 'cognitive', 'perceptualmotor']  # options = ['all', 'cognitive', 'perceptualmotor']
null_models = ['None']  # options = ['shuffle-tasks', 'shuffle-features', 'None']
num_null_iterations = 20
subjects = [2, 3, 4, 6, 8, 9, 10, 12, 14, 15, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31]
excluded_subs = [29]
subjects = ['sub-%02d' % s for s in subjects if s not in excluded_subs]       

params = {
    'firstlevel_label': firstlevel_label,
    'analysis_name': analysis_name,

    'num_parcels': None,
    'feature_set': None,
    'model_name': None,
    'null_model': None,
    'sub': None,
    
    'save': True,
    'scale_x': True,
    'scale_y': False,
    'scale_x_type': 'maxabs',

    'save_coefs': False,
    'cv': True,
    
    'corr_Preds_TrainingMean': False,
    'corr_twoPreds': False
}


## Generate and save commands
for num_parcels in parcellations:
    for feature_set in feature_sets:
        for null_model in null_models:
            num_iter = 1
            if 'shuffle' in null_model:
                num_iter = num_null_iterations
            for model_name in model_names:
                for sub in subjects:
                    params['num_parcels'] = num_parcels
                    params['feature_set'] = feature_set
                    params['model_name'] = model_name
                    params['null_model'] = null_model
                    params['sub'] = sub

                    for i in range(num_iter):
                        seed = i
                        cmd = 'python3 train-test.py '
                        for k in params:
                            if type(params[k])==bool:
                                if params[k]:
                                    cmd += f'-{k} '
                            else:
                                cmd += f'-{k} {params[k]} '
                        cmd += f'-seed {seed} '
                        writeCommand(cmd, params['firstlevel_label'], params['analysis_name'], params['model_name'])