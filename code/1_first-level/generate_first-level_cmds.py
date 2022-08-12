import os

def saveCommand(cmd):
    outfile = 'first-level_cmds.sh'
    with open(outfile, 'a') as f:
        f.write(cmd + '\n')
        
        
## Define parameters
ANALYSIS_NAME = 'first_level_bySession'
BIDS_DIR = 'Y_data/bids'

SCs = [1, 2]
SESSIONs = [1, 2]

SUBS = [2, 3, 4, 6, 8, 9, 10, 12, 14, 15, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31]
excluded_subs = []
SUBS = [s for s in SUBS if s not in excluded_subs]       


## Generate and save commands
for SUB in SUBS:
    for SC in SCs:
        for SESSION in SESSIONs:
            cmd = f'python3 first_level.py -sc {SC} -session {SESSION} -sub {SUB} -analysis_name {ANALYSIS_NAME} -bids_dir {BIDS_DIR}'
            saveCommand(cmd)