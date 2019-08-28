#
# Example: how to use the conversion script
# ------------------------------------------------------------------------------

from datetime import datetime
from dateutil.tz import tzlocal
from axel_lab_to_nwb import npz_to_nwb

fpath = r'C:\Users\Luiz\Google Drive (luiz@taufferconsulting.com)\client_ben\project_caim'
#fpath = '/Users/bendichter/Desktop/Axel Lab/data/2019_07_01_fly2'
f1 = '2019_07_01_Nsyb_NLS6s_walk_fly2.npz'
f2 = '2019_07_01_Nsyb_NLS6s_walk_fly2_A.npz'
f3 = '2019_07_01_Nsyb_NLS6s_walk_fly2_ref_im.npz'
fnpz = [f1, f2, f3]
fnwb = 'fly2.nwb'
info = {'session_description':'my CaIm recording',
        'identifier':'EXAMPLE_ID',
        'session_id':'IDX',
        'session_start_time':datetime.now(tzlocal()),
        'notes':'NOTES',
        'stimulus_notes':'STIMULUS_NOTES',
        'data_collection':'DATA_COLLECTION',
        'experimenter':'Evan Schaffer',
        'lab':'Axel lab',
        'institution':'Columbia University',
        'experiment_description':'EXPERIMENT_DESCRIPTION',
        'protocol':'EXAMPLE_PROTOCOL',
        'keywords':['KW1','KW2','KW3'],
        }

npz_to_nwb(fpath=fpath, fnpz=fnpz, fnwb=fnwb, info=info, plot_rois=False)
