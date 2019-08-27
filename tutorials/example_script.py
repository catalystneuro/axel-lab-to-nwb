#
# Example: how to use the conversion script
# ------------------------------------------------------------------------------

from datetime import datetime
from dateutil.tz import tzlocal
from axel_lab_to_nwb import npz_to_nwb

fpath = '/Users/bendichter/Desktop/Axel Lab/data/2019_07_01_fly2'
f1 = '2019_07_01_Nsyb_NLS6s_walk_fly2.npz'
f2 = '2019_07_01_Nsyb_NLS6s_walk_fly2_A.npz'
f3 = '2019_07_01_Nsyb_NLS6s_walk_fly2_ref_im.npz'
fnpz = [f1, f2, f3]
fnwb = 'fly2.nwb'
info = {'session_description':'my CaIm recording',
        'identifier':'EXAMPLE_ID',
        'session_start_time':datetime.now(tzlocal()),
        'experimenter':'Evan Schaffer',
        'lab':'Axel lab',
        'institution':'Columbia University',
        'experiment_description':'EXPERIMENT_DESCRIPTION',
        'session_id':'IDX'}

npz_to_nwb(fpath=fpath, fnpz=fnpz, fnwb=fnwb, info=info, plot_rois=False)
