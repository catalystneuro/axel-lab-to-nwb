# axel-lab-to-nwb
NWB conversion scripts and tutorials.
A collaboration with [Axel Lab](https://www.axellab.columbia.edu/).

# Install
To clone the repository and set up a conda environment, do:
```
$ git clone https://github.com/ben-dichter-consulting/axel-lab-to-nwb.git
$ conda env create -f axel-lab-to-nwb/make_env.yml
$ source activate convert_to_nwb
```

Alternatively, to install directly in an existing environment:
```
$ pip install git+https://github.com/ben-dichter-consulting/axel-lab-to-nwb.git
```

# Use
After activating the correct environment, the conversion functions can be imported and run from python. 
Here's an example: we'll grab the data from the same experiment but stored in different `.npz` files and save it to a single `.nwb` file.
```python
from datetime import datetime
from dateutil.tz import tzlocal
from axel_lab_to_nwb import npz_to_nwb

fpath = '/path/to/files'
f1 = '2019_07_01_Nsyb_NLS6s_walk_fly2.npz'
f2 = '2019_07_01_Nsyb_NLS6s_walk_fly2_A.npz'
f3 = '2019_07_01_Nsyb_NLS6s_walk_fly2_ref_im.npz'
fnpz = [f1, f2, f3]
fnwb = 'fly2.nwb'
info = {'session_description':'my CaIm recording',
        'identifier':'EXAMPLE_ID',
        'session_start_time':datetime.now(tzlocal()),
        'experimenter':'My Name',
        'lab':'Axel lab',
        'institution':'Columbia University',
        'experiment_description':'EXPERIMENT_DESCRIPTION',
        'session_id':'IDX'}

npz_to_nwb(fpath=fpath, fnpz=fnpz, fnwb=fnwb, info=info, plot_rois=False)
```

At [tutorials](https://github.com/ben-dichter-consulting/axel-lab-to-nwb/tree/master/tutorials) you can also find Jupyter notebooks with the step-by-step process of conversion.
