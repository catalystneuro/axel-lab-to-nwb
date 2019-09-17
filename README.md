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
from axel_lab_to_nwb import conversion_function

fpath = '/path/to/files'
f1 = '2019_07_01_Nsyb_NLS6s_walk_fly2.npz'
f2 = '2019_07_01_Nsyb_NLS6s_walk_fly2_A.npz'
f3 = '2019_07_01_Nsyb_NLS6s_walk_fly2_ref_im.npz'
f_source = [f1, f2, f3]
f_nwb = 'fly2.nwb'
metafile = 'metafile.yml'
plot_rois = False
conversion_function(f_source=f_source,
                    f_nwb=f_nwb,
                    metafile=metafile,
                    plot_rois=plot_rois)
```

At [tutorials](https://github.com/ben-dichter-consulting/axel-lab-to-nwb/tree/master/tutorials) you can also find Jupyter notebooks with the step-by-step process of conversion.

To use the GUI, just run the auxiliary function `nwb_gui.py` from terminal:
```
$ python nwb_gui.py
```
