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
$ pip install axel-lab-to-nwb
```

# Use
After activating the correct environment, the conversion function can be used in different forms:

**1. Imported and run from a python script:** <br/>
Here's an example: we'll grab the data from the same experiment but stored in different `.npz` and `.mat` files and save it to a single `.nwb` file.
```python
from axel_lab_to_nwb import conversion_function
import yaml

# Nwb file
f_nwb = 'output.nwb'

# Source files
source_paths = {}
source_paths['raw data'] = {'type': 'file', 'path': PATH_TO_FILE}
source_paths['raw info'] = {'type': 'file', 'path': PATH_TO_FILE}
source_paths['processed data'] = {'type': 'file', 'path': PATH_TO_FILE}
source_paths['sparse matrix'] = {'type': 'file', 'path': PATH_TO_FILE}
source_paths['ref image'] = {'type': 'file', 'path': PATH_TO_FILE}

# Load metadata from YAML file
metafile = 'metafile.yml'
with open(metafile) as f:
   metadata = yaml.safe_load(f)

# Other options
kwargs = {
   'raw': False, 
   'processed': True, 
   'behavior': True
   'plot_rois': False
}

conversion_function(source_paths=source_paths,
                   f_nwb=f_nwb,
                   metadata=metadata,
                   **kwargs)
```
<br/>

**2. Command line:** <br/>
Similarly, the conversion function can be called from the command line in terminal:
```
$ python conversion_module.py [processed_data_file] [sparse_matrix_file] [ref_image_file]
  [output_file] [metadata_file]
```
<br/>

**3. Graphical User Interface:** <br/>
To use the GUI, just run the auxiliary function `nwb_gui.py` from terminal:
```
$ python nwb_gui.py
```
The GUI eases the task of editing the metadata of the resulting `.nwb` file, it is integrated with the conversion module (conversion on-click) and allows for visually exploring the data in the end file with [nwb-jupyter-widgets](https://github.com/NeurodataWithoutBorders/nwb-jupyter-widgets).

![](media/gif_axel.gif)

**4. Tutorial:** <br/>
At [tutorials](https://github.com/ben-dichter-consulting/axel-lab-to-nwb/tree/master/tutorials) you can also find Jupyter notebooks with the step-by-step process of conversion.
