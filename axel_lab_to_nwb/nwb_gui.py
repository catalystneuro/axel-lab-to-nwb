# Opens the NWB conversion GUI
# authors: Luiz Tauffer and Ben Dichter
# written for Axel Lab
# ------------------------------------------------------------------------------
from nwbn_conversion_tools.gui.nwbn_conversion_gui import nwbn_conversion_gui

metafile = 'metafile.yml'
conversion_module = 'conversion_module.py'

nwbn_conversion_gui(metafile=metafile, conversion_module=conversion_module)
