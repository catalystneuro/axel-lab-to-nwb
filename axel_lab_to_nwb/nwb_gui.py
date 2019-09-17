# Opens the NWB conversion GUI
# authors: Luiz Tauffer and Ben Dichter
# written for Axel Lab
# ------------------------------------------------------------------------------
from nwbn_conversion_tools.gui.main_gui import main

metafile = 'metafile.yml'
conversion_module = 'conversion_module.py'

main(metafile=metafile, conversion_module=conversion_module)
