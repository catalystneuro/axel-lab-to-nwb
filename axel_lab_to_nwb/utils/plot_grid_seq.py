# Plot function for compression results

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_grid_seq(df, columns, legend_columns, ax_lims = None, markers = None, add_gridline = False):

    if ax_lims != None:
        if len(ax_lims) != len(columns):
            raise IndexError("number of ax_lims is not equal to the number of columns")
        # remove data not in axis limits
        df = df[ functools.reduce(   lambda x,y: x & ( df[y] < ax_lims[y][1] ) & ( df[y] > ax_lims[y][0] ), columns, True ) ]

    # add markers
    if markers == None:
        markers = sns.mpl.lines.Line2D.filled_markers

    df_mod = df[legend_columns]
    df_mod = df_mod.drop_duplicates()

    _lightest_col = 0.4 # > 0 and < 1
    df_mod_group_compression = df_mod.sort_values(legend_columns).groupby(["Compression"])
    palette_saturation = df_mod_group_compression.apply(lambda x: pd.Series(
            _lightest_col + ( 1 - _lightest_col )*(np.arange(len(x))+1)/len(x) )).values.flatten() # saturation > 0 and < 1
    current_palette = sns.color_palette()
    palette_colors = df_mod_group_compression.ngroup().apply(lambda x: current_palette[x] ).tolist()
    sns_ncolors_percompression = 10 # larger than total count of each algorithm's compression options
    palette_order = [sns.light_palette(color = palette_colors[i],
                       n_colors=sns_ncolors_percompression)[ int(palette_saturation[i]*sns_ncolors_percompression-1) ]
                     for i in range(len(palette_saturation)) ]

    # expand markers list if not of sufficient length
    if len(markers) < len(df_mod_group_compression):
        markers += tuple(set(matplotlib.lines.Line2D.filled_markers) - set(markers))
    marker_order = df_mod_group_compression.ngroup().apply(lambda x: markers[x] ).tolist()

    # format legends
    LegendFormatFunc = lambda x: ''.join(str(i).ljust(12) for i in x)
    legend_title = "Compression Option"
    legend_title = LegendFormatFunc(legend_title.split())
    df[legend_title] = df[legend_columns].astype(str).apply(LegendFormatFunc, axis = 1)
    legend_order = df_mod_group_compression.apply(lambda x: pd.Series(
            sorted(x.astype(str).apply(LegendFormatFunc, axis = 1))  )).values.flatten() # saturation > 0 and < 1

    # remove unused elements from legend_order, palette_order, and marker_order
    legend_present = df[legend_title].unique()
    legend_order,palette_order,marker_order = map(list, zip(*[(i,j,k) for i,j,k in
                                             zip( legend_order,palette_order,marker_order) if i in legend_present]) )

    plot_dpi = 300
    sns.set_style("darkgrid")
    sns.set(rc={'figure.facecolor': '#F8F8F8'})
    sns.set_context(context="notebook", font_scale=1.2)

    # workaround to skip kde plots with single data point
    single_elements = df.groupby(legend_columns).size() == 1
    ###import pdb; pdb.set_trace()
    sns_plot = None
    if not single_elements.any():
        try:
            sns_plot = sns.pairplot(vars=columns, hue = legend_title , hue_order = legend_order,
                                    palette = palette_order, data = df,
                                    markers= marker_order, plot_kws={ "s": 150, "linewidth": 0.05 }, height = 5 )
            # set axis limits
            if ax_lims != None:
                for i in range(len(columns)):
                    sns_plot.axes[i,i].set_xbound( ax_lims[columns[i]] )
                    sns_plot.axes[i,i].set_ybound( ax_lims[columns[i]] )

            for i in range(len(columns)):
                pass
                sns_plot.axes[i,i].set_xbound( ax_lims[columns[i]] )
                sns_plot.axes[i,i].set_ybound( ax_lims[columns[i]] )

            # save plot
            sns_plot.savefig("/home/e/figure.png", dpi=plot_dpi)

        except:
            pass
  
    return sns_plot
