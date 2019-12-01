# Plot function for compression results

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import functools

def generate_param_list(compression_opts_list, chunks_list, shuffle_list):
    compression_param_product = itertools.chain.from_iterable(
            itertools.product(compression_opts_list[compression_alg],
                              [compression_alg],
                              chunks_list,
                              shuffle_list)
                              for compression_alg in compression_opts_list)

    return compression_param_product
def plot_grid_seq(df, columns, legend_columns, legend_data=None, ax_lims=None,
                  markers=None, file_name="figure",
                  legend_title=None, color_palette=None):

    sns.set_style("darkgrid")
    sns.set(rc={'figure.facecolor': '#F8F8F8', 'legend.markerscale': 1.5, "axes.edgecolor": "k"})
    sns.set_context(context="notebook", font_scale=1.9,
                    rc={"xtick.labelsize": 19, "ytick.labelsize": 19})
    plot_dpi = 100

    if ax_lims is not None:
        if len(ax_lims) != len(columns):
            raise IndexError("number of ax_lims is not equal to the number of columns")
        # remove data not in axis limits
        df = df[functools.reduce(lambda x, y: x & (df[y] < ax_lims[y][1]) & (df[y] > ax_lims[y][0]), columns, True)]

    # add markers
    if markers is None:
        markers = sns.mpl.lines.Line2D.filled_markers

    if legend_data is not None:
        df_mod = pd.DataFrame([{legend_columns[0]: str(c), legend_columns[1]: str(el)}
                               for c, opts in legend_data.items() for el in opts])
    else:
        df_mod = df[legend_columns]
    df_mod = df_mod.drop_duplicates()

    if legend_title is None:
        legend_title = ''.join(str(lc) for lc in legend_columns)

    if color_palette is None:
        color_palette = sns.color_palette()

    _lightest_col = 0.4  # > 0 and < 1
    df_mod_group_compression = df_mod.sort_values(legend_columns).groupby([legend_columns[0]])
    palette_saturation = df_mod_group_compression.apply(lambda x: pd.Series(_lightest_col +
                    (1 - _lightest_col)*(np.arange(len(x))+1)/len(x))).values.flatten()  # saturation > 0 and < 1
    palette_colors = df_mod_group_compression.ngroup().apply(lambda x: color_palette[x]).tolist()
    sns_ncolors_percompression = 10  # larger than total count of each algorithm's compression options
    palette_order = [sns.light_palette(color=palette_colors[i],
                       n_colors=sns_ncolors_percompression)[int(palette_saturation[i]*sns_ncolors_percompression-1)]
                     for i in range(len(palette_saturation))]

    # expand markers list if not of sufficient length
    if len(markers) < len(df_mod_group_compression):
        markers += tuple(set(sns.mpl.lines.Line2D.filled_markers) - set(markers))
    marker_order = df_mod_group_compression.ngroup().apply(lambda x: markers[x]).tolist()

    # format legends
    LegendFormatFunc = lambda x: ''.join(str(i).ljust(12) for i in x)
    legend_title = LegendFormatFunc(legend_title.split())
    df[legend_title] = df[legend_columns].astype(str).apply(LegendFormatFunc, axis=1)
    legend_order = df_mod_group_compression.apply(lambda x: pd.Series(
            sorted(x.astype(str).apply(LegendFormatFunc, axis=1)))).values.flatten()  # saturation > 0 and < 1

    # remove unused elements from legend_order, palette_order, and marker_order
    legend_present = df[legend_title].unique()
    legend_order, palette_order, marker_order = map(list, zip(*[(i, j, k) for i, j, k in
                                             zip(legend_order, palette_order, marker_order) if i in legend_present]))

    # workaround to skip kde plots with single data point
    single_elements = df.groupby(legend_columns).size() == 1
    sns_plot = None
    if not single_elements.any():
        try:
            sns_plot = sns.pairplot(vars=columns, hue=legend_title, hue_order=legend_order,
                                    palette=palette_order, data=df,
                                    markers=marker_order, plot_kws={"s": 150, "linewidth": 0.05}, height=5)
        except:
            pass

    if sns_plot is not None:
        # set axis limits
        for i in range(len(columns)):
            for j in range(len(columns)):
                if i != j:
                    sns_plot.axes[i,j].axhline(y=1, linewidth=1.5, color='#282828', alpha=1, linestyle="--")
                sns_plot.axes[i,j].axvline(x=1, linewidth=1.5, color='#282828', alpha=1, linestyle="--")
                if i == j and i != 2:
                    sns_plot.axes[i,j].set_xbound([0, sns_plot.axes[i,j].get_xbound()[1]])
                    sns_plot.axes[i,j].set_ybound([0, sns_plot.axes[i,j].get_ybound()[1]])
                    sns_plot.axes[i,j].set_xticks([0, 1, *np.arange(2, sns_plot.axes[i,j].get_xbound()[1], 2)])
                    sns_plot.axes[i,j].set_yticks([0, 1, *np.arange(2, sns_plot.axes[i,j].get_ybound()[1], 2)])

        # set axis limits
        if ax_lims != None:
            for i in range(len(columns)):
                sns_plot.axes[i,i].set_xbound(ax_lims[columns[i]])
                sns_plot.axes[i,i].set_ybound(ax_lims[columns[i]])

        # save plot
        sns_plot.savefig(file_name+".png", dpi=plot_dpi)

    return sns_plot
