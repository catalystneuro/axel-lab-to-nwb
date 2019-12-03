# Plot function for compression results

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import functools
import itertools
from tqdm import tqdm

from IPython.display import clear_output, Markdown, update_display
import pandas as pd
import itertools
import time
import functools
import warnings
warnings.simplefilter('ignore')
import ipyvolume.pylab as p3
from ipyvolume import ipv
from ipywidgets import interact, FloatSlider, jslink, GridBox
from nwbwidgets.utils.cmaps import linear_transfer_function
from .sparse_volume_widget import sparse_volume_widget

class show_data():
    
    def __init__(self, channels, compression_opts_list = None, uncompressed_results = None, show_results = True,
                 sync_volume_plot = False, marker_list = None, results_file = None):

        self.channels = channels
        self.compression_opts_list = compression_opts_list
        self.chunk_opacity = FloatSlider(min=0, max=1, step=0.01,
                                    value=0, description="Data Opacity :",
                                    style={'description_width': 'initial'})
        self.free_space_opacity = FloatSlider(min=0, max=1, step=0.01,
                                    value = 0, description = "Inverted Opacity :",
                                    style={'description_width': 'initial'})
        self.initial_display = True
        self.uncompressed_results = uncompressed_results
        self.show_results = show_results
        self.sync_volume_plot = sync_volume_plot
        self.header_str = ("Chunk Size","Compression","Compression Options",
                           "Shuffle","Write Time (sec)","Read Time (sec)",
                           "File Size (MB)")
        if self.uncompressed_results is not None:
            self.header_str +=  ("Write Time Ratio",
                                 "Read Time Ratio",
                                 "Size Ratio")
        self.header_str += ("Written / Total Chunks Ratio", )

        self.data_frame = pd.DataFrame(columns = self.header_str)

        if results_file != None:
            self.results_file = results_file
        else:
            self.results_file = 'results.tsv'

        if marker_list == None:
            self.marker_list = ["v","d","o","X","s"]
            
        self.display_style = {h: "{:.2f}" for h in self.header_str[4:]}    

        clear_output()
        self.volume_widget = sparse_volume_widget(channels, [1,1,1], np.zeros([1,3], dtype=int))                 
        p3.display(self.volume_widget,display_id="data selection")
        display(Markdown(''), display_id="figure")
        self.sns_plot_last = None

        if self.show_results == True:
            display(Markdown(''), display_id="data frame")


    def update(self, run_nwb_result, compression_opts, compression, chunk_shape, shuffle, channels):
        time_write, time_read, file_size, sparse_written_chunks = run_nwb_result

        # prepare benchmark results
        output_list =  [str(chunk_shape).replace(" ",""),
                        compression,
                        str(compression_opts),
                        str(shuffle),
                        time_write,
                        time_read,
                        file_size]

        # add comp/uncompressed ratios
        if self.uncompressed_results is not None:
            output_ratio = np.array([time_write, time_read, file_size])/self.uncompressed_results
            output_list += list(output_ratio)

        # add ratio of written chunks
        if type(channels) == dict:
            data_shape = list(channels.values())[0].shape
        else:
            data_shape = channels.shape
        if type(sparse_written_chunks) == list and sparse_written_chunks != []:
            sparse_written_chunks = np.vstack(sparse_written_chunks)
            # remove duplicates
            sparse_written_chunks = np.unique( sparse_written_chunks, axis=0 )

        if chunk_shape != None:
            chunks_index_shape = np.ceil( np.divide( data_shape, chunk_shape ) ).astype(int)
            chunk_ratio = len(sparse_written_chunks)/np.prod( chunks_index_shape )
            chunk_shape_display = chunk_shape
        else:
            chunks_index_shape = np.ones(len(data_shape), dtype=int)
            chunk_ratio = 1
            chunk_shape_display = data_shape
        output_list += [chunk_ratio]

        # add results to data frame
        self.data_frame.loc[len(self.data_frame)] = output_list

        if self.sync_volume_plot == True or self.initial_display == True:
            chunks_index_boolean = np.zeros( chunks_index_shape, dtype = bool )
            chunks_index_boolean[tuple(np.array(sparse_written_chunks).T)] = True
            chunks_boolean_max = np.max( chunks_index_boolean, axis = 0)
            chunks_quantized_bool = np.kron( chunks_boolean_max, np.ones( chunk_shape_display[1:], dtype = bool ) )
            checkered_grid = functools.reduce( lambda x,y : np.logical_xor(x%2,y%2),
                                              np.ogrid[0:chunks_index_shape[1],
                                              0:chunks_index_shape[2],
                                              0:chunks_index_shape[3] ] )
            checkered_grid_quantized = np.kron( checkered_grid, np.ones( chunk_shape_display[1:] ) )
            chunks_quantized_bool_a = checkered_grid_quantized*chunks_quantized_bool
            chunks_quantized_bool_b = np.logical_not(checkered_grid_quantized)*chunks_quantized_bool
            free_space_quantized_bool_a = checkered_grid_quantized*np.logical_not(chunks_quantized_bool)
            free_space_quantized_bool_b = np.logical_not(checkered_grid_quantized
                                                        )*np.logical_not(chunks_quantized_bool)
            self.volume_widget.children[0].volumes[1].opacity_scale = 1
            self.volume_widget.children[0].volumes[2].opacity_scale = 1
            if chunk_shape == None:
                self.volume_widget.children[1].value=0
            self.volume_widget.children[0].volumes[1].data = chunks_quantized_bool_a
            self.volume_widget.children[0].volumes[2].data = chunks_quantized_bool_b
            self.initial_display = False

        if self.show_results == True:
            update_display(self.data_frame.style.format(self.display_style).set_properties(width='110px'),
                           display_id="data frame")            

            # store results
            with open(self.results_file, 'a') as output_file:
                output_line = '\t'.join( str(el) for el in output_list)+'\n' 
                output_file.write( output_line )

            # plot results 
            if self.uncompressed_results is None:
                columns = self.header_str[4:7]
            else:
                columns = self.header_str[7:10]
            self.sns_plot = plot_grid_seq(self.data_frame, columns=columns,
                        legend_columns=["Compression","Compression Options"], legend_data=self.compression_opts_list,
                        markers=self.marker_list, legend_title="Compression Option" )

            if self.sns_plot != None:
                update_display(self.sns_plot.fig, display_id="figure")
                if self.sns_plot_last is not None:
                    plt.close(self.sns_plot_last.fig)
                self.sns_plot_last = self.sns_plot

        return self.data_frame

def generate_param_list(compression_opts_list, chunks_list, shuffle_list):
    # Iterator for nested loop of compression variables
    compression_param_product = itertools.chain.from_iterable(
            itertools.product(compression_opts_list[compression_alg],
                              [compression_alg],
                              chunks_list,
                              shuffle_list)
                              for compression_alg in compression_opts_list)
    compression_params_list_withprogressbar = tqdm(list(compression_param_product))
    compression_params_list_withprogressbar.clear()
    return compression_params_list_withprogressbar

def plot_grid_seq(data_frame, columns, legend_columns, legend_data=None, ax_lims=None,
                  markers=None, file_name="figure",
                  legend_title=None, color_palette=None):

    sns.set_style("darkgrid")
    sns.set(rc={'figure.facecolor': '#F8F8F8', 'legend.markerscale': 1.5, "axes.edgecolor": "k"})
    sns.set_context(context="notebook", font_scale=1.9,
                    rc={"xtick.labelsize": 19, "ytick.labelsize": 19})
    plot_dpi = 100

    df = data_frame.copy(deep=True) # copy data_frame for plotting
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
