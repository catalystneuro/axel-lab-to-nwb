import numpy as np
import functools
import ipyvolume.pylab as p3
import ipywidgets

from nwbwidgets.utils.cmaps import linear_transfer_function


def sparse_volume_widget(data, chunk_shape, chunks_written):
    fig = p3.figure(width=800, controls=False)

    chunks_index_shape = np.ceil(np.divide(data.shape, chunk_shape)).astype(int)
    chunks_index_boolean = np.zeros(chunks_index_shape, dtype=bool)
    chunks_index_boolean[tuple(chunks_written.T)] = True
    chunks_boolean_max = np.max(chunks_index_boolean, axis=0)
    chunks_quantized_bool = np.kron(chunks_boolean_max, np.ones(chunk_shape[1:], dtype=bool))

    checkered_grid = functools.reduce(lambda x, y: np.logical_xor(x % 2, y % 2),
                                      np.ogrid[0:chunks_index_shape[1],
                                      0:chunks_index_shape[2],
                                      0:chunks_index_shape[3]])
    checkered_grid_quantized = np.kron(checkered_grid, np.ones(chunk_shape[1:]))
    chunks_quantized_bool_a = checkered_grid_quantized * chunks_quantized_bool
    chunks_quantized_bool_b = np.logical_not(checkered_grid_quantized) * chunks_quantized_bool

    vol1 = p3.volshow(np.max(data, axis=0), controls=False,
                      tf=linear_transfer_function([0.6, 0.6, 0.6], max_opacity=0.1))
    vol2 = p3.volshow(chunks_quantized_bool_a, controls=False,
                      tf=linear_transfer_function([0.45, 0.45, 1], max_opacity=0.75),
                      specular_exponent=5, lighting=True)
    vol3 = p3.volshow(chunks_quantized_bool_b, controls=False,
                      tf=linear_transfer_function([0.85, 0.75, 0.6], max_opacity=0.75),
                      specular_exponent=5, lighting=True)

    fig.volumes += [vol1, vol2, vol3]

    widget_opacity_scale = ipywidgets.FloatLogSlider(value=1., min=-4, max=.1, description="opacity")
    ipywidgets.jslink((vol2, 'opacity_scale'), (widget_opacity_scale, 'value'))
    ipywidgets.jslink((vol3, 'opacity_scale'), (widget_opacity_scale, 'value'))

    widget = ipywidgets.VBox(children=(fig, widget_opacity_scale))
    return widget


