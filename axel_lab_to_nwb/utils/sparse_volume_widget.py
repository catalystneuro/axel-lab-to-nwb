import numpy as np
import functools
import ipyvolume.pylab as p3
from ipyvolume import ipv
import ipywidgets

from nwbwidgets.utils.cmaps import linear_transfer_function


def sparse_volume_widget(data, chunk_shape, chunks_written):

    p3.clear()
    fig = p3.figure(width=480, controls=False)
    ipv.style.use('seaborn-whitegrid')
    ipv.style.box_off()
    downscale = 10

    if type(data) == dict:
        if len(data) > 1:
            data = np.maximum(*data.values())
        else:
            data = list(data.values())[0]
    while len(data.shape) > 3:
        data = np.max(data, axis=0)

    chunk_shape = chunk_shape[-len(data.shape):]  # last 3 elements
    chunks_written = chunks_written[:, -len(data.shape):]  # last 3 columns
    chunks_index_shape = np.ceil(np.divide(data.shape, chunk_shape)).astype(int)
    chunks_index_boolean = np.zeros(chunks_index_shape, dtype=bool)
    chunks_index_boolean[tuple(chunks_written.T)] = True
    chunks_quantized_bool = np.kron(chunks_index_boolean, np.ones(chunk_shape, dtype=bool))

    checkered_grid = functools.reduce(lambda x, y: np.logical_xor(x % 2, y % 2),
                                      np.ogrid[0:chunks_index_shape[0],
                                      0:chunks_index_shape[1],
                                      0:chunks_index_shape[2]])
    checkered_grid_quantized = np.kron(checkered_grid, np.ones(chunk_shape))
    chunks_quantized_bool_a = checkered_grid_quantized * chunks_quantized_bool
    chunks_quantized_bool_b = np.logical_not(checkered_grid_quantized) * chunks_quantized_bool
    chunks_quantized_bool_a[0, 0, 0] = True
    chunks_quantized_bool_b[0, 0, 0] = True
    vol1 = p3.volshow(data, controls=False,
                      tf=linear_transfer_function([0.6, 0.6, 0.6], max_opacity=0.1),
                      downscale=downscale)
    vol2 = p3.volshow(chunks_quantized_bool_a, controls=False,
                      tf=linear_transfer_function([0.45, 0.45, 1], max_opacity=0.75),
                      specular_exponent=5, lighting=True, downscale=downscale)
    vol3 = p3.volshow(chunks_quantized_bool_b, controls=False,
                      tf=linear_transfer_function([0.85, 0.75, 0.6], max_opacity=0.75),
                      specular_exponent=5, lighting=True, downscale=downscale)

    widget_opacity_scale = ipywidgets.FloatLogSlider(value=1., min=-4, max=.1, description="Opacity")
    ipywidgets.jslink((vol2, 'opacity_scale'), (widget_opacity_scale, 'value'))
    ipywidgets.jslink((vol3, 'opacity_scale'), (widget_opacity_scale, 'value'))

    widget = ipywidgets.GridBox(children=(fig, widget_opacity_scale))
    widget.layout.justify_content = "center"
    return widget


