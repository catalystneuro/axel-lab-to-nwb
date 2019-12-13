# This script grabs Calcium Imaging data stored in .npz files and stores it in
# .nwb files.
# authors: Luiz Tauffer and Ben Dichter
# written for Axel Lab
# ------------------------------------------------------------------------------
import argparse
import os
import sys
from itertools import cycle

import h5py
import numpy as np
import scipy.io
import yaml
from ndx_grayscalevolume import GrayscaleVolume
from pynwb import NWBFile, NWBHDF5IO, ProcessingModule
from pynwb.base import TimeSeries
from pynwb.behavior import Position
from pynwb.device import Device
from pynwb.ophys import OpticalChannel, ImageSegmentation, DfOverF, TwoPhotonSeries
from hdmf.data_utils import DataChunkIterator


def conversion_function(source_paths, f_nwb, metadata, add_raw=False, add_processed=True,
                        add_behavior=True, plot_rois=False):
    """
    Copy data stored in a set of .npz files to a single NWB file.

    Parameters
    ----------
    source_paths : dict
        Dictionary with paths to source files/directories. e.g.:
        {'raw_data': {'type': 'file', 'path': ''},
         'raw_info': {'type': 'file', 'path': ''}
         'processed_data': {'type': 'file', 'path': ''},
         'sparse_matrix': {'type': 'file', 'path': ''},
         'ref_image',: {'type': 'file', 'path': ''}}
    f_nwb : str
        Path to output NWB file, e.g. 'my_file.nwb'.
    metadata : dict
        Metadata dictionary
    add_raw : bool
        Whether to convert raw data or not.
    add_processed : bool
        Whether to convert processed data or not.
    add_behavior : bool
        Whether to convert behavior data or not.
    plot_rois : bool
        Plot ROIs
    """

    # Source files
    file_raw = None
    file_info = None
    file_processed = None
    file_sparse_matrix = None
    file_reference_image = None
    for k, v in source_paths.items():
        if source_paths[k]['path'] != '':
            fname = source_paths[k]['path']
            if k == 'raw_data':
                file_raw = h5py.File(fname, 'r')
            if k == 'raw_info':
                file_info = scipy.io.loadmat(fname, struct_as_record=False, squeeze_me=True)
            if k == 'processed_data':
                file_processed = np.load(fname)
            if k == 'sparse_matrix':
                file_sparse_matrix = np.load(fname)
            if k == 'ref_image':
                file_reference_image = np.load(fname)

    # Initialize a NWB object
    nwb = NWBFile(**metadata['NWBFile'])

    # Create and add device
    device = Device(name=metadata['Ophys']['Device'][0]['name'])
    nwb.add_device(device)

    # Creates one Imaging Plane for each channel
    fs = 1. / (file_processed['time'][0][1]-file_processed['time'][0][0])
    for meta_ip in metadata['Ophys']['ImagingPlane']:
        # Optical channel
        opt_ch = OpticalChannel(
            name=meta_ip['optical_channel'][0]['name'],
            description=meta_ip['optical_channel'][0]['description'],
            emission_lambda=meta_ip['optical_channel'][0]['emission_lambda']
        )
        nwb.create_imaging_plane(
            name=meta_ip['name'],
            optical_channel=opt_ch,
            description=meta_ip['description'],
            device=device,
            excitation_lambda=meta_ip['excitation_lambda'],
            imaging_rate=fs,
            indicator=meta_ip['indicator'],
            location=meta_ip['location'],
        )

    # Raw optical data
    if add_raw:
        print('Adding raw data...')
        for meta_tps in metadata['Ophys']['TwoPhotonSeries']:
            if meta_tps['name'][-1] == 'R':
                raw_data = file_raw['R']
            else:
                raw_data = file_raw['Y']

            def data_gen(data):
                xl, yl, zl, tl = data.shape
                chunk = 0
                while chunk < tl:
                    val = data[:, :, :, chunk]
                    chunk += 1
                    print(chunk)
                    yield val

            xl, yl, zl, tl = raw_data.shape
            tps_data = DataChunkIterator(data=data_gen(data=raw_data),
                                         iter_axis=0,
                                         maxshape=(tl, xl, yl, zl),
                                         dtype=np.dtype('int16'))

            # Change dimensions from (X,Y,Z,T) in mat file to (T,X,Y,Z) nwb standard
            raw_data = np.moveaxis(raw_data, -1, 0)
            tps = TwoPhotonSeries(
                name=meta_tps['name'],
                imaging_plane=nwb.imaging_planes[meta_tps['imaging_plane']],
                data=tps_data,
                rate=file_info['info'].daq.scanRate
            )
            nwb.add_acquisition(tps)

    # Processed data
    if add_processed:
        print('Adding processed data...')
        ophys_module = ProcessingModule(
            name='Ophys',
            description='contains optical physiology processed data.',
        )
        nwb.add_processing_module(ophys_module)

        # Create Image Segmentation compartment
        img_seg = ImageSegmentation(
            name=metadata['Ophys']['ImageSegmentation']['name']
        )
        ophys_module.add(img_seg)

        # Create plane segmentation and add ROIs
        meta_ps = metadata['Ophys']['ImageSegmentation']['plane_segmentations'][0]
        ps = img_seg.create_plane_segmentation(
            name=meta_ps['name'],
            description=meta_ps['description'],
            imaging_plane=nwb.imaging_planes[meta_ps['imaging_plane']],
        )

        # Add ROIs
        indices = file_sparse_matrix['indices']
        indptr = file_sparse_matrix['indptr']
        dims = np.squeeze(file_processed['dims'])
        for start, stop in zip(indptr, indptr[1:]):
            voxel_mask = make_voxel_mask(indices[start:stop], dims)
            ps.add_roi(voxel_mask=voxel_mask)

        # Visualize 3D voxel masks
        if plot_rois:
            plot_rois_function(plane_segmentation=ps, indptr=indptr)

        # DFF measures
        dff = DfOverF(name=metadata['Ophys']['DfOverF']['name'])
        ophys_module.add(dff)

        # create ROI regions
        n_cells = file_processed['dFF'].shape[0]
        roi_region = ps.create_roi_table_region(
            description='RoiTableRegion',
            region=list(range(n_cells))
        )

        # create ROI response series
        dff_data = file_processed['dFF']
        tt = file_processed['time'].ravel()
        meta_rrs = metadata['Ophys']['DfOverF']['roi_response_series'][0]
        meta_rrs['data'] = dff_data.T
        meta_rrs['rois'] = roi_region
        meta_rrs['timestamps'] = tt
        dff.create_roi_response_series(**meta_rrs)

        # Creates GrayscaleVolume containers and add a reference image
        grayscale_volume = GrayscaleVolume(
            name=metadata['Ophys']['GrayscaleVolume']['name'],
            data=file_reference_image['im']
        )
        ophys_module.add(grayscale_volume)

    # Behavior data
    if add_behavior:
        print('Adding behavior data...')
        # Ball motion
        behavior_mod = nwb.create_processing_module(
            name='Behavior',
            description='holds processed behavior data',
        )
        meta_ts = metadata['Behavior']['TimeSeries'][0]
        meta_ts['data'] = file_processed['ball'].ravel()
        tt = file_processed['time'].ravel()
        meta_ts['timestamps'] = tt
        behavior_ts = TimeSeries(**meta_ts)
        behavior_mod.add(behavior_ts)

        # Re-arranges spatial data of body-points positions tracking
        pos = file_processed['dlc']
        n_points = 8
        pos_reshaped = pos.reshape((-1, n_points, 3))  # dims=(nSamples,n_points,3)

        # Creates a Position object and add one SpatialSeries for each body-point position
        position = Position()
        for i in range(n_points):
            position.create_spatial_series(name='SpatialSeries_' + str(i),
                                           data=pos_reshaped[:, i, :],
                                           timestamps=tt,
                                           reference_frame='Description defining what the zero-position is.',
                                           conversion=np.nan)
        behavior_mod.add(position)

    # Trial times
    trialFlag = file_processed['trialFlag'].ravel()
    trial_inds = np.hstack((0, np.where(np.diff(trialFlag))[0], trialFlag.shape[0]-1))
    trial_times = tt[trial_inds]

    for start, stop in zip(trial_times, trial_times[1:]):
        nwb.add_trial(start_time=start, stop_time=stop)

    # Saves to NWB file
    with NWBHDF5IO(f_nwb, mode='w') as io:
        io.write(nwb)
    print('NWB file saved with size: ', os.stat(f_nwb).st_size/1e6, ' mb')


def make_voxel_mask(indices, dims):
    """
    indices - List with voxels indices, e.g. [64371, 89300, 89301, ..., 3763753, 3763842, 3763843]
    dims - (height, width, depth) in pixels
    """
    voxel_mask = []
    for ind in indices:
        zp = np.floor(ind/(dims[0]*dims[1])).astype('int')
        rest = ind % (dims[0]*dims[1])
        yp = np.floor(rest/dims[0]).astype('int')
        xp = rest % dims[0]
        voxel_mask.append((xp, yp, zp, 1))
    return voxel_mask


def plot_rois_function(plane_segmentation, indptr):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for select, c in zip(range(len(indptr)-1), cycle(['r', 'g', 'k', 'b', 'm', 'w', 'y', 'brown'])):
        x, y, z, _ = np.array(plane_segmentation['voxel_mask'][select]).T
        ax.scatter(x, y, z, c=c, marker='.')
    plt.show()


# If called directly fom terminal
if __name__ == '__main__':
    """
    Usage: python conversion_module.py [raw_data] [raw_info] [processed_data]
    [sparse_matrix] [ref_image] [-add_raw] [-add_processed] [-add_behavior] [-plot_rois]
    """

    parser = argparse.ArgumentParser(description='A package for converting Axel Lab data to the NWB standard.')

    parser.add_argument(
        "raw_data", help="The path to the .mat file holding raw data."
    )
    parser.add_argument(
        "raw_info", help="The path to the .mat file holding raw data info."
    )
    parser.add_argument(
        "processed_data", help="The path to the .npz file holding processed data."
    )
    parser.add_argument(
        "sparse_matrix", help="The path to the .npz file holding sparse matrix data."
    )
    parser.add_argument(
        "ref_image", help="The path to the .npz file holding reference image data."
    )
    parser.add_argument(
        "metafile", help="The path to the metadata YAML file."
    )
    parser.add_argument(
        "output_file", help="Output file to be created."
    )
    parser.add_argument(
        "--add_raw",
        action="store_true",
        default=False,
        help="Whether to add the raw data to the NWB file or not",
    )
    parser.add_argument(
        "--add_processed",
        action="store_true",
        default=False,
        help="Whether to add the processed data to the NWB file or not",
    )
    parser.add_argument(
        "--add_behavior",
        action="store_true",
        default=False,
        help="Whether to add the behavior data to the NWB file or not",
    )
    parser.add_argument(
        "--plot_rois",
        action="store_true",
        default=False,
        help="Whether to plot the ROIs or not",
    )

    if not sys.argv[1:]:
        args = parser.parse_args(["--help"])
    else:
        args = parser.parse_args()

    source_paths = {
        'raw_data': {'type': 'file', 'path': args.raw_data},
        'raw_info': {'type': 'file', 'path': args.raw_info},
        'processed_data': {'type': 'file', 'path': args.processed_data},
        'sparse_matrix': {'type': 'file', 'path': args.sparse_matrix},
        'ref_image': {'type': 'file', 'path': args.ref_image}
    }

    f_nwb = args.output_file

    # Load metadata from YAML file
    metafile = args.metafile
    with open(metafile) as f:
       metadata = yaml.safe_load(f)

    plot_rois = False

    conversion_function(source_paths=source_paths,
                        f_nwb=f_nwb,
                        metadata=metadata,
                        add_raw=args.add_raw,
                        add_processed=args.add_processed,
                        add_behavior=args.add_behavior,
                        plot_rois=args.plot_rois)
