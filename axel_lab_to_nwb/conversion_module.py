# This script grabs Calcium Imaging data stored in .npz files and stores it in
# .nwb files.
# authors: Luiz Tauffer and Ben Dichter
# written for Axel Lab
# ------------------------------------------------------------------------------
from pynwb import NWBFile, NWBHDF5IO, ProcessingModule
from pynwb.ophys import OpticalChannel, ImageSegmentation, DfOverF, TwoPhotonSeries
from pynwb.device import Device
from pynwb.base import TimeSeries
from pynwb.behavior import Position
from ndx_grayscalevolume import GrayscaleVolume

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import cycle
import h5py
import yaml
import numpy as np
import os


def conversion_function(source_paths, f_nwb, metadata, plot_rois=False, **kwargs):
    """
    Copy data stored in a set of .npz files to a single NWB file.

    Parameters
    ----------
    source_paths : dict
        Dictionary with paths to source files/directories. e.g.:
        {'processed data': {'type': 'file', 'path': ''},
         'sparse matrix': {'type': 'file', 'path': ''},
         'ref image',: {'type': 'file', 'path': ''}}
    f_nwb : str
        Path to output NWB file, e.g. 'my_file.nwb'.
    metadata : dict
        Metadata dictionary
    plot_rois : bool
        Plot ROIs
    **kwargs : key, value pairs
        Extra keyword arguments, e.g.:
        {'raw': False, 'processed': True, 'behavior': True}
    """

    # Optional keywords
    add_raw = False
    add_processed = False
    add_behavior = False
    for key, value in kwargs.items():
        if key == 'raw':
            add_raw = kwargs[key]
        if key == 'processed':
            add_processed = kwargs[key]
        if key == 'behavior':
            add_behavior= kwargs[key]

    # Source files
    file_raw = None
    file_processed_1 = None
    file_processed_2 = None
    file_processed_3 = None
    for k, v in source_paths.items():
        if source_paths[k]['path'] != '':
            fname = source_paths[k]['path']
            if k == 'raw data':
                file_raw = h5py.File(fname, 'r')
            if k == 'processed data':
                file_processed_1 = np.load(fname)
            if k == 'sparse matrix':
                file_processed_2 = np.load(fname)
            if k == 'ref image':
                file_processed_3 = np.load(fname)

    # Initialize a NWB object
    nwb = NWBFile(**metadata['NWBFile'])

    # Create and add device
    device = Device(name=metadata['Ophys']['Device'][0]['name'])
    nwb.add_device(device)

    # Creates one Imaging Plane for each channel
    fs = 1. / (file_processed_1['time'][0][1]-file_processed_1['time'][0][0])
    for meta_ip in metadata['Ophys']['ImagingPlane']:
        # Optical channel
        opt_ch = OpticalChannel(
            name=meta_ip['optical_channel'][0]['name'],
            description=meta_ip['optical_channel'][0]['description'],
            emission_lambda=meta_ip['optical_channel'][0]['emission_lambda']
        )
        imaging_plane = nwb.create_imaging_plane(
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
        raw_options = file['options']
        for meta_tps in metadata['Ophys']['TwoPhotonSeries']:
            if meta_tps['name'][-1] == 'R':
                raw_data = file['R']
            else:
                raw_data = file['Y']
            # Change dimensions from (X,Y,Z,T) in mat file to (T,X,Y,Z) nwb standard
            raw_data = np.moveaxis(raw_data, -1, 0)
            tps = TwoPhotonSeries(
                name=meta_tps['name'],
                imaging_plane=nwb.imaging_planes[meta_tps['imaging_plane']],
                data=raw_data
            )

    # Processed data
    if add_processed:
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
        indices = file_processed_2['indices']
        indptr = file_processed_2['indptr']
        dims = np.squeeze(file_processed_1['dims'])
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
        nCells = file_processed_1['dFF'].shape[0]
        roi_region = ps.create_roi_table_region(
            description='RoiTableRegion',
            region=list(range(nCells))
        )

        # create ROI response series
        dff_data = file_processed_1['dFF']
        tt = file_processed_1['time'].ravel()
        meta_rrs = metadata['Ophys']['DfOverF']['roi_response_series'][0]
        meta_rrs['data'] = dff_data.T
        meta_rrs['rois'] = roi_region
        meta_rrs['timestamps'] = tt
        dff.create_roi_response_series(**meta_rrs)

        # Creates GrayscaleVolume containers and add a reference image
        grayscale_volume = GrayscaleVolume(
            name=metadata['Ophys']['GrayscaleVolume']['name'],
            data=file_processed_3['im']
        )
        ophys_module.add(grayscale_volume)

    # Behavior data
    if add_behavior:
        # Ball motion
        behavior_mod = nwb.create_processing_module(
            name='Behavior',
            description='holds processed behavior data',
        )
        meta_ts = metadata['Behavior']['TimeSeries'][0]
        meta_ts['data'] = file_processed_1['ball'].ravel()
        meta_ts['timestamps'] = tt
        behavior_ts = TimeSeries(**meta_ts)
        behavior_mod.add(behavior_ts)

        # Re-arranges spatial data of body-points positions tracking
        pos = file_processed_1['dlc']
        nPoints = 8
        pos_reshaped = pos.reshape((-1, nPoints, 3))  # dims=(nSamples,nPoints,3)

        # Creates a Position object and add one SpatialSeries for each body-point position
        position = Position()
        for i in range(nPoints):
            position.create_spatial_series(name='SpatialSeries_' + str(i),
                                           data=pos_reshaped[:, i, :],
                                           timestamps=tt,
                                           reference_frame='Description defining what the zero-position is.',
                                           conversion=np.nan)
        behavior_mod.add(position)

    # Trial times
    trialFlag = file_processed_1['trialFlag'].ravel()
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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for select, c in zip(range(len(indptr)-1), cycle(['r', 'g', 'k', 'b', 'm', 'w', 'y', 'brown'])):
        x, y, z, _ = np.array(plane_segmentation['voxel_mask'][select]).T
        ax.scatter(x, y, z, c=c, marker='.')
    plt.show()


# If called directly fom terminal
if __name__ == '__main__':
    import sys
    import yaml

    if len(sys.argv) < 6:
        print('Error: Please provide source files, nwb file name and metafile.')

    f1 = sys.argv[1]
    f2 = sys.argv[2]
    f3 = sys.argv[3]
    source_paths = {
        'processed data': {'type': 'file', 'path': f1},
        'sparse matrix': {'type': 'file', 'path': f2},
        'ref image': {'type': 'file', 'path': f3}
    }
    f_nwb = sys.argv[4]
    metafile = sys.argv[5]
    plot_rois = False

    # Load metadata from YAML file
    metafile = sys.argv[3]
    with open(metafile) as f:
       metadata = yaml.safe_load(f)

    conversion_function(source_paths=source_paths,
                        f_nwb=f_nwb,
                        metadata=metadata,
                        plot_rois=plot_rois)
