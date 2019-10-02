# This script grabs Calcium Imaging data stored in .npz files and stores it in
# .nwb files.
# authors: Luiz Tauffer and Ben Dichter
# written for Axel Lab
# ------------------------------------------------------------------------------
from pynwb import NWBFile, NWBHDF5IO, ProcessingModule
from pynwb.ophys import OpticalChannel, ImageSegmentation, DfOverF
from pynwb.device import Device
from pynwb.base import TimeSeries
from pynwb.behavior import Position
from ndx_grayscalevolume import GrayscaleVolume

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import cycle
import yaml
import numpy as np
import os


def conversion_function(source_paths, f_nwb, metafile, **kwargs):
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
    metafile : str
        Path to .yml meta data file
    **kwargs : key, value pairs
        Extra keyword arguments, e.g. {'plot_rois':True}
    """

    plot_rois = False
    for key, value in kwargs.items():
        if key == 'plot_rois':
            plot_rois = kwargs[key]

    # Source files
    file1 = None
    file2 = None
    file3 = None
    for k, v in source_paths.items():
        if source_paths[k]['path'] != '':
            fname = source_paths[k]['path']
            if k == 'processed data':
                file1 = np.load(fname)
            if k == 'sparse matrix':
                file2 = np.load(fname)
            if k == 'ref image':
                file3 = np.load(fname)

    # Load meta data from YAML file
    with open(metafile) as f:
        meta = yaml.safe_load(f)

    # Initialize a NWB object
    nwb = NWBFile(
        session_description=meta['NWBFile']['session_description'],
        identifier=meta['NWBFile']['identifier'],
        session_id=meta['NWBFile']['session_id'],
        session_start_time=meta['NWBFile']['session_start_time'],
        notes=meta['NWBFile']["notes"],
        stimulus_notes=meta['NWBFile']["stimulus_notes"],
        data_collection=meta['NWBFile']["data_collection"],
        experimenter=meta['NWBFile']['experimenter'],
        lab=meta['NWBFile']['lab'],
        institution=meta['NWBFile']['institution'],
        experiment_description=meta['NWBFile']['experiment_description'],
        protocol=meta['NWBFile']["protocol"],
        keywords=meta['NWBFile']["keywords"],
    )

    # Create and add device
    device = Device(name=meta['Ophys']['Device']['name'])
    nwb.add_device(device)

    # Create an Imaging Plane
    fs = 1. / (file1['time'][0][1]-file1['time'][0][0])
    tt = file1['time'].ravel()
    optical_channel = OpticalChannel(
        name=meta['Ophys']['OpticalChannel']['name'],
        description=meta['Ophys']['OpticalChannel']['description'],
        emission_lambda=meta['Ophys']['OpticalChannel']['emission_lambda'],
    )
    imaging_plane = nwb.create_imaging_plane(
        name=meta['Ophys']['ImagingPlane']['name'],
        optical_channel=optical_channel,
        description=meta['Ophys']['ImagingPlane']['description'],
        device=device,
        excitation_lambda=meta['Ophys']['ImagingPlane']['excitation_lambda'],
        imaging_rate=fs,
        indicator=meta['Ophys']['ImagingPlane']['indicator'],
        location=meta['Ophys']['ImagingPlane']['location'],
    )

    nCells = file1['dFF'].shape[0]

    # Creates ophys ProcessingModule and add to file
    ophys_module = ProcessingModule(
        name='Ophys',
        description='contains optical physiology processed data.',
    )
    nwb.add_processing_module(ophys_module)

    # Create Image Segmentation compartment
    img_seg = ImageSegmentation(
        name=meta['Ophys']['ImageSegmentation']['name']
    )
    ophys_module.add(img_seg)

    # Create plane segmentation and add ROIs
    ps = img_seg.create_plane_segmentation(
        name=meta['Ophys']['PlaneSegmentation']['name'],
        description=meta['Ophys']['PlaneSegmentation']['description'],
        imaging_plane=imaging_plane,
    )

    # Call function
    indices = file2['indices']
    indptr = file2['indptr']
    dims = np.squeeze(file1['dims'])
    for start, stop in zip(indptr, indptr[1:]):
        voxel_mask = make_voxel_mask(indices[start:stop], dims)
        ps.add_roi(voxel_mask=voxel_mask)

    # Visualize 3D voxel masks
    if plot_rois:
        plot_rois_function(plane_segmentation=ps, indptr=indptr)

    # DFF measures
    dff = DfOverF(name=meta['Ophys']['DfOverF']['name'])
    ophys_module.add(dff)

    # create ROI regions
    roi_region = ps.create_roi_table_region(
        description='RoiTableRegion',
        region=list(range(nCells))
    )

    # create ROI response series
    dff_data = file1['dFF']
    dff.create_roi_response_series(
        name=meta['Ophys']['RoiResponseSeries']['name'],
        description=meta['Ophys']['RoiResponseSeries']['description'],
        data=dff_data.T,
        unit=meta['Ophys']['RoiResponseSeries']['unit'],
        rois=roi_region,
        timestamps=tt
    )

    # Creates GrayscaleVolume containers and add a reference image
    grayscale_volume = GrayscaleVolume(name=meta['Ophys']['GrayscaleVolume']['name'],
                                       data=file3['im'])
    ophys_module.add(grayscale_volume)

    # Trial times
    tt = file1['time'].ravel()
    trialFlag = file1['trialFlag'].ravel()
    trial_inds = np.hstack((0, np.where(np.diff(trialFlag))[0], trialFlag.shape[0]-1))
    trial_times = tt[trial_inds]

    for start, stop in zip(trial_times, trial_times[1:]):
        nwb.add_trial(start_time=start, stop_time=stop)

    # Behavior data - ball motion
    behavior_mod = nwb.create_processing_module(
        name='Behavior',
        description='holds processed behavior data',
    )
    behavior_ts = TimeSeries(name=meta['Behavior']['TimeSeries']['name'],
                             data=file1['ball'].ravel(),
                             timestamps=tt,
                             unit=meta['Behavior']['TimeSeries']['unit'])
    behavior_mod.add(behavior_ts)

    # Re-arranges spatial data of body-points positions tracking
    pos = file1['dlc']
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
    conversion_function(source_paths=source_paths,
                        f_nwb=f_nwb,
                        metafile=metafile,
                        plot_rois=plot_rois)
