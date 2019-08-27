# This script grabs Calcium Imaging data stored in .npz files and stores it in
# .nwb files.
# authors: Luiz Tauffer and Ben Dichter
# written for Axel Lab
# ------------------------------------------------------------------------------
from pynwb import NWBFile, NWBHDF5IO, ProcessingModule
from pynwb.ophys import TwoPhotonSeries, OpticalChannel, ImageSegmentation, Fluorescence, DfOverF, MotionCorrection
from pynwb.device import Device
from pynwb.base import TimeSeries
from pynwb.behavior import SpatialSeries, Position
from ndx_grayscalevolume import GrayscaleVolume

from mpl_toolkits.mplot3d import Axes3D
from itertools import cycle

import scipy.io
import numpy as np
import os
import matplotlib.pyplot as plt


def npz_to_nwb(fpath, fnpz, fnwb, info, plot_rois=False):
    """
    Copy data stored in a set of .npz files to a single NWB file.

    Parameters
    ----------
    fpath : str, path
        Directory path to files.
    fnpz : list of str
        List of .npz files names, e.g. ['file1.npz', 'file2.npz', 'file3.npz'].
    fnwb : str
        NWB file name, e.g. 'my_file.nwb'.
    info : dict
        Name:Value pairs of lab/experiment information.
    plot_rois : boolean
        If True plots 3D ROIs, if False skips it.
    """
    # Load data from .npz files
    files_path = fpath

    fname1 = fnpz[0]
    fpath1 = os.path.join(files_path, fname1)
    file1 = np.load(fpath1)

    fname2 = fnpz[1]
    fpath2 = os.path.join(files_path, fname2)
    file2 = np.load(fpath2)

    fname3 = fnpz[2]
    fpath3 = os.path.join(files_path, fname3)
    file3 = np.load(fpath3)

    # Initialize a NWB object
    nwb = NWBFile(
        session_description=info['session_description'],
        identifier=info['identifier'],
        session_start_time=info['session_start_time'],
        experimenter=info['experimenter'],
        lab=info['lab'],
        institution=info['institution'],
        experiment_description=info['experiment_description'],
        session_id=info['session_id'],
    )

    # Create and add device
    device = Device('Device')
    nwb.add_device(device)

    # Create an Imaging Plane
    fs = 1. / (file1['time'][0][1]-file1['time'][0][0])
    tt = file1['time'].ravel()
    optical_channel = OpticalChannel(
        name='OpticalChannel',
        description='2P Optical Channel',
        emission_lambda=510.,
    )
    imaging_plane = nwb.create_imaging_plane(
        name='ImagingPlane',
        optical_channel=optical_channel,
        description='Imaging plane',
        device=device,
        excitation_lambda=488.,
        imaging_rate=fs,
        indicator='NLS-GCaMP6s',
        location='whole central brain',
    )

    # Dimensions
    Xp = file1['dims'][0][0]
    Yp = file1['dims'][0][1]
    Zp = file1['dims'][0][2]
    T = file1['dFF'].shape[1]
    nCells = file1['dFF'].shape[0]

    # Creates ophys ProcessingModule and add to file
    ophys_module = ProcessingModule(
        name='ophys',
        description='contains optical physiology processed data.',
    )
    nwb.add_processing_module(ophys_module)

    # Create Image Segmentation compartment
    img_seg = ImageSegmentation()
    ophys_module.add(img_seg)

    # Create plane segmentation and add ROIs
    ps = img_seg.create_plane_segmentation(
        description='plane segmentation',
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
    dff = DfOverF(name='DfOverF')
    ophys_module.add(dff)

    # create ROI regions
    roi_region = ps.create_roi_table_region(
        description='RoiTableRegion',
        region=list(range(nCells))
    )

    # create ROI response series
    dff_data = file1['dFF']
    dff.create_roi_response_series(
        name='RoiResponseSeries',
        data=dff_data.T,
        unit='NA',
        rois=roi_region,
        timestamps=tt
    )

    # Creates GrayscaleVolume containers and add a reference image
    grayscale_volume = GrayscaleVolume(name='GrayscaleVolume',
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
        name='behavior',
        description='holds processed behavior data',
    )
    behavior_mod.add(TimeSeries(name='ball_motion',
                                data=file1['ball'].ravel(),
                                timestamps=tt,
                                unit='unknown'))

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
    fpath_nwb = os.path.join(fpath, fnwb)
    with NWBHDF5IO(fpath_nwb, mode='w') as io:
        io.write(nwb)
    print('NWB file saved with size: ', os.stat(fpath_nwb).st_size/1e6, ' mb')


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
    for select, c in zip(range(len(indptr)-1),cycle(['r','g','k','b','m','w','y','brown'])):
        x, y, z, _ = np.array(plane_segmentation['voxel_mask'][select]).T
        ax.scatter(x, y, z, c=c, marker='.')
    plt.show()
