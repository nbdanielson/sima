"""
Neuropil subtraction algorithm.  A boolean neuropil mask is defined for each
ROI object, and the corresponding neuropil timecourse is calculated.  The
corrected timecourse s(t) is defined as

s(t) = r(t) - a * n(t)

where r is the raw ROI timecourse and a is scalar-valued constant


References
----------
* Gobel and Helmchen. 2007. Physiology. 22: 358-65.
* Kerlin et al. 2010. Neuron. 67(5): 858-71.

"""

from scipy import ndimage
from matplotlib.mlab import dist
import numpy as np
import pandas as pd
import itertools as it
from sima.ROI import ROI, ROIList
from sima.extract import extract_rois
from pudb import set_trace


def subtract_neuropil(imset, channel, label, frame_rate=15, min_distance=0, grid_dim=(3, 3),
                      contamination_ratio = 1, apply_smoothing = False, algorithm="global"):
     # pulling out the raw imaging signals (one t-series per sequence per ROI)
     signals = imset.signals(channel=channel)
     raw_signals = signals[label]['raw']
     rois = ROIList(signals[label]['rois'])
     # Initialize mask (all True).  shape is zyx
     nonROI_mask = []  # one level for each z-plane
     for plane in xrange(len(rois[0].mask)):
         nonROI_mask.append(np.ones(rois[0].mask[plane].todense().shape))
     nonROI_mask = np.array(nonROI_mask)
     # Create the non-ROI mask
     neuropil_rois = []
     # mean_neuropil_levels = np.zeros(grid_dim)
     # mean_roi_levels = []
     all_roi_mask = []

     assert len(rois[0].mask) == 1

     # TODO: Below is incorrect for multi-plane datasets and will fail!
     for plane in xrange(len(rois[0].mask)):
         for roi in rois:
             roi_mask = np.array(roi.mask[plane].todense())
             all_roi_mask.append(roi_mask)
             # mean_roi_levels.append(
             #     np.sum(roi_mask * mean_frame) / np.sum(roi_mask))
         all_roi_mask = sum(all_roi_mask)
         all_roi_mask = all_roi_mask.astype(bool)
         dilated_mask = ndimage.binary_dilation(
             all_roi_mask, iterations=min_distance)
         nonROI_mask[plane] -= dilated_mask

     # SPATIAL MASK: LOOK FOR AXON SHADOWS
         # Vertically track raw signals (rois x frames) for unlikely events
     set_trace()
     for seq_idx in xrange(len(raw_signals)):
         dff = np.apply_along_axis(lambda x: (x - np.mean(x))/np.mean(x),axis=1, arr=raw_signals[seq_idx])
         binary_dff = np.apply_along_axis(lambda x: x >= np.mean(x), axis = 1, arr=dff)
         highs_in_column = binary_dff.sum(axis=0)
         #if more than half of ROIs are 1, neuropil event; array of bool indicating which
         half = 0.5*binary_dff.shape[0]
         np_frames = np.where(highs_in_column >= half)[0]
         np_indices = np.where(np.frames)[0]
         # Get rid of nans
         np_frame_list = [np.nan_to_num(imset[seq_idx, idx,:,:,:,:].time_averages)
                 for idx in np_indices]
         np_mean_frame = sum(np_frame_list) / len(np_frame_list)
         np_mask = np.squeeze(np_mean_frame - signals[label]['mean_frame'])
         # Zero negative values
         np_mask = np_mask * (np_mask > 0)

     # Gets coordinates of tile boundaries; bounds right and bottom inclusive
         x = nonROI_mask[plane].shape[0]
         y = nonROI_mask[plane].shape[1]
         x_bounds = map(int, np.linspace(-1, x, grid_dim[0] + 1))
         y_bounds = map(int, np.linspace(-1, y, grid_dim[1] + 1))
         grid = [[np.zeros([x, y]) for i in xrange(grid_dim[0])]
                 for j in xrange(grid_dim[1])]
         for i in xrange(len(x_bounds) - 1):
             for j in xrange(len(y_bounds) - 1):
                 grid[i][j][x_bounds[i] + 1:x_bounds[i + 1],
                            y_bounds[j] + 1:y_bounds[j + 1]] = 1
                 neuropil_mask = (grid[i][j] * np_mask)
                 neuropil_rois.append(ROI(mask=neuropil_mask))
                 # mean_neuropil_levels[i, j] = np.sum(
                 #     neuropil_mask * mean_frame) / np.sum(neuropil_mask)
         neuropil_rois = ROIList(neuropil_rois)
        # neuropil_rois = ROIList([ROI(np_mask)])

     # Calculate neuropil timecourse
     neuropil_extracted = extract_rois(
         imset, neuropil_rois, channel, remove_overlap=True)
     neuropil_signals = neuropil_extracted['raw']
     if(apply_smoothing == True):
         for seq_idx in xrange(len(neuropil_signals)):
            df = pd.DataFrame(neuropil_signals[seq_idx].T)
            WINDOW_SIZE = int(1.5*frame_rate)
            SIGMA = frame_rate/1.5
            neuropil_smoothed.append(pd.stats.moments.rolling_window(
              df, window=WINDOW_SIZE, min_periods=WINDOW_SIZE / 2., center=True,
              win_type='gaussian', std=SIGMA).values.T)
     else:
        neuropil_smoothed = neuropil_signals

     # should be a list of 2D numpy arrays (rois x time)
     corrected_timecourses = []
     # Calculate centroid of neuropil ROIs
     neuropil_centroids = [[((x_bounds[i] + x_bounds[i + 1]) / 2.0,
                             (y_bounds[j] + y_bounds[j + 1]) / 2.0)
                           for j in xrange(len(y_bounds) - 1)]
                           for i in xrange(len(x_bounds) - 1)]

     neuropil_centroids = np.array(neuropil_centroids)
     roi_centroids = []
     for roi in rois:
         roi_centroid = np.array(roi.polygons.centroid.coords)[0]
         roi_centroids.append(roi_centroid)

  #  f = np.load('/home/nathan/neuropil_vars.npz')
  #  raw_signals = f['raw_signals']
  #  roi_centroids = f['roi_centroids']
  #  neuropil_signals = f['neuropil_signals']
  #  neuropil_smoothed = f['neuropil_smoothed']
  #  neuropil_centroids = f['neuropil_centroids']

    # Calculate correction factors
     corrected_signals = []
     for seq_idx in xrange(len(raw_signals)):
         sequence_signals = []
         for raw_timecourse, roi_centroid in it.izip(
                 raw_signals[seq_idx], roi_centroids):
             distances = np.zeros(grid_dim)
             for i in xrange(grid_dim[0]):
                 for j in xrange(grid_dim[1]):
                     distances[i, j] = dist(
                         roi_centroid, neuropil_centroids[i, j])
             raw_weights = 1.0 / (1 + distances ** 2)
             weights = raw_weights / raw_weights.sum()
             # weights *= mean_neuropil_levels
             correction = np.array(neuropil_smoothed[seq_idx]).reshape(
                 grid_dim + (len(neuropil_smoothed[seq_idx][0, :]),), order='F')
             weighted_correction = map(lambda x: weights * np.squeeze(x),
                 np.dsplit(correction, len(neuropil_smoothed[seq_idx][0, :])))
             neuropil_timecourse = np.array([correction_frame.sum() \
                     for correction_frame in weighted_correction])
             neuropil_timecourse = neuropil_timecourse / np.mean(neuropil_timecourse) - 1
             corrected_timecourse = raw_timecourse - 0.05*neuropil_timecourse
             sequence_signals.append(corrected_timecourse)
         sequence_signals.append(np.array(corrected_signals))
     return sequence_signals


