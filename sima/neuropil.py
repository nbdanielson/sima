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

#  THIS IS A COMMENT FROM ZHENRUI

from scipy import ndimage
import numpy as np
import pandas as pd
import itertools as it
from sima.ROI import ROI, ROIList
from sima.extract import extract_rois
from pudb import set_trace

def subtract_neuropil(imSet, channel, label, min_distance = 0, grid_dim =  (3,3), contamination_ratio = 0.5):

    # pulling out the raw imaging signals (one t-series per sequence per ROI)
    signals = imSet.signals(channel=channel)
    raw_signals = signals[label]['raw']
    rois = ROIList(signals[label]['rois'])
    #OK
    # Initialize mask (all True).  shape is zyx
    nonROI_mask = []  # one level for each z-plane
    for plane in xrange(len(rois[0].mask)):
        nonROI_mask.append(np.ones(rois[0].mask[plane].todense().shape))
    nonROI_mask = np.array(nonROI_mask)
    #OK
    # Create the non-ROI mask
    neuropil_rois= []
    for plane in xrange(len(rois[0].mask)):
        all_roi_mask = sum([roi.mask[plane].tocsr() for roi in rois]).todense()
        all_roi_mask = all_roi_mask.astype(bool)
        dilated_mask = ndimage.binary_dilation(all_roi_mask, iterations=min_distance)
        nonROI_mask[plane] -= dilated_mask

    # Gets coordinates of ROI boundaries; bounds right inclusive, bottom inclusive
        x = nonROI_mask[plane].shape[0]
        y = nonROI_mask[plane].shape[1]
        x_bounds = map(int,np.linspace(-1, x, grid_dim[0]+1))
        y_bounds = map(int, np.linspace(-1, y, grid_dim[1]+1))
        grid = [[np.zeros([x,y]) for i in xrange(grid_dim[0])] for j in xrange(grid_dim[1])]
        for i in xrange(len(x_bounds)-1):
            for j in xrange(len(y_bounds)-1):
                grid[i][j][x_bounds[i]+1:x_bounds[i+1],y_bounds[j]+1:y_bounds[j+1]] = 1
                neuropil_rois.append(ROI(mask=(grid[i][j]*nonROI_mask[plane]).astype(bool)))
        neuropil_rois = ROIList(neuropil_rois)

    # Calculate neuropil timecourse
    neuropil_signals = extract_rois( \
        imSet, neuropil_rois, channel, remove_overlap=True)['raw']
    set_trace()
    neuropil_smoothed = []
    for seq_idx in xrange(len(neuropil_signals)):
        df = pd.DataFrame(neuropil_signals[seq_idx].T)

        WINDOW_SIZE = 45
        SIGMA = 5
        neuropil_smoothed.append(pd.stats.moments.rolling_window(
            df, window=WINDOW_SIZE, min_periods=WINDOW_SIZE / 2., center=True,
            win_type='gaussian', std=SIGMA).values.T)

    # should be a list of 2D numpy arrays (rois x time)
    corrected_timecourses = []
 # Calculate centroid of neuropil ROIs
    neuropil_centroids = [[((x_bounds[i]+x_bounds[i+1])/2,(y_bounds[i]+y_bounds[i+1])/2)\
            for j in xrange(len(y_bounds)-1)] for i in xrange(len(x_bounds)-1)]
    neuropil_centroids = np.array(neuropil_centroids)
    roi_info = []
    for roi in rois:
        roi_centroid = np.array(roi.MultiPolygon.centroid.coords)
        roi_tile = (np.where(roi_centroid[0] < x_bounds)[0][0], np.where(roi_centroid[1] < y_bounds)[0][0])
        roi_info.append((roi_centroid, roi_tile))
    for seq_idx in xrange(len(raw_signals)):
        sequence_signals = []
        for raw_timecourse, roi in it.izip(raw_signals[seq_idx], roi_info):
            roi_centroid = roi[0]
            roi_tile = roi[1]
            corrected_timecourse = raw_timecourse
            distances = map(lambda x: dist(x,array(roi_centroid)), neuropil_centroids)
            distances = np.array(distances)
            rel_weights = 1/(1+distances**2)
            weights = rel_weights / sum(1/(1+distances**2))
            correction = np.array(neuropil_smoothed[seq_idx]).reshape(grid_dim, order='F')
            weighted_correction <- weights*correction
            corrected_timecourse = raw_timecourse - contamination_ratio*sum(weighted_correction)
            corrected_timecourses.append(np.array(corrected_timecourse))
    return corrected_timecourses

