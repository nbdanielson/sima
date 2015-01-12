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
import timeit as ti

def subtract_neuropil(imSet, channel, label, min_distance = 0, max_distance = None, buffer_rois = True,\
        contamination_ratio = 0.5):

    set_trace()

    # pulling out the raw imaging signals (one t-series per sequence per ROI)
    signals = imSet.signals(channel=channel)
    raw_signals = signals[label]['raw']
    rois = ROIList(signals[label]['rois'])

    # Initialize mask (all True).  shape is zyx
    nonROI_mask = []  # one level for each z-plane
    for plane in xrange(len(rois[0].mask)):
        nonROI_mask.append(np.ones(rois[0].mask[plane].todense().shape))
    nonROI_mask = np.array(nonROI_mask)

    # Create the non-ROI mask

    for plane in xrange(len(roi.mask)):
        all_roi_mask = sum([roi.mask[plane].tocsr() for roi in rois]).todense()
        all_roi_mask = all_roi_mask.astype(bool)
        nonROI_mask[plane] -= all_roi_mask

    if buffer_rois:
        for plane in xrange(len(nonROI_mask)):
            inv_mask = ~nonROI_mask[plane].astype(bool)
            for iteration in xrange(min_distance):
                inv_mask = ndimage.binary_dilation(inv_mask)
            nonROI_mask[plane] = (~inv_mask).astype(float)

    neuropil_rois = []
    # One neuropil mask for all ROIs
    if max_distance is None and \
            (min_distance == 0 or buffer_rois is True):
        neuropil_rois.append(ROI(mask=nonROI_mask.astype(bool)))
    # Each ROI has a unique neuropil mask
    else:
        for roi in rois:
            roi_mask = []
            for plane in range(len(roi.mask)):
                roi_mask.append(roi.mask[plane].todense())
            roi_mask = np.array(roi_mask)

            # Subtract the dilated ROI from the neuropil mask if necessary
            if not buffer_rois and min_distance > 0:
                dilated_mask = []
                for plane_mask in roi_mask:
                    dilated_plane = plane_mask
                    for iteration in range(min_distance):
                        dilated_plane = ndimage.binary_dilation(dilated_plane)
                    dilated_mask.append(dilated_plane)
                dilated_mask = np.array(dilated_mask)
                neuropil_mask = nonROI_mask - dilated_mask
                neuropil_mask[neuropil_mask < 0] = 0
            else:
                neuropil_mask = np.copy(nonROI_mask)

            # Apply max distance threshold.
            # Note this procedure enforces neuropil signal is taken only from
            # planes on which the ROI is defined
            if max_distance is not None:
                include_mask = []
                for plane_mask in roi_mask:
                    include_plane = plane_mask
                    for iteration in range(max_distance):
                        include_plane = ndimage.binary_dilation(include_plane)
                    include_mask.append(include_plane)
                include_mask = np.array(include_mask)
                neuropil_mask *= include_mask
            neuropil_rois.append(ROI(mask=neuropil_mask.astype(bool)))
    neuropil_rois = ROIList(neuropil_rois)

    for roi in neuropil_rois:
        assert np.any([m.nnz > 0 for m in roi.mask])

    # Calculate neuropil timecourse
    neuropil_signals = extract_rois(
        imSet, neuropil_rois, channel, remove_overlap=False)['raw']

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
    for seq_idx in xrange(len(raw_signals)):
        if len(neuropil_signals[seq_idx]) == 1:
            neuropil_smoothed[seq_idx] = \
                [neuropil_smoothed[seq_idx][0]] * len(raw_signals[seq_idx])

        sequence_signals = []
        for raw_timecourse, neuropil_timecourse in it.izip(
                raw_signals[seq_idx], neuropil_smoothed[seq_idx]):

            normed_neuropil = neuropil_timecourse * \
                (np.median(raw_timecourse) / np.median(neuropil_timecourse))

            sequence_signals.append(
                raw_timecourse - normed_neuropil + np.median(raw_timecourse))
        corrected_timecourses.append(np.array(sequence_signals))
    return corrected_timecourses

