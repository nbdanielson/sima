from scipy import ndimage
import numpy as np
import itertools as it
from sima.ROI import ROI, ROIList
from sima.extract import extract_rois
from pudb import set_trace


def subtract_neuropil(imSet, channel, label, kwargs):

    signals = imSet.signals(channel=channel)
    raw_signals = signals[label]['raw']
    rois = ROIList(signals[label]['rois'])

    # Set mask parameter defaults
    params = {'min_distance': 0, 'max_distance': None, 'buffer_rois': True,
              'contamination_ratio': 0.5}

    for param, value in kwargs.iteritems():
        if param in params:
            params[param] = value

    nonROI_mask = []  # one level for each z-plane
    for plane in xrange(len(rois[0].mask)):
        nonROI_mask.append(np.ones(rois[0].mask[plane].todense().shape))
    nonROI_mask = np.array(nonROI_mask)

    # Create the non-ROI mask
    for roi in rois:
        for plane in xrange(len(roi.mask)):
            nonROI_mask[plane] -= roi.mask[plane].todense()
        nonROI_mask[nonROI_mask < 0] = 0
    if params['buffer_rois']:
        for plane in xrange(len(nonROI_mask)):
            inv_mask = ~nonROI_mask[plane].astype(bool)
            for iteration in xrange(params['min_distance']):
                inv_mask = ndimage.binary_dilation(inv_mask)
            nonROI_mask[plane] = (~inv_mask).astype(float)

    neuropil_rois = []
    # One neuropil mask for all ROIs
    if params['max_distance'] is None and \
            (params['min_distance'] == 0 or params['buffer_rois'] is True):
        neuropil_rois.append(ROI(mask=nonROI_mask))
    # Each ROI has a unique neuropil mask
    else:
        for roi in rois:
            roi_mask = []
            for plane in range(len(roi.mask)):
                roi_mask.append(roi.mask[plane].todense())
            roi_mask = np.array(roi_mask)

            # Subtract the dilated ROI from the neuropil mask if necessary
            if not params['buffer_rois'] and params['min_distance'] > 0:
                dilated_mask = []
                for plane_mask in roi_mask:
                    dilated_plane = plane_mask
                    for iteration in range(params['min_distance']):
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
            if params['max_distance'] is not None:
                include_mask = []
                for plane_mask in roi_mask:
                    include_plane = plane_mask
                    for iteration in range(params['max_distance']):
                        include_plane = ndimage.binary_dilation(include_plane)
                    include_mask.append(include_plane)
                include_mask = np.array(include_mask)
                neuropil_mask *= include_mask
            neuropil_rois.append(ROI(mask=neuropil_mask))
    neuropil_rois = ROIList(neuropil_rois)
    # TODO: enforce all neuropil ROIs are not empty?  Will cause extract error

    # Calculate neuropil timecourse
    neuropil_signals = extract_rois(imSet, neuropil_rois, channel)['raw']

    # should be a list of 2D numpy arrays (rois x time)
    corrected_timecourses = []
    for seq_idx in xrange(len(raw_signals)):
        if len(neuropil_signals[seq_idx]) == 1:
            neuropil_signals[seq_idx] = \
                [neuropil_signals[seq_idx][0]] * len(raw_signals[seq_idx])

        sequence_signals = []
        for raw_timecourse, neuropil_timecourse in it.izip(
                raw_signals, neuropil_signals):
            sequence_signals.append(
                raw_timecourse - params['contamination_ratio'] *
                neuropil_timecourse)
        corrected_timecourses.append(np.array(sequence_signals))
    set_trace()
