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
    neuropil_mask_params = {'min_distance': 0, 'max_distance': None}
                            # 'buffer_rois': False}
    for param, value in kwargs.iteritems():
        if param in neuropil_mask_params:
            neuropil_mask_params[param] = value

    nonROI_mask = []
    for plane in range(len(rois[0].mask)):
        nonROI_mask.append(np.ones(rois[0].mask[plane].todense().shape))
    nonROI_mask = np.array(nonROI_mask)

    for roi in rois:
        for plane in range(len(roi.mask)):
            nonROI_mask -= roi.mask[plane].todense()
            nonROI_mask[nonROI_mask < 0] = 0

    for roi, raw_signal in it.izip(rois, raw_signal):
        roi_mask = []
        for plane in range(len(roi.mask)):
            roi_mask.append(roi.mask[plane].todense())
        roi_mask = np.array(roi_mask)

        # Subtract the dilated ROI from the neuropil mask
        dilated_mask = []
        for plane_mask in roi_mask:
            dilated_plane = plane_mask
            for iteration in range(neuropil_mask_params['min_distance']):
                dilated_plane = ndimage.binary_dilation(dilated_plane)
            dilated_mask.append(dilated_plane)

        neuropil_mask = nonROI_mask - dilated_mask
        neuropil_mask[neuropil_mask < 0] = 0

        # Apply max distance threshold
        if neuropil_mask_params['max_distance'] is not None:
            include_mask = []
            for plane_mask in roi_mask:
                include_plane = plane_mask
                for iteration in range(neuropil_mask_params['max_distance']):
                    include_plane = ndimage.binary_dilation(include_plane)
            include_mask = np.array(include_mask)
            neuropil_mask *= include_mask

        neuropil_roi = ROIList([ROI(mask=neuropil_mask)])

        # Calculate neuropil timecourse
        neuropil_timecourse = extract_rois(imSet, neuropil_roi, channel)['raw']

        # Iterate over roi pixels to calculate a weights mask
        for plane_idx, plane in enumerate(roi.mask):
            (x_coords, y_coords) = np.where(plane.todense())

            weights = 0.
            for x, y in it.izip(x_coords, y_coords):
                for sequence_idx in xrange(len(raw_signal)):
                    #TODO: MAKE SURE Y, X ORDER IS CORRECT
                    imSet.sequences[sequence_idx][:, plane_idx, y, x, channel]

                    #MAKE A NEW SLICED IMAGING DATASET OF 1 PIXEL, THEN EXTRACT A 1-PIXEL ROI! TO GET TIMECOURSE



        # Extract a contamination timecourse (normalized sum of weights) * neuropil timecourse

        # return a list (of lists?  Need to deal with cycles)