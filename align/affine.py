import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.ndimage.morphology as morph
from scipy.ndimage.measurements import label, labeled_comprehension, center_of_mass
from pudb import set_trace
from skimage.transform import PiecewiseAffineTransform, warp
import sima
from sima.imaging import ImagingDataset
from sima.ROI import ROIList, ROI, mask2poly, poly2mask
from sima.segment import ca1pc
from sima.misc import TransformError, estimate_array_transform, \
            estimate_coordinate_transform

def naive_align(ref, target):
    """
    Uses the method implemented for single plane datasets in
    ROIBuddy to automatically attempt to fit ref to target, plane
    by plane.

    Parameters:
    ref (numpy 3d array)
    target (numpy 3d array)

    Notes:
    - Assumes exact correspondence of planes (i.e., ith plane in initSet
      corresponds to ith plane in targetSet)
    """
    transforms = []
    # Hold target constant throughout
    # Planewise alignment
    # Automatic; assumes ith plane in ref corresponds to ith plane in target
    # LOW CONFIDENCE
    for ref_plane, target_plane in zip(np.dsplit(ref), np.dsplit(target)):
        transform = estimate_array_transform(ref, target, method='affine')
        transforms.append(transform)
    return transforms

def structure_align(ref, target, erode=2, grid=None):
    """
    Looks for macroscopic structures to align to

    Parameters:
    ref (numpy 3d-array)
    target (numpy 3d-array)
    erode (int): Use binary erosion to clean up sets first. 0 or False for
        no erosion
    grid (tuple, optional): breaks image up into tiles, looks at structures
        in each tile for greater accuracy (approaches piecewise affine transform)

    Notes:
    - Assumes exact correspondence of planes
    - Assumes underlying landmarks, e.g. blood vessels or other macro features
    """

    def feature_dict(labeled, idx):
        feature = {}
        feature['centroid'] = np.array(center_of_mass(labeled, labeled, idx))
        feature['size'] = (labeled == idx).sum()
        feature['index'] = index
        return feature

    transforms = []

    for ref_plane, target_plane in zip(np.dsplit(ref), np.dsplit(target)):
        bin_ref = ref_plane > ref_plane.mean()
        bin_target = target_plane > target_plane.mean()
        if erode:
            bin_ref = morph.binary_erosion(bin_ref, iterations=erode)
            bin_tgt = morph.binary_erosion(bin_target, iterations=erode)
        isct = bin_ref * bin_ref*bin_target
        ref_lbl, ref_features = label(bin_ref)
        tgt_lbl, tgt_features = label(bin_tgt)
        isct_lbl, isct_features = label(isct)
        # Construct dictionaries from the intersection of feature sizes
        # and locations (center of mass)
        tgt_features = [feature_dict(tgt_lbl, idx) for idx in xrange(tgt_features)]
        ref_features = [feature_dict(ref_lbl, idx) for idx in xrange(ref_features)]
        features = [feature_dict(isct_lbl, idx) for idx in xrange(isct_features)]
        features = sorted(features, key=lambda d: d['size'], reverse=True)
        features = filter(lambda d: d['size'] > 16, features)
        # Associate top 5 features in tgt and ref and isct to use as anchors
        anchors = []
        for feature in features[0:4]:
            id_point = feature['centroid']
            tgt_distances = []
            ref_distances = []
            for tgt_feature in tgt_features:
                point = tgt_feature['centroid']
                dist = np.linalg.norm(id_point-point)
                tgt_distances.append({'dist':dist, 'idx':tgt_feature['index']})
            for ref_feature in ref_features:
                point = ref_feature['centroid']
                ref_distances.append({'dist':dist, 'idx':ref_feature['index']})
            ref_idx = min(ref_distances, key=lambda x: x['dist'])['idx']
            tgt_idx = min(tgt_distances, key=lambda x: x['dist'])['idx']
            feature['ref_idx'] = ref_idx
            feature['tgt_idx'] = tgt_idx
            anchors.append(feature)
    return anchors


# TODO: WARP() isn't working right. Figure out what function is used in ROIBuddy
def apply_transform(initSets, targetSet, tf_method, **kwargs):
    """
    Uses the specified method to attempt to fit initSets to targetSet

    Parameters:
    initSet (List of ImagingDataset objects)
    targetSet (A reference ImagingDataset object to transform into)
    tf_method (Function to calculate the transformation to targetSet)

    Note:
    - Any **kwargs will be passed directly to tf_method, if implemented.
    """
    target = np.dsplit(targetSet.time_averages)
    transformed_sets = []
    for initSet in initSets:
        tf_planes = []
        ref = np.dsplit(initSet.time_averages)
        transforms = tf_method(ref, target, **kwargs)
        for ref_plane, tf in zip(ref, transforms):
            tf_plane = warp(ref_plane, tf)
            tf_planes.append(tf_plane)
        transformed_set = np.dstack(tf_planes)
    transformed_sets.append(transformed_set)
    return transformed_sets

