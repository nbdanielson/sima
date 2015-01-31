from neuropil import subtract_neuropil
from sima import ImagingDataset

import matplotlib.pyplot as plt

from pudb import set_trace

TEST_DIR = '/data/Nathan/2photon/acuteRemapping/nd113/12102014/ctxA-002/ctxA-002.sima'
CHANNEL = 'Ch2'
LABEL = 'rois'

imSet = ImagingDataset.load(TEST_DIR)

raw_signals = imSet.signals(channel=CHANNEL)[LABEL]['raw']

subtracted_signals = subtract_neuropil(
    imSet, CHANNEL, LABEL, min_distance=0, grid_dim=(3, 3),
    contamination_ratio = 1)

set_trace()
