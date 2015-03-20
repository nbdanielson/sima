import numpy as np
import scipy as sp
from sima import ImagingDataset
import matplotlib.pyplot as plt
import scipy.ndimage.morphology as morph
import pandas as pd
import itertools as it
from sima.ROI import ROI, ROIList
from sima.extract import extract_rois
from pudb import set_trace

def show(matrix, **kwargs):
    plt.imshow(matrix, **kwargs)
    plt.show()

imSet = ImagingDataset.load('D1S1.sima')
slice0 = np.squeeze(imSet.time_averages[0])
# slice0 = np.squeeze(imSet[0,0,0,:,:].time_average)
bin_slice0 = slice0 > np.nanmean(slice0)


