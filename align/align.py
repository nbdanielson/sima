from matplotlib.mlab import dist
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.ndimage.morphology as morph
import pandas as pd
import itertools as it
from sima.ROI import ROI, ROIList
from sima.extract import extract_rois
from pudb import set_trace
from align_testing import *
# Search for macro structures
#imSet
# imSet[channel, frame, z, y, x]

slice0 = np.squeeze(imSet[0,0,0,:,:].time_averages)
bin_slice0 = slice0 > np.nanmean(slice0)

height, width = bin_slice0.shape
# Find cross-frame structures (blood vessels, etc.)
structures = ~bin_slice0
iter = 2
clean = morph.binary_erosion(structures, iterations = iter)

top = np.where(clean[iter,:] > 0)[0]
bottom = np.where(clean[height-iter-1,:] > 0)[0]
left = np.where(clean[:,iter] > 0)[0]
right = np.where(clean[:,width-iter-1] > 0)[0]

sides = {}

sides['left'] = np.array([iter*np.ones(len(left)), left])
sides['right'] = np.array([(width-iter)*np.ones(len(right)), right])
sides['top'] = np.array([top, iter*np.ones(len(top))])
sides['bottom'] = np.array([bottom, (height-iter)*np.ones(len(bottom))])

# Find contiguous blocks - best candidates
# Returns list of blocks sorted by size
def get_points(hi_pixels):
    # Temporarily only consider the line that changes
    pixel_gap = np.diff(hi_pixels)
    pixel_gap = pixel_gap[np.any(pixel_gap != 0, axis = 1)]
    blocks = np.split(hi_pixels, np.where(pixel_gap > 5)[0], axis=1)
    blocks_sorted = sorted(blocks, key=lambda x: x.shape[1],\
        reverse=True)
    blocks_filtered = filter(lambda x: x.shape[1] > 5, blocks_sorted)
    terminals = [{'point':np.array([int(np.median(block[0,:])),
                                    int(np.median(block[1,:]))]),
                  'block':block,
                  'width':len(block)} for block in blocks_filtered]
    return terminals

side_endpoints = [get_points(side) for side in sides.values()]


# List of tuples (slope, point)
lines = []
for i in xrange(4): # 4 sides
    for j in xrange(i+1,4):
        for origin in side_endpoints[i]:
            for destination in side_endpoints[j]:
                Dx, Dy = origin['point'] - destination['point']
                slope = Dy / float(Dx)
                lines.append({'slope':slope, 'origin':origin['point'], 'destination':destination['point']})

# Calculate scores
# No of pixels passed through
# Score, i.e., no of hi
# Stability
set_trace()
for line in lines:
    m = line['slope']
    x0, y0 = line['origin']
    xs = range(width)
    points_x =  [np.array([x,round(m*(x-x0)+y0)]) for x in xs]
    ys = range(height)
    points_y =  [np.array([round((y-y0)/float(m) + x0), y]) for y in ys]
    points = set(map(tuple,points_x+points_y))
    points = np.vstack(map(np.array,points))
    points = np.transpose(points)
    all_points = points[:,np.array(map(lambda x: 0 <= x[0] and x[0] <= width and\
                                               0 <= x[1] and x[1] <= height, points))]
    line['all_points'] = all_points
    line['score'] = np.nansum(np.array([structures[y,x] for x,y in zip(all_points[0,:], all_points[1,:])]))
    line['total_pixels'] = len(all_points[0,:])
    line['confidence'] = line['score'] / float(line['total_pixels'])
