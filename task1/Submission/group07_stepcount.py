# This is the template for the submission. If you want, you can develop your algorithm in a Jupyter Notebook and copy the code here for submission. Don't forget to test according to specification below

# Team members (e-mail, legi):
# examplestudent1@ethz.ch, 12-345-678
# examplestudent2@ethz.ch, 12-345-679
# examplestudent3@ethz.ch, 12-345-670

import sys
from Lilygo.Recording import Recording
from Lilygo.Dataset import Dataset

filename = sys.argv[1] # e.g. 'data/someDataTrace.json'
print(filename)
# IMPORTANT: To allow grading, the two arguments no_labels and mute must be set True in the constructor when loading the data
trace = Recording(filename, no_labels=True, mute=True)
stepCount = 0 # <- here goes your detected stepcount
#print(trace.data["ax"].timestamps)

import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
import math

x_axis = np.array(trace.data["ax"].timestamps)
magnitude_x = np.array(trace.data["ax"].values)
magnitude_y = np.array(trace.data["ay"].values)
magnitude_z = np.array(trace.data["az"].values)
magnitude = np.sqrt(magnitude_x**2 + magnitude_y**2 + magnitude_z**2)
avg_magnitude = np.mean(magnitude)
net_magnitude = magnitude - avg_magnitude

# Phase 1: Filter

b, a = signal.butter(12, 4, fs=250)
y = signal.lfilter(b, a, net_magnitude)

# PHASE 2: iterate over everything, find peaks and valleys and combine into steps
if len(x_axis) < 4:
    exit(1)
    
MAX_STEP_LENGTH = 1.6
halfstep = None # (int: x1, int: x2, bool: upwards) if upwards: valley then peak, otherwise opposite
curr_walk = [] # list of steps. step: (int: x1, int: x2, int: x3, bool: VPV) VPV=> valley peak valley, otherwise PVP
recent_valley_indexes = []
recent_peak_indexes = []
walk_list = []
for i in range(1, len(x_axis)-2):

    height = y[i]
    type = 2 # 0 for peak, 1 for valley, 2 for neither
    if y[i-1] < height and y[i+1] < height:
        type = 0
    elif y[i-1] > height and y[i+1] > height:
        type = 1
    if type == 2:
         continue
    timestamp = x_axis[i]
    if len(recent_valley_indexes) > 0:
        oldest_acceptable_valley_index = -1
        for index_index, index in enumerate(recent_valley_indexes):
            if timestamp-x_axis[index] < MAX_STEP_LENGTH:
                oldest_acceptable_valley_index = index_index
                break
        if oldest_acceptable_valley_index >= 0:
            recent_valley_indexes = recent_valley_indexes[oldest_acceptable_valley_index:]
        else:
            recent_valley_indexes = []

    if len(recent_peak_indexes) > 0:
        oldest_acceptable_peak_index = -1
        for index_index, index in enumerate(recent_peak_indexes):
            if timestamp-x_axis[index] < MAX_STEP_LENGTH:
                oldest_acceptable_peak_index = index_index
                break
        if oldest_acceptable_peak_index >= 0:
            recent_peak_indexes = recent_peak_indexes[oldest_acceptable_peak_index:]
        else:
            recent_peak_indexes = []

    if halfstep != None and timestamp-x_axis[halfstep[0]] > MAX_STEP_LENGTH:
        halfstep = None
    if len(curr_walk) > 0 and timestamp-x_axis[curr_walk[-1][0]] > MAX_STEP_LENGTH:
        walk_list.append(curr_walk)
        curr_walk = []

    if type == 0:
        recent_peak_indexes.append(i)
        if len(recent_valley_indexes) > 0:
            last_halfstep_index = 0 if halfstep == None else halfstep[0]
            for j in range(len(recent_valley_indexes)-1, 0, -1):
                if recent_valley_indexes[j] < last_halfstep_index:
                    break
                if len(curr_walk) > 0 and recent_valley_indexes[j] < curr_walk[-1][1]:
                    break
                if height - y[recent_valley_indexes[j]] > 0.08: #might have found a half step
                    if halfstep != None: #does it continue a previous halfstep?
                        if not halfstep[2]: #if it went downwards
                            if i - halfstep[1] < 4*(i-recent_valley_indexes[j]): 
                                # the distance from halfstep valley to new peak should not be disproportionately large compared to the distance between valley and peak (also, halfstep valley and valley might be the same)
                                step = (halfstep[0], halfstep[1], i, False)
                                if curr_walk == None:
                                    curr_walk = []
                                curr_walk.append(step)
                                halfstep = None
                        else: # if it went upwards, but might be longer than expected?
                            if i - halfstep[0] < 4*(i-recent_valley_indexes[j]): 
                                halfstep = (halfstep[0], i, True)
                    elif len(curr_walk) > 0: # does it continue a previous step?
                        step = curr_walk[-1]
                        if step[3]: # if the step ended with a valley
                            if i - step[2] < 4*(i-recent_valley_indexes[j]): 
                                halfstep = (step[2], i, True)
                        else: # if it ended with a peak, but might be longer than expected?
                            if i - step[1] < 3*(i-recent_valley_indexes[j]) and i - step[1] < 4*(step[2]-step[1]): 
                                step = (step[0], step[1], i, False)
                                curr_walk[-1] = step
                    else:
                        halfstep = (recent_valley_indexes[j], i, True)
    else:
        recent_valley_indexes.append(i)
        if len(recent_peak_indexes) > 0:
            last_halfstep_index = 0 if halfstep == None else halfstep[0]
            for j in range(len(recent_peak_indexes)-1, 0, -1):
                if recent_peak_indexes[j] < last_halfstep_index:
                    break
                if len(curr_walk) > 0 and recent_peak_indexes[j] < curr_walk[-1][1]:
                    break
                if y[recent_peak_indexes[j]] - height > 0.08: #might have found a half step
                    if halfstep != None: #does it continue a previous halfstep?
                        if halfstep[2]: #if it went upwards
                            if i - halfstep[1] < 4*(i-recent_peak_indexes[j]): 
                                # the distance from halfstep peak to new valley should not be disproportionately large compared to the distance between peak and valley (also, halfstep peak and peak might be the same)
                                step = (halfstep[0], halfstep[1], i, True)
                                if curr_walk == None:
                                    curr_walk = []
                                curr_walk.append(step)
                                halfstep = None
                        else: # if it went downwards, but might be longer than expected?
                            if i - halfstep[0] < 4*(i-recent_peak_indexes[j]): 
                                halfstep = (halfstep[0], i, False)
                    elif len(curr_walk) > 0: # does it continue a previous step?
                        step = curr_walk[-1]
                        if not step[3]: # if the step ended with a peak
                            if i - step[2] < 4*(i-recent_peak_indexes[j]): 
                                halfstep = (step[2], i, False)
                        else: # if it ended with a valley, but might be longer than expected?
                            if i - step[1] < 3*(i-recent_peak_indexes[j]) and i - step[1] < 4*(step[2]-step[1]): 
                                step = (step[0], step[1], i, True)
                                curr_walk[-1] = step
                    else:
                        halfstep = (recent_peak_indexes[j], i, False)
walk_list.append(curr_walk)

# PHASE 3: Compute energy over 1s long windows intersecting neighbours over 0.5s each, find average over all windows where steps were counted, 
# prune steps with less than half or more than twice the average energy on both their windows
start0 = 0
start1 = -0.5
energies = {0: 0, -0.5: 0}
for i in range(0, len(x_axis)):
    ts = x_axis[i]
    if ts >= start0 + 1:
        start0 = int(ts*2)/2
        energies[start0] = 0
    if ts >= start1 + 1:
        start1 = int(ts*2)/2
        energies[start1] = 0
    energies[start0] += net_magnitude[i]**2
    energies[start1] += net_magnitude[i]**2
step_sum = 0
to_average = []
for walk in walk_list:
    if len(walk) <= 3: continue
    for step in walk:
        ts = int(x_axis[step[0]]*2)/2.0
        to_average.append(ts)
avg = 0
for ts in to_average:
    avg += energies[ts]
divide_factor = len(to_average)
avg = avg / divide_factor
step_sum = 0
for walk in walk_list:
    if len(walk) <= 3: continue
    for i, step in enumerate(walk):
        ts = int(x_axis[step[0]]*2)/2.0
        en1 = energies[ts]
        en2 = energies[ts-0.5]
        if (en1 > avg*0.5 or (ts > 0 and en2 > avg*0.5) or (i == 0 and energies[ts+0.5] > 0.5)) and (en1 < avg*2 or (ts > 0 and en2 < avg*2)): 
            # the i==0 part is because the first step might have less energy, so we compensate by looking at the subsequent window
            step_sum += 1
        
stepCount = step_sum


#Print result as Integer, do not change!
print(stepCount)


# Test this file before submission with the data we provide to you

# 1. In the console or Anaconda Prompt execute:
# python --version

# Output should look something like (displaying your python version, which must be 3.8.x):
# Python 3.8.10
# If not, check your python installation or command 

# 2. In the console execute:
# python [thisfilename.py] path/to/datafile.json

# Output should be an integer corresponding to the step count you calculated
