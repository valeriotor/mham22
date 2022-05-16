from ast import walk
import sys
from Lilygo.Recording import Recording
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal as signal
import task1
import utils

def __magnitude(trace, dataset="accelerometer"):
    prefix = "g" if dataset == "gyroscope" else "a"
    magnitude_x = np.array(trace.data[prefix + "x"].values)
    magnitude_y = np.array(trace.data[prefix + "y"].values)
    magnitude_z = np.array(trace.data[prefix + "z"].values)
    mag = np.sqrt(magnitude_x**2 + magnitude_y**2 + magnitude_z**2)
    return mag


def __net_magnitude(trace, dataset="accelerometer"):
    mag = __magnitude(trace, dataset)
    return mag - np.mean(mag)

def __fft(trace, dataset="accelerometer"):
    x_axis = np.array(trace.data["ax"].timestamps)
    L = len(x_axis)
    L1 = 0
    L2 = L
    x_axis = x_axis[L1:L2]
    fft_freq = scipy.fft.fft(__net_magnitude(trace, dataset)[L1:L2])
    L = len(x_axis)
    Ts = np.mean(np.diff(x_axis))
    Fs = 1.0 / Ts
    P2 = abs(fft_freq)
    P1 = P2[0:L//2]
    P1 = 2*P1
    P1[0] = P1[0]/2
    f = Fs*range(L//2)/L
    return f, P1

def __compute_energy(frequency, amplitude, start, end):
    started = False
    energy = 0
    for i, freq in enumerate(frequency):
        if started == False and freq > start:
            started = True
        if started == True:
            energy += amplitude[i]
            if freq > end:
                break
    return energy

def __fits_in_range(sub_range, window):
    if len(sub_range["windows"]) <  3:
        return True
    return window["energy"] < 1.3*sub_range["energy_mean"] and window["energy"] > 0.77*sub_range["energy_mean"]

def __add_to_range(sub_range, window):
    sub_range["windows"].append(window)
    sub_range["energy_mean"] = (sub_range["energy_mean"]*0.8 + window["energy"]) / 1.8

def __combine_ranges(range1, range2):
    """
    Outputs a range whose window list is the concatenation of the window lists of the 2 inputs and the energy mean is the average of the energy means of the
    inputs weighted on their number of windows.
    """
    new_mean = (range1["energy_mean"]*len(range1["windows"])+range2["energy_mean"]*len(range2["windows"]))/(len(range1["windows"])+len(range2["windows"]))
    new_range = {"windows": range1["windows"]+range2["windows"], "energy_mean": new_mean}
    return new_range

def __find_frequency_interval_indexes(f, low_freq_indexes, low_freq2_indexes, walk_freq_indexes, run_freq_indexes):
    """
    Finds the integer indexes corresponding to the start and end of the various frequency intervals we look at in the time domain
    """
    for i, freq in enumerate(f):
        if len(low_freq_indexes) == 0 and freq > 0.7:
            low_freq_indexes.append(i)
        if len(low_freq_indexes) == 1 and freq > 1.15:
            low_freq_indexes.append(i)
        if len(low_freq2_indexes) == 0 and freq > 1.1:
            low_freq2_indexes.append(i)
        if len(low_freq2_indexes) == 1 and freq > 1.55:
            low_freq2_indexes.append(i)
        if len(walk_freq_indexes) == 0 and freq > 1.4:
            walk_freq_indexes.append(i)
        if len(walk_freq_indexes) == 1 and freq > 2.25:
            walk_freq_indexes.append(i)
        if len(run_freq_indexes) == 0 and freq > 2.45:
            run_freq_indexes.append(i)
        if len(run_freq_indexes) == 1 and freq > 3.45:
            run_freq_indexes.append(i)
            break


def __compute_sub_ranges(timestamps, y):
    """
    Splits the time domain in a list of ranges. Iterates over the filtered magnitude and takes 5 second windows,
    """
    sub_ranges = [] # a range is a list of windows and a numeric energy mean
    current_sub_range = {"windows":[],"energy_mean":0} #the energy mean is slightly weighted towards more recent windows
    even_window = {"start":-1, "start_index":0, "energy":0}
    odd_window = {"start":-0.5, "start_index":0, "energy":0}
    WINDOW_LENGTH = 5

    for i, time in enumerate(np.array(timestamps)):
        new_even = (time >= even_window["start"]+WINDOW_LENGTH)
        new_odd = (time >= odd_window["start"]+WINDOW_LENGTH)
        if new_even or new_odd:
            if new_even:
                if even_window["start"] > -0.5:
                    if __fits_in_range(current_sub_range, even_window):
                        __add_to_range(current_sub_range, even_window)
                    else:
                        sub_ranges.append(current_sub_range)
                        current_sub_range = {"windows":[],"energy_mean":0}
                        __add_to_range(current_sub_range, even_window)
                even_window = {"start":time, "start_index":i, "energy":0}
            if new_odd:
                if odd_window["start"] > -0.5:
                    if __fits_in_range(current_sub_range, odd_window):
                        __add_to_range(current_sub_range, odd_window)
                    else:
                        sub_ranges.append(current_sub_range)
                        current_sub_range = {"windows":[],"energy_mean":0}
                        __add_to_range(current_sub_range, odd_window)
                odd_window = {"start":time, "start_index":i, "energy":0}
        local_energy = y[i]**2
        even_window["energy"] += local_energy
        odd_window["energy"] += local_energy

    sub_ranges.append(current_sub_range)
    return sub_ranges

def __compress_ranges(sub_ranges):
    """
    Compresses some consecutive ranges with similar energies
    """
    compressed_ranges = []
    for i, sub_range in enumerate(sub_ranges):
        if i == 0:
            compressed_ranges.append(sub_range)
        else:
            plus_fac = 1.2 if sub_range["energy_mean"] > 100 else 1.5 # more lenient with low energy ranges
            minus_fac = 0.8 if sub_range["energy_mean"] > 100 else 0.66
            if (sub_range["energy_mean"] < plus_fac*compressed_ranges[-1]["energy_mean"] and sub_range["energy_mean"] > minus_fac*compressed_ranges[-1]["energy_mean"]) or (sub_range["energy_mean"] < 5 and compressed_ranges[-1]["energy_mean"] < 10):
                new_mean = (compressed_ranges[-1]["energy_mean"]*len(compressed_ranges[-1]["windows"])+sub_range["energy_mean"]*len(sub_range["windows"]))/(len(compressed_ranges[-1]["windows"])+len(sub_range["windows"]))
                new_range = {"windows": compressed_ranges[-1]["windows"]+sub_range["windows"], "energy_mean": new_mean}
                compressed_ranges[-1] = new_range
            else:
                compressed_ranges.append(sub_range)
    return compressed_ranges

def __catalogue_ranges(compressed_ranges):
    """
    Divides the ranges into categories depending on their energies
    """
    range_height = 0
    low_ranges = []
    medium_low_ranges = []
    medium_ranges = []
    medium_high_ranges = []
    high_ranges = []
    may_have_stood = False
    
    for sub_range in compressed_ranges:
        if sub_range["energy_mean"] < 5 and len(sub_range["windows"]) > 3:
            may_have_stood = True
            break

    for sub_range in compressed_ranges:
        if (may_have_stood and sub_range["energy_mean"] < 50) or (not may_have_stood and sub_range["energy_mean"] < 10): 
            # if the person may have stood we increase the threshold in the if statement to capture this better, otherwise we leave more space for the medium_low category
            if range_height == 1:
                low_ranges[-1] = __combine_ranges(low_ranges[-1], sub_range)
            else:
                low_ranges.append(sub_range)
            range_height = 1
        elif sub_range["energy_mean"] < 90:
            if range_height == 2:
                medium_low_ranges[-1] = __combine_ranges(medium_low_ranges[-1], sub_range)
            else:
                medium_low_ranges.append(sub_range)
            range_height = 2
        elif sub_range["energy_mean"] < 200:
            if range_height == 3:
                medium_ranges[-1] = __combine_ranges(medium_ranges[-1], sub_range)
            else:
                medium_ranges.append(sub_range)
            range_height = 3
        elif sub_range["energy_mean"] < 570:
            if range_height == 4:
                medium_high_ranges[-1] = __combine_ranges(medium_high_ranges[-1], sub_range)
            else:
                medium_high_ranges.append(sub_range)
            range_height = 4
        elif sub_range["energy_mean"] < 900:
            if range_height == 5:
                high_ranges[-1] = __combine_ranges(high_ranges[-1], sub_range)
            else:
                high_ranges.append(sub_range)
            range_height = 5

    return low_ranges, medium_low_ranges, medium_ranges, medium_high_ranges, high_ranges


#filename = "data/trace_{:03d}.json".format(int(sys.argv[1]))
#trace = Recording(filename, no_labels=True, mute=True)

def all_traces():
    activity_false_positives = [0, 0, 0, 0]
    activity_false_negatives = [0, 0, 0, 0]
    activity_count = [0, 0, 0, 0]
    location_error = {(1, 2) : 0, (2, 1) : 0, (1, 0) : 0, (0, 1) : 0, (2, 0) : 0, (0, 2) : 0}
    path_errors = 0
    for i in range(0, 179):
        filename = "data/trace_{:03d}.json".format(i)
        trace = Recording(filename, no_labels=True, mute=True)
        print("Trace " + str(i))
        confidences_activities = utils.get_activities_confidences(trace)
        confidences_location = utils.get_locations_confidences(trace)
        predicted_path = utils.get_path(trace)        
        activities, band_position = task(trace, confidences_activities, confidences_location)
        if predicted_path != trace.labels["path_idx"]:
            print("Chose path " + str(predicted_path) + " instead of " + str(trace.labels["path_idx"]))
            path_errors += 1
        for i in range(4):
            if i in activities and i not in trace.labels["activities"]:
                activity_false_positives[i] += 1
            elif i not in activities and i in trace.labels["activities"]:
                activity_false_negatives[i] += 1
            if i in trace.labels["activities"]:
                activity_count[i] += 1
        if band_position != trace.labels["board_loc"]:
            location_error[(band_position, trace.labels["board_loc"])] += 1
    for i in range(4):
        print("Activity count for " + str(i) + ": " + str(activity_count[i]))
        print("False negatives: " + str(activity_false_negatives[i]))
        print("False positives" + str(activity_false_positives[i]))
        print("")
    for i in range(3):
        for j in range(3):
            if i != j:
                print("Chose location " + str(i) + " instead of " + str(j) + " " + str(location_error[(i, j)]) + " times.")
    print(str(path_errors) + " path errors")
        

def task(trace, conf1, conf2):
    """
    Returns a tuple containing predicted activities (list), predicted band location (integer)
	Asks as input for confidences in activities (conf1) and location (conf2)
    """
    activities = []
    ankle_points = 0
    wrist_points = 0
    band_position = -1

    f, acc_fft = __fft(trace, dataset="accelerometer")
    timestamps = trace.data["ax"].timestamps
    mag = __net_magnitude(trace)

    low_freq_indexes = []
    low_freq2_indexes = []
    walk_freq_indexes = []
    run_freq_indexes = []
    
    __find_frequency_interval_indexes(f, low_freq_indexes, low_freq2_indexes, walk_freq_indexes, run_freq_indexes)

    b, a = signal.butter(4, [0.7], fs=200, btype="highpass")
    y = signal.lfilter(b, a, mag)

    # the following are computed over the fft of the accelerometer's energy
    total_energy = __compute_energy(f, acc_fft, 0.7, 10)
    walk_energy = __compute_energy(f, acc_fft, 1.4, 2.25)
    run_energy = __compute_energy(f, acc_fft, 2.45, 3.45)
    low_freq_energy = __compute_energy(f, acc_fft, 0.7, 1.15)
    low_freq_energy2 = __compute_energy(f, acc_fft, 1.1, 1.55)

    walk_energy_ratio = 100*walk_energy/total_energy
    run_energy_ratio = 100*run_energy/total_energy
    low_freq_energy_ratio = 100*low_freq_energy/total_energy
    low_freq_energy2_ratio = 100*low_freq_energy2/total_energy

    max_walk_freq_peak = np.max(acc_fft[walk_freq_indexes[0]:walk_freq_indexes[1]])
    max_run_freq_peak = np.max(acc_fft[run_freq_indexes[0]:run_freq_indexes[1]])
    max_low_freq_peak = np.max(acc_fft[low_freq_indexes[0]:low_freq_indexes[1]])
    max_low_freq2_peak = np.max(acc_fft[low_freq2_indexes[0]:low_freq2_indexes[1]])

    # looking at the fft plots this seemed like a reliable way to identify walking
    if walk_energy/total_energy > 0.11:
        activities.append(1)

    sub_ranges = __compute_sub_ranges(timestamps, y)
    compressed_ranges = __compress_ranges(sub_ranges)
    low_ranges, medium_low_ranges, medium_ranges, medium_high_ranges, high_ranges = __catalogue_ranges(compressed_ranges)
            
    # the following are all heuristic approaches trying to mirror the reasoning done mentally when analysing the plots visually

    # any ranges with very high energy probably suggest running as well as band positioning on the ankle
    for sub_range in high_ranges:
        if len(sub_range["windows"]) > 13:
            ankle_points += 1
            activities.append(2)

    # ranges in the medium high range could indicate walking with band on the ankle or running. The second if statement tries to check for various clues in both the frequency and 
    # time domain that could indicate the presence of running, e.g. high energy or peak in the running frequency interval, high subrange energy mean, lack of walking etc. 
    for sub_range in medium_high_ranges:
        if len(sub_range["windows"]) > 13:
            if run_energy_ratio < 15 and 1 in activities:
                ankle_points += 1
            elif 2 not in activities and ((1 in activities and run_energy_ratio > walk_energy_ratio) or (run_energy_ratio > 16 and sub_range["energy_mean"] > 450 and max_run_freq_peak > max_walk_freq_peak/2) or 1 not in activities or max_run_freq_peak > 0.95*max_walk_freq_peak):
                activities.append(2)


    # medium energy ranges usually indicate walking with the band not on the ankle. If we know the person is walking, it's a good time to bet on wrist or belt, while looking 
    # at the frequency domain
    for sub_range in medium_ranges:
        if len(sub_range["windows"]) > 13:
            if 1 in activities:
                if low_freq_energy_ratio < 3.5 and (low_freq_energy2_ratio < 3.5 or 2 not in activities) and max_low_freq_peak < max_walk_freq_peak/5:
                    band_position = 1
                else:
                    wrist_points += 1
            else:
                wrist_points += 1

    # same as above
    for sub_range in medium_low_ranges:
        if len(sub_range["windows"]) > 13:
            if 1 in activities:
                if low_freq_energy_ratio < 3.5 and (low_freq_energy2_ratio < 3.5 or 2 not in activities) and max_low_freq_peak < max_walk_freq_peak/5:
                    band_position = 1
                else:
                    wrist_points += 1
            else:
                wrist_points += 1

    # low energy ranges are almost synonymous with standing
    for sub_range in low_ranges:
        if len(sub_range["windows"]) > 13:
            activities.append(0)
            break
    
    # different approach: looking at the time domain without filtering (but after having removed the total mean) we sometime notice that if both walking and running are present 
    # they sometimes have clearly different means, with the running one being clearly higher up. Here we take a look at 30s windows and check if any have very high mean
    if 1 in activities and 2 not in activities:
        last_window_start = 0
        last_window_start_index = 0
        for i, time in enumerate(timestamps):
            if time > last_window_start + 30:
                mean = np.mean(mag[last_window_start_index:i])
                last_window_start_index = i
                last_window_start = time
                if mean > 0.7:
                    activities.append(2)
                    break

    # deciding band position if not already chosen
    if band_position == -1:
        if ((low_freq_energy_ratio < 3.5 and (low_freq_energy2_ratio < 3.5 or 2 not in activities)) and (max_low_freq_peak < max_walk_freq_peak/5 and 1 in activities or max_low_freq2_peak < max_run_freq_peak/5 and 2 in activities )) or max_low_freq_peak < max_walk_freq_peak/10:
            band_position = 1
        elif ankle_points > wrist_points or (ankle_points == 0 and wrist_points == 0):
            band_position = 2
        else:
            band_position = 0

    # yet another approach: in the time domain cycling ranges often have irregular patterns. This code block looks at the peaks in the various sub_ranges (except the high energy
    # ones) and threshold the standard deviation ofthe peaks' heights and the range between max and min
    for range_type in [low_ranges, medium_low_ranges, medium_ranges, medium_high_ranges]:
        for sub_range in range_type:
            start = sub_range["windows"][0]["start_index"]
            end = sub_range["windows"][-1]["start_index"]
            if len(sub_range["windows"]) > 10 and sub_range["energy_mean"] > 15:
                mean = np.mean(mag[start:end])
                peak = mean
                last_peak_timestamp = -1
                peaks = []
                i = start
                for value in mag[start:end]:
                    timestamp = timestamps[i]
                    if value > peak:
                        if timestamp - last_peak_timestamp > 0.15:
                            peak = value
                            last_peak_timestamp = timestamp
                        elif len(peaks) > 0 and value > peaks[-1]:
                            peaks[-1] = value
                            last_peak_timestamp = timestamp
                    elif value < mean and peak > mean:
                        if peak > mean+0.1:
                            peaks.append(peak)
                        peak = mean
                    i+=1
                if np.std(np.array(peaks)) > 0.35 and (max(peaks) - min(peaks))/np.mean(np.array(peaks)) > 1.68:
                    if 3 not in activities:
                        activities.append(3)
                    break
    
    
    factor1 = 0.5 if 2 not in activities else 0.15
    factor2 = 2 if 2 not in activities else 2.5
    #step_count = task1.step_count(timestamps, mag, factor1, factor2)
    print(activities)
    print(band_position)
    print(conf1)
    if 0 in activities and conf1[0] < 0.22:
        activities.remove(0)
    elif 0 not in activities and conf1[0] > 0.90:
        activities.append(0)
    if 1 in activities and conf1[1] < 0.08:
        activities.remove(1)
    elif 1 not in activities and conf1[1] > 0.90:
        activities.append(1)
    if 2 in activities and conf1[2] < 0.25:
        activities.remove(2)
    elif 2 not in activities and conf1[2] > 0.75:
        activities.append(2)
    if 3 in activities and conf1[3] < 0.70:
        activities.remove(3)
    elif 3 not in activities and conf1[3] >= 0.83:
        activities.append(3)
    print(conf2)
    print("Final activities: " + str(activities))

    max_loc_conf = max(conf2)
    max_loc_conf_ind = conf2.index(max_loc_conf)
    if ((max_loc_conf_ind != band_position and max_loc_conf > 0.80 and conf2[(max_loc_conf_ind+1)%3] < 0.60 and conf2[(max_loc_conf_ind+2)%3] < 0.60)
        or (max_loc_conf_ind != band_position and max_loc_conf > 0.95 and conf2[(max_loc_conf_ind+1)%3] < 0.90 and conf2[(max_loc_conf_ind+2)%3] < 0.90)):
        band_position = max_loc_conf_ind
    print("Final position: " + str(band_position))
    #print("Steps: " + str(step_count))
    return activities, band_position
    