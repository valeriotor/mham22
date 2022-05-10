from ast import walk
import sys
from Lilygo.Recording import Recording
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal as signal


def slim_trace(trace):
    data = {}
    data["ax"] = trace.data["ax"]
    data["ay"] = trace.data["ay"]
    data["az"] = trace.data["az"]
    trace.data = data
    return trace

def magnitude(trace, dataset="accelerometer"):
    prefix = "g" if dataset == "gyroscope" else "a"
    magnitude_x = np.array(trace.data[prefix + "x"].values)
    magnitude_y = np.array(trace.data[prefix + "y"].values)
    magnitude_z = np.array(trace.data[prefix + "z"].values)
    mag = np.sqrt(magnitude_x**2 + magnitude_y**2 + magnitude_z**2)
    return mag


def net_magnitude(trace, dataset="accelerometer"):
    mag = magnitude(trace, dataset)
    return mag - np.mean(mag)

def fft(trace, dataset="accelerometer"):
    x_axis = np.array(trace.data["ax"].timestamps)
    L = len(x_axis)
    L1 = 0
    L2 = L
    x_axis = x_axis[L1:L2]
    fft_freq = scipy.fft.fft(net_magnitude(trace, dataset)[L1:L2])
    L = len(x_axis)
    Ts = np.mean(np.diff(x_axis))
    Fs = 1.0 / Ts
    P2 = abs(fft_freq)
    P1 = P2[0:L//2]
    P1 = 2*P1
    P1[0] = P1[0]/2
    f = Fs*range(L//2)/L
    return f, P1

def compute_energy(frequency, amplitude, start, end):
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

def fits_in_range(range, window):
    if len(range["windows"]) <  3:
        return True
    return window["energy"] < 1.3*range["energy_mean"] and window["energy"] > 0.77*range["energy_mean"]

def add_to_range(range, window):
    range["windows"].append(window)
    range["energy_mean"] = (range["energy_mean"]*0.8 + window["energy"]) / 1.8

def update_ranges(sub_ranges, current_sub_range, window):
    if window["start"] > -0.5:
        if fits_in_range(current_sub_range, window):
            add_to_range(current_sub_range, window)
        else:
            sub_ranges.append(current_sub_range)
            current_sub_range = {"windows":[],"energy_mean":0}
            add_to_range(current_sub_range, window)

def combine_ranges(range1, range2):
    new_mean = (range1["energy_mean"]*len(range1["windows"])+range2["energy_mean"]*len(range2["windows"]))/(len(range1["windows"])+len(range2["windows"]))
    new_range = {"windows": range1["windows"]+range2["windows"], "energy_mean": new_mean}
    return new_range

#filename = "data/trace_{:03d}.json".format(int(sys.argv[1]))
#trace = Recording(filename, no_labels=True, mute=True)

def task(trace):
    activities = []

    f, acc_fft = fft(trace, dataset="accelerometer")
    #_, gyr_fft = fft(trace, dataset="gyroscope")
    mag = net_magnitude(trace)

    low_freq_indexes = []
    low_freq2_indexes = []
    walk_freq_indexes = []
    run_freq_indexes = []
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
        if len(run_freq_indexes) == 0 and freq > 2.55:
            run_freq_indexes.append(i)
        if len(run_freq_indexes) == 1 and freq > 3.45:
            run_freq_indexes.append(i)
            break


    b, a = signal.butter(4, [0.7], fs=200, btype="highpass")
    y = signal.lfilter(b, a, mag)

    # the following are computed over the fft of the accelerometer's energy
    total_energy = compute_energy(f, acc_fft, 0.7, 10)
    walk_energy = compute_energy(f, acc_fft, 1.4, 2.25)
    run_energy = compute_energy(f, acc_fft, 2.55, 3.45)
    low_freq_energy = compute_energy(f, acc_fft, 0.7, 1.15)
    low_freq_energy2 = compute_energy(f, acc_fft, 1.1, 1.55)

    if walk_energy/total_energy > 0.11:
        activities.append(1)

    sub_ranges = []
    current_sub_range = {"windows":[],"energy_mean":0} #the energy mean is slightly weighted towards more recent windows
    even_window = {"start":-1, "start_index":0, "energy":0}
    odd_window = {"start":-0.5, "start_index":0, "energy":0}

    for i, time in enumerate(np.array(trace.data["ax"].timestamps)):
        new_even = (time >= even_window["start"]+5)
        new_odd = (time >= odd_window["start"]+5)
        if new_even or new_odd:
            if new_even:
                if even_window["start"] > -0.5:
                    if fits_in_range(current_sub_range, even_window):
                        add_to_range(current_sub_range, even_window)
                    else:
                        sub_ranges.append(current_sub_range)
                        current_sub_range = {"windows":[],"energy_mean":0}
                        add_to_range(current_sub_range, even_window)
                even_window = {"start":time, "start_index":i, "energy":0}
            if new_odd:
                if odd_window["start"] > -0.5:
                    if fits_in_range(current_sub_range, odd_window):
                        add_to_range(current_sub_range, odd_window)
                    else:
                        sub_ranges.append(current_sub_range)
                        current_sub_range = {"windows":[],"energy_mean":0}
                        add_to_range(current_sub_range, odd_window)
                odd_window = {"start":time, "start_index":i, "energy":0}
        local_energy = y[i]**2
        even_window["energy"] += local_energy
        odd_window["energy"] += local_energy

    sub_ranges.append(current_sub_range)


    # compress some ranges
    compressed_ranges = []
    for i, range in enumerate(sub_ranges):
        if i == 0:
            compressed_ranges.append(range)
        else:
            plus_fac = 1.2 if range["energy_mean"] > 100 else 1.5
            minus_fac = 0.8 if range["energy_mean"] > 100 else 0.66
            if (range["energy_mean"] < plus_fac*compressed_ranges[-1]["energy_mean"] and range["energy_mean"] > minus_fac*compressed_ranges[-1]["energy_mean"]) or (range["energy_mean"] < 5 and compressed_ranges[-1]["energy_mean"] < 10):
                new_mean = (compressed_ranges[-1]["energy_mean"]*len(compressed_ranges[-1]["windows"])+range["energy_mean"]*len(range["windows"]))/(len(compressed_ranges[-1]["windows"])+len(range["windows"]))
                new_range = {"windows": compressed_ranges[-1]["windows"]+range["windows"], "energy_mean": new_mean}
                compressed_ranges[-1] = new_range
            else:
                compressed_ranges.append(range)
    #print("")
    #for range in compressed_ranges:
    #    print(str(range["energy_mean"]) + " " + str(len(range["windows"])))

    walk_energy_ratio = 100*walk_energy/total_energy
    run_energy_ratio = 100*run_energy/total_energy
    low_freq_energy_ratio = 100*low_freq_energy/total_energy
    low_freq_energy2_ratio = 100*low_freq_energy2/total_energy

    #print("Walk Energy Ratio: " + str(walk_energy_ratio))
    #print("Run Energy Ratio: " + str(run_energy_ratio))
    #print("LF Energy Ratio: " + str(low_freq_energy_ratio))
    #print("LF2 Energy Ratio: " + str(low_freq_energy2_ratio))


    max_walk_freq_peak = np.max(acc_fft[walk_freq_indexes[0]:walk_freq_indexes[1]])
    max_run_freq_peak = np.max(acc_fft[run_freq_indexes[0]:run_freq_indexes[1]])
    max_low_freq_peak = np.max(acc_fft[low_freq_indexes[0]:low_freq_indexes[1]])
    max_low_freq2_peak = np.max(acc_fft[low_freq2_indexes[0]:low_freq2_indexes[1]])


    range_height = 0
    low_ranges = []
    medium_low_ranges = []
    medium_ranges = []
    medium_high_ranges = []
    high_ranges = []
    ankle_points = 0
    wrist_points = 0
    may_have_stood = False


    for range in compressed_ranges:
        if range["energy_mean"] < 5 and len(range["windows"]) > 3:
            may_have_stood = True
            break
    for range in compressed_ranges:
        if (may_have_stood and range["energy_mean"] < 50) or (not may_have_stood and range["energy_mean"] < 10):
            if range_height == 1:
                low_ranges[-1] = combine_ranges(low_ranges[-1], range)
            else:
                low_ranges.append(range)
            range_height = 1
        elif range["energy_mean"] < 90:
            if range_height == 2:
                medium_low_ranges[-1] = combine_ranges(medium_low_ranges[-1], range)
            else:
                medium_low_ranges.append(range)
            range_height = 2
        elif range["energy_mean"] < 200:
            if range_height == 3:
                medium_ranges[-1] = combine_ranges(medium_ranges[-1], range)
            else:
                medium_ranges.append(range)
            range_height = 3
        elif range["energy_mean"] < 570:
            if range_height == 4:
                medium_high_ranges[-1] = combine_ranges(medium_high_ranges[-1], range)
            else:
                medium_high_ranges.append(range)
            range_height = 4
        elif range["energy_mean"] < 900:
            if range_height == 5:
                high_ranges[-1] = combine_ranges(high_ranges[-1], range)
            else:
                high_ranges.append(range)
            range_height = 5
            
    band_position = -1

    for range in high_ranges:
        if len(range["windows"]) > 13:
            ankle_points += 1
            activities.append(2)

    for range in medium_high_ranges:
        if len(range["windows"]) > 13:
            if run_energy_ratio < 15 and 1 in activities:
                ankle_points += 1
            elif 2 not in activities and ((1 in activities and run_energy_ratio > walk_energy_ratio) or (run_energy_ratio > 16 and range["energy_mean"] > 450 and max_run_freq_peak > max_walk_freq_peak/2) or 1 not in activities):
                activities.append(2)


    for range in medium_ranges:
        if len(range["windows"]) > 13:
            if 1 in activities:
                if low_freq_energy_ratio < 3.5 and (low_freq_energy2_ratio < 3.5 or 2 not in activities) and max_low_freq_peak < max_walk_freq_peak/5:
                    band_position = 1
                else:
                    wrist_points += 1
            else:
                #activities.append(3)
                break

    

    for range in low_ranges:
        if len(range["windows"]) > 13:
            activities.append(0)
            break


    if band_position == -1:
        if ((low_freq_energy_ratio < 3.5 and (low_freq_energy2_ratio < 3.5 or 2 not in activities)) and (max_low_freq_peak < max_walk_freq_peak/5 and 1 in activities or max_low_freq2_peak < max_run_freq_peak/5 and 2 in activities )) or max_low_freq_peak < max_walk_freq_peak/10:
            band_position = 1
        elif ankle_points >= wrist_points:
            band_position = 2
        else:
            band_position = 0

    #for range in medium_low_ranges:
    #    if len(range["windows"]) > 13:
    #        if band_position == 1:
    #            activities.append(3)
    #        break
    
    for range_type in [low_ranges, medium_low_ranges, medium_ranges, medium_high_ranges]:
        for sub_range in range_type:
            #print("mean " + str(sub_range["energy_mean"]))
            #print(len(sub_range["windows"]))
            start = sub_range["windows"][0]["start_index"]
            end = sub_range["windows"][0-1]["start_index"]
            #print("Time:")
            #print(trace.data["ax"].timestamps[start])
            #print(trace.data["ax"].timestamps[end])
            if len(sub_range["windows"]) > 10 and sub_range["energy_mean"] > 15:
                #print(sub_range["energy_mean"])
                #start = sub_range["windows"][0]["start_index"]
                #end = sub_range["windows"][0-1]["start_index"]
                #print("Time:")
                #print(trace.data["ax"].timestamps[start])
                #print(trace.data["ax"].timestamps[end])
                mean = np.mean(mag[start:end])
                peak = mean
                last_peak_timestamp = -1
                peaks = []
                i = start
                for value in mag[start:end]:
                    timestamp = trace.data["ax"].timestamps[i]
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
        #        print(peaks)
                #print(np.std(np.array(peaks)))
                #print((max(peaks) - min(peaks))/np.mean(np.array(peaks)))
                if np.std(np.array(peaks)) > 0.35 and (max(peaks) - min(peaks))/np.mean(np.array(peaks)) > 1.68:
                    activities.append(3)
                    print((max(peaks) - min(peaks))/np.mean(np.array(peaks)))
                    if 3 not in trace.labels["activities"]:
                        print(np.std(np.array(peaks)))
                    break
        if 3 in activities:
            break
    
    
    #print(max_walk_freq_peak)
    #print(max_low_freq_peak)
#
    print("Activities: " + str(activities))
    print("Band position: " + str(band_position))
    #if 3 in activities and 3 not in trace.labels["activities"]:
    #    print("Oh NOOOOOOOOOOOOOOOOOOOOOOOOOOOo!")
    #elif 3 not in activities and 3 in trace.labels["activities"]:
    #    print("Oh wooooooooooooooooot!")
    