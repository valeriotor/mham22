import json
import numpy as np

class Dataset:    
    # name: used for axis names
    # values: list of all sensor values in chronological order
    # timestamps: list of tuples (data_index, timestamp)
    def __init__(self, name, values, timestamps):
        self.title = name # Title for this collections of data
        self.values = values # list of numpy time series (for example [ax, ay az])
        self.raw_timestamps = timestamps # for plotting and processing, list of numpy array of timestamps for the axis
        
        # calculate interpolated timestamps 
        # timestamps go linearly from first time to last time
        # seconds from start

        # seconds between first and last sample
        self.total_time = (self.raw_timestamps[-1][1] - self.raw_timestamps[0][1]) / 1000
        # linearly interpolated offset in seconds from the first sample
        self.timestamps = np.linspace(0, self.total_time, num=len(values))
        
        # Calculate average sample rate
        # samples/s
        self.samplerate = len(self.values) / self.total_time
        
        
    #skip raw timestamp conversion
    #directly pass values and corresponding timestamps
    @classmethod
    def fromLists(cls, name, values, timestamps):
        raw_timestamps = [(n, 1000 * ts) for n, ts in zip(range(len(timestamps)), timestamps)]
        return cls(name, values, raw_timestamps)