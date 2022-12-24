from b2analysis import StackedHistogram
import numpy as np
import matplotlib.pyplot as plt

data = np.random.normal(0,2,10000)
background = np.random.normal(1,5,100000)

stacked_hist = StackedHistogram(0, "data+bg")
stacked_hist.create_histogram("background", background, 0, overflow_bin=True, bins=50)
stacked_hist.create_histogram("data", data, 0, overflow_bin=True, bins=50)