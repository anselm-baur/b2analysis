from b2analysis import Histogram
import numpy as np
import matplotlib.pyplot as plt

data = np.random.normal(0,2,10000)

hist1 = Histogram("data", data, 0, bins=100)
hist2 = Histogram("data", data, 0, range=(-5,5), overflow_bin=True, bins=100)

hist1.plot(histtype="step")
hist2.plot(histtype="step")
plt.show()

