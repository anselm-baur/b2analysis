from b2analysis import Histogram
import numpy as np

class Test_Histogram():
    def test_init(self):
        data = np.random.rand(100)
        h = Histogram(name="test", data=data, bins=5, range=(0.2, 0.8), overflow_bin=True, lumi=1)
        assert np.sum(h.entries) == 100, "Number of entries not 100!"
        assert h.bin_centers.size == 5, "Number of bin centers not 5!"

    def test_weights(sel):
        data = np.random.rand(100)
        h = Histogram(name="test", data=data, bins=5, range=(0.2, 0.8), weights=np.full(100, 2), overflow_bin=True, lumi=1)
        assert np.sum(h.entries) == 200, "Number of entries not 100!"
        assert h.bin_centers.size == 5, "Number of bin centers not 5!"

    def test_uncert(self):
        data = np.random.rand(9)
        h = Histogram(name="test", data=data, bins=1, range=(0,1), overflow_bin=True, lumi=1)
        assert h.bin_centers.size == 1, "Number of bin centers not 1!"
        assert h.err[0] == 3, f"uncert not 3 (sqrt(9)), but {h.err}"

        h = Histogram(name="test", data=data, bins=1, range=(0,1), weights=np.full(9, 2), overflow_bin=True, lumi=1)
        assert h.bin_centers.size == 1, "Number of bin centers not 1!"
        #we expect the relative error to be unchanged after applying weights
        assert h.err[0] == 6, f"uncert not 6 (2*sqrt(9)), but {h.err}"