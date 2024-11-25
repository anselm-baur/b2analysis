from b2analysis import HistogramBase2D, Histogram2D, StackedHistogram2D
import numpy as np



class Test_Histogram2D():
    def test_init(self):
        data = [np.random.rand(100), np.random.rand(100)]
        h = HistogramBase2D(name="test2d", data=data, bins=5)
        assert np.sum(h.entries.flatten()) == 100, "Number of entries not 100!"
        assert h.entries.shape == (5,5), "Shape does not match (5,5)"

        print("hello world")
        print(h.weights.T)


    def test_overflow_bin(self):
        data = [np.random.rand(100), np.random.rand(100)]
        h = HistogramBase2D(name="test2d", data=data, bins=[5, 6], range=[(0.2, 0.8), (0.3, 0.7)], overflow_bin=True)
        assert np.sum(h.entries.flatten()) == 100, "Number of entries not 100!"
        assert h.entries.shape == (5,6), "Shape does not match (5,6)"


    def test_uncert(self):
        data = [4*[0.5] + 4*[1.5] + 4*[2.5] ,
                4*[0.5] + 4*[1.5] + 4*[2.5]]
        h = HistogramBase2D(name="test2d", data=data, bins=[[0,1,2,3],[0,1,2,3]], overflow_bin=True)
        assert np.all(np.diag(h.err) == 2), f"uncertainty is not 2 (sqrt(4), {h.err}"

        # how the uncertainty change with weights, expected: relative uncert stays const.
        h = HistogramBase2D(name="test2d", data=data, bins=[[0,1,2,3],[0,1,2,3]], weights=np.full(12, 2), overflow_bin=True)
        assert  np.all(np.diag(h.err) == 4), f"uncertainty is not 4 (2*sqrt(4), {h.err}"


    def test_rebin(self):
        data = [4*[0.5] + 4*[1.5] + 4*[2.5] ,
                4*[0.5] + 4*[1.5] + 4*[2.5]]
        h = HistogramBase2D(name="test2d", data=data, bins=[[0,1,2,3],[0,1,2,3]], overflow_bin=True)
        entries_before = h.entries
        h.rebin(bin_edges=[[0,2,3],[0,1,3]])
        assert h.entries.sum() == entries_before.sum(), f"entries after rebin not equeal ({entries_before.sum()} != {h.entries.sum()})"
        


class Test_StackedHistogram2D:
    def test_init(self):
        data = [np.random.rand(100), np.random.rand(100)]
        hs = StackedHistogram2D()
        h = Histogram2D(name="test2d_1", data=data, bins=[6, 6], range=[(0.2, 0.8), (0.3, 0.7)], overflow_bin=True, lumi=1)
        hs.add_histogram(h)
        h = Histogram2D(name="test2d_2", data=data, bins=[6, 6], range=[(0.2, 0.8), (0.3, 0.7)], overflow_bin=True, lumi=1)
        hs.add_histogram(h)
        assert np.sum(hs.entries.flatten()) == 200, "Total number of entries is not 200"
        print(h.entries)

    def test_add(self):
        data = [np.random.rand(100), np.random.rand(100)]
        hs1 = StackedHistogram2D()
        h1 = Histogram2D(name="test2d_1", data=data, bins=[6, 6], range=[(0.2, 0.8), (0.3, 0.7)], overflow_bin=True, lumi=1, is_signal=True)
        #print(h.err)
        hs1.add_histogram(h1)
        #print(h.entries)
        #print(hs1.err)
        hs2 = StackedHistogram2D()
        h2 = Histogram2D(name="test2d_2", data=data, bins=[6, 6], range=[(0.2, 0.8), (0.3, 0.7)], overflow_bin=True, lumi=1)
        #print(h.err)
        hs2.add_histogram(h2)
        #print(hs2.err)
        hs1 = hs1 + hs2
        assert np.sum(hs1.entries.flatten()) == 200, "Total number of entries is not 200"
        h_sig = hs1.get_sig_hist()
        h_bkg = hs1.get_bkg_hist()
        assert (h1.entries - h_sig.entries).sum() == 0, "Signal hist entries not equal to initial signal hist"
        assert (h2.entries - h_bkg.entries).sum() == 0, "bkg hist entries not equal to initial bkg hist"
        
        

