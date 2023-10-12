from b2analysis import Histogram, StackedHistogram
import pytest
import numpy as np


class Test_StackedHistogram():

    def _create_stacked_histogram(self, hist_args={}, lumi=10):
        stacked_hist = StackedHistogram(lumi, "x")
        stacked_hist.add_histogram(Histogram("data1", np.array([0,0,0,1,1,1, 2,2, 3,4,4,5]), lumi, var="x", weights=np.array([1,1,1,1,1,1,3/2,3/2,3,1,1,1]), **hist_args))
        stacked_hist.add_histogram(Histogram("data2", np.array([0,0,1,1,1, 2,2, 3,4,4,4]), lumi/2, var="x", lumi_scale=2, weights=np.array([1,1,1,1,1,3/2,3/2,3,1,1,1]), **hist_args))
        stacked_hist.add_histogram(Histogram("data3", np.array([1,1,1, 2,2, 3]), lumi, var="x", **hist_args))
        stacked_hist.plot()
        hist = stacked_hist.get_hist()
        assert hist.var == "x", "var between histograms mismatch"
        assert hist.lumi == lumi, "lumi not as expected"
        assert np.all(stacked_hist.bin_edges == hist.bin_edges), "bin edges mismatch"
        return hist, None


    def create_stacked_histogram(self, hist_args={}, lumi=10):
        d1 = np.array([0,0,0,1,1,1, 2,2, 3,4,4,5])
        w1 = np.array([1,1,1,1,1,1,3/2,3/2,3,1,1,1])

        d2 = np.array([0,0,1,1,1, 2,2, 3,4,4,4])
        w2 = np.array([1,1,1,1,1,3/2,3/2,3,1,1,1])

        d3 = np.array([1,1,1, 2,2, 3])

        stacked_hist = StackedHistogram(lumi, "x")
        stacked_hist.add_histogram(Histogram("data1", d1, lumi, var="x", weights=w1, **hist_args))
        stacked_hist.add_histogram(Histogram("data2", d2, lumi/2, var="x", lumi_scale=2, weights=w2, **hist_args))
        stacked_hist.add_histogram(Histogram("data3", d3, lumi, var="x", **hist_args))
        stacked_hist.add_data_histogram(Histogram("data", np.array([20,15,17]), err=np.sqrt([20,15,17]), lumi=lumi, bins=stacked_hist.bin_edges, is_hist=True))
        stacked_hist.plot()
        hist = stacked_hist.get_hist()
        data = stacked_hist.get_data_hist()

        assert hist.var == "x", "var between histograms mismatch"
        assert hist.lumi == lumi, "lumi not as expected"
        assert np.all(stacked_hist.bin_edges == hist.bin_edges), "bin edges mismatch"

        return stacked_hist


    def check_uncertainty_overflow_hist(self, u):
        """Check the uncertainty of the stacked hist with overflow bins.
        :param u: bin uncertainties
        :type u: list with size of number of bins
        """
        u0 = np.sqrt((6*(1*1)**2) + (5*(2*1)**2) + (3*(1*1)**2))
        assert np.round(u[0],4)==np.round(u0, 4), f"0. bin magnitude {u[0]} of uncertainties wrong, expected {u0}"
        u1 = np.sqrt((2*(1*3/2)**2) + (2*(2*3/2)**2) + (2*(1*1)**2))
        assert np.round(u[1],4)==np.round(u1, 4), f"0. bin magnitude {u[1]} of uncertainties wrong, expected {u1} "
        u2 = np.sqrt((1*(1*3)**2) + (3*(1*1)**2) + (1*(2*3)**2) + (3*(2*1)**2) + (1*(1*1)**2))
        assert np.round(u[2],4)==np.round(u2, 4), f"0. bin magnitude {u[2]} of uncertainties wrong, expected {u2} "




    def test_overflow_bins(self):
        """Check the number of entries per bin in in an histogram with overflow bins is as expected.
        """
        hist_args = {"overflow_bin": True, "bins": 3, "range": (1,3)}
        stacked_hist  = self.create_stacked_histogram(hist_args=hist_args)
        hist = stacked_hist.get_hist()

        assert hist.entries[0]==19, f"0. bin ({hist.entries[0]}) not matching required number of 19"
        assert hist.entries[1]==11, f"1. bin ({hist.entries[1]}) not matching required number of 11"
        assert hist.entries[2]==19, f"2. bin ({hist.entries[2]}) not matching required number of 11"


    def test_wo_overflow_bins(self):
        """Check the number of entries per bin in in an histogram without overflow bins is as expected.
        """
        hist_args = {"overflow_bin": False, "bins": 3, "range": (1,3)}
        stacked_hist  = self.create_stacked_histogram(hist_args=hist_args)
        hist = stacked_hist.get_hist()

        e = hist.entries
        u = hist.err

        e0 = (3*1 + 3*1*2 + 3*1)
        assert e[0]==e0, f"0. bin ({e[0]}) not matching required number of {e0}"
        assert e[1]==(2*3/2 + 2*3/2*2 + 2*1), f"1. bin ({e[1]}) not matching required number of {(2*3/2 + 2*3/2*2 + 2*1)}"
        assert e[2]==(1*3 + 1*3*2 + 1*1), f"2. bin ({e[2]}) not matching required number of {(1*3/2 + 1*3/2*2 + 1*1)}"


    def test_uncertainty_overflow_bins(self):
        """Check the statistical uncertainty per bin in in an histogram with overflow bins is as large as expected.
        """
        hist_args = {"overflow_bin": True, "bins": 3, "range": (1,3)}
        stacked_hist  = self.create_stacked_histogram(hist_args=hist_args)
        hist = stacked_hist.get_hist()

        self.check_uncertainty_overflow_hist(hist.err)


    def test_serialization(self):
        """Check the serialization of the stacked histogram to make sure we save it correctly
        """
        hist_args = {"overflow_bin": True, "bins": 3, "range": (1,3)}
        stacked_hist  = self.create_stacked_histogram(hist_args=hist_args)

        serial_hist = stacked_hist.serialize()
        assert serial_hist["name"] == "stacked_histogram"
        assert serial_hist["var"] == "x"
        assert serial_hist["bin_edges"] == list(stacked_hist.bin_edges)

        for h in ["data1", "data2", "data3"]:
            assert serial_hist["hists"][h]["entries"] == list(stacked_hist.hists[h].entries)
            assert serial_hist["hists"][h]["err"] == list(stacked_hist.hists[h].err)


    def test_stacked_hist_from_serial(self):
        """Check the crated stacked histogram to make sure we produce it correctly
        """
        hist_args = {"overflow_bin": True, "bins": 3, "range": (1,3)}
        stacked_hist  = self.create_stacked_histogram(hist_args=hist_args)

        serial_hist = stacked_hist.serialize()
        stacked_hist_from_serial = StackedHistogram.from_serial(serial_hist)
        assert stacked_hist_from_serial.name == stacked_hist.name
        assert stacked_hist_from_serial.var == stacked_hist.var
        assert stacked_hist_from_serial.lumi == stacked_hist.lumi
        for hist1, hist2 in zip(stacked_hist.hists, stacked_hist_from_serial.hists):
            assert hist1 == hist2
            assert stacked_hist[hist1].entries == stacked_hist_from_serial[hist2].entries
            assert stacked_hist[hist1].err == stacked_hist_from_serial[hist2].err
            assert stacked_hist[hist1].bin_edges == stacked_hist_from_serial[hist2].bin_edges
















