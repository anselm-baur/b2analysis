from b2analysis import HistogramBase, StackedHistogram, Histogram
from b2style import B2Figure
import matplotlib.pyplot as plt
import numpy as np
import copy

class HistogramBase2D(HistogramBase):
    """We expect now the data and bins to be a 2D array, first x, second y"""

    @property
    def bin_edges(self):
        return self._bin_edges

    @bin_edges.setter
    def bin_edges(self, value):
        if value is None:
            value = [np.array([]), np.array([])]
        elif isinstance(value, np.ndarray) and value.size==0:
            #make sure we have the right format
            value = [np.array([]), np.array([])]
        #print("parsing bin edges...")
        #print(f"value: {value}")
        self._bin_edges = [self.parse_bin_edges(value[0]),
                           self.parse_bin_edges(value[1])]
        self._update_bins()


    def init_without_overflow_bins(self, data, kwargs):
        np_hist = np.histogram2d(data[0], data[1],  weights=self.weights, **kwargs)
        self.entries = np_hist[0]
        self.bin_edges = [np.array(np_hist[1]), (np_hist[2])]
        #self._update_bins()
        self.err = self.calc_weighted_uncert(data=data, weights=self.weights, bin_edges=self.bin_edges)

    def init_with_overflow_bins(self, data, kwargs):
        #first add to the first and last bins bins which go to -/+ inf
        #then we trim the histograms and add the content of the first and last inf bins
        #to the neighbor bins and cut the inf bins out
        if not "bins" in kwargs:
            kwargs["bins"] = 50
        if isinstance(kwargs["bins"], int):
            _bins = np.array(kwargs["bins"], dtype=int)
        elif len(list(kwargs["bins"]))==2:
            if isinstance(kwargs["bins"][0], int) or isinstance(kwargs["bins"][0], float) \
                and isinstance(kwargs["bins"][1], int) or isinstance(kwargs["bins"][1], float):
                # provided bins are number of bins in each dimension e.g. [10, 5]
                _bins = np.array(kwargs["bins"], dtype=np.float32)
            else:
                # provided bins are two lists with (un)equal elements e.g. [[0,1,2,3],[0,3,4]]
                _bins = np.array(kwargs["bins"], dtype=object)
        else:
            raise RuntimeError("got bad bin format, don't know what to do. use e.g. 3, [3, 4], [[1,2,3], [2, 5, 6, 7]]")

        if isinstance(kwargs["bins"], int) or (_bins.ndim==1 and _bins.size==2 and not _bins.dtype==object):
            if not "range" in kwargs:
                bin_ranges = []
                for i, _data in enumerate(data):
                    if np.min(_data) == np.max(_data):
                        bin_ranges.append((_data[0]-1, _data[0]+1))
                    else:
                        bin_ranges.append((np.min(_data), np.max(_data)))
                kwargs["range"] = bin_ranges
            bins = []
            #print(kwargs["bins"])
            for i in range(2):
                # creae the 2D bin edges
                if (np.array(kwargs["bins"]).ndim==1 and np.array(kwargs["bins"]).size==2):
                    #print("got two bin vals")
                    _bins = kwargs["bins"][i]+1
                else:
                    _bins = kwargs["bins"]+1

                #print(np.linspace(*kwargs["range"][i], _bins))
                bins.append(np.concatenate([[-np.inf],
                                        np.linspace(*kwargs["range"][i], _bins),
                                        [np.inf]]))

        else:
            #print(kwargs["bins"])
            bins = []
            for i in range(2):
                bins.append(np.concatenate([[-np.inf],kwargs["bins"][i],[np.inf]]))
        kwargs["bins"] = bins

        np_hist = np.histogram2d(data[0], data[1],  weights=self.weights, **kwargs)
        self.entries = np_hist[0]
        self.bin_edges = [np.array(np_hist[1]), (np_hist[2])]
        #self._update_bins()
        self.concatinate_bins(new_x_bins=np_hist[1][1:-1] , new_y_bins=np_hist[2][1:-1], overflow_bin=True)
        #print(self.bin_edges)
        self.err = self.calc_weighted_uncert(data=data, weights=self.weights, bin_edges=self.bin_edges)


    def init_from_hist(self, data, kwargs):
        """Create the Histogram object from an existing Histogram, basically we are just copying the values.
        """
        if (not "bins" in kwargs or len(list(kwargs["bins"][0])) != len(list(data[0]))+1) and \
           (not "bins" in kwargs or len(list(kwargs["bins"][1])) != np.array(data).shape[1]+1):
            print(f"failed, len(data)+1 = {len(list(data[0]))+1} == len(bins) = {len(list(kwargs['bins'][0]))}")
            print(f"failed, len(data)+1 = {np.array(data).shape[1]+1} == len(bins) = {len(list(kwargs['bins'][1]))}")
            raise ValueError("bins expectes when is_hist is true, with len(data)+1 == len(bins)!")
        self.bin_edges = np.array(kwargs["bins"])
        self.entries = np.array(data)
        self._update_bins()
        if "err" in kwargs:
            self.err = copy.deepcopy(np.array(kwargs["err"]))
            self.stat_uncert = None #omit wrong uncertainty calculation
        else:
            self.update_hist()
        #self.size = self.entries.size


    def _update_bins(self):
        #self.bin_centers = [np.around(np.array((bin_edges[1:]+bin_edges[:-1])/2, dtype=np.float64), 5) for bin_edges in self.bin_edges]
        self.bin_centers = []
        for i, bin_edges in enumerate(self.bin_edges):
            #print(bin_edges)
            self.bin_centers.append(np.around(np.array((bin_edges[1:]+bin_edges[:-1])/2, dtype=np.float64), 5))
        if self.bin_edges[0].size == 0:
            self.range = [np.array([]), np.array([])]
        else:
            self.range = [(bin_edges[0], bin_edges[-1]) for bin_edges in self.bin_edges]


    def concatinate_bins(self, new_x_bins=None, new_y_bins=None, overflow_bin=False):
        if not self.bins_intersect(new_x_bins=new_x_bins, new_y_bins=new_y_bins):
            raise RuntimeError("new bins don't intersect with the old bins")

        new_entries = np.zeros([new_x_bins.size-1, new_y_bins.size-1])

        _new_x_bins = self.parse_bin_edges(new_x_bins)
        _new_y_bins = self.parse_bin_edges(new_y_bins)

        if overflow_bin:
            _new_x_bins[0] = -np.inf
            _new_x_bins[-1] = +np.inf
            _new_y_bins[0] = -np.inf
            _new_y_bins[-1] = +np.inf
        else:
            _new_x_bins[-1] += 1
            _new_y_bins[-1] += 1

        #print(f"new_x_bins:, {_new_x_bins}")
        #print(f"new_y_bins:, {_new_y_bins}")
        #print("before",  np.sum(self.entries.flatten()))
        #print(self.entries)
        for x in range(new_x_bins.size-1):
            for y in range(new_y_bins.size-1):
                x_min, x_max = _new_x_bins[x], _new_x_bins[x + 1]
                y_min, y_max = _new_y_bins[y], _new_y_bins[y + 1]
                # Find the original bins that fall into the current new bin
                x_indices = np.where((self.bin_edges[0] >= x_min) & (self.bin_edges[0] < x_max))[0]
                y_indices = np.where((self.bin_edges[1] >= y_min) & (self.bin_edges[1] < y_max))[0]
                #print(f"x {x} [{x_min}, {x_max}]: {x_indices}", self.bin_edges[0][:-1])
                #print(f"y {y} [{y_min}, {y_max}]: {y_indices}", self.bin_edges[1][:-1])

                # Sum the contents of these original bins
                if len(x_indices) > 0 and len(y_indices) > 0:
                    for xi in x_indices:
                        for yj in y_indices:
                            new_entries[x, y] += self.entries[xi, yj]
                            #print(new_entries)

        self.entries =  new_entries
        self.bin_edges = [new_x_bins, new_y_bins]
        #print("after", np.sum(self.entries.flatten()))
        #print(self.entries)
        #self._update_bins()


    def bins_intersect(self, new_x_bins=None, new_y_bins=None):
        bins_intersect = True
        def check_intersection(new_bins, axis):
            new_bins = self.parse_bin_edges(new_bins)
            for new_bin_edge in new_bins:
                if not new_bin_edge in self.bin_edges[axis]:
                    raise ValueError(f"{new_bin_edge} not in self.bin_edges[axis]")
                    return False
            return True
        if new_x_bins is not None:
            bins_intersect = bins_intersect and check_intersection(new_x_bins, 0)
        if new_y_bins is not None:
            bins_intersect = bins_intersect and check_intersection(new_y_bins, 1)
        return bins_intersect


    def calc_weighted_uncert(self, data, weights, bin_edges):
        """Caluculate the statistical uncertainty of each bin regarding the weighted data

        :param data: data set from which the histogram is built
        :type data: numpy array of length n
        :param weights: weights per data event
        :type weights: numpy array of length n
        :param bin_edges: bin edges of the histogram
        :type bin_edges: numpy array of length k
        :return: statistical uncertainty per bin
        :rtype: nunmpy array of length k-1
        """
        w_uncert = np.zeros((bin_edges[0].size-1, bin_edges[1].size-1))
        for x, [x_lower_edge, x_upper_edge] in enumerate(zip(bin_edges[0][0:-1], bin_edges[0][1:])):
            if x_upper_edge == bin_edges[0][-1]:
                    x_upper_edge += 1 # we want to take in the last bin the bin border into account
            for y, [y_lower_edge, y_upper_edge] in enumerate(zip(bin_edges[1][0:-1], bin_edges[1][1:])):
                if y_upper_edge == bin_edges[1][-1]:
                    y_upper_edge += 1 # we want to take in the last bin the bin border into account
                weight_idxs = np.where(((data[0,:]>=x_lower_edge) & (data[0,:]<x_upper_edge)) &
                                       ((data[1,:]>=y_lower_edge) & (data[1,:]<y_upper_edge)))
                w_i = np.array(weights[weight_idxs])
                w_uncert[x,y] = np.sqrt(np.sum(w_i**2))
        return np.array(w_uncert)


    def check_compatibility(self, other):
        for other_bin_edges, self_bin_edges in zip(other.bin_edges, self.bin_edges):
            assert np.array_equal(np.array(other_bin_edges, dtype=np.float32), self_bin_edges.astype(np.float32)), f"Hist bin edges not compatible! ({other_bin_edges} vs. {self_bin_edges})"
            assert self.unit == other.unit, "Hist units not compatible!"


    def plot(self, fig=None, ax=None, xlabel=None, ylabel=None, zlabel='events', figsize=None, additional_info="", cut_0=True, contour=None, level=None, contour_label=None, **kwargs):
        if not xlabel:
            xlabel=""
        if not ylabel:
            ylabel=""

        if additional_info:
            self.description["additional_info"] = additional_info
        self.b2fig = B2Figure(auto_description=True, description=self.description)
        if not fig and not ax:
            self.fig, self.ax = self.b2fig.create(figsize=figsize)
        else:
            self.fig = fig
            self.ax = ax
        # Create a meshgrid for the plot
        _xedges, _yedges = np.meshgrid(self.bin_edges[0], self.bin_edges[1])

        # Create a masked array where zero values are masked
        if cut_0:
            masked_data = np.ma.masked_where(self.entries == 0, self.entries)
        else:
            print("cut0 false")
            masked_data = self.entries

        # Create a custom colormap based on 'viridis'
        cmap = plt.cm.viridis
        cmap.set_bad(color='white')

        # Plot the 2D histogram
        cm = self.ax.pcolormesh(_xedges, _yedges, masked_data.T, cmap=self.b2fig.colors.cm["blue_green_yellow"], **kwargs)

        if contour is not None:
            self.check_compatibility(contour)
            if level is None:
                level = [np.array(contour.entries).max()/10]
            c= self.ax.contour(contour.entries.T, extent=[contour.bin_edges[0][0], contour.bin_edges[0][-1],
                                                    contour.bin_edges[1][0], contour.bin_edges[1][-1]],
                        levels=level, colors=["red"])
            le, _ = c.legend_elements()
            self.ax.legend([le[0]], [f'{contour_label}'], fontsize=9, frameon=True, framealpha=0.7)


        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

        if not fig and not ax:
            self.fig.colorbar(cm, label=zlabel)
            return self.fig, self.ax
        return cm

class Histogram2D(HistogramBase2D, Histogram):
    
    def create(self, *args, **kwargs):
       """Creates an instance of this class.
       """
       return Histogram2D(*args, **kwargs)


class StackedHistogram2D(StackedHistogram, HistogramBase2D):
    #@property
    #def bin_edges(self):
    #    return self._bin_edges
    #
    #@bin_edges.setter
    #def bin_edges(self, value):
    #    self._bin_edges = [self.parse_bin_edges(value[0]),
    #                       self.parse_bin_edges(value[1])]
    #    self._update_bins()

    def __init__(self, *args, **kwargs):
        StackedHistogram.__init__(self, *args, **kwargs)


    def create(self, *args, **kwargs):
       """Creates an instance of this class.
       """
       return StackedHistogram2D(*args, **kwargs)


    def get_stacked_entries(self):
        """Get the sum of the stacked entries per bin."""
        #print(self.bin_edges)
        entries = np.array([[]])
        if len(self.hists) > 0:
            #print(self.bin_centers)
            entries = np.zeros((self.bin_centers[0].size, self.bin_centers[1].size))
            if not self.data_hist:
                if len(self.hists) > 0:
                    for name, hist in self.hists.items():
                        entries += hist.entries
            else:
                if len(self.hists) > 1:
                    for name, hist in self.hists.items():
                        if not name == "data":
                            entries += hist.entries
                #else:
                #    entries = self.data_hist.entries
        return entries

    def _update_bins(self):
        HistogramBase2D._update_bins(self)


    def get_stat_uncert(self):
        """Calculate the stacked uncertainty of the stacked histogram using sqrt(sum(sum(lumi_scale_1**2), sum(lumi_scale_2**2), ..., sum(lumi_scale_n**2)))
        of the n histograms."""
        #print(self.bin_centers)
        if len(self.bin_centers)==2:
            uncert = np.zeros([self.bin_centers[0].size, self.bin_centers[1].size])
        else:
            uncert=np.array([])
        #print(uncert)
        if len(self.hists) > 0:
            for name, hist in self.hists.items():
                # sigma = n2/lumi_scale *lumi_scale**2
                #uncert += hist.entries/hist.lumi_scale*hist.lumi_scale**2
                uncert += hist.err**2 #quadratic sum of each uncertainty component
        elif len(self.hists) == 0 and self.data_hist:
            uncert = self.data_hist.err**2
        return np.sqrt(uncert)


    def empty(self):
        """Method to check if any histograms have been already added. If not the bin
        edges are empty!"""
        return not self.bin_edges[0].any()


    def bin_edges_compatible(self, bin_edges):
        compatible = True
        for i, _bin_edges in enumerate(bin_edges):
            compatible = compatible and np.array_equal(self.parse_bin_edges(_bin_edges), self.bin_edges[i])
            if not compatible:
                print("Bin edges not compatible:")
                print(_bin_edges, self.bin_edges[i])
                break
        assert compatible, f"Hist bin edges not compatible with the rest of the stack [({bin_edges[0].size}, {self.bin_edges[0].size}), ({bin_edges[1].size}, {self.bin_edges[1].size})]!"


    def _parse_bin_edges_hack(self, bin_edges):
        """Hack to deal with parsing the bin edges in the 2D Histograms
        """
        return bin_edges

    def plot(self, *args, **kwargs):
        return HistogramBase2D.plot(self, *args, **kwargs)

