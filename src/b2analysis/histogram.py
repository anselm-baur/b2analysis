import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from b2style.b2figure import B2Figure
import copy
import pickle


def parse_bin_edges(bin_edges):
    """parse the bin edges to a numpy array which has a certain precise round to avoide incompatibility
    due to bit precision limits.
    """
    return np.around(np.array(copy.deepcopy(bin_edges), dtype=np.float64), 5)


class PickleBase(object):
    def copy(self):
        return copy.deepcopy(self)


    @staticmethod
    def load(file_name, lumi=None, bins=None, serialized=True):
        if serialized:
            with open(file_name, "rb") as pkl:
                hist_dict = pickle.load(pkl)
            if hist_dict["class"] == "Histogram":
                del hist_dict["class"]
                hist = Histogram(**hist_dict)
            elif hist_dict["class"] == "StackedHistogram":
                hist = StackedHistogram.from_serial(hist_dict)
        else:
            with open(file_name, "rb") as pkl:
                hist = pickle.load(pkl)
        if lumi:
            hist.re_scale_to(lumi, update_lumi=True)
        if not bins is None:
            hist.rebin(bins)
        return hist


    def save(self, file_name):
        if not file_name.endswith(".pickle") and not file_name.endswith(".pkl"):
            file_name += ".pkl"
        path = Path(file_name).parent
        path.mkdir(parents=True, exist_ok=True)
        with open(file_name, "wb") as pkl:
            pickle.dump(self.serialize(), pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"pickled serialized: {file_name}")


    def serialize(self):
        raise NotImplementedError


    def pickle(self, file_name):
        """If none serializable atributes are in one of the child classes, overwrite this method where a copy is created and the
        respective atributes are removed and then dumped.
        """
        self.pickle_dump(file_name)


    def pickle_dump(self, file_name):
        if not file_name.endswith(".pickle") and not file_name.endswith(".pkl"):
            file_name += ".pkl"

        with open(file_name, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"pickled: {file_name}")



class CanvasBase(PickleBase):

    def __init__(self, name="hist_canvas", output_dir = "", plot_state=None) -> None:
        self.output_dir = output_dir
        self.name = name
        self.fig = None
        self.ax = None
        self.b2fig = None
        self.description = {}

        self.errorbar_args = {"fmt":'o',
                              "color": "black",
                              "markersize": 3.3, #2.2,
                              "elinewidth": 1.5, #0.5
                              }

        self.setp_args = {"lw": 1.5}
        self.hatch_args = {}

        if plot_state is not None:
            #print("got plot state")
            self.plot_state = plot_state

    def savefig(self, filename=""):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not filename:
            filename = f"{self.name}.pdf"
        self.fig.savefig(os.path.join(self.output_dir, filename))


    def pickle(self, file_name):
        """Remove none searializable atributes.
        """
        self_copy = self.copy()
        self_copy.fig = None
        self_copy.ax = None
        self_copy.b2fig = None
        self_copy.pickle_dump(file_name)

    @property
    def plot_state(self):
        return self._plot_state

    @plot_state.setter
    def plot_state(self, value):
        self._plot_state = value
        self.description["plot_state"] = value



class HistogramBase(CanvasBase):
    def __init__(self, name, data, scale=1, var="", unit="", overflow_bin=False, label="", is_hist=False, weights=np.array([]), output_dir="", plot_state=None, **kwargs):
        """Creates a HistogramBase object from either data points wich gets histogramed (is_hist=False) or form already binned data
        (is_hist=True).
        """
        super().__init__(name=name, output_dir=output_dir, plot_state=plot_state)
        self.var = var
        self.scale = scale # basically the weight of each event
        self.unit = unit
        self.label = label if label else name
        self.overflow_bin = overflow_bin

        weights = copy.deepcopy(weights)
        data = copy.deepcopy(data)

        #make sure data are a numpy array
        if isinstance(data, list):
            data = np.array(data)

        #make sure weights are a numpy array
        #print(weights, type(weights), weights.shape)
        if data.ndim == 1:
            weights_shape = data.size
        else:
            weights_shape = data.shape[-1]
        if isinstance(weights, int) or isinstance(weights, float):
            weights = np.full(weights_shape, weights)
        elif isinstance(weights, list):
            weights = np.array(weights)

        if weights is None:
            self.weights = np.full(weights_shape, self.scale)
        elif not weights.any():
            #print("create weights")
            self.weights = np.full(weights_shape, self.scale)
        else:
            #if not weights.shape == weights_shape:
            #    raise ValueError(f"data and weights not same size ({weights.shape}/{weights_shape})!")
            self.weights = weights * scale
        #print(scale, self.weights, weights)
        #print(data.size)

        # We create a Histogram from an existing Histogram
        if is_hist:
            self.init_from_hist(data=data, kwargs=kwargs)

        # We create a new Histogram from a data sample
        else:
            #print(kwargs)
            if not overflow_bin:
                self.init_without_overflow_bins(data=data, kwargs=kwargs)
            else:
                self.init_with_overflow_bins(data=data, kwargs=kwargs)

            #self.size = self.bin_centers.size
            self.update_hist()

        #self.bin_edges = np.around(np.array(self.bin_edges, dtype=np.float64), 5)

    @property
    def bin_edges(self):
        return self._bin_edges

    @bin_edges.setter
    def bin_edges(self, value):
        self._bin_edges = self.parse_bin_edges(value)
        self._update_bins()


    @property
    def size(self):
        return self.bin_centers.size


    def __setstate__(self, state):
        # Convert old attribute to new property format
        if 'bin_edges' in state:
            state['_bin_edges'] = state.pop('bin_edges')
        self.__dict__.update(state)


    def init_from_hist(self, data, kwargs):
        """Create the Histogram object from an existing Histogram, basically we are just copying the values.
        """
        if not "bins" in kwargs or len(list(kwargs["bins"])) != len(list(data))+1 :
            print(f"failed, len(data)+1 = {len(list(data))+1} == len(bins) = {len(list(kwargs['bins']))}")
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


    def init_without_overflow_bins(self, data, kwargs):
        np_hist = np.histogram(data, weights=self.weights, **kwargs)
        self.entries = np_hist[0]
        self.bin_edges = np.array(np_hist[1])
        self._update_bins()
        self.err = self.calc_weighted_uncert(data=data, weights=self.weights, bin_edges=self.bin_edges)


    def init_with_overflow_bins(self, data, kwargs):
        #first add to the first and last bins bins which go to -/+ inf
        #then we trim the histograms and ad at content of the first and last inf bins
        #to the neighbor bins and cut the inf bins out
        if not "bins" in kwargs:
            kwargs["bins"] = 50
        if isinstance(kwargs["bins"], int):
            if not "range" in kwargs:
                if np.min(data) == np.max(data):
                    kwargs["range"] = (data[0]-1, data[0]+1)
                else:
                    kwargs["range"] = (np.min(data), np.max(data))
            kwargs["bins"] = np.concatenate([[-np.inf],
                                        np.linspace(*kwargs["range"], kwargs["bins"]+1),
                                        [np.inf]])
        else:
            kwargs["bins"] = np.concatenate([[-np.inf],kwargs["bins"],[np.inf]])

        np_hist = np.histogram(data, weights=self.weights, **kwargs)
        self.entries = np_hist[0]
        self.bin_edges = np_hist[1]
        self.err = self.calc_weighted_uncert(data=data, weights=self.weights, bin_edges=self.bin_edges)
        self._trim_hist(0,1)
        self._trim_hist(-2,-1)
        self._update_bins()


   # breaks the pickled histogram object
   #@property
   #def bin_edges(self):
   #    return self._bin_edges
   #
   #@bin_edges.setter
   #def bin_edges(self, value):
   #    self._bin_edges = np.around(np.array(value, dtype=np.float64), 3)


    def create(self, *args, **kwargs):
       """Creates an instance of this class.
       """
       return HistogramBase(*args, **kwargs)


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
        w_uncert = []
        for lower_edge, upper_edge in zip(bin_edges[0:-1], bin_edges[1:]):
            if upper_edge != bin_edges[-1]:
                w_i = weights[(data>=lower_edge) & (data<upper_edge)]
            else:
                w_i = weights[(data>=lower_edge) & (data<=upper_edge)]
            w_uncert+= [np.sqrt(np.sum(w_i**2))]
        return np.array(w_uncert)


    def update_hist(self):
        """Recalculate the uncertainty and set the entries atribute.
        """
        #self.err = self.stat_uncert()
        pass


    def _trim_hist(self, a, b):
        # start with first bin
        if a == 0:
            #add bin content and error of the cut bins to the first new bin b
            self.entries[b] = np.sum(self.entries[:b+1])
            self.err[b] = np.sqrt(np.sum(self.err[:b+1]**2))
            self.entries = self.entries[b:]
            self.err = self.err[b:]
            self.bin_edges = self.bin_edges[b:]
        # end with last bin
        elif b == -1:
            self.entries[a] = np.sum(self.entries[a:])
            self.err[a] = np.sqrt(np.sum(self.err[a:]**2))
            self.entries = self.entries[:a+1]
            self.err = self.err[:a+1]
            self.bin_edges = self.bin_edges[:a+1]
        # s.th. in the center is cut out
        else:
            self.entries[a] = np.sum(self.entries[a:b+1])
            self.err[a] = np.sqrt(np.sum(self.err[a:b+1]**2))
            self.entries = np.concatenate([self.entries[:a+1], self.entries[b+1:]])
            self.err[a] = np.concatenate([self.err[:a+1], self.err[b+1:]])
            self.bin_edges = np.concatenate([self.bin_edges[:a+1], self.bin_edges[b+1:]])
        self._update_bins()


    def _update_bins(self):
        self.bin_centers = np.around(np.array((self.bin_edges[1:]+self.bin_edges[:-1])/2, dtype=np.float64), 5)
        self.range = (self.bin_edges[0], self.bin_edges[-1])
        #self.bins = self.bin_centers.size


    def rebin(self, new_bin_edges):
        new_bin_edges = np.around(np.array(new_bin_edges, dtype=np.float64),5)
        for nbin in new_bin_edges:
            if nbin not in self.bin_edges:
                print("present bins", self.bin_edges, 2, type(self.bin_edges))
                print("new bins", new_bin_edges, 2, type(new_bin_edges))
                raise RuntimeError("New bin edges need to intersect with old bin edges!")

        combinedbins = self.get_combined_bins(new_bin_edges)
        tmpdata=[]
        tmperr=[]
        for i in range(0,len(combinedbins)):
            tmpdata.append(0)
            tmperr.append(0)
            for u in combinedbins[i]:
                if np.round(self.bin_edges[u],4)<new_bin_edges[0] or np.round(self.bin_edges[u],4)>new_bin_edges[-1]:
                    #print("continue because")
                    edges = self.bin_edges[u]
                    #print(f"{edges}<{new_bin_edges[0]} or {edges}>{new_bin_edges[-1]}")
                    continue
                #print(u, self.entries[u])
                tmpdata[i] += self.entries[u]
                tmperr[i] = tmperr[i]+pow(self.err[u],2)
            tmperr[i] = np.sqrt(tmperr[i])
        self.entries = np.array(tmpdata)
        self.err = np.array(tmperr)
        self.bin_edges = new_bin_edges
        self._update_bins()
        #print(len(list(self.entries)), len(list(self.bin_edges)))


    def get_combined_bins(self, new_bin_edges):
        combinedbins=[]
        n=0
        for i in range(0,len(new_bin_edges)-1):
            combinedbins.append([])
            for u in range(n,len(self.bin_edges)):
                if new_bin_edges[i+1] > self.bin_edges[u]:
                    combinedbins[i].append(u)
                else:
                    n=u
                    break
        return combinedbins


    def re_scale(self, factor):
        self.entries = self.entries*factor
        """
        sig_1/n_1 = sig_2/n_2
        sig_2 = sig_1 * n_2/n_1 = sig_1*factor
        """
        self.err *= factor
        self.scale *= factor
        # the stat_uncet function would return wrong results, so we delete it after re_scaling
        self.stat_uncert = None


    def plot(self, fig=None, ax=None, histtype="errorbar", dpi=100, uncert_label=True, log=False):
        if not fig and not ax:
            fig, ax = plt.subplots(ncols=1, nrows=1, dpi=dpi)

        if histtype == "errorbar":
            ax.errorbar(self.bin_centers, self.entries, yerr=self.err, label=self.label, **self.errorbar_args)
        elif histtype == "step":
            x = np.concatenate([self.bin_edges, [self.bin_edges[-1]]])
            y = np.concatenate([[0], self.entries, [0]])
            ax.step(x, y, label=self.label, lw=0.9)
            uncert = self.err
            print(uncert)
            bin_width = self.bin_edges[1:]-self.bin_edges[0:-1]
            ax.bar(x=self.bin_centers, height=2*uncert, width=bin_width, bottom=self.entries-uncert,
                    edgecolor="grey",hatch="///////", fill=False,
                    lw=0,label="stat. unc." if uncert_label else "")
            if uncert_label: uncert_label = False
        unit = f" in {self.unit}"
        ax.set_xlim((*self.range))
        ax.set_xlabel(f"{self.var}{unit if self.unit else ''}")
        ax.set_ylabel("events")
        if log:
            ax.set_yscale("log")
        ax.legend()

        return fig, ax

    def stat_uncert(self):
        """relative error stays constant with scaling:
        sqrt(n_1) / n_1 = x / n_2
        -> x = sqrt(n_1) * n_2 / n_1
             = lumi_scale * sqrt(n_1)

        we need to calculate the bin_contents without weights:
        n_2 = n_1 * lumi_scale
        -> n_1 = n_2 / lumi_scale

        so the scaled uncertainty is:
        x = sigma = lumi_scale * sqrt(n_2) / sqrt(lumi_scale)
        """
        return np.sqrt(self.entries)*(self.scale**(1/2))


    def __sub__(self, other):
        self.check_compatibility(other)
        diff_hist = self.copy()
        diff_hist.entries = diff_hist.entries - other.entries
        diff_hist.scale = 1
        diff_hist.weights = None
        if not self.name == other.name:
            diff_hist.name += " - " + other.name
        diff_hist.update_hist()
        return diff_hist


    def __add__(self, other):
        if other is None:
            print("Try to add emtpy Histogram... skip!")
            return self.copy()
        self.check_compatibility(other)
        add_hist = self.copy()
        add_hist.entries = add_hist.entries + other.entries
        add_hist.err = np.sqrt(self.err**2 + other.err**2)
        add_hist.scale = 1
        add_hist.weights = None
        if not self.name == other.name:
            add_hist.name += " + " + other.name
        add_hist.update_hist()
        return add_hist


    def __truediv__(self, other):
        """Division of two histograms without error correlation in the
        uncertainties.
        """
        return self.corr_div(other, corr=0)


    def corr_div(self, other, corr=1):
        """Division of two histograms with allowed error correlation in the
        uncertainties.
        """
        self.check_compatibility(other)
        with np.errstate(divide='ignore',invalid='ignore'):
            name = f"{self.name}/{other.name}"
            data = np.array(self.entries)/np.array(other.entries)
            if hasattr(self, "err") and hasattr(other, "err"):
                err = np.sqrt((self.err/other.entries)**2+(other.err*self.entries/other.entries**2)**2-2*self.entries/other.entries**3*self.err*other.err*corr)
            else:
                err = None
            #print(self.entries)
            #print(other.entries)
            res_hist = self.create(name, data, err=err, lumi=self.lumi, is_hist=True, bins=self.bin_edges )
        return res_hist


    def check_compatibility(self, other):
        assert np.array_equal(np.array(other.bin_edges, dtype=np.float32), self.bin_edges.astype(np.float32)), "Hist bin edges not compatible!"
        assert self.unit == other.unit, "Hist units not compatible!"


    def parse_bin_edges(self, bin_edges):
        return parse_bin_edges(bin_edges=bin_edges)


class Histogram(HistogramBase):
    """Analysis Histogram Class."""

    def __init__(self, name, data, lumi, lumi_scale=1, is_signal=False, is_hist=False, color=None, is_simulation=True, is_preliminary=False,
                 additional_info=None, b2color=None, **kwargs):
        super().__init__(name=name, data=data, scale=lumi_scale, is_hist=is_hist, **kwargs)
        self.b2fig = B2Figure()
        self.is_signal = is_signal
        self.lumi = lumi
        self.lumi_scale = lumi_scale # basically the weight of each event
        self.color = color
        if b2color is not None:
            self.b2color = b2color

        self.description.update({"luminosity": self.lumi,
                            "simulation": is_simulation,
                            "additional_info": additional_info,
                            "preliminary": not is_simulation and is_preliminary})


    @property
    def b2color(self):
        return self.color

    @b2color.setter
    def b2color(self, color):
        self.color = self.b2fig.color(color)


    def create(self, *args, **kwargs):
       """Creates an instance of this class.
       """
       return Histogram(*args, **kwargs)


    def re_scale(self, factor, update_lumi=False):
        """Rescale the histogram by a given factor.
        """
        super().re_scale(factor)
        if update_lumi:
            self.lumi_scale *= factor


    def re_scale_to(self, target_lumi, update_lumi=True):
        """Rescale the histogram to the given target luminosity.
        """
        factor = target_lumi/self.lumi
        self.re_scale(factor=factor, update_lumi=update_lumi)


    def plot(self, fig=None, ax=None, figsize=(6,6), histtype="hatch", dpi=100, uncert_label=True, log=False, ylim=False, color="blue",
             additional_info=None, **kwargs):

        if additional_info:
                self.description["additional_info"] = additional_info
        b2fig = B2Figure(auto_description=True, description=self.description)
        if not fig and not ax:
            fig, ax = b2fig.create(ncols=1, nrows=1, dpi=dpi, figsize=figsize)

        if histtype == "errorbar":
            ax.errorbar(self.bin_centers, self.entries, yerr=self.err, label=self.label, **self.errorbar_args)
        elif histtype == "step":
            x = np.concatenate([self.bin_edges, [self.bin_edges[-1]]])
            y = np.concatenate([[0], self.entries, [0]])
            ax.step(x, y, label=self.label, **self.setp_args, **kwargs)
            uncert = self.err
            bin_width = self.bin_edges[1:]-self.bin_edges[0:-1]
            ax.bar(x=self.bin_centers, height=2*uncert, width=bin_width, bottom=self.entries-uncert,
                    edgecolor="grey",hatch="///////", fill=False, lw=0,label="MC stat. unc." if uncert_label else "")
            if uncert_label: uncert_label = False
        elif histtype == "hatch":
            x = np.concatenate([self.bin_edges, [self.bin_edges[-1]]])
            y1 =np.concatenate([[0], self.entries, [0]])
            hatch = "\\\\\\\\\\\\"
            ax.fill_between(x, y1, lw=0.9, hatch=hatch, color=color, step='pre', facecolor="white", alpha=0.5)
            ax.step(x, y1, label=self.name, color=color, lw=1.2)
            uncert = self.err
            bin_width = self.bin_edges[1:]-self.bin_edges[0:-1]
            ax.bar(x=self.bin_centers, height=2*uncert, width=bin_width, bottom=self.entries-uncert,
                edgecolor="black",hatch="///////", fill=False, lw=0,label="MC stat. unc." if uncert_label else "")
            if uncert_label: uncert_label = False
        unit = f" in {self.unit}"
        ax.set_xlim((*self.range))
        if not log:
            ax.set_ylim([0, ax.get_ylim()[1]])
        if ylim:
            span = self.entries.max() - self.entries.min()
            ax.set_ylim([(self.entries-self.err).min()-span*0.3, (self.entries+self.err).max()+span*0.3])
        ax.set_xlabel(f"{self.var}{unit if self.unit else ''}")

        b2fig.shift_offset_text_position_old(ax)
        ax.set_ylabel("events")
        if log:
            ax.set_yscale("log")
        ax.legend()

        return fig, ax


    def compare(self, other, **kwargs):
        return compare_histograms(f"compare {self.label}", other, self, **kwargs)


    def __str__(self):
        ret_str  = "Histogram Object\n"
        ret_str += "================\n"
        ret_str += f"name: {self.name}\n"
        ret_str += f"var: {self.var}\n"
        #ret_str += f"bins: {self.bins}\n"
        ret_str += f"entries: {np.sum(self.entries):.0f}\n"
        if self.weights is not None:
            ret_str += f"weights: {np.mean(self.weights):.3f}\n"
        else:
            ret_str += f"weights: None \n"
        ret_str += f"lumi: {self.lumi}\n"
        ret_str += f"lumi scale factor: {self.lumi_scale:.4f}\n"
        ret_str += f"normalized lumi: {self.lumi*self.lumi_scale:.2f}\n"
        return ret_str


    def info(self):
        print(self.__str__())


    def serialize(self):
        """Create a serialized version of the histogram for storage in a file.

        :return: the histogram content in strings and lists
        :rtype: dict
        """
        ser_hist = {"class": "Histogram",
                    "name": self.name,
                    "data": list(self.entries),
                    "err": list(self.err),
                    "lumi": self.lumi,
                    "bins": list(self.bin_edges),
                    "lumi_scale": self.lumi_scale,
                    #"weights": list(self.weights), # makes no sence to write out the weigts e.g. from single events... event information is anyways already lost
                    "var": self.var,
                    "unit": self.unit,
                    "color": self.color,
                    "label": self.label,
                    "is_signal": self.is_signal,
                    "is_hist": True,
                    "overflow_bin": self.overflow_bin
                    }
        return ser_hist


class HistogramCanvas(CanvasBase):
    """Class to aggregate the Histogram objects and plot them
    for comparison."""

    def __init__(self, lumi=None, var="", unit="", additional_info="", is_simulation=True, is_preliminary=False, **kwargs):
        CanvasBase.__init__(self,**kwargs)

        self.lumi = lumi
        self.var = var
        self.hists = {}
        self.unit = unit
        self.bin_edges = np.array([])
        self.bin_centers = np.array([])
        self.description.update({"luminosity": self.lumi,
                            "simulation": is_simulation,
                            "additional_info": additional_info,
                            "preliminary": not is_simulation and is_preliminary})

        self.b2fig = B2Figure(color_theme="phd")
        self.fig = None
        self.ax = None

        self.signals = 0
        self.labels = {}
        self.colors = {}

        self.pull_args = {"hist_name": [],
                "nom_hist_name": None,
                "ratio": True,
                "ylim": [0.99, 1.01],
                "pull_bar": False,
                "corr": 0}


    #@property
    #def lumi(self):
    #    return self._lumi

    #@lumi.setter
    #def lumi(self, value):
    #    self._lumi = value
    #    self.description["luminosity"] = self.lumi

    def create(self, *args, **kwargs):
       """Creates an instance of this class.
       """
       return HistogramCanvas(*args, **kwargs)

    def empty(self):
        """Method to check if any histograms have been already added. If not the bin
        edges are empty!"""
        return not self.bin_edges.any()


    def bin_edges_compatible(self, bin_edges):
        #compatible = np.array_equal(np.array(bin_edges, dtype=np.float32), self.bin_edges)
        compatible = np.array_equal(parse_bin_edges(bin_edges), parse_bin_edges(self.bin_edges))
        assert compatible, f"Hist bin edges not compatible with the rest of the stack ({bin_edges.size}, {self.bin_edges.size})!"


    def _parse_bin_edges_hack(self, bin_edges):
        """Hack to deal with parsing the bin edges in the 2D Histograms
        """
        return parse_bin_edges(bin_edges=bin_edges)


    def add_histogram(self, hist, label=True, color=None, bins=None):
        """Add a histogram to the canvas."""

        if bins is not None:
            hist.rebin(bins)

        if self.empty():
            #print(f"adding bin edges: {hist.bin_edges}")
            #self.bin_edges = np.array(hist.bin_edges, dtype=np.float32)
            self.bin_edges = self._parse_bin_edges_hack(hist.bin_edges)
            if not self.unit and hist.unit:
                self.unit = hist.unit
            #self.bins = hist.bins
            #self.size = self.bin_centers.size
            self._update_bins()
        else:
            self.bin_edges_compatible(hist.bin_edges)
        if not self.lumi:
            self.set_lumi(hist.lumi * hist.lumi_scale)
        else:
            if not np.round(hist.lumi * hist.lumi_scale, 3) == np.round(self.lumi, 3):
                raise ValueError(f"Histogram luminosity {hist.lumi} and histogram luminosity scale {hist.lumi_scale} ({np.round(hist.lumi * hist.lumi_scale, 3)}) not compatible with desired luminosity {np.round(self.lumi, 3)}")
        if color is not None:
            #print(self.b2fig.colors.color)
            #print(self.b2fig.colors.cm)
            if isinstance(color, str) and  color in self.b2fig.colors.color.keys():
                color = self.b2fig.colors[color]
            self.colors[hist.name] = color
            hist.color = color
        elif hist.color:
            self.colors[hist.name] = hist.color
        if hist.is_signal:
            self.signals += 1

        self.hists[hist.name] = hist
        self.labels[hist.name] = label

        self.__update()

        #pre-fill pull args for pull plot
        if hasattr(self, "pull_args"):
            if self.pull_args["nom_hist_name"] is None:
                self.pull_args["nom_hist_name"]=hist.name
            else:
                self.pull_args["hist_name"].append(hist.name)

    def __update(self):
        pass


    def _update_bins(self):
        self.bin_centers = (self.bin_edges[1:]+self.bin_edges[:-1])/2
        self.range = (self.bin_edges[0], self.bin_edges[-1])
        #self.bins = self.bin_centers.size


    def create_histogram(self, name, data, lumi, lumi_scale=1, is_signal=False, **kwargs):
        """Create a histogram from data and add it to the stack"""
        self.add_histogram(Histogram(name, data, lumi, lumi_scale, is_signal=is_signal, **kwargs))


    def rebin(self, new_bin_edges):
        for name, hist in self.hists.items():
            hist.rebin(new_bin_edges)
        self.bin_edges = np.array(new_bin_edges)
        self._update_bins()


    def re_scale(self, factor, update_lumi=True):
        for hist in self.hists.values():
            hist.re_scale(factor=factor, update_lumi=update_lumi)
        if update_lumi:
            self.set_lumi(self.lumi * factor)


    def re_scale_to(self, target_lumi, update_lumi=True):
        for hist in self.hists.values():
            hist.re_scale_to(target_lumi=target_lumi, update_lumi=update_lumi)
        if update_lumi:
            self.set_lumi(target_lumi)


    def set_lumi(self, lumi):
        """When updating the luminosity we also want to set the description.
        """
        self.lumi = lumi
        self.description["luminosity"] = self.lumi


    def plot(self, dpi=90, figsize=(6,5), pull_args={}, additional_info="", ylim=None, **kwargs):
        """Plot the histogram canvas."""

        # make sure we have colors for our histograms
        if not "colors" in kwargs and len(self.colors) != len(self.hists) or ("colors" in kwargs and len(kwargs["colors"]) != len(self.hists)):
            print("create colors...")
            reverse_colors = False if not "reverse_colors" in kwargs else kwargs["reverse_colors"]
            self.color_scheme(reverse=reverse_colors)

        if "hist_name" in pull_args and "nom_hist_name" in pull_args:
            self.b2fig = B2Figure(auto_description=False)
            self.fig, ax = self.b2fig.create(ncols=1, nrows=2, dpi=dpi, figsize=figsize, gridspec_kw={'height_ratios': [2, 1]})
            self.ax = ax[0]
            self.b2fig.add_descriptions(ax=self.ax, **self.description)
            self.ax_pull = ax[1]
            #move the xlabel to the pull plot
            if "xlabel" in kwargs:
                pull_args["xlabel"] = kwargs["xlabel"]
                kwargs["xlabel"] = ""
            self.pull_plot(self.ax_pull, **pull_args)
            self.ax.set_xticklabels([])
            self.fig.subplots_adjust(hspace=0.05)
        else:
            if additional_info:
                self.description["additional_info"] = additional_info
            self.b2fig = B2Figure(auto_description=True, description=self.description)
            self.fig, self.ax = self.b2fig.create(ncols=1, nrows=1, dpi=dpi, figsize=figsize)
        #self.fig.subplots_adjust(right=1)

        self.plot_ax(self.ax, colors=self.colors, ylim=ylim, **kwargs)
        self.b2fig.shift_offset_text_position(self.ax)
        if not pull_args:
            xlabel = kwargs.get("xlabel", "")
            print("xlabel:", xlabel)
            self.add_labels(ax=self.ax, xlabel=xlabel)

        return self.fig, self.ax


    def plot_ax(self, ax, xlabel="", histtype="hatch", log=False, x_log=False, colors=None, reverse_colors=False, errorbar_args=None, ylim=None, **kwargs):
        if not colors:
            colors = []
        if len(colors) <  len(self.hists):
            if len(self.colors) <  len(self.hists):
                print("create colors in plot_ax...")
                self.color_scheme(reverse=reverse_colors)
            colors = self.colors
            print(colors)
            print(self.hists.keys())

        uncert_label = True
        plot_args = copy.deepcopy(kwargs.get("plot_args", {}))
        ncols_legend = plot_args.get("ncols", 1)
        loc_legend = plot_args.get("loc", None)
        if "ncols" in plot_args:
            del plot_args["ncols"]


        base_line = np.zeros(self.bin_centers.size+2)
        for i, (name, hist) in enumerate(self.hists.items()):
            label = self.get_label(name)
            if type(colors) == dict:
                color = color = self.signal_color if hist.is_signal else colors[name]
            else:
                color = colors[i]
            if histtype == "errorbar":
                if not errorbar_args:
                    errorbar_args = self.b2fig.errorbar_args
                ax.errorbar(self.bin_centers, hist.entries, yerr=hist.err, label=name, color=color, zorder=i*1000, **errorbar_args)
            elif histtype == "step":
                x = np.concatenate([self.bin_edges, [self.bin_edges[-1]]])
                y = np.concatenate([[0], hist.entries, [0]])
                if not "linewidth" in plot_args and not "lw" in plot_args:
                    plot_args["lw"] = self.setp_args["lw"]
                ax.step(x, y, label=label, color=color, **plot_args)
                uncert = hist.err
                bin_width = self.bin_edges[1:]-self.bin_edges[0:-1]
                ax.bar(x=self.bin_centers, height=2*uncert, width=bin_width, bottom=hist.entries-uncert,
                       edgecolor="grey",hatch="///////", fill=False, lw=0,label="MC stat. unc." if uncert_label else "")
                if uncert_label: uncert_label = False
            elif histtype == "hatch":
                x = np.concatenate([self.bin_edges, [self.bin_edges[-1]]])
                y1 = base_line
                y2 = np.concatenate([[0], hist.entries, [0]])
                if not "linewidth" in plot_args and not "lw" in plot_args:
                    plot_args["lw"] = 0.9
                if i%2 == 0:
                    hatch = "/////"
                else:
                    hatch = "\\\\\\\\\\"
                ax.fill_between(x, y1, y2,  color=color, **plot_args, hatch=hatch,  step='pre', facecolor="white", alpha=0.5)
                ax.step(x, y2, label=name, color=color, lw=1.2)
            else:
                raise ValueError(f"histtype {histtype} not implemented!")
        ax.set_xlim((*self.range))
        if xlabel:
            #ax.set_xlabel(xlabel)
            # is now set outside
            pass
        if x_log:
            ax.set_xscale("symlog")
        else:
            ax.set_xscale("linear")
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            if not log:
                ax.set_ylim((0, ax.get_ylim()[1]))
        if log:
                ax.set_yscale("log")
        ax.set_ylabel("events")
        self.b2fig.shift_offset_text_position(ax)
        #ax.legend(loc='upper left', prop={'size': 7})
        ax.legend(loc=loc_legend, fontsize=9, frameon=False, framealpha=0.7, ncols=ncols_legend)
        self.b2fig.shift_offset_text_position_old(ax)
        if self.description["additional_info"]:
            head_room_fac = 1.1
            if log:
                head_room_fac = 10
            ylim=ax.get_ylim()
            ax.set_ylim([ylim[0], ylim[1]*head_room_fac])


    def get_label(self, name):
        if self.labels[name]:
            if type(self.labels[name]) == bool:
                return self.hists[name].label
            else:
                # we allow to give a new label here if we don't have a
                # True or False label
                return f"{self.labels[name]}"
        else:
            return None


    def add_labels(self, ax, xlabel="", ylabel="events"):
        if xlabel:
            ax.set_xlabel(xlabel)
        else:
            unit = f" in {self.unit}"
            ax.set_xlabel(f"{self.var}{unit if self.unit else ''}")
        ax.set_ylabel(ylabel)


    def get_signal_names(self):
        """Return a list of the names of the signal histograms in self.hists.
        """
        signal_names = []
        for name, hist in self.hists.items():
            if hist.is_signal:
                signal_names.append(name)
        return signal_names


    def color_scheme(self, reverse=False, exclude_signals=True,  cm=None):
        #cm = plt.cm.seismic
        #plt.cm.gist_earth
        if cm is None:
            cm = self.b2fig.colors.cm["default"]
        cm_low = 0
        cm_high = 1

        self.signal_color = self.b2fig.colors.color["dark_red"] #plt.cm.seismic(0.9)

        if exclude_signals:
            nhists = len(self.hists)-self.signals
            signals = self.get_signal_names()
            iter_histsts = copy.deepcopy(self.hists)
            for sig in signals:
                del(iter_histsts[sig])
                self.colors[sig] = self.signal_color
        else:
            nhists = len(self.hists)
        linspace = np.linspace(cm_low,cm_high,nhists)
        sorted_iter_hists = sorted(iter_histsts.items(), key=lambda item: item[1].entries.sum(), reverse=False)
        for name_hist, color in zip(sorted_iter_hists, cm(np.flip(linspace) if reverse else linspace)):
            name = name_hist[0]
            if self.hists[name].color:
                self.colors[name] = self.hists[name].color
            else:
                self.colors[name] = color




    def pull_plot(self, dpi=90, figsize=(6,5), pull_args=None, additional_info="", height_ratios=None, ylim=None, pull_ylim=None, plot_state=None, empty_pull=False, **kwargs):
        """
        :param dpi: _description_, defaults to 90
        :type dpi: int, optional
        :param figsize: _description_, defaults to (6,6)
        :type figsize: tuple, optional
        :param pull_args: _description_, defaults to {}
        :type pull_args: dict, optional
        :param additional_info: _description_, defaults to ""
        :type additional_info: str, optional
        :param height_ratios: set the height ratios of the subplots
        :type height_args: list, tupel, optional, defaults to [2,1]
        :return: _description_
        :rtype: _type_

        Example how to use the pull_args:
        pull_args = {"hist_name": [hist_name_1, hist_name_2, ...],
                     "nom_hist_name": nom_hist_name,
                     "ratio": True,
                     "ylim": pull_ylim,
                     "pull_bar": pull_bar,
                     "fmt": fmt,
                     "xlabel": xlabel,
                     "ylabel":  r"$\\mathbf{\\frac{hist}{nom_hist}}$"
                     "corr": corr}
        """

        pull_args = copy.deepcopy(pull_args)
        if pull_args is None:
            pull_args = self.pull_args
        else:
            for pull_arg_key in self.pull_args.keys():
                if not pull_arg_key in pull_args:
                    pull_args[pull_arg_key] = self.pull_args[pull_arg_key]
        if not pull_ylim is None:
            pull_args["ylim"] = pull_ylim
        if plot_state is not None:
            self.plot_state = plot_state
        self.b2fig = B2Figure(auto_description=False)
        if height_ratios is None:
            height_ratios = [2, 1]
        self.fig, ax = self.b2fig.create(ncols=1, nrows=2, dpi=dpi, figsize=figsize, gridspec_kw={'height_ratios': height_ratios})
        self.ax = ax[0]
        if additional_info:
                self.description["additional_info"] = additional_info
        #print(self.description)
        self.b2fig.add_descriptions(ax=self.ax, **self.description)
        self.ax_pull = ax[1]
        #move the xlabel to the pull plot

        self.ax.set_xticklabels([])
        self.fig.subplots_adjust(hspace=0.05)

        self.plot_ax(self.ax, ylim=ylim, **kwargs)
        self.b2fig.shift_offset_text_position(self.ax)
        self.add_labels(ax=self.ax)
        if "x_log" in kwargs:
            pull_args["x_log"] = kwargs["x_log"]
            #print(pull_args["x_log"])
        if not empty_pull:
            self.plot_pull_ax(self.ax_pull, **pull_args, **kwargs)

        self.ax.set_xticklabels([])
        self.fig.subplots_adjust(hspace=0.05)

        return self.fig, ax


    def plot_pull_ax(self, ax, hist_name, nom_hist_name, color='black', ratio=True, corr=0, xlabel="", ylabel="", ylim=None, fmt="o--",
                     pull_bar=False, x_log=False, **kwargs):
        nom_hist = self.hists[nom_hist_name]
        bin_centers = nom_hist.bin_centers
        bin_edges = nom_hist.bin_edges
        bins = nom_hist.size
        normalized = kwargs.get("normalized", False)

        def iterate_plot(hist, hist_color):
            nonlocal ylabel # idk why we need this here but without it will not find ylabel variable
            nonlocal pull_bar
            if ratio:
                with np.errstate(divide='ignore',invalid='ignore'):
                    _hist = copy.deepcopy(hist)
                    if normalized:
                        scale_factor = nom_hist.entries.sum()/hist.entries.sum()
                        _hist.re_scale(factor=scale_factor, update_lumi=False)

                    plot = _hist.entries/nom_hist.entries
                if pull_bar:
                    self.plot_pull_bars(ax, bin_edges, plot-1, 1)
                ax.plot((bin_edges[0], bin_edges[-1]),[1,1], color='black', ls="-")
                if not ylabel:
                    ylabel = r"$\mathbf{\frac{"+hist.name.replace("_",r"\_").replace(" ",r"\;")+r"}{"+nom_hist.name.replace("_",r"\_").replace(" ",r"\;")+r"}}$"

            else:
                with np.errstate(divide='ignore',invalid='ignore'):
                    plot = (hist.entries-nom_hist.entries)/nom_hist.entries
                if pull_bar:
                    self.plot_pull_bars(ax, bin_edges, plot)
                ax.plot((bin_edges[0], bin_edges[-1]),[0,0], color='black', ls="-")
                if not ylabel:
                    hist_label = hist.name.replace("_",r"\_").replace(" ",r"\;")
                    nom_hist_label = nom_hist.name.replace("_",r"\_").replace(" ",r"\;")
                    ylabel = r"$\mathbf{\frac{"+hist_label+r"-"+nom_hist_label+r"}{"+nom_hist_label+r"}}$"
            with np.errstate(divide='ignore',invalid='ignore'):
                plot_err = np.sqrt((_hist.err/nom_hist.entries)**2+(nom_hist.err*_hist.entries/nom_hist.entries**2)**2-2*_hist.entries/nom_hist.entries**3*_hist.err*nom_hist.err*corr)
            ax.errorbar(bin_centers, plot, yerr=plot_err, fmt=fmt, color=hist_color, markersize='2.2', elinewidth=0.5)

        if type(hist_name) == list:
            if type(color) != dict or len(color) != len(hist_name):
                #print("WARNING: color must have the same type and size as hist_name! -> gonna create new color scheme...")
                #self.color_scheme()
                # we use the default canvas colors
               color = self.colors
            for name in hist_name:
                #print(color)
                #print(hist_name)
                hist = self.hists[name]
                iterate_plot(hist, color[name])
        else:
            iterate_plot(self.hists[hist_name], color)

        if normalized:
            ylabel = ylabel + " Norm."

        self.b2fig.shift_offset_text_position_pull_old(ax)

        if x_log:
            ax.set_xscale("symlog")
        else:
            ax.set_xscale("linear")
        ax.set_xlim(bin_edges[0], bin_edges[-1])
        if ylim:
            ax.set_ylim(ylim)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)





    def plot_pull_bars(self, ax, bin_edges, y, bottom=0, color="lightgrey"):
            """Plot grey bars between bottom and y"""
            widths = bin_edges[1:]-bin_edges[:-1]
            ax.bar(bin_edges[:-1], y, widths, align="edge", color=color, bottom=bottom, zorder=0)


    def any(self):
        """Check if any hitograms have already been added."""
        if len(self.hists) > 0:
            return True
        else:
            return False



class StackedHistogram(HistogramCanvas):
    """Class to aggregate the Histogram objects and stack them."""

    def __init__(self, lumi=None, var="", unit="", additional_info="", is_simulation=True, is_preliminary=False, name="stacked_histogram", **kwargs):
        super().__init__(lumi, var, unit, additional_info, is_simulation, is_preliminary, name=name, **kwargs)
        self.data_hist = None
        self.__update()


    def create(self, *args, **kwargs):
       """Creates an instance of this class.
       """
       return StackedHistogram(*args, **kwargs)


    @staticmethod
    def from_serial(serial_hist):
        """Create a StackedHistogram object from the sereial input.

        :param serial_hist: serial histogram information
        :type serial_hist: dict
        :return: stacked histogram
        :rtype: StackedHistogram
        """
        args = ["name", "var", "lumi", "unit"]
        init_kwargs = {arg: serial_hist[arg] for arg in args}
        stacked_hist = StackedHistogram(**init_kwargs)

        for name, hist in serial_hist["hists"].items():
            if not hist["class"] == "Histogram":
                raise RuntimeError("Class Histogram expected, not {hist['class']}")
            del hist["class"] # Histogram does not have an argurment class
            stacked_hist.add_histogram(Histogram(**hist))
        if "data_hist" in serial_hist:
                if "data" in serial_hist["data_hist"]:
                    print(serial_hist["data_hist"])
                    stacked_hist.add_data_histogram(Histogram(**serial_hist["data_hist"]))
        return stacked_hist


    def add_histogram(self, hist):
        super().add_histogram(hist)
        self.__update()

        # TODO: unify data hist
        #if hist.name == "data":
        #    self.data_hist = self.hists["data"]


    def add_data_histogram(self, hist):
        if not self.bin_edges.any():
            self.bin_edges = hist.bin_edges
            self.bin_centers = hist.bin_centers
            if not self.unit and hist.unit:
                self.unit = hist.unit
            #self.bins = hist.bins
            self.range = hist.range
            #self.size = self.bin_centers.size
        else:
            assert np.array_equal(np.array(hist.bin_edges, dtype=np.float32), np.array(self.bin_edges, dtype=np.float32)), f"Hist bin edges not compatible with the rest of the stack! {hist.bin_edges} {hist.bin_edges.dtype} and {self.bin_edges} {self.bin_edges.dtype}"
        self.data_hist = hist


    def get_hist(self, name="", label="", bins=None, **kwargs):
        """Return the stacked entries as histogram

        :return: stacked histogram
        :rtype: Histogram
        """
        if not name:
            name=self.name
        if not label:
            label=name

        hist = Histogram(name, self.entries, var=self.var, lumi=self.lumi, bins=self.bin_edges, err=self.err, is_hist=True, label=label, **kwargs)

        if bins is not None:
            hist.rebin(bins)
        return hist


    def get_sig_hist(self, name=""):
        """Return the stacked signal entries as histogram

        :return: stacked histogram
        :rtype: Histogram
        """
        if not name:
            name=self.name+"_sig"
        h = None
        for _h_name, _h in self.hists.items():
            if _h.is_signal:
                if h is None:
                    h = copy.deepcopy(_h)
                else:
                    h = h + _h
        h.label = name
        h.name = name
        return h


    def get_bkg_hist(self, name=""):
        """Return the stacked backgournd entries as histogram

        :return: stacked histogram
        :rtype: Histogram
        """
        if not name:
            name=self.name+"_bkg"
        h = None
        for _h_name, _h in self.hists.items():
            if not _h.is_signal and not _h_name == "data":
                if h is None:
                    h = copy.deepcopy(_h)
                else:
                    h = h + _h
        if h:
            h.label = name
            h.name = name
        return h


    def get_data_hist(self, name=""):
        """Return the data histogram

        :return: data point histogram
        :rtype: Histogram
        """
        data_hist = copy.deepcopy(self.data_hist)
        if name:
            data_hist.name=name
            data_hist.label=name
        return data_hist


    def rebin(self, new_bin_edges):
        for name, hist in self.hists.items():
            #print(f"rebin {name} hist")
            hist.rebin(new_bin_edges)

        if self.data_hist:
            #print(f"rebin {self.data_hist.name} hist")
            self.data_hist.rebin(new_bin_edges)

        self.bin_edges = np.array(new_bin_edges)
        self._update_bins()
        self.__update()


    def re_scale(self, factor, update_lumi=True):
        super().re_scale(factor=factor, update_lumi=update_lumi)
        if self.data_hist:
            self.data_hist.re_scale(factor=factor, update_lumi=update_lumi)
        self.__update()


    def re_scale_to(self, target_lumi, update_lumi=True):
        super().re_scale_to(target_lumi=target_lumi, update_lumi=update_lumi)
        if self.data_hist:
            self.data_hist.re_scale_to(target_lumi=target_lumi, update_lumi=update_lumi)
        self.__update()


    #def plot(self, dpi=90,  xlabel="", ylabel="events", **kwargs):
    #    """Plot the stacked histogram"""
    #
    #    self.b2fig = B2Figure(auto_description=True, description=self.description)
    #    self.fig, self.ax = self.b2fig.create(ncols=1, nrows=1, dpi=dpi)
    #
    #    self.plot_ax(self.ax, **kwargs)
    #    self.b2fig.shift_offset_text_position(self.ax)
    #    self.add_labels(ax=self.ax, xlabel=xlabel, ylabel=ylabel)
    #
    #    return self.fig, self.ax



    def plot_ax(self, ax, reverse_colors=True, log=False, ylim=None, uncert_color="black", uncert_label="MC stat. unc.",  cm=None, histtype= "hatch", sort_hists=True, reverse_order=False, **kwargs):
        #colors = plt.cm.summer(np.linspace(0.1,0.8,len(self.hists)))
        #plt.cm.gist_earth
        if not self.b2fig:
            print("create b2figure")
            self.b2fig = B2Figure()

        self.color_scheme(reverse=reverse_colors, cm=cm)
        colors=self.colors
        plot_args = copy.deepcopy(kwargs.get("plot_args", {}))
        ncols_legend = plot_args.pop("ncols", 1)
        loc_legend = plot_args.pop("loc", None)

        bin_width = self.bin_edges[1:]-self.bin_edges[0:-1]
        stack = np.zeros(self.bin_centers.size)
        i=0

        hists_items = self.hists.items()
        if sort_hists:
            hists_items = sorted(self.hists.items(), key=lambda item: item[1].entries.sum(), reverse=reverse_order)
        else:
            if reverse_order:
                hists_items = reversed(hists_items)
        for name, hist in hists_items:
            # handle the date hists separate
            if name == "data":
                continue

            if hist.label:
                label = hist.label
            else:
                label = name

            color = self.signal_color if hist.is_signal else colors[name]

            if histtype == "step":
                    x = np.concatenate([self.bin_edges, [self.bin_edges[-1]]])
                    y = np.concatenate([[0], stack+hist.entries, [0]])
                    if not "linewidth" in plot_args and not "lw" in plot_args:
                        plot_args["lw"] = 0.9
                    ax.step(x, y, label=label, color=color, **plot_args)
                    #uncert = hist.err
                    #bin_width = self.bin_edges[1:]-self.bin_edges[0:-1]
                    #ax.bar(x=self.bin_centers, height=2*uncert, width=bin_width, bottom=hist.entries-uncert,
                    #    edgecolor="grey",hatch="///////", fill=False, lw=0,label="MC stat. unc." if uncert_label else "")
                    #if uncert_label: uncert_label = False
            elif histtype == "hatch":
                x = np.concatenate([self.bin_edges, [self.bin_edges[-1]]])
                y1 = copy.deepcopy(np.concatenate([[0], stack, [0]]))
                y2 = np.concatenate([[0], stack+hist.entries, [0]])
                if not "linewidth" in plot_args and not "lw" in plot_args:
                    plot_args["lw"] = 0.9
                if i%2 == 0:
                    hatch = "/////"
                else:
                    hatch = "\\\\\\\\\\"
                ax.fill_between(x, y1, y2,  color=color, **plot_args, hatch=hatch,  step='pre', facecolor="white", alpha=0.5)
                ax.step(x, y2, label=label, color=color, lw=1.2)
            else:

                    #print(f"stack {name}")
                    #ax.plot(self.bin_centers, stack+hist.entries, drawstyle="steps", color=colors[i], linewidth=0.5)
                    #ax.fill_between(self.bin_centers, stack, stack+hist.entries, label=name, step="mid",
                    #                linewidth=0, linestyle="-", color=color)
                    #ax.fill_between(self.bin_centers, stack, stack+hist.entries, label=name, step="mid",
                    #                linewidth=0, linestyle="-", color=color)
                    ax.bar(x=self.bin_centers, height=hist.entries, width=bin_width, bottom=stack,
                        color=color, edgecolor=color, lw=1,label=name, fill=False, hatch="/")

            stack += hist.entries
            i += 1

        if (len(self.hists) > 1 and self.data_hist) or (len(self.hists) > 0 and not self.data_hist):
            uncert = self.get_stat_uncert()
            ax.bar(x=self.bin_centers, height=2*uncert, width=bin_width, bottom=stack-uncert,
                    edgecolor=uncert_color,hatch="///////", fill=False, lw=0,label=uncert_label)

        if self.data_hist:
            label = self.data_hist.label if self.data_hist.label else self.data_hist.name
            ax.errorbar(self.data_hist.bin_centers, self.data_hist.entries, yerr=self.data_hist.err,
                        label=label, **self.errorbar_args)

        if log:
            ax.set_yscale("log")
            if not ylim:
                ax.set_ylim((0.5, ax.get_ylim()[1]))
        if ylim:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim((0, ax.get_ylim()[1]))

        ax.legend(loc=loc_legend, fontsize=9, frameon=False, framealpha=0.7, ncols=ncols_legend)
        self.add_labels(ax=ax)
        ax.set_xlim(self.range)
        self.b2fig.shift_offset_text_position_old(ax)


    def plot_pull_ax(self, ax, color='black', ratio=True, corr=0, xlabel="", ylabel="", ylim=None, normalized=False, **kwargs):
        data_hist = self.data_hist
        bin_centers = data_hist.bin_centers
        bin_edges = data_hist.bin_edges

        if ratio:
            with np.errstate(divide='ignore',invalid='ignore'):
                #print(self.get_stacked_entries())
                scale_factor = 1
                norm_label = ""
                if normalized:
                    # we are only interested in the shape difference
                    scale_factor = self.get_stacked_entries().sum()/data_hist.entries.sum()
                    norm_label = "norm. "
                plot = data_hist.entries/self.get_stacked_entries()*scale_factor
            ax.plot((bin_edges[0], bin_edges[-1]),[1,1], color='black', ls="-")
            self.plot_pull_bars(ax, bin_edges, plot-1, 1)
            if not ylabel:
                ylabel = norm_label + r"$\mathbf{\frac{data}{MC}}$"
        else:
            # FIXME: this is still mc over data -> data over mc
            with np.errstate(divide='ignore',invalid='ignore'):
                plot = (self.get_stacked_entries().entries-data_hist.entries)/data_hist.entries
            ax.plot((bin_edges[0], bin_edges[-1]),[0,0], color='black', ls="-")
            self.plot_pull_bars(ax, bin_edges, plot, 0)
            if not ylabel:
                ylabel = r"$\mathbf{\frac{mc-data}{data}}$"
        with np.errstate(divide='ignore',invalid='ignore'):
            plot_err = np.sqrt((self.get_stat_uncert()/data_hist.entries)**2+(data_hist.err*self.get_stacked_entries()/data_hist.entries**2)**2-2*self.get_stacked_entries()/data_hist.entries**3*self.get_stat_uncert()*data_hist.err*corr)
        #print(plot)
        #print(plot_err)
        ax.errorbar(bin_centers, plot, yerr=plot_err, **self.errorbar_args)

        ax.set_xlim(bin_edges[0], bin_edges[-1])
        if ylim:
            ax.set_ylim(ylim)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)


    def rank_bar_plot(self, name, n_show=20, log=False, color=None, plot_state="(Simulation)", xlabel="events", figsize=[6,5], parse_labels=None, fraction=True, precission=2, cm=None, reverse_colors=False):
        from matplotlib.ticker import NullLocator
        if color is None:
            self.color_scheme(reverse=reverse_colors, cm=cm)
            color = self.colors[name]
        else:
            color = self.b2fig.color(color)

        # Sorting the values and labels based on values
        labels, values = self.rank_entries(name=name, n_show=n_show, parse_labels=parse_labels, precission=precission)
        print(values)
        # Create the horizontal bar plot

        description = {"plot_state": plot_state,
                    "luminosity": self.lumi
                    }
        b2fig = B2Figure(description=description, auto_description=True)
        fig,ax = b2fig.create(figsize=figsize)
        #canvas.b2fig.add_descriptions(ax, **description)

        ax.barh(labels[::-1], values[::-1], color=color)
        #ax.barh(["A", "B"], [5, 15])

        # Ensure the bars evolve from the y-axis
        if log:
            ax.set_xscale("log")
        ax.yaxis.set_minor_locator(NullLocator())
        #ax.set_xlim([0, 2*1e7])

        ax.set_xlabel(xlabel)
        if fraction:
            ticklabels = []
            tot = self[name].entries.sum()
            for frac in values[::-1]:
                ticklabels.append(f"{np.round(frac/tot*100, precission)}")
            ax2 = ax.twinx()
            # Copying the y-ticks from the first axis to the second
            ax2.set_ylim(ax.get_ylim())
            ax2.set_yticks(ax.get_yticks())
            ax2.set_yticklabels(ticklabels)
            ax2.yaxis.set_minor_locator(NullLocator())
            ax2.set_ylabel("fraction in %")
        return fig, ax


    def rank_entries(self, name, n_show=20, parse_labels=None, precission=2):
        def sort_indices(input_list):
            # Create a list of tuples where each tuple is (index, value)
            indexed_list = list(enumerate(input_list))

            # Sort the indexed list based on the values (second element of each tuple)
            sorted_indexed_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)

            # Extract the indices from the sorted list
            sorted_indices = [index for index, value in sorted_indexed_list]

            return sorted_indices

        tot = self[name].entries.sum()
        other = 0
        sorted_indices = sort_indices(self[name].entries)
        labels = []
        values = []
        for n, i in enumerate(sorted_indices):
            entry = self[name].entries[i]
            if n >= n_show:
                other += entry
                continue
            label = parse_labels[f"{self.bin_edges[i]:.0f}"] if parse_labels is not None else f"{self.bin_edges[i]:.0f}"
            labels.append(label)
            values.append(entry)
            print(f"{self.bin_edges[i]:.0f}: \t {entry:.0f} ({np.round(entry/tot*100, precission)}%)")
        if other > 0:
            labels.append(f"other")
            values.append(other)
            print(f"other: \t {other:.0f} ({np.round(other/tot*100, precission)}%)")
        return labels, values


    def get_stat_uncert(self):
        """Calculate the stacked uncertainty of the stacked histogram using sqrt(sum(sum(lumi_scale_1**2), sum(lumi_scale_2**2), ..., sum(lumi_scale_n**2)))
        of the n histograms."""
        uncert = np.zeros(self.bin_centers.size)
        if len(self.hists) > 0:
            for name, hist in self.hists.items():
                # sigma = n2/lumi_scale *lumi_scale**2
                #uncert += hist.entries/hist.lumi_scale*hist.lumi_scale**2
                uncert += hist.err**2 #quadratic sum of each uncertainty component
        elif len(self.hists) == 0 and self.data_hist:
            uncert = self.data_hist.err**2
        return np.sqrt(uncert)


    def get_stacked_entries(self):
        """Get the sum of the stacked entries per bin."""
        entries = np.zeros(self.bin_centers.size)
        if not self.data_hist:
            if len(self.hists) > 0:
                for name, hist in self.hists.items():
                    entries += hist.entries
        else:
            if len(self.hists) >= 1:
                for name, hist in self.hists.items():
                    if not name == "data":
                        entries += hist.entries
            #else:
            #    entries = self.data_hist.entries
        return entries


    def get_ratio(self):
        """Get the ratio of stack/data per bin. This ratio can be used to apply to the simulation data
        to reweight the histogram.
        """
        return self.data_hist.entries/self.entries


    def get_ratio_err(self, corr=0):
        data_hist = self.data_hist
        with np.errstate(divide='ignore',invalid='ignore'):
            return np.sqrt((self.get_stat_uncert()/data_hist.entries)**2+(data_hist.err*self.get_stacked_entries()/data_hist.entries**2)**2-2*self.get_stacked_entries()/data_hist.entries**3*self.get_stat_uncert()*data_hist.err*corr)


    def compare(self, other, **kwargs):
        for name, hist in self.hists.items():
            if not name in other.hists:
                print(f"{name} not in other hists!")
                continue
            else:
                hist.compare(other[name], **kwargs)


    def __update(self):
        self.entries = self.get_stacked_entries()
        self.err = self.get_stat_uncert()


    def serialize(self):
        serial_hist = {}
        serial_hist["name"] = self.name
        serial_hist["class"] = "StackedHistogram"
        serial_hist["var"] = self.var
        serial_hist["unit"] = self.unit
        serial_hist["lumi"] = self.lumi
        #serial_hist["bins"] = self.bins
        serial_hist["range"] = self.range
        serial_hist["bin_edges"] = list(self.bin_edges)
        serial_hist["bin_centers"] = list(self.bin_centers)

        serial_hist["hists"] = {}
        for h_name, h in self.hists.items():
            serial_hist["hists"][h_name] = h.serialize()

        if self.data_hist:
            serial_hist["data_hist"] = self.data_hist.serialize()
        else:
            serial_hist["data_hist"] = {}

        return serial_hist


    def __add__(self, other):
        #FIXME: Check if copy or deepcopy
        self_copy = self.copy()
        for name, hist in other.hists.items():
            if name in self_copy.hists:
                this_hist = self_copy.hists[name]
                #print(f"adding {name} {this_hist.entries} + {hist.entries}")
                this_hist.check_compatibility(hist)
                this_hist.entries += hist.entries
                #add uncertainty quadratically
                this_hist.err = np.sqrt(this_hist.err*this_hist.err + hist.err*hist.err)
                this_hist.scale = 1
                this_hist.weights = None
                this_hist.update_hist()
                self_copy.hists[name] = this_hist
            else:
                #print("adding histogram")
                self_copy.add_histogram(hist)

        if self.data_hist and other.data_hist:
            self_copy.data_hist += other.data_hist
            self_copy.data_hist.name = self.data_hist.name
        elif not self.data_hist and other.data_hist:
            self_copy.add_data_histogram(other.data_hist)

        self_copy.__update()
        return self_copy


    def __getitem__(self, item):
        if item in self.hists:
            return self.hists[item]
        elif item == "data":
            return self.data_hist
        #elif item == self.data_hist.name:
        #    return self.data_hist
        else:
            pass
            #removed ValueError because test=True in tau lifetime analysis framework can reurn empty histograms
            #raise ValueError(f"{item} not a valid histogram!")


    def __str__(self):
        ret_str = "StackedHistogram Object:\n"
        ret_str +="========================\n"
        if self.lumi is None:
            ret_str +=f"lumi: None\n\n"
        else:
            ret_str +=f"lumi: {self.lumi:.2f}\n\n"

        if self.data_hist:
            ret_str += self.data_hist.__str__()
            ret_str += "\n\n"
        for name, hist in self.hists.items():
            ret_str += hist.__str__()
            ret_str += "\n\n"
        return ret_str


    def info(self):
        print(self.__str__())






def compare_histograms(name, hist_1, hist_2, name_1=None, name_2=None, additional_info="", output_dir="", log=True, pull_ylim=(0.8,1.2), pull_bar=True, fmt="o", savefig=False, suffix="",
                        callback=None, xlabel="", corr=0, normalized=False, colors=None, **kwargs):
    """Creates a histogramCanvos object from noth histograms with a pull plot.

    :param name: name of the histogram canvas
    :type name: str
    :param hist_1: first histogram
    :type hist_1: Histogram
    :param hist_2: second histogram
    :type hist_2: Histogram
    :param name_1: name/label of the first histogram, if None the name is extracted from the Histogram object
    :type name_1: str, optional
    :param name_2: name/label of the second histogram, if None the name is extracted from the Histogram object
    :type name_2: str, optional
    :param additional_info: text which show up in the top region of the plot, defaults to ""
    :type additional_info: str, optional
    :param log: log scale for the comparison plot
    :type log: bool, optional
    :param output_dir: directory wich is used to save the plot, defaults to ""
    :type output_dir: str, optional
    :param pull_ylim: range of the y axis of the pull pot, defaults to (0.8,1.2)
    :type pull_ylim: tuple, optional
    :param pull_bar: plot pullbars on top of the pull points
    :type pull_bar: bool, optional
    :param fmt: the pull data points
    :type fmt: str, optional
    :param savefig: save the resulting plot to the output drectory, defaults to False
    :type savefig: boolean, optional
    :param suffix: suffix for the figure file name
    :type suffix: str, optional
    :param callback: a callback function for ax
    :type callback: func, optional
    :param xlabel: x label text for the comparison plot
    :type xlabel: str, optional
    :param corr: correlation between the histograms, for same statistical sample use 1, independent samples use 0
    type corr: int, float
    :param colos: colors the two plots are assigned
    type colors: list of colors, optional, defaults None
    :return: The resulting HistogramCanvas from both histograms
    :rtype: HistogramCanvas
    """
    _hist_1 = copy.deepcopy(hist_1)
    _hist_2 = copy.deepcopy(hist_2)

    if _hist_1.name == _hist_2.name:
        _hist_1.name += "_1"
        _hist_1.label += "_1"
        _hist_2.name += "_2"
        _hist_2.label += "_2"

    if not name_1:
        name_1 = _hist_1.name
    else:
        _hist_1.name = name_1
        _hist_1.label = name_1
    if not name_2:
        name_2 = _hist_2.name
    else:
        _hist_2.name = name_2
        _hist_2.label = name_2


    if normalized:
        scale_factor = _hist_2.entries.sum()/_hist_1.entries.sum()
        _hist_1.re_scale(factor=scale_factor, update_lumi=False)

    hist_canvas = HistogramCanvas(lumi=_hist_1.lumi*_hist_1.lumi_scale, name=name, output_dir=output_dir, **kwargs)
    hist_canvas.add_histogram(_hist_1)
    hist_canvas.add_histogram(_hist_2)

    unit_label = r" in $\mathbf{" + _hist_1.unit + r"}$" if _hist_1.unit else ""

    pull_args = {"hist_name": name_1,
                "nom_hist_name": name_2,
                "ratio": True,
                "ylim": pull_ylim,
                "pull_bar": pull_bar,
                "fmt": fmt,
                "xlabel": xlabel if xlabel else f"{_hist_1.var}" + unit_label,
                "corr": corr}

    if additional_info:
        hist_canvas.description["additional_info"] = additional_info
    fig, ax = hist_canvas.pull_plot(histtype="step", figsize=(6,5),
                    log=log, colors=colors, pull_args=pull_args)

    if callback:
        callback(ax)

    if savefig:
        os.system("mkdir -p " + output_dir)
        file_name = f"{os.path.join(output_dir,hist_canvas.name)}{suffix}.pdf"
        fig.savefig(file_name)
        print(f"figure saved: {file_name}")
    return hist_canvas, fig, ax


def reweight(data, bin_edges, reweights, weights=np.array([])):
    """Update the weights of each event for a given data set

    :param data: data set
    :type data: list of size n
    :param bin_edges: bin edges of a histogram where the data set is sorted into
    :type bin_edges: list of size k
    :param reweights: list of weights for each of the k bins which should be applied to the events
                      in the respective bin
    :type reweights: list of size k
    :param weights: optional, weights which sould be already applied to each event
    :type weights: list of size k or None
    :return: list of the new weights
    :rtype: list of size n
    """
    if not weights.any():
        weights=np.full(data.size, 1.0)
    for wi, bi0, bi1 in zip(reweights, bin_edges[:-1], bin_edges[1:]):
        #first bin
        if bi0==bin_edges[0]:
            weights[(data<bi1)] *= wi
        #last bin
        elif bi1==bin_edges[-1]:
            weights[(data>=bi0)] *= wi
        #all bins in between
        else:
            weights[(data>=bi0) & (data<bi1)] *= wi
    return weights
