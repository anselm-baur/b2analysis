import os
import numpy as np
import matplotlib.pyplot as plt
from b2style.b2figure import B2Figure
import copy


class CanvasBase(object):

    def __init__(self, name="hist_canvas", output_dir = "") -> None:
        self.output_dir = output_dir
        self.name=name
        self.fig = None
        self.ax = None

    def savefig(self, filename=""):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not filename:
            filename = f"{self.name}.pdf"
        self.fig.savefig(os.path.join(self.output_dir, filename))

    def copy(self):
        return copy.deepcopy(self)



class HistogramBase(object):
    def __init__(self, name, data, scale=1, var="", unit="", overflow_bin=False, label="", is_hist=False, weights=np.array([]), **kwargs):
        """Creates a HistogramBase object from either data points wich gets histogramed (is_hist=False) or form already binned data
        (is_hist=True).
        """
        self.name = name
        self.var = var
        self.scale = scale # basically the weight of each event
        self.unit = unit
        self.label = label if label else name
        self.overflow_bin = overflow_bin

        #print(weights, type(weights))
        if isinstance(weights, int) or isinstance(weights, float):
            weights = np.full(data.size, weights)
        if isinstance(weights, list):
            weights = np.array(weights)
        if isinstance(data, list):
            data = np.array(data)

        if not weights.any():
            self.weights = np.full(data.size, self.scale)
        else:
            if not weights.size == data.size:
                raise ValueError(f"data and weights not same size ({weights.size}/{data.size})!")
            self.weights = weights * scale


        # We create a Histogram from an existing Histogram
        if is_hist:
            if not "bins" in kwargs or len(list(kwargs["bins"])) != len(list(data))+1 :
                raise ValueError("bins expectes when is_hist is true, with len(data)+1 == len(bins)!")
            self.bin_edges = np.array(kwargs["bins"])
            self.bin_counts = np.array(data)
            self._update_bins()
            if "err" in kwargs:
                self.err = np.array(kwargs["err"])
                self.stat_uncert = None #omit wrong uncertainty calculation
                self.entries = self.bin_counts
            else:
                self.update_hist()
            self.size = self.bin_counts.size

        # We create a new Histogram from a data sample
        else:
            #print(kwargs)
            if not overflow_bin:
                np_hist = np.histogram(data, weights=self.weights, **kwargs)
                self.bin_counts = np_hist[0]
                self.bin_edges = np_hist[1]
                self._update_bins()
                self.err = self.calc_weighted_uncert(data=data, weights=self.weights, bin_edges=self.bin_edges)

            else:
                #first add to the first and last bins bins which go to -/+ inf
                #then we trim the histograms and ad at content of the first and last inf bins
                #to the neighbor bins and cut the inf bins out
                if not "bins" in kwargs:
                    kwargs["bins"] = 50
                if type(kwargs["bins"]) is int:
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
                self.bin_counts = np_hist[0]
                self.bin_edges = np_hist[1]
                self.err = self.calc_weighted_uncert(data=data, weights=self.weights, bin_edges=self.bin_edges)
                self._trim_hist(0,1)
                self._trim_hist(-2,-1)

            self.size = self.bin_centers.size
            self.update_hist()


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
        self.entries = self.bin_counts
        pass


    def _trim_hist(self, a, b):
        # start with first bin
        if a == 0:
            #add bin content and error of the cut bins to the first new bin b
            self.bin_counts[b] = np.sum(self.bin_counts[:b+1])
            self.err[b] = np.sqrt(np.sum(self.err[:b+1]**2))
            self.bin_counts = self.bin_counts[b:]
            self.err = self.err[b:]
            self.bin_edges = self.bin_edges[b:]
        # end with last bin
        elif b == -1:
            self.bin_counts[a] = np.sum(self.bin_counts[a:])
            self.err[a] = np.sqrt(np.sum(self.err[a:]**2))
            self.bin_counts = self.bin_counts[:a+1]
            self.err = self.err[:a+1]
            self.bin_edges = self.bin_edges[:a+1]
        # s.th. in the center is cut out
        else:
            self.bin_counts[a] = np.sum(self.bin_counts[a:b+1])
            self.err[a] = np.sqrt(np.sum(self.err[a:b+1]**2))
            self.bin_counts = np.concatenate([self.bin_counts[:a+1], self.bin_counts[b+1:]])
            self.err[a] = np.concatenate([self.err[:a+1], self.err[b+1:]])
            self.bin_edges = np.concatenate([self.bin_edges[:a+1], self.bin_edges[b+1:]])
        self._update_bins()


    def _update_bins(self):
        self.bin_centers = (self.bin_edges[1:]+self.bin_edges[:-1])/2
        self.range = (self.bin_edges[0], self.bin_edges[-1])
        self.bins = self.bin_centers.size



    def plot(self, fig=None, ax=None, histtype="errorbar", dpi=100, uncert_label=True, log=False):
        if not fig and not ax:
            fig, ax = plt.subplots(ncols=1, nrows=1, dpi=dpi)

        if histtype == "errorbar":
            ax.errorbar(self.bin_centers, self.bin_counts, yerr=self.err, label=self.label,)
        elif histtype == "step":
            x = np.concatenate([self.bin_edges, [self.bin_edges[-1]]])
            y = np.concatenate([[0], self.bin_counts, [0]])
            ax.step(x, y, label=self.label, lw=0.9)
            uncert = np.sqrt(self.bin_counts)
            bin_width = self.bin_edges[1:]-self.bin_edges[0:-1]
            ax.bar(x=self.bin_centers, height=2*uncert, width=bin_width, bottom=self.bin_counts-uncert,
                    edgecolor="grey",hatch="///////", fill=False, lw=0,label="stat. unc." if uncert_label else "")
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
        return np.sqrt(self.bin_counts)*(self.scale**(1/2))


    def __sub__(self, other):
        self.check_compatibility(other)
        diff_hist = self.copy()
        diff_hist.bin_counts -= other.bin_counts
        diff_hist.scale = 1
        diff_hist.weights = None
        diff_hist.name += " - " + other.name
        diff_hist.update_hist()
        return diff_hist


    def __add__(self, other):
        self.check_compatibility(other)
        add_hist = self.copy()
        add_hist.bin_counts += other.bin_counts
        add_hist.err = np.sqrt(self.err**2 + other.err**2)
        add_hist.scale = 1
        add_hist.weights = None
        add_hist.name += " + " + other.name
        add_hist.update_hist()
        return add_hist

    def __truediv__(self, other):
        self.check_compatibility(other)
        return np.array(self.entries)/np.array(other.entries)


    def check_compatibility(self, other):
        assert np.array_equal(np.array(other.bin_edges, dtype=np.float32), self.bin_edges.astype(np.float32)), "Hist bin edges not compatible!"
        assert self.unit == other.unit, "Hist units not compatible!"


    def copy(self):
        return copy.deepcopy(self)




class Histogram(HistogramBase):
    """Analysis Histogram Class."""

    def __init__(self, name, data, lumi, lumi_scale=1, is_signal=False, is_hist=False, color=None,  **kwargs):
        super().__init__(name=name, data=data, scale=lumi_scale, is_hist=is_hist, **kwargs)
        self.is_signal = is_signal
        self.lumi = lumi
        self.lumi_scale = lumi_scale # basically the weight of each event
        self.color = color


    def plot(self, fig=None, ax=None, histtype="errorbar", dpi=100, uncert_label=True, log=False):
        b2fig = B2Figure()
        if not fig and not ax:
            fig, ax = b2fig.create(ncols=1, nrows=1, dpi=dpi)

        if histtype == "errorbar":
            ax.errorbar(self.bin_centers, self.bin_counts, yerr=self.err, label=self.label, **b2fig.errorbar_args)
        elif histtype == "step":
            x = np.concatenate([self.bin_edges, [self.bin_edges[-1]]])
            y = np.concatenate([[0], self.bin_counts, [0]])
            ax.step(x, y, label=self.label, lw=0.9)
            uncert = np.sqrt(self.bin_counts)
            bin_width = self.bin_edges[1:]-self.bin_edges[0:-1]
            ax.bar(x=self.bin_centers, height=2*uncert, width=bin_width, bottom=self.bin_counts-uncert,
                    edgecolor="grey",hatch="///////", fill=False, lw=0,label="MC stat. unc." if uncert_label else "")
            if uncert_label: uncert_label = False
        unit = f" in {self.unit}"
        ax.set_xlim((*self.range))
        ax.set_xlabel(f"{self.var}{unit if self.unit else ''}")
        ax.set_ylabel("events")
        if log:
            ax.set_yscale("log")
        ax.legend()

        return fig, ax


    def __str__(self):
        ret_str  = "Histogram Object\n"
        ret_str += "================\n"
        ret_str += f"name: {self.name}\n"
        ret_str += f"var: {self.var}\n"
        ret_str += f"bins: {self.bins}\n"
        ret_str += f"entries: {np.sum(self.entries):.0f}\n"
        ret_str += f"weights: {np.mean(self.weights):.3f}\n"
        ret_str += f"lumi: {self.lumi}"
        return ret_str


    def serialize(self):
        """Create a serialized version of the histogram for storage in a file.

        :return: the histogram content in strings and lists
        :rtype: dict
        """
        ser_hist = {"name": self.name,
                    "data": list(self.entries),
                    "err": list(self.err),
                    "lumi": self.lumi,
                    "bins": list(self.bin_edges),
                    "lumi_scale": self.scale,
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

    def __init__(self, lumi, var="", unit="", additional_info="", is_simulation=True, is_preliminary=False, **kwargs):
        super().__init__(**kwargs)
        self.lumi = lumi
        self.var = var
        self.hists = {}
        self.unit = unit
        self.bin_edges = np.array([])
        self.bin_centers = np.array([])
        self.description = {"luminosity": self.lumi,
                            "simulation": is_simulation,
                            "additional_info": additional_info,
                            "preliminary": not is_simulation and is_preliminary}

        self.b2fig = None
        self.fig = None
        self.ax = None

        self.signals = 0
        self.labels = {}
        self.colors = {}

    def add_histogram(self, hist, label=True, color=None):
        """Add a histogram to the canvas."""
        if not self.bin_edges.any():
            self.bin_edges = np.array(hist.bin_edges, dtype=np.float32)
            self.bin_centers = hist.bin_centers
            if not self.unit and hist.unit:
                self.unit = hist.unit
            self.bins = hist.bins
            self.range = hist.range
            self.size = self.bin_centers.size
        else:
            assert np.array_equal(np.array(hist.bin_edges, dtype=np.float32), self.bin_edges), "Hist bin edges not compatible with the rest of the stack!"
        if not np.round(hist.lumi * hist.lumi_scale, 1) == np.round(self.lumi, 1):
            raise ValueError(f"Histogram luminosity {hist.lumi} and histogram luminosity scale {hist.lumi_scale} not compatible with desired luminosity {self.lumi}")
        self.hists[hist.name] = hist
        self.labels[hist.name] = label
        if color:
            self.colors[hist.name] = color
        elif hist.color:
            self.colors[hist.name] = hist.color
        if hist.is_signal:
            self.signals += 1
        self.__update()


    def __update(self):
        pass


    def create_histogram(self, name, data, lumi, lumi_scale=1, is_signal=False, **kwargs):
        """Create a histogram from data and add it to the stack"""
        self.add_histogram(Histogram(name, data, lumi, lumi_scale, is_signal=is_signal, **kwargs))


    def plot(self, dpi=90, figsize=(6,6), pull_args={}, additional_info="", **kwargs):
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

        self.plot_ax(self.ax, **kwargs)
        self.b2fig.shift_offset_text_position(self.ax)
        self.add_labels(ax=self.ax)

        return self.fig, self.ax

    def plot_ax(self, ax, xlabel="", histtype="errorbar", log=False, x_log=False, colors=None, reverse_colors=False):
        if not colors:
            colors = []
        if len(colors) <  len(self.hists):
            if len(self.colors) <  len(self.hists):
                print("create colors...")
                self.color_scheme(reverse=reverse_colors)
            colors = self.colors
            print(colors)
            print(self.hists.keys())

        uncert_label = True
        for i, (name, hist) in enumerate(self.hists.items()):
            label = self.get_label(name)
            if histtype == "errorbar":
                ax.errorbar(self.bin_centers, hist.entries, yerr=hist.err, label=name, **self.b2fig.errorbar_args)
            elif histtype == "step":
                x = np.concatenate([self.bin_edges, [self.bin_edges[-1]]])
                y = np.concatenate([[0], hist.entries, [0]])
                if type(colors) == dict:
                    ax.step(x, y, label=label, lw=0.9, color=colors[name])
                else:
                    ax.step(x, y, label=label, lw=0.9, color=colors[i])
                uncert = hist.err
                bin_width = self.bin_edges[1:]-self.bin_edges[0:-1]
                ax.bar(x=self.bin_centers, height=2*uncert, width=bin_width, bottom=hist.entries-uncert,
                       edgecolor="grey",hatch="///////", fill=False, lw=0,label="MC stat. unc." if uncert_label else "")
                if uncert_label: uncert_label = False
            else:
                raise ValueError(f"histtype {histtype} not implemented!")
        ax.set_xlim((*self.range))
        if xlabel:
            ax.set_xlabel(xlabel)
        if x_log:
            ax.set_xscale("symlog")
        else:
            ax.set_xscale("linear")
        if log:
            ax.set_yscale("log")
            ax.set_ylim((0.5, ax.get_ylim()[1]))
        ax.set_ylabel("events")
        self.b2fig.shift_offset_text_position(ax)
        #ax.legend(loc='upper left', prop={'size': 7})
        ax.legend()
        self.b2fig.shift_offset_text_position_old(ax)


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


    def color_scheme(self, reverse=False, exclude_signals=True,  cm=plt.cm.gist_earth):
        #cm = plt.cm.seismic
        cm_low = 0.1
        cm_high = 0.8

        if exclude_signals:
            nhists = len(self.hists)-self.signals
            signals = self.get_signal_names()
            iter_histsts = copy.deepcopy(self.hists)
            for sig in signals:
                del(iter_histsts[sig])
        else:
            nhists = len(self.hists)
        linspace = np.linspace(cm_low,cm_high,nhists)

        for name, color in zip(iter_histsts, cm(np.flip(linspace) if reverse else linspace)):
            if self.hists[name].color:
                self.colors[name] = self.hists[name].color
            else:
                self.colors[name] = color
        self.signal_color = plt.cm.seismic(0.9)


    def pull_plot_old(self, ax, hist_name, nom_hist_name, color='black', ratio=True, corr=0, xlabel="", ylabel="", ylim=None, fmt="o--", pull_bar=False):
        nom_hist = self.hists[nom_hist_name]
        bin_centers = nom_hist.bin_centers
        bin_edges = nom_hist.bin_edges
        bins = nom_hist.size

        def plot_pull_bars(y, bottom=0):
            widths = bin_edges[1:]-bin_edges[:-1]
            ax.bar(bin_edges[:-1], y, widths, align="edge", color="lightgrey", bottom=bottom, zorder=0)

        def iterate_plot(hist, hist_color):
            nonlocal ylabel # idk why we need this here but without it will not find ylabel variable
            nonlocal pull_bar
            if ratio:
                plot = hist.entries/nom_hist.entries
                if pull_bar:
                    self.plot_pull_bars(ax, bin_edges, plot-1, 1)
                ax.plot((bin_edges[0], bin_edges[-1]),[1,1], color='black', ls="-")
                if not ylabel:
                    ylabel = r"$\mathbf{\frac{"+hist.name.replace("_",r"\_").replace(" ",r"\;")+r"}{"+nom_hist.name.replace("_",r"\_").replace(" ",r"\;")+r"}}$"
            else:
                plot = (hist.entries-nom_hist.entries)/nom_hist.entries
                if pull_bar:
                    self.plot_pull_bars(ax, bin_edges, plot)
                ax.plot((bin_edges[0], bin_edges[-1]),[0,0], color='black', ls="-")
                if not ylabel:
                    hist_label = hist.name.replace("_",r"\_").replace(" ",r"\;")
                    nom_hist_label = nom_hist.name.replace("_",r"\_").replace(" ",r"\;")
                    ylabel = r"$\mathbf{\frac{"+hist_label+r"-"+nom_hist_label+r"}{"+nom_hist_label+r"}}$"
            plot_err = np.sqrt((hist.err/nom_hist.entries)**2+(nom_hist.err*hist.entries/nom_hist.entries**2)**2-2*hist.entries/nom_hist.entries**3*hist.err*nom_hist.err*corr)
            ax.errorbar(bin_centers, plot, yerr=plot_err, fmt=fmt, color=hist_color, markersize='2.2', elinewidth=0.5)

        if type(hist_name) == list:
            if type(color) != dict and len(color) != len(hist_name):
                #print("WARNING: color must have the same type and size as hist_name! -> gonna create new color scheme...")
                #self.color_scheme()
                # we use the default canvas colors
               color = self.colors
            for name in hist_name:
                hist = self.hists[name]
                iterate_plot(hist, color[name])
        else:
            iterate_plot(self.hists[hist_name], color)


        ax.set_xlim(bin_edges[0], bin_edges[-1])
        if ylim:
            ax.set_ylim(ylim)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)


    def pull_plot(self, dpi=90, figsize=(6,6), pull_args={}, additional_info="", **kwargs):
        pull_args = copy.deepcopy(pull_args)
        self.b2fig = B2Figure(auto_description=False)
        self.fig, ax = self.b2fig.create(ncols=1, nrows=2, dpi=dpi, figsize=figsize, gridspec_kw={'height_ratios': [2, 1]})
        self.ax = ax[0]
        if additional_info:
                self.description["additional_info"] = additional_info
        self.b2fig.add_descriptions(ax=self.ax, **self.description)
        self.ax_pull = ax[1]
        #move the xlabel to the pull plot

        self.ax.set_xticklabels([])
        self.fig.subplots_adjust(hspace=0.05)

        self.plot_ax(self.ax, **kwargs)
        self.b2fig.shift_offset_text_position(self.ax)
        self.add_labels(ax=self.ax)
        if "x_log" in kwargs:
            pull_args["x_log"] = kwargs["x_log"]
            #print(pull_args["x_log"])
        self.plot_pull_ax(self.ax_pull, **pull_args)

        self.ax.set_xticklabels([])
        self.fig.subplots_adjust(hspace=0.05)

        return self.fig, ax


    def plot_pull_ax(self, ax, hist_name, nom_hist_name, color='black', ratio=True, corr=0, xlabel="", ylabel="", ylim=None, fmt="o--",
                     pull_bar=False, x_log=False):
        nom_hist = self.hists[nom_hist_name]
        bin_centers = nom_hist.bin_centers
        bin_edges = nom_hist.bin_edges
        bins = nom_hist.size

        def iterate_plot(hist, hist_color):
            nonlocal ylabel # idk why we need this here but without it will not find ylabel variable
            nonlocal pull_bar
            if ratio:
                plot = hist.entries/nom_hist.entries
                if pull_bar:
                    self.plot_pull_bars(ax, bin_edges, plot-1, 1)
                ax.plot((bin_edges[0], bin_edges[-1]),[1,1], color='black', ls="-")
                if not ylabel:
                    ylabel = r"$\mathbf{\frac{"+hist.name.replace("_",r"\_").replace(" ",r"\;")+r"}{"+nom_hist.name.replace("_",r"\_").replace(" ",r"\;")+r"}}$"
            else:
                plot = (hist.entries-nom_hist.entries)/nom_hist.entries
                if pull_bar:
                    self.plot_pull_bars(ax, bin_edges, plot)
                ax.plot((bin_edges[0], bin_edges[-1]),[0,0], color='black', ls="-")
                if not ylabel:
                    hist_label = hist.name.replace("_",r"\_").replace(" ",r"\;")
                    nom_hist_label = nom_hist.name.replace("_",r"\_").replace(" ",r"\;")
                    ylabel = r"$\mathbf{\frac{"+hist_label+r"-"+nom_hist_label+r"}{"+nom_hist_label+r"}}$"
            plot_err = np.sqrt((hist.err/nom_hist.entries)**2+(nom_hist.err*hist.entries/nom_hist.entries**2)**2-2*hist.entries/nom_hist.entries**3*hist.err*nom_hist.err*corr)
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

    def __init__(self, lumi, var="", unit="", additional_info="", is_simulation=True, is_preliminary=False, name="stacked_histogram", **kwargs):
        super().__init__(lumi, var, unit, additional_info, is_simulation, is_preliminary, name=name, **kwargs)
        self.data_hist = None
        self.errorbar_args = {"fmt":'o',
                              "color": "black",
                              "markersize": 2.2,
                              "elinewidth": 0.5
                              }

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
            stacked_hist.add_histogram(Histogram(**hist))
        if "data_hist" in serial_hist:
                stacked_hist.add_data_histogram(Histogram(**serial_hist["data_hist"]))
        return stacked_hist


    def add_histogram(self, hist):
        super().add_histogram(hist)
        self.__update()


    def add_data_histogram(self, hist):
        if not self.bin_edges.any():
            self.bin_edges = hist.bin_edges
            self.bin_centers = hist.bin_centers
            if not self.unit and hist.unit:
                self.unit = hist.unit
            self.bins = hist.bins
            self.range = hist.range
            self.size = self.bin_centers.size
        else:
            assert np.array_equal(hist.bin_edges, self.bin_edges), "Hist bin edges not compatible with the rest of the stack!"
        self.data_hist = hist


    def get_hist(self, name=""):
        """Return the stacked entries as histogram

        :return: stacked histogram
        :rtype: Histogram
        """
        if not name:
            name=self.name
        return Histogram(name, self.entries, var=self.var, lumi=self.lumi, bins=self.bin_edges, err=self.err, is_hist=True, label=name)


    def get_data_hist(self, name=""):
        """Return the data histogram

        :return: data point histogram
        :rtype: Histogram
        """
        if name:
            data_hist = copy.deepcopy(self.data_hist)
            data_hist.name=name
            data_hist.label=name
            return data_hist
        else:
            return self.data_hist


    def plot(self, dpi=90,  xlabel="", ylabel="events", **kwargs):
        """Plot the stacked histogram"""

        self.b2fig = B2Figure(auto_description=True, description=self.description)
        self.fig, self.ax = self.b2fig.create(ncols=1, nrows=1, dpi=dpi)

        self.plot_ax(self.ax, **kwargs)
        self.b2fig.shift_offset_text_position(self.ax)
        self.add_labels(ax=self.ax, xlabel=xlabel, ylabel=ylabel)

        return self.fig, self.ax


    def pull_plot(self, dpi=90, figsize=(6,6), pull_args={}, **kwargs):
        """Plot stacked histogram and a pull distribution.
        """
        self.b2fig = B2Figure(auto_description=False)
        self.fig, ax = self.b2fig.create(ncols=1, nrows=2, dpi=dpi, figsize=figsize, gridspec_kw={'height_ratios': [2, 1]})
        self.ax = ax[0]
        self.b2fig.add_descriptions(ax=self.ax, **self.description)
        self.ax_pull = ax[1]
        #move the xlabel to the pull plot

        self.ax.set_xticklabels([])
        self.fig.subplots_adjust(hspace=0.05)

        self.plot_ax(self.ax, **kwargs)
        self.b2fig.shift_offset_text_position(self.ax)
        self.add_labels(ax=self.ax)
        self.plot_pull_ax(self.ax_pull, pull_args=pull_args, **kwargs)

        return self.fig, ax


    def plot_ax(self, ax, reverse_colors=False, log=False, ylim=None, uncert_color="black", uncert_label="MC stat. unc.",  cm=plt.cm.gist_earth, **kwargs):
        #colors = plt.cm.summer(np.linspace(0.1,0.8,len(self.hists)))
        if not self.b2fig:
            self.b2fig = B2Figure()

        self.color_scheme(reverse=reverse_colors, cm=cm)
        colors=self.colors

        bin_width = self.bin_edges[1:]-self.bin_edges[0:-1]
        stack = np.zeros(self.bin_centers.size)
        i=0
        for name, hist in self.hists.items():
            color = self.signal_color if hist.is_signal else colors[name]
            #print(f"stack {name}")
            #ax.plot(self.bin_centers, stack+hist.bin_counts, drawstyle="steps", color=colors[i], linewidth=0.5)
            #ax.fill_between(self.bin_centers, stack, stack+hist.bin_counts, label=name, step="mid",
            #                linewidth=0, linestyle="-", color=color)
            #ax.fill_between(self.bin_centers, stack, stack+hist.bin_counts, label=name, step="mid",
            #                linewidth=0, linestyle="-", color=color)
            ax.bar(x=self.bin_centers, height=hist.bin_counts, width=bin_width, bottom=stack,
                color=color, edgecolor=color, lw=0.1,label=name)

            stack += hist.bin_counts
            i += 1

        uncert = self.get_stat_uncert()
        ax.bar(x=self.bin_centers, height=2*uncert, width=bin_width, bottom=stack-uncert,
                edgecolor=uncert_color,hatch="///////", fill=False, lw=0,label=uncert_label)

        if self.data_hist:
            ax.errorbar(self.data_hist.bin_centers, self.data_hist.entries, yerr=self.data_hist.err,
                        label="data", **self.errorbar_args)

        if log:
            ax.set_yscale("log")
            if not ylim:
                ax.set_ylim((0.5, ax.get_ylim()[1]))
        if ylim:
            ax.set_ylim(ylim)

        ax.legend()
        self.add_labels(ax=ax)
        ax.set_xlim(self.range)
        self.b2fig.shift_offset_text_position_old(ax)


    def plot_pull_ax(self, ax, color='black', ratio=True, corr=0, xlabel="", ylabel="", ylim=None, pull_args={}, **kwargs):
        data_hist = self.data_hist
        bin_centers = data_hist.bin_centers
        bin_edges = data_hist.bin_edges

        if ratio:
            plot = self.get_stacked_entries()/data_hist.entries
            ax.plot((bin_edges[0], bin_edges[-1]),[1,1], color='black', ls="-")
            self.plot_pull_bars(ax, bin_edges, plot-1, 1)
            if not ylabel:
                ylabel = r"$\mathbf{\frac{MC}{data}}$"
        else:
            plot = (self.get_stacked_entries().entries-data_hist.entries)/data_hist.entries
            ax.plot((bin_edges[0], bin_edges[-1]),[0,0], color='black', ls="-")
            self.plot_pull_bars(ax, bin_edges, plot, 0)
            if not ylabel:
                ylabel = r"$\mathbf{\frac{mc-data}{data}}$"
        plot_err = np.sqrt((self.get_stat_uncert()/data_hist.entries)**2+(data_hist.err*self.get_stacked_entries()/data_hist.entries**2)**2-2*self.get_stacked_entries()/data_hist.entries**3*self.get_stat_uncert()*data_hist.err*corr)
        ax.errorbar(bin_centers, plot, yerr=plot_err, **self.errorbar_args)

        ax.set_xlim(bin_edges[0], bin_edges[-1])
        if ylim:
            ax.set_ylim(ylim)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)


    def get_stat_uncert(self):
        """Calculate the stacked uncertainty of the stacked histogram using sqrt(sum(sum(lumi_scale_1**2), sum(lumi_scale_2**2), ..., sum(lumi_scale_n**2)))
        of the n histograms."""
        uncert = np.zeros(self.bin_centers.size)
        if len(self.hists) > 0:
            for name, hist in self.hists.items():
                # sigma = n2/lumi_scale *lumi_scale**2
                #uncert += hist.entries/hist.lumi_scale*hist.lumi_scale**2
                uncert += hist.err**2 #quadratic sum of each uncertainty component
        elif len(self.hists) ==0 and self.data_hist:
            uncert = self.data_hist.err**2
        return np.sqrt(uncert)


    def get_stacked_entries(self):
        """Get the sum of the stacked entries per bin."""
        entries = np.zeros(self.bin_centers.size)
        if len(self.hists) > 0:
            for name, hist in self.hists.items():
                entries += hist.entries
        elif len(self.hists) ==0 and self.data_hist:
            entries = self.data_hist.entries
        return entries


    def __update(self):
        self.entries = self.get_stacked_entries()
        self.err = self.get_stat_uncert()


    def serialize(self):
        serial_hist = {}
        serial_hist["name"] = self.name
        serial_hist["var"] = self.var
        serial_hist["unit"] = self.unit
        serial_hist["lumi"] = self.lumi
        serial_hist["bins"] = self.bins
        serial_hist["range"] = self.range
        serial_hist["bin_edges"] = list(self.bin_edges)
        serial_hist["bin_centers"] = list(self.bin_centers)

        serial_hist["hists"] = {}
        for h_name, h in self.hists.items():
            #serial_hist["hists"][h_name] = {"entries": list(h.entries),
            #                                "err": list(h.err),
            #                                "color": h.color}
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
                print("adding histogram")
                self_copy.add_histogram(hist)

        if self.data_hist and other.data_hist:
            self_copy.data_hist += other.data_hist
            self_copy.name = self.data_hist.name
        elif not self.data_hist and other.data_hist:
            self_copy.add_data_histogram(other.data_hist)

        self_copy.__update()
        return self_copy


    def __getitem__(self, item):
        if item in self.hists:
            return self.hists[item]
        elif item == "data":
            return self.data_hist
        elif item == self.data_hist.name:
            return self.data_hist
        else:
            raise ValueError(f"{item} not a valid histogram!")




def compare_histograms(name, hist_1, hist_2, name_1=None, name_2=None, additional_info="", output_dir="", log=True, pull_ylim=(0.8,1.2), pull_bar=True, fmt="o", savefig=False, suffix="",
                        callback=None, xlabel="", **kwargs):
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
    :return: The resulting HistogramCanvas from both histograms
    :rtype: HistogramCanvas
    """
    _hist_1 = copy.deepcopy(hist_1)
    _hist_2 = copy.deepcopy(hist_2)
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

    hist_canvas = HistogramCanvas(lumi=_hist_1.lumi, name=name, output_dir=output_dir, **kwargs)
    hist_canvas.add_histogram(_hist_1)
    hist_canvas.add_histogram(_hist_2)

    pull_args = {"hist_name": name_1,
                "nom_hist_name": name_2,
                "ratio": True,
                "ylim": pull_ylim,
                "pull_bar": pull_bar,
                "fmt": fmt,
                "xlabel": xlabel}
    unit_label = r" in $\mathbf{" + _hist_1.unit + r"}$" if _hist_1.unit else ""
    if additional_info:
        hist_canvas.description["additional_info"] = additional_info
    fig, ax = hist_canvas.pull_plot(xlabel=f"{_hist_1.var}" + unit_label, histtype="step", figsize=(6,5),
                    log=log, colors=["blue", "red"], pull_args=pull_args)

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
