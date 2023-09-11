import os
import numpy as np
import matplotlib.pyplot as plt
from b2style import B2Figure
import copy


class CanvasBase:

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



class HistogramBase:
    def __init__(self, name, data, scale=1, var="", unit="", overflow_bin=False, label="", is_hist=False, **kwargs):
        """Creates a HistogramBase object from either data points wich gets histogramed (is_hist=False) or form already binned data
        (is_hist=True).
        """
        self.name = name
        self.var = var
        self.scale = scale # basically the weight of each event
        self.unit = unit
        self.label = label if label else name
        self.overflow_bin = overflow_bin

        self.weights = np.full(data.size, self.scale)

        if is_hist:
            if not "bins" in kwargs or len(list(kwargs["bins"])) != len(list(data))+1 :
                raise ValueError("bins expectes when is_hist is true, with len(data)+1 == len(bins)!")
            self.bin_edges = kwargs["bins"]
            self.bin_counts = data
            self._update_bins()
            if "err" in kwargs:
                self.err = kwargs["err"]
                self.entries = self.bin_counts
            else:
                self.update_hist()
            self.size = self.bin_counts.size

        else:

            if not overflow_bin:
                np_hist = np.histogram(data, weights=self.weights, **kwargs)
                self.bin_counts = np_hist[0]
                self.bin_edges = np_hist[1]
                self._update_bins()

            else:
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
                self._trim_hist(0,1)
                self._trim_hist(-2,-1)

            self.size = self.bin_centers.size
            self.update_hist()


    def update_hist(self):
        """Recalculate the uncertainty and set the entries atribute.
        """
        self.err = self.stat_uncert()
        self.entries = self.bin_counts


    def _trim_hist(self, a, b):
        if a == 0:
            self.bin_counts[b] = np.sum(self.bin_counts[:b+1])
            self.bin_counts = self.bin_counts[b:]
            self.bin_edges = self.bin_edges[b:]
        elif b == -1:
            self.bin_counts[a] = np.sum(self.bin_counts[a:])
            self.bin_counts = self.bin_counts[:a+1]
            self.bin_edges = self.bin_edges[:a+1]
        else:
            self.bin_counts[a] = np.sum(self.bin_counts[a:b+1])
            self.bin_counts = np.concatenate([self.bin_counts[:a+1], self.bin_counts[b+1:]])
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
            ax.errorbar(self.bin_centers, self.bin_counts, yerr=self.stat_uncert(), label=self.label,)
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
        diff_hist = self.copy()
        diff_hist.bin_counts += other.bin_counts
        diff_hist.scale = 1
        diff_hist.weights = None
        diff_hist.name += " + " + other.name
        diff_hist.update_hist()
        return diff_hist


    def check_compatibility(self, other):
        assert np.array_equal(np.array(other.bin_edges, dtype=np.float32), self.bin_edges.astype(np.float32)), "Hist bin edges not compatible!"
        assert self.unit == other.unit, "Hist units not compatible!"


    def copy(self):
        return copy.deepcopy(self)



class Histogram(HistogramBase):
    """Analysis Histogram Class."""

    def __init__(self, name, data, lumi, lumi_scale=1, is_signal=False, is_hist=False, **kwargs):
        super().__init__(name=name, data=data, scale=lumi_scale, is_hist=is_hist, **kwargs)
        self.is_signal = is_signal
        self.lumi = lumi
        self.lumi_scale = lumi_scale # basically the weight of each event


    def plot(self, fig=None, ax=None, histtype="errorbar", dpi=100, uncert_label=True, log=False):
        b2fig = B2Figure()
        if not fig and not ax:
            fig, ax = b2fig.create(ncols=1, nrows=1, dpi=dpi)

        if histtype == "errorbar":
            ax.errorbar(self.bin_centers, self.bin_counts, yerr=self.stat_uncert(), label=self.label, **b2fig.errorbar_args)
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
        if not "colors" in kwargs and len(self.colors) != len(self.hists) or len(kwargs["colors"]) != len(self.hists):
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

    def plot_ax(self, ax, xlabel="", histtype="errorbar", log=False, colors=[], reverse_colors=False):
        if len(colors) < 1 :
            if len(self.colors) < 1:
                print("create colors...")
                self.color_scheme(reverse=reverse_colors)
            colors = self.colors

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


    def color_scheme(self, reverse=False, exclude_signals=True,  cm=plt.cm.gist_earth):
        #cm = plt.cm.seismic
        cm_low = 0.1
        cm_high = 0.8

        if exclude_signals:
            nhists = len(self.hists)-self.signals
        else:
            nhists = len(self.hists)
        linspace = np.linspace(cm_low,cm_high,nhists)
        for name, color in zip(self.hists, cm(np.flip(linspace) if reverse else linspace)):
            self.colors[name] = color
        self.signal_color = plt.cm.seismic(0.9)


    def pull_plot(self, ax, hist_name, nom_hist_name, color='black', ratio=True, corr=0, xlabel="", ylabel="", ylim=None, fmt="o--", pull_bar=False):
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
                    ylabel = r"$\mathbf{\frac{"+hist.name.replace("_","\_").replace(" ","\;")+r"}{"+nom_hist.name.replace("_","\_").replace(" ","\;")+r"}}$"
            else:
                plot = (hist.entries-nom_hist.entries)/nom_hist.entries
                if pull_bar:
                    self.plot_pull_bars(ax, bin_edges, plot)
                ax.plot((bin_edges[0], bin_edges[-1]),[0,0], color='black', ls="-")
                if not ylabel:
                    hist_label = hist.name.replace("_","\_").replace(" ","\;")
                    nom_hist_label = nom_hist.name.replace("_","\_").replace(" ","\;")
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

    def __init__(self, lumi, var="", unit="", additional_info="", is_simulation=True, is_preliminary=False, **kwargs):
        super().__init__(lumi, var, unit, additional_info, is_simulation, is_preliminary, **kwargs)
        self.data_hist = None
        self.errorbar_args = {"fmt":'o',
                              "color": "black",
                              "markersize": 2.2,
                              "elinewidth": 0.5
                              }

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
        return Histogram(name, self.entries, lumi=self.lumi, bins=self.bin_edges, err=self.err, is_hist=True)

    def get_data_hist(self, name=""):
        """Return the data histogram

        :return: data point histogram
        :rtype: Histogram
        """
        if name:
            data_hist = copy.deepcopy(self.data_hist)
            data_hist.name=name
            return data_hist
        else:
            return self.data_hist


    def plot(self, dpi=90, **kwargs):
        """Plot the stacked histogram"""

        self.b2fig = B2Figure(auto_description=True, description=self.description)
        self.fig, self.ax = self.b2fig.create(ncols=1, nrows=1, dpi=dpi)

        self.plot_ax(self.ax, **kwargs)
        self.b2fig.shift_offset_text_position(self.ax)
        self.add_labels(ax=self.ax)

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
            ax.errorbar(self.data_hist.bin_centers, self.data_hist.entries, yerr=self.data_hist.stat_uncert(),
                        label="data", **self.errorbar_args)

        if log:
            ax.set_yscale("log")
            if not ylim:
                ax.set_ylim((0.5, ax.get_ylim()[1]))
        if ylim:
            ax.set_ylim(ylim)

        ax.legend()
        self.add_labels(ax=ax)
        ax.set_xlim((*self.range))
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
                uncert += hist.entries/hist.lumi_scale*hist.lumi_scale**2
        elif len(self.hists) ==0 and self.data_hist:
            uncert = self.data_hist.err
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

    #@property
    #def entries(self):
    #    return self.get_stacked_entries()

    #@property
    #def err(self):
    #    return self.get_stat_uncert()


class StackedDataHistogram(StackedHistogram):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_hist

    def add_data_histogram(self, hist):
        if not self.bin_edges.any():
            self.bin_edges = hist.bin_edges
            self.bin_centers = hist.bin_centers
        else:
            assert np.array_equal(hist.bin_edges, self.bin_edges), "Hist bin edges not compatible with the rest of the stack!"

        self.data_hist = hist

    def create_data_histogram(self,  name, data, lumi, lumi_scale=1, **kwargs):
        self.add_data_histogram(Histogram(name, data, lumi, lumi_scale=1, **kwargs))




def compare_histograms(name, hist_1, hist_2, name_1=None, name_2=None, additional_info="", output_dir="", log=True, pull_ylim=(0.8,1.2), pull_bar=True, fmt="o", savefig=False, suffix="", callback=None):
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

    hist_canvas = HistogramCanvas(lumi=_hist_1.lumi, name=name, output_dir=output_dir)
    hist_canvas.add_histogram(_hist_1)
    hist_canvas.add_histogram(_hist_2)

    pull_args = {"hist_name": name_1,
                "nom_hist_name": name_2,
                "ratio": True,
                "ylim": pull_ylim,
                "pull_bar": pull_bar,
                "fmt": fmt}
    unit_label = r" in $\mathbf{" + _hist_1.unit + r"}$" if _hist_1.unit else ""
    if additional_info:
        hist_canvas.description["additional_info"] = additional_info
    fig, ax = hist_canvas.plot(xlabel=f"{_hist_1.var}" + unit_label, histtype="step", figsize=(6,5),
                    log=log, colors=["blue", "red"], pull_args=pull_args)

    if callback:
        callback(ax)

    if savefig:
        os.system("mkdir -p " + output_dir)
        file_name = f"{os.path.join(output_dir,hist_canvas.name)}{suffix}.pdf"
        fig.savefig(file_name)
        print(f"figure saved: {file_name}")
    return hist_canvas, fig, ax