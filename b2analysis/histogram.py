import numpy as np
import matplotlib.pyplot as plt
from b2style import B2Figure

class Histogram:
    """Analysis Histogram Class."""

    def __init__(self, name, data, lumi, lumi_scale=1, var="", unit="", is_signal=False, overflow_bin=False, **kwargs):
        self.is_signal = is_signal
        self.name = name
        self.var = var
        self.lumi = lumi
        self.lumi_scale = lumi_scale # basically the weight of each event
        self.unit = unit
        self.overflow_bin = overflow_bin

        if not overflow_bin:
            np_hist = np.histogram(data, **kwargs)
            self.bin_counts = np_hist[0]
            self.bin_edges = np_hist[1]
            self.bin_centers = (self.bin_edges[1:]+self.bin_edges[:-1])/2

            self.range = (self.bin_edges[0], self.bin_edges[-1])
            self.bins = self.bin_centers.size
            
        else:
            if not "range" in kwargs:
                kwargs["range"] = (np.min(data), np.max(data))
            if not "bins" in kwargs:
                kwargs["bins"] = 50
            kwargs["bins"] = np.concatenate([[-np.inf], 
                                            np.linspace(*kwargs["range"], kwargs["bins"]),
                                            [np.inf]])
            np_hist = np.histogram(data, **kwargs)
            self.bin_counts = np_hist[0]
            self.bin_edges = np_hist[1]
            
            
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

    def plot(self, fig=None, ax=None, dpi=100):
        b2fig = B2Figure()
        if not fig and not ax:
            fig, ax = b2fig.create(ncols=1, nrows=1, dpi=dpi)
        ax.errorbar(self.bin_centers, self.bin_counts, yerr=self.stat_uncert(), label=self.name, **b2fig.errorbar_args)
        unit = f" in {self.unit}"
        ax.set_xlabel(f"{self.var}{unit if self.unit else ''}")
        ax.set_ylabel("events")
        ax.legend()

        return fig, ax

    def stat_uncert(self):
        return np.sqrt(self.bin_counts)


class HistogramCanvas:
    """Class to aggregate the Histogram objects and plot them for comparison."""

    def __init__(self, lumi, var="", simulation=True, unit=""):
        self.lumi = lumi
        self.var = var
        self.hists = {}
        self.unit = unit
        self.bin_edges = np.array([])
        self.bin_centers = np.array([])
        self.description = {"luminosity": self.lumi,
                            "simulation": True}

        self.b2fig = None
        self.fig = None
        self.ax = None

    def add_histogram(self, hist):
        """Add a histogram to the stack"""
        if not self.bin_edges.any():
            self.bin_edges = np.array(hist.bin_edges, dtype=np.float32)
            self.bin_centers = hist.bin_centers
            self.unit = hist.unit
        else:
            assert np.array_equal(np.array(hist.bin_edges, dtype=np.float32), self.bin_edges), "Hist bin edges not compatible with the rest of the stack!"
        if not hist.lumi * hist.lumi_scale == self.lumi:
            raise ValueError(f"Histogram luminosity {hist.lumi} and histogram luminosity scale {hist.lumi_scale} not compatible with desired luminosity {self.lumi}")
        self.hists[hist.name] = hist

    def create_histogram(self, name, data, lumi, lumi_scale=1, is_signal=False, **kwargs):
        """Create a histogram from data and add it to the stack"""
        self.add_histogram(Histogram(name, data, lumi, lumi_scale, is_signal=is_signal, **kwargs))


    def plot(self, xlabel="", histtype="errorbar", dpi=100):
        self.b2fig = B2Figure(auto_description=True, description=self.description)
        self.fig, self.ax = self.b2fig.create(ncols=1, nrows=1, dpi=dpi)

        ax=self.ax
        uncert_label = True
        for name, hist in self.hists.items():
            if histtype == "errorbar":
                ax.errorbar(self.bin_centers, hist.bin_counts, yerr=np.sqrt(hist.bin_counts), label=name, **self.b2fig.errorbar_args)
            elif histtype == "step":
                x = np.concatenate([self.bin_edges, [self.bin_edges[-1]]])
                y = np.concatenate([[0], hist.bin_counts, [0]])
                ax.step(x, y, label=name, lw=0.9)
                uncert = np.sqrt(hist.bin_counts)
                bin_width = self.bin_edges[1:]-self.bin_edges[0:-1]
                ax.bar(x=self.bin_centers, height=2*uncert, width=bin_width, bottom=hist.bin_counts-uncert,
                       edgecolor="grey",hatch="///////", fill=False, lw=0,label="MC stat. unc." if uncert_label else "")
                if uncert_label: uncert_label = False
            else:
                raise ValueError(f"histtype {histtype} not implemented!")
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.set_ylabel("events")
        self.b2fig.shift_offset_text_position(ax)
        ax.legend(loc='upper left', prop={'size': 7})

    def add_labels(self, ax, xlabel="", ylabel="events"):
        if xlabel:
            ax.set_xlabel(xlabel)
        else:
            unit = f" in {self.unit}"
            ax.set_xlabel(f"{self.var}{unit if self.unit else ''}")
        ax.set_ylabel(ylabel)


class StackedHistogram(HistogramCanvas):
    """Class to aggregate the Histogram objects and stack them."""

    def plot(self, dpi=90):
        """Plot the stacked histogram"""

        self.b2fig = B2Figure(auto_description=True, description=self.description)
        self.fig, self.ax = self.b2fig.create(ncols=1, nrows=1, dpi=dpi)

        self.plot_ax(self.ax)
        self.b2fig.shift_offset_text_position(self.ax)
        self.add_labels(ax=self.ax)

        return self.fig, self.ax

    def color_scheme(self, reverse=False):
        if reverse:
            self.colors = plt.cm.gist_earth(np.flip(np.linspace(0.1,0.75,len(self.hists))))
        else:
            self.colors = plt.cm.gist_earth(np.linspace(0.1,0.75,len(self.hists)))

    def plot_ax(self, ax, reverse_colors=False):
        #colors = plt.cm.summer(np.linspace(0.1,0.8,len(self.hists)))
        if not self.b2fig:
            self.b2fig = B2Figure()

        self.color_scheme(reverse=reverse_colors)
        colors=self.colors

        stack = np.zeros(self.bin_centers.size)
        i=0
        for name, hist in self.hists.items():
            color = "red" if hist.is_signal else colors[i]
            #print(f"stack {name}")
            #ax.plot(self.bin_centers, stack+hist.bin_counts, drawstyle="steps", color=colors[i], linewidth=0.5)
            ax.fill_between(self.bin_centers, stack, stack+hist.bin_counts, label=name, step="mid",
                            linewidth=0, linestyle="-", color=color)
            stack += hist.bin_counts
            i += 1

        uncert = self.get_stat_uncert()
        bin_width = self.bin_edges[1:]-self.bin_edges[0:-1]
        ax.bar(x=self.bin_centers, height=2*uncert, width=bin_width, bottom=stack-uncert,
                edgecolor="grey",hatch="///////", fill=False, lw=0,label="MC stat. unc.")

        ax.legend()
        self.add_labels(ax=ax)
        self.b2fig.shift_offset_text_position_old(ax)




    def get_stat_uncert(self):
        """Calculate the stacked uncertainty of the stacked histogram using sqrt(sum(sum(lumi_scale_1**2), sum(lumi_scale_2**2), ..., sum(lumi_scale_n**2)))
        of the n histograms."""
        uncert = np.zeros(self.bin_centers.size)
        for name, hist in self.hists.items():
            uncert += hist.bin_counts*hist.lumi_scale**2
        return np.sqrt(uncert)


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