import os
import numpy as np
import matplotlib.pyplot as plt
from b2style import B2Figure

class HistogramBase:

    def __init__(self, output_dir = "") -> None:
        self.output_dir = output_dir
        self.fig = None
        self.ax = None

    def savefig(self, filename):
        self.fig.savefig(os.path.join(self.output_dir, filename))

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

        self.weights = np.full(data.size, self.lumi_scale)

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
        self.err = self.stat_uncert()
        self.entries = self.bin_counts



    def _update_bins(self):
        self.bin_centers = (self.bin_edges[1:]+self.bin_edges[:-1])/2
        self.range = (self.bin_edges[0], self.bin_edges[-1])
        self.bins = self.bin_centers.size


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


    def plot(self, fig=None, ax=None, histtype="errorbar", dpi=100, uncert_label=True):
        b2fig = B2Figure()
        if not fig and not ax:
            fig, ax = b2fig.create(ncols=1, nrows=1, dpi=dpi)

        if histtype == "errorbar":
            ax.errorbar(self.bin_centers, self.bin_counts, yerr=self.stat_uncert(), label=self.name, **b2fig.errorbar_args)
        elif histtype == "step":
            x = np.concatenate([self.bin_edges, [self.bin_edges[-1]]])
            y = np.concatenate([[0], self.bin_counts, [0]])
            ax.step(x, y, label=self.name, lw=0.9)
            uncert = np.sqrt(self.bin_counts)
            bin_width = self.bin_edges[1:]-self.bin_edges[0:-1]
            ax.bar(x=self.bin_centers, height=2*uncert, width=bin_width, bottom=self.bin_counts-uncert,
                    edgecolor="grey",hatch="///////", fill=False, lw=0,label="MC stat. unc." if uncert_label else "")
            if uncert_label: uncert_label = False
        unit = f" in {self.unit}"
        ax.set_xlim((*self.range))
        ax.set_xlabel(f"{self.var}{unit if self.unit else ''}")
        ax.set_ylabel("events")
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
        return np.sqrt(self.bin_counts)*(self.lumi_scale**(3/2))


class HistogramCanvas(HistogramBase):
    """Class to aggregate the Histogram objects and plot them
    for comparison."""

    def __init__(self, lumi, var="", simulation=True, unit="", mc_campaign="", is_simulation=True, **kwargs):
        super().__init__(**kwargs)
        self.lumi = lumi
        self.var = var
        self.hists = {}
        self.unit = unit
        self.bin_edges = np.array([])
        self.bin_centers = np.array([])
        self.description = {"luminosity": self.lumi,
                            "simulation": is_simulation,
                            "additional_info": mc_campaign}

        self.b2fig = None
        self.fig = None
        self.ax = None

    def add_histogram(self, hist):
        """Add a histogram to the canvas."""
        if not self.bin_edges.any():
            self.bin_edges = np.array(hist.bin_edges, dtype=np.float32)
            self.bin_centers = hist.bin_centers
            self.unit = hist.unit
            self.bins = hist.bins
            self.range = hist.range
            self.size = self.bin_centers.size
        else:
            assert np.array_equal(np.array(hist.bin_edges, dtype=np.float32), self.bin_edges), "Hist bin edges not compatible with the rest of the stack!"
        if not np.round(hist.lumi * hist.lumi_scale, 1) == np.round(self.lumi, 1):
            raise ValueError(f"Histogram luminosity {hist.lumi} and histogram luminosity scale {hist.lumi_scale} not compatible with desired luminosity {self.lumi}")
        self.hists[hist.name] = hist
        self.__update()


    def __update(self):
        pass


    def create_histogram(self, name, data, lumi, lumi_scale=1, is_signal=False, **kwargs):
        """Create a histogram from data and add it to the stack"""
        self.add_histogram(Histogram(name, data, lumi, lumi_scale, is_signal=is_signal, **kwargs))


    def plot(self, dpi=90, figsize=(6,6), pull_args={}, **kwargs):
        """Plot the histogram canvas."""


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
            self.b2fig = B2Figure(auto_description=True, description=self.description)
            self.fig, self.ax = self.b2fig.create(ncols=1, nrows=1, dpi=dpi, figsize=figsize)

        self.plot_ax(self.ax, **kwargs)
        self.b2fig.shift_offset_text_position(self.ax)
        self.add_labels(ax=self.ax)

        return self.fig, self.ax

    def plot_ax(self, ax, xlabel="", histtype="errorbar", log=False, colors=[], reverse_colors=False):
        if len(colors) < 1:
            self.color_scheme(reverse=reverse_colors)
            colors = self.colors

        uncert_label = True
        for i, (name, hist) in enumerate(self.hists.items()):
            if histtype == "errorbar":
                ax.errorbar(self.bin_centers, hist.entries, yerr=hist.err, label=name, **self.b2fig.errorbar_args)
            elif histtype == "step":
                x = np.concatenate([self.bin_edges, [self.bin_edges[-1]]])
                y = np.concatenate([[0], hist.entries, [0]])
                ax.step(x, y, label=name, lw=0.9, color=colors[i])
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


    def add_labels(self, ax, xlabel="", ylabel="events"):
        if xlabel:
            ax.set_xlabel(xlabel)
        else:
            unit = f" in {self.unit}"
            ax.set_xlabel(f"{self.var}{unit if self.unit else ''}")
        ax.set_ylabel(ylabel)


    def color_scheme(self, reverse=False):
        if reverse:
            self.colors = plt.cm.gist_earth(np.flip(np.linspace(0.1,0.75,len(self.hists))))
        else:
            self.colors = plt.cm.gist_earth(np.linspace(0.1,0.75,len(self.hists)))

    def pull_plot(self, ax, hist_name, nom_hist_name, color='black', ratio=True, corr=0, xlabel="", ylabel="", ylim=None):
        hist = self.hists[hist_name]
        nom_hist = self.hists[nom_hist_name]
        bin_centers = hist.bin_centers
        bin_edges = hist.bin_edges
        bins = hist.size
        if ratio:
            plot = hist.entries/nom_hist.entries
            ax.plot((bin_edges[0], bin_edges[-1]),[1,1], color='black', ls="-")
            if not ylabel:
                ylabel = r"$\mathbf{\frac{"+hist.name.replace("_","\_").replace(" ","\;")+r"}{"+nom_hist.name.replace("_","\_").replace(" ","\;")+r"}}$"
        else:
            plot = (hist.entries-nom_hist.entries)/nom_hist.entries
            ax.plot((bin_edges[0], bin_edges[-1]),[0,0], color='black', ls="-")
            if not ylabel:
                hist_label = hist.name.replace("_","\_").replace(" ","\;")
                nom_hist_label = nom_hist.name.replace("_","\_").replace(" ","\;")
                ylabel = r"$\mathbf{\frac{"+hist_label+r"-"+nom_hist_label+r"}{"+nom_hist_label+r"}}$"
        plot_err = np.sqrt((hist.err/nom_hist.entries)**2+(nom_hist.err*hist.entries/nom_hist.entries**2)**2-2*hist.entries/nom_hist.entries**3*hist.err*nom_hist.err*corr)
        ax.errorbar(bin_centers, plot, yerr=plot_err, fmt='o--', color=color, markersize='2.8', elinewidth=1)
        ax.set_xlim(bin_edges[0], bin_edges[-1])
        if ylim:
            ax.set_ylim(ylim)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)


    def any(self):
        """Check if any hitograms have already been added."""
        if len(self.hists) > 0:
            return True
        else:
            return False



class StackedHistogram(HistogramCanvas):
    """Class to aggregate the Histogram objects and stack them."""

    def add_histogram(self, hist):
        super().add_histogram(hist)
        self.__update()

    def plot(self, dpi=90, **kwargs):
        """Plot the stacked histogram"""

        self.b2fig = B2Figure(auto_description=True, description=self.description)
        self.fig, self.ax = self.b2fig.create(ncols=1, nrows=1, dpi=dpi)

        self.plot_ax(self.ax, **kwargs)
        self.b2fig.shift_offset_text_position(self.ax)
        self.add_labels(ax=self.ax)

        return self.fig, self.ax


    def plot_ax(self, ax, reverse_colors=False, log=False, ylim=None):
        #colors = plt.cm.summer(np.linspace(0.1,0.8,len(self.hists)))
        if not self.b2fig:
            self.b2fig = B2Figure()

        self.color_scheme(reverse=reverse_colors)
        colors=self.colors

        bin_width = self.bin_edges[1:]-self.bin_edges[0:-1]
        stack = np.zeros(self.bin_centers.size)
        i=0
        for name, hist in self.hists.items():
            color = "red" if hist.is_signal else colors[i]
            #print(f"stack {name}")
            #ax.plot(self.bin_centers, stack+hist.bin_counts, drawstyle="steps", color=colors[i], linewidth=0.5)
            #ax.fill_between(self.bin_centers, stack, stack+hist.bin_counts, label=name, step="mid",
            #                linewidth=0, linestyle="-", color=color)
            #ax.fill_between(self.bin_centers, stack, stack+hist.bin_counts, label=name, step="mid",
            #                linewidth=0, linestyle="-", color=color)
            ax.bar(x=self.bin_centers, height=hist.bin_counts, width=bin_width, bottom=stack,
                color=color, lw=0,label=name)

            stack += hist.bin_counts
            i += 1

        uncert = self.get_stat_uncert()
        ax.bar(x=self.bin_centers, height=2*uncert, width=bin_width, bottom=stack-uncert,
                edgecolor="grey",hatch="///////", fill=False, lw=0,label="MC stat. unc.")

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


    def get_stat_uncert(self):
        """Calculate the stacked uncertainty of the stacked histogram using sqrt(sum(sum(lumi_scale_1**2), sum(lumi_scale_2**2), ..., sum(lumi_scale_n**2)))
        of the n histograms."""
        uncert = np.zeros(self.bin_centers.size)
        for name, hist in self.hists.items():
            # sigma = n2/lumi_scale *lumi_scale**2
            uncert += hist.entries/hist.lumi_scale*hist.lumi_scale**2
        return np.sqrt(uncert)


    def get_stacked_entries(self):
        """Get the sum of the stacked entries per bin."""
        entries = np.zeros(self.bin_centers.size)
        for name, hist in self.hists.items():
            entries += hist.entries
        return entries

    def __update(self):
        self.entries = self.get_stacked_entries()
        self.err = self.get_stat_uncert()


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