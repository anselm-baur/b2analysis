import ROOT
import numpy as np

class Efficiency:
    def __init__(self, df, selection):
        self.df = df
        self.selection = selection
        self.eff_dict = {}

    def get_hist(self, variable_list, x_range=(), n_bins=50,**kwargs):
        weight = kwargs['weight'] if 'weight' in kwargs.keys() else 1
        xerr = kwargs['xerr'] if 'xerr' in kwargs.keys() else False

        _kwargs = {'bins': n_bins}
        if not x_range == ():
            _kwargs['range'] = x_range
        else:
            _kwargs['range'] = (int(min([self.df[var].min() for var in variable_list])),
                                int(max([self.df[var].max() for var in variable_list])))

        eff = []
        eff_yerr_low = []
        eff_yerr_up = []

        n_df_list = []
        k_df_list = []

        for i in range(len(variable_list)):
            # copy the relevant variabable column and the weight column
            n_df_list.append(self.df[[variable_list[i], '__weight__']].copy())
            # select the k subset and copy only the relevant variable column and weight column
            k_df_list.append(self.df.loc[self.selection][[variable_list[i], '__weight__']].copy())


            _n_hist, bins = np.histogram(n_df_list[i][variable_list[i]], **_kwargs)
            _k_hist, bins = np.histogram(k_df_list[i][variable_list[i]], **_kwargs)

            _eff, _eff_yerr_low, _eff_yerr_up = self.calc_TEff(_k_hist,  # k
                                                          _n_hist,  # N
                                                          weight)
            eff.append(_eff)
            eff_yerr_low.append(_eff_yerr_low)
            eff_yerr_up.append(_eff_yerr_up)

        width = np.diff(bins)
        center = (bins[:-1] + bins[1:]) / 2

        eff_x_err = width / 2 if xerr else None
        for i, var in enumerate(variable_list):
            self.eff_dict[var] = {'x_values': center,
                                  'y_values': eff[i],
                                  'x_errors': eff_x_err,
                                  'y_errors': (eff_yerr_low[i], eff_yerr_up[i])
                                  }
        return self.eff_dict, width, center


    def calc_TEff(self, k, N, weight=1):
        ignore_denominator_zero = True

        n = k.size
        eff = np.zeros(n)
        err_low = np.zeros(n)
        err_up = np.zeros(n)
        h1f_k = ROOT.TH1F('k', 'number passed events', n, 0, n)
        h1f_N = ROOT.TH1F('N', 'number total events', n, 0, n)

        for i in range(n):
            # print("[{}] {}/{},{}".format(i+1,k[i],N[i],k[i]/N[i]))
            h1f_k.SetBinContent(i + 1, int(k[i] * weight))
            h1f_N.SetBinContent(i + 1, int(N[i] * weight))

        TEff = ROOT.TEfficiency(h1f_k, h1f_N)

        for i in range(n):
            if ignore_denominator_zero:
                if h1f_N[i + 1] == 0:
                    eff[i] = np.nan
                    err_low[i] = np.nan
                    err_up[i] = np.nan
                else:
                    eff[i] = TEff.GetEfficiency(i + 1)
                    err_low[i] = TEff.GetEfficiencyErrorLow(i + 1)
                    err_up[i] = TEff.GetEfficiencyErrorUp(i + 1)
            else:
                eff[i] = TEff.GetEfficiency(i + 1)
                err_low[i] = TEff.GetEfficiencyErrorLow(i + 1)
                err_up[i] = TEff.GetEfficiencyErrorUp(i + 1)

        return eff, err_low, err_up