from b2analysis import scale_integrated_luminosity_by_luminosity, scale_integrated_luminosity_by_cross_section, invert_root_mangling

import uproot
import numpy as np

class Ntuple:
    def __init__(self,ntuple_file, tree_name, luminosity, enable_invert_root_mangilng=True, is_mc=False, n_mc_events=0, cross_section=0, l_int_mc=0):

        #load ntuple
        self.ntuple = uproot.open(ntuple_file)[tree_name].pandas.df()
        self.data = self.ntuple # a data alias

        if enable_invert_root_mangilng:
            # change column names to human readable (remove root escaping)
            self.ntuple = self.ntuple.rename(columns=invert_root_mangling(self.ntuple.columns))

        # flag to indicate mc ntuple
        self.is_mc = is_mc

        #luminosity of sample
        self.integrated_luminosity = luminosity
        self.luminosity_advertise = r"$\int \mathcal{L} \,dt=" + "{:.0f}".format(np.round(self.integrated_luminosity,0)) +"\,\mathrm{pb}^{-1}$"

        self.weight=1
        if is_mc:
            # scale mc to data integrated luminosity
            # TODO: make variable mc sizes 
            if l_int_mc > 0:
                self.mc_weight = scale_integrated_luminosity_by_luminosity(l_int_data=self.integrated_luminosity,l_int_mc=l_int_mc)
            elif n_mc_events > 0 and cross_section > 0:
                self.mc_weight = scale_integrated_luminosity_by_cross_section(n_mc_events=n_mc_events,cross_section=cross_section,l_int_data=self.integrated_luminosity)
            else:
                raise ValueError("Don't now how to scale the mc sample!")
            self.ntuple['__weight__'] = self.mc_weight
            self.weight = self.mc_weight
            
            
'''
class pipiyNtuples:
    def __init__(self,directory):
        parent_dir = directory
        
        self.integ_lumi = 2381.9071136829994
        self.luminosity = r"$\int \mathcal{L} \,dt=" + "{:.0f}".format(np.round(self.integ_lumi,0)) +"\,\mathrm{pb}^{-1}$"
    
        
        data_dir = f'{parent_dir}process=data/nocut={variable}/'
        kky_dir = f'{parent_dir}process=kky/nocut={variable}/'
        mumuy_dir = f'{parent_dir}process=mumuy/nocut={variable}/' 
        pipiy_dir = f'{parent_dir}process=pipiy/nocut={variable}/'
        
        self.data = uproot.open(data_dir + "skimmed_pipiy.root")['skimmed_pipiy'].pandas.df()
        self.mc_kk = uproot.open(kky_dir + "skimmed_pipiy.root")['skimmed_pipiy'].pandas.df()
        self.mc_pipi = uproot.open(pipiy_dir + "skimmed_pipiy.root")['skimmed_pipiy'].pandas.df()
        self.mc_mumu = uproot.open(mumuy_dir + "skimmed_pipiy.root")['skimmed_pipiy'].pandas.df()
        
        self.data = self.data.rename(columns=invert_root_mangling(self.data.columns))
        self.mc_kk = self.mc_kk.rename(columns=invert_root_mangling(self.mc_kk.columns))
        self.mc_pipi = self.mc_pipi.rename(columns=invert_root_mangling(self.mc_pipi.columns))
        self.mc_mumu = self.mc_mumu.rename(columns=invert_root_mangling(self.mc_mumu.columns))
        
        mc_kk_weight = scale_integrated_luminosity_by_cross_section(n_mc_events=1e7,cross_section=15.65,l_int_data=self.integ_lumi)
        mc_pipi_weight = scale_integrated_luminosity_by_cross_section(n_mc_events=1e7,cross_section=167,l_int_data=self.integ_lumi)
        mc_mumu_weight = scale_integrated_luminosity_by_cross_section(n_mc_events=980000+500000+10*1000000,cross_section=1148,l_int_data=self.integ_lumi)

        self.mc_kk['__weight__']   = mc_kk_weight
        self.mc_pipi['__weight__'] = mc_pipi_weight
        self.mc_mumu['__weight__'] = mc_mumu_weight
        


class pipiyHist2DPlot:
       def __init__(self,ntuples):
        
        self.target_dir = 'analysis/plots/variables_no_pvalue_cut/'

        self.b2fig = B2Figure()
        self.fig, self.axes = self.b2fig.create_figure(figsize=(11, 5),n_x_subfigures=2)
        
        self.ntuples = ntuples
              
        self.data = ntuples.data.copy()
        self.mc_kk = ntuples.mc_kk.copy()
        self.mc_pipi = ntuples.mc_pipi.copy()
        self.mc_mumu = ntuples.mc_mumu.copy()

        parent_dir = '/work/abaur/trigger_eff/pipiy_trigger/analysis/variable_cuts_out/' 

        self.translated_variable = {"pValue": ['daughter(0,daughter(0,pValue))','daughter(0,daughter(1,pValue))'],
                                    "ndf": ['daughter(0,daughter(0,nCDCHits))','daughter(0,daughter(1,nCDCHits))']}
        
            
    def do_plot(self,variables, range=[[0,0.1],[1,66]],bins=(100,65),
               additional_info='',suffix=''):
        
        integ_lumi = 2381.9071136829994
        luminosity = r"$\int \mathcal{L} \,dt=" + "{:.0f}".format(np.round(integ_lumi,0)) +"\,\mathrm{pb}^{-1}$"
        plot_target_dir = self.target_dir
        
        _data = self.data
        _mc_kk = self.mc_kk
        _mc_pipi = self.mc_pipi
        _mc_mumu = self.mc_mumu


        ax = self.axes[0]
        bx = self.axes[1]


        self.b2fig.add_descriptions(ax,luminosity=luminosity,small_title=True,additional_info=additional_info)
        self.b2fig.add_descriptions(bx,luminosity=luminosity,small_title=True,additional_info=additional_info)

        variable_x = self.translated_variable[variables[0]][0]
        variable_y= self.translated_variable[variables[1]][0]
        h = ax.hist2d(_data[variable_x],_data[variable_y],bins=bins, range=range, label='n', cmap='viridis', cmin=1e-7)
        plt.colorbar(h[3], ax=ax)

        ax.set_xlabel(r'$pValue$ of $\pi^{-}$')
        ax.set_ylabel(r'$nCDCHits$ of $\pi^{-}$')

        variable_x = self.translated_variable[variables[0]][1]
        variable_y= self.translated_variable[variables[1]][1]
        h = bx.hist2d(_data[variable_x],_data[variable_y],bins=bins, range=range, label='n', cmap='viridis', cmin=1e-7)
        plt.colorbar(h[3], ax=bx)

        bx.set_xlabel(r'$pValue$ of $\pi^{+}$')
        bx.set_ylabel(r'$nCDCHits$ of $\pi^{+}$')
        
        self.b2fig.save(self.fig,f'2dhis_{variables[1]}_vs_{variables[0]}{suffix}',target_dir=plot_target_dir)
'''