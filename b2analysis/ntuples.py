from b2analysis import scale_integrated_luminosity_by_luminosity, scale_integrated_luminosity_by_cross_section, invert_root_mangling

import uproot
import numpy as np

class Ntuple:
    def __init__(self, ntuple_file, tree_name, luminosity, is_mc=False, n_mc_events=0, cross_section=0, l_int_mc=0):

        self.is_mc = is_mc # flag to indicate mc ntuple
        self.n_mc_events = n_mc_events
        self.cross_section = cross_section
        self.l_int_mc = l_int_mc

        #load ntuple
        #self.ntuple = uproot.open(ntuple_file)[tree_name].pandas.df()
        self.tree = uproot.open(ntuple_file)[tree_name]

        # change column names to human readable (remove root escaping)
        #self.ntuple = self.ntuple.rename(columns=invert_root_mangling(self.ntuple.columns))
        #self.data = self.ntuple # a data alias


        #luminosity of sample
        self.integrated_luminosity = luminosity
        self.luminosity_advertise = r"$\int \mathcal{L} \,dt=" + "{:.0f}".format(np.round(self.integrated_luminosity,0)) +"\,\mathrm{pb}^{-1}$"

        if is_mc:
            self.weight = self.__calculate_weight__()
        else:
            self.weight = 1


    def __calculate_weight__(self):
        # scale mc to data integrated luminosity
            # TODO: make variable mc sizes
            if self.l_int_mc > 0:
                self.mc_weight = scale_integrated_luminosity_by_luminosity(l_int_data=self.integrated_luminosity,
                                                                           l_int_mc=self.l_int_mc)
            elif self.n_mc_events > 0 and self.cross_section > 0:
                self.mc_weight = scale_integrated_luminosity_by_cross_section(n_mc_events=self.n_mc_events,
                                                                              cross_section=self.cross_section,
                                                                              l_int_data=self.integrated_luminosity)
            else:
                raise ValueError("Don't now how to scale the mc sample!")
            self.ntuple[v.weight] = self.mc_weight
            return self.mc_weight

    def __scale_mc__(self, weight):
        """Fill the weight
        """
        raise NotImplementedError("The behavoir of mc reweighting is not implemented")

    def get_variable_list(self):
        return self.tree.keys()

    def load_variables(self, variable_list):
        df = self.tree.arrays(variable_list, library='pd')
        self.df = df


class NtupleToDf:
    def __init__(self, file_name, tree, variable_list=[]):
        self.file_name = file_name
        self.tree = tree
        self.ntuple = Ntuple(file_name, tree, 1, is_mc=False) # to avoide automatic scaling, mc turned to false
        if len(variable_list) < 1:
            self.variable_list = self.ntuple.get_variable_list()
            self.variable_list = list(dict.fromkeys(self.variable_list)) # remove dublicate entries``
        else:
            self.variable_listv = variable_list
        self.ntuple.load_variables(self.variable_list)
        self.df = self.ntuple.df


class NtupleTau(Ntuple):

    def __init__(self, ntuple_file, tree_name, luminosity, is_mc=False, n_mc_events=0, cross_section=0, l_int_mc=0):
        super().__init__(ntuple_file, tree_name, luminosity, is_mc=is_mc, n_mc_events=n_mc_events,
                                                             cross_section=cross_section, l_int_mc=l_int_mc)


class NtuplePiPiY(Ntuple):

    def __init__(self, ntuple_file, tree_name, luminosity, is_mc=False, n_mc_events=0, cross_section=0, l_int_mc=0):
        super().__init__(ntuple_file, tree_name, luminosity, is_mc=is_mc, n_mc_events=n_mc_events, cross_section=cross_section, l_int_mc=l_int_mc)

        # transform radians in degrees
        rad2deg_list = [v.theta_y,v.phi_y,v.phi_cms_y,v.theta_cms_y,
                        v.phi_rho,v.theta_rho,v.phi_cms_rho,v.theta_cms_rho,
                        v.phi_pi_0,v.phi_pi_1,v.theta_pi_0,v.theta_pi_1,
                        v.phi_cms_pi_0,v.phi_cms_pi_1,v.theta_cms_pi_0,v.theta_cms_pi_1]

        for angle in rad2deg_list:
            self.ntuple[angle] = np.rad2deg(self.ntuple[angle])
