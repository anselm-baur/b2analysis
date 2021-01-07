from typing import Dict,  Optional, Tuple, Union, List
import re
import os, os.path
import numpy as np
from getpass import getpass, getuser
import pandas as pd


def invert_root_mangling(variables: List[str]):
    """
    stolen from felix
    invert escape for specific characters and remove 'extrainfo()'

    :param variables: list of root compatible variable names.
    :return: list of basf2 variable names.
    """

    # noinspection PyProtectedMember
    from ROOT import gSystem
    gSystem.Load('libanalysis.so')
    from ROOT import Belle2
    Belle2.Variable.Manager.Instance()
    return {v: re.sub(r'extraInfo\(([^)]*)\)', r'\1', Belle2.invertMakeROOTCompatible(v)) for v in variables}


def scale_integrated_luminosity_by_cross_section(n_mc_events, cross_section, l_int_data):
    '''
    this function can be used to scale mc to data integrated luminosity if number of mc events and the corsssection of the mc events is known
    '''
    return l_int_data*cross_section/n_mc_events

def scale_integrated_luminosity_by_luminosity(l_int_mc, l_int_data):
    '''
    this function can be used to scale mc to data integrated luminosity if the luminosity of the mc_sample is known
    '''
    return l_int_data/l_int_mc

def get_number_of_files(path):
    return len([name for name in os.listdir('.') if os.path.isfile(name)])

def get_lumi_in_pb(l):
    # submit in cm^-2 return pb-1
    p = 1e-12 # pico
    b = 1e-24 # barn in cm^2
    return l*b*p

def get_integrated_luminosity(run_nr, exp_nr, username='', password=''):
    import requests
    from requests.auth import HTTPBasicAuth
    from lxml import html
    from collections import namedtuple

    username = getuser()
    username = input(f'\nUsername [{username}]: ') if username == '' else username
    password = getpass() if password == '' else password


    with requests.Session() as s:
        p = s.post('https://elog.belle2.org/elog/Beam+run', auth=HTTPBasicAuth(username, password),
               files={'uname': username, 'upassword': password})
        if p.status_code != 200:
            raise IOError('Bad password :(')

        #get the id of the run
        request_url = 'https://elog.belle2.org/elog/Beam+run/?Exp+number={}&Run+number={}'.format(run_nr,exp_nr)
        #print(request_url)
        page = s.get(request_url,
                      auth=HTTPBasicAuth(username, password))

        tree = html.fromstring(page.content)

        entries = [[j.strip() for j in i.text.split('\n')] for i in tree.xpath("//td[@class='messagelist']/pre") if i.text]
        Summary = namedtuple('Summary', 'ID JSTTime Author Subject Type Category Exp_number Run_number Num_events Run_time Solenoid Subdetectors DQMPlots Text Attach')
        db_id = [Summary(*[j.getchildren()[0].text for j in i.getchildren()]) for i in tree.xpath("//tr[td/@class='list1'] | //tr[td/@class='list2']")][0][0].replace('\xa0','')
       
        #open db entry of id and get the luminosity
        page = s.get('https://elog.belle2.org/elog/Beam+run/{}'.format(db_id),
                        auth=HTTPBasicAuth(username, password))
        tree = html.fromstring(page.content)

        content = tree.xpath("//td[@class='messageframe']/pre")
        #print(content[0].text)
        lumi = [line.replace(' ','').split(':')[-1] for line in (content[0].text).split('\n') if 'Integrated Luminosity' in line][0]
        #print(lumi)
        return float(lumi)*1e33
   

def get_bin_centers(bins):
    return np.mean(np.vstack([bins[0:-1],bins[1:]]), axis=0)

if __name__ == '__main__':

    username = getuser()
    password = getpass()
    
    runs = [2639,1976,2266,1539,2630,2569,1315,2608,1553,2064]
    #runs = [1539]
    total_lumi = 0
    for run in runs:
        total_lumi += get_integrated_luminosity(8,run,username,password)
    print(get_lumi_in_pb(total_lumi))
    #print(get_lumi_in_pb(get_integrated_luminosity(8,1539,username,password)))
    #print(get_lumi_in_pb(get_integrated_luminosity(8,1540,username,password)))

# Particle merger 
# -------------------
def do_merge(df, col1, col2, col_merged):
    # merge 2 cols of a pandas dataframe in a new dataframe
    merged_df1 = pd.DataFrame({col_merged: df[col1]})
    merged_df2 = pd.DataFrame({col_merged: df[col2]})
    return pd.concat([merged_df1, merged_df2], ignore_index=True)

def merge_cols(df, col_list_1, col_list_2, col_list_out, weight=1, additional_variables=[]):
    # merge the dataframe variable cols of 2 particles to a single particle
    df_merged = pd.DataFrame()
    for col_1, col_2, col_out in zip(col_list_1,col_list_2,col_list_out):
        df_merged[col_out] = do_merge(df, col_1, col_2, col_out)[col_out]
    
    # we want additinal variables independent from the merged variables also in the new merged data frame
    for var in additional_variables:
        df_merged = pd.concat([df_merged,df.loc[:,var].reset_index(drop=True)],ignore_index=False, axis=1, join="outer", sort=False)


    df_merged['__weight__'] = weight

    return df_merged


def makedirs(path):
        """
        Create all parent folders if they do not exist.
        """
        normpath = os.path.normpath(path)
        parentfolder = os.path.dirname(normpath)
        if parentfolder:
            try:
                os.makedirs(parentfolder)
            except OSError:
                pass

