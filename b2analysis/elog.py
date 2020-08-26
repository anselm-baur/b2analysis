from getpass import getpass, getuser

import requests
from requests.auth import HTTPBasicAuth

from lxml import html
from collections import namedtuple

import numpy as np


class ELog:
    def __init__(self,username='',password=''):
        self.username = input(f'\nUsername: ') if username == '' else username
        self.password = getpass() if password == '' else password

        # set up https session
        self.session = requests.Session()

        p = self.session.post('https://elog.belle2.org/elog/Beam+run', auth=HTTPBasicAuth(self.username, self.password),
            files={'uname': self.username, 'upassword': self.password})
        if p.status_code != 200:
            raise IOError('Bad password :(')


    def _get_run_db_id(self,run_nr, exp_nr):
        #get the id of the run

        request_url = 'https://elog.belle2.org/elog/Beam+run/?Exp+number={}&Run+number={}'.format(exp_nr,run_nr)
        #print(request_url)

        #open db entry of db id 
        page = self.session.get(request_url,
                      auth=HTTPBasicAuth(self.username, self.password))

        tree = html.fromstring(page.content)

        try:
            # find the needed content in the dedicated html <td> tag
            entries = [[j.strip() for j in i.text.split('\n')] for i in tree.xpath("//td[@class='messagelist']/pre") if i.text]
            # build namedtuple containing the columns
            Summary = namedtuple('Summary', 'ID JSTTime Author Subject Type Category Exp_number Run_number Num_events Run_time Solenoid Subdetectors DQMPlots Text Attach')
            db_id = [Summary(*[j.getchildren()[0].text for j in i.getchildren()]) for i in tree.xpath("//tr[td/@class='list1'] | //tr[td/@class='list2']")][0][0].replace('\xa0','')
            return db_id
        except:
            return None


    def get_integrated_luminosity(self,run_nr, exp_nr):
        # collect integrated luminosity from the elog entry of a specific run from a specific experiment
        
        db_id = self._get_run_db_id(run_nr,exp_nr)
         
        #if not db_id:
        #    # something went wrong
        #    return np.nan

        #open db entry of db id 
        page = self.session.get('https://elog.belle2.org/elog/Beam+run/{}'.format(db_id),
                        auth=HTTPBasicAuth(self.username, self.password))
        tree = html.fromstring(page.content)

        # find the needed content in the dedicated html <td> tag
        content = tree.xpath("//td[@class='messageframe']/pre")
        #print(content[0].text)
        # find the dedicated line containing the luyminosity
        lumi = [line.replace(' ','').split(':')[-1] for line in (content[0].text).split('\n') if 'Integrated Luminosity' in line][0]
        #print(lumi)
        return float(lumi)*1e33