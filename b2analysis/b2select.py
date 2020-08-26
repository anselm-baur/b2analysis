import numpy as np

class PiPiY:
    @staticmethod
    def y(df, Ecms=3, theta=np.radians([50,110])):
        return ((df["daughter(1,useCMSFrame(E))"] > Ecms) &
                (df["daughter(1,theta)"] > theta[0]) & (df["daughter(1,theta)"] < theta[1]))  

    @staticmethod
    def pipi(df, p=1, d0=2, z0=4, EoP=0.8, pValue=0.001, nCDCHits=4, mask=[]):

        return ((df["daughter(0,daughter(0,p))"] > p) & (df["daughter(0,daughter(1,p))"] > p) & 
                (df["daughter(0,daughter(0,d0))"].abs() < d0) & (df["daughter(0,daughter(1,d0))"].abs() < d0) &
                (df["daughter(0,daughter(0,z0))"].abs() < z0) & (df["daughter(0,daughter(1,z0))"].abs() < z0) &
                (df["daughter(0,daughter(0,clusterEoP))"] < EoP) & (df["daughter(0,daughter(1,clusterEoP))"] < EoP) &
                (df["daughter(0,daughter(0,nCDCHits))"] > nCDCHits) & (df["daughter(0,daughter(1,nCDCHits))"] > nCDCHits) &
                (df["daughter(0,daughter(0,pValue))"] > pValue) & (df["daughter(0,daughter(1,pValue))"] > pValue))

    @staticmethod
    def pipiy(df, M=[10,11]):
        return ((df["M"] > M[0]) & (df["M"] < M[1]))

def rank_by_highest(df, rank_variable, group_by_variables=['__event__','__run__','__experiment__']):
    '''
    rank_variable: variable used to rank the candidates grouped
    group_by_variables: variables used to group the events used for ranking, standard is the event number, run number and, experiment number
    '''
    #return df.sort_values(rank_variable, ascending=False).drop_duplicates(dublicate_variable).sort_index()
    return df[rank_variable] == df.groupby(by=group_by_variables)[rank_variable].transform(max)


