
import os
import sys
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(r'C:\Users\User\Desktop\yocahi\master\kobi')
import  myFunctions as mf
import GaitReport as gr
import matplotlib.pyplot as plt
from tqdm import tqdm
path='1st exp'
# DRAW A FIGURE WITH MATPLOTLIB
path='1st report\\1st exp'

class Switcher(object):
    def indirect(self, i):
        method_name = 'number_' + str(i)
        method = getattr(self, method_name, lambda: 'Invalid')
        return method()

    # running readXsens(xsens_file=,cols=,sheet=)
    def number_1(self):
        # load xsens data
        import sys
        sys.path.insert(1, '../../code')
        import myFunctions as mf
        path = os.getcwd().replace('\\', '/') + '/files/'
        file = 'ayalaholtsman001.csv'
        df1=mf.readXsens(path,file,cols=['Right Forearm x','Right Forearm y','Right Forearm z'],sheet='Segment Position')

    # checks gaitReport['stepsArray']
    def number_2(self):
        path = os.getcwd().replace('\\', '/') + '/files/'
        file = 'ayalaholtsman001.csv'
        df = mf.readXsens(path,file, cols=['Right Foot x', 'Right Foot y', 'Right Foot z'], sheet='Segment Position')
        gaitReport = {}
        gaitReport['stepsArray'] = gr.findSteps(df['Right Foot z'],show=True,on_spot=True)

    #checks getReport
    def number_3(self):
        path=os.getcwd().replace('\\','/')+'/'
        files= os.listdir(path+'files/')
        files=['check.xlsx']

        if True:
            for i in tqdm(range(len(files))):
                item=files[i]
                _ = gr.getReport(path,item[:-5]+'.csv')

def main():
    s = Switcher()
    s.indirect(3)
if __name__ == "__main__":
    main()

