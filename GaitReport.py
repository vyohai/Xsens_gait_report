import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import myFunctions as mf
import datetime
import pandas as pd
import math
from scipy.signal import find_peaks, peak_prominences
import os

# find starting steps index
def findSteps(x, show=False,on_spot=True):
    # find distance and prominance
    steps = []
    peaksMin, _ = find_peaks(-x, prominence=0.01)

    # calc distance
    height = max(x[peaksMin]) + 0.01

    # calc max peaks
    peaksMax, _ = find_peaks(x, height=height,distance=30)

    diffSteps=[]
    for indPeakMin, peakMin in enumerate(peaksMin):
        if indPeakMin == len(peaksMin) - 1:
            break
        if len(np.where((peaksMin[indPeakMin] < peaksMax) & (peaksMin[indPeakMin + 1] > peaksMax))[0]) == 0:
            continue
        peak_max_between = \
        peaksMax[np.where((peaksMin[indPeakMin] < peaksMax) & (peaksMin[indPeakMin + 1] > peaksMax))][0]
        diffSteps.append(x[peak_max_between]-x[peakMin])
    for indPeakMin, peakMin in enumerate(peaksMin):
        if indPeakMin == len(peaksMin) - 1:
            break
        if len(np.where((peaksMin[indPeakMin] < peaksMax) & (peaksMin[indPeakMin + 1] > peaksMax))[0]) == 0:
            continue
        peak_max_between = \
        peaksMax[np.where((peaksMin[indPeakMin] < peaksMax) & (peaksMin[indPeakMin + 1] > peaksMax))][0]
        if x[peak_max_between] - x[peakMin] > 0.5*np.mean(diffSteps):
            steps.append([peaksMin[indPeakMin], peaksMin[indPeakMin + 1]])

    # find peaks heights
    steps2=[]
    for i in range(len(peaksMax)-1):
        steps2.append([peaksMax[i],peaksMax[i+1]])
    if show:
        figure(figsize=(14, 12), dpi=80)
        a1 = plt.subplot(1, 1, 1)
        a1.plot(x)
        for i in range(len(steps2)):
            a1.plot(steps2[i][0], x[steps2[i][0]], "x")
            a1.plot(steps2[i][1], x[steps2[i][1]], "o")
        plt.legend(['signal','start step','end step'])
        plt.show()
    return steps2

# get a report from a file
##  the report produce the following calculations for each foot(mean and std for each calculation):
### single support, double support, gait speed, gait length,
### gait width, gait frequency, stepsHeight, cadence, number of steps,
### stance precentage, swing precentage
def getReport(path,file):

    ## load data and create a csv file for "segment position" xsens calculations
    df = mf.readXsens(path+'files/',file, cols=['Right Foot x', 'Right Foot y', 'Right Foot z',\
                                  'Left Foot x', 'Left Foot y', 'Left Foot z'], sheet='Segment Position')

    ## load data and create a csv file for "segment orientation" xsens calculations
    df2 = mf.readXsens(path+'files/',file,
                       cols=['Right Foot x', 'Right Foot y', 'Right Foot z', 'Right Upper Leg y', 'Right Lower Leg y',\
                             'Left Foot x', 'Left Foot y', 'Left Foot z', 'Left Upper Leg y', 'Left Lower Leg y'],
                       sheet='Segment Orientation - Euler')

    ## load data and create a csv file for "segment angular velocity" xsens calculations
    df3=mf.readXsens(path+'files/',file,cols=['Right Foot x', 'Right Foot y', 'Right Foot z',\
                                  'Left Foot x', 'Left Foot y', 'Left Foot z'], sheet='Segment Angular Velocity')

    # creating a timeVec
    ts = np.arange(0,0.01*len(df),0.01)

    # create report dictionary
    gaitReport = {}
    gaitReport['right'] = {}
    gaitReport['left'] = {}

    # find steps indexes
    gaitReport['right']['stepsArray'] = findSteps(df['Right Foot z'],show=False, on_spot=True)
    gaitReport['left']['stepsArray'] = findSteps(df['Left Foot z'],show=False, on_spot=True)

    # calculate single and double support
    single, double = [], []
    stance_r,stance_l=[],[]
    for i in range(min([len(gaitReport['left']['stepsArray']), len(gaitReport['left']['stepsArray'])])-1):

        # define the step window
        z_position_right=df['Right Foot z'][gaitReport['right']['stepsArray'][i][0]:gaitReport['right']['stepsArray'][i][1]]
        z_position_left = df['Left Foot z'][gaitReport['left']['stepsArray'][i][0]:gaitReport['left']['stepsArray'][i][1]]

        # define the stance threshold
        gait_min_right=min(z_position_right)
        gait_max_right = max(z_position_right)
        gait_min_left = min(z_position_left)
        gait_max_left = max(z_position_left)

        stance_threshold_right=gait_min_right+(gait_max_right-gait_min_right) / 4
        stance_threshold_left = gait_min_left + (gait_max_left - gait_min_left) / 4

        ind_stance_right = np.where(z_position_right <= stance_threshold_right)
        ind_stance_left = np.where(z_position_left <= stance_threshold_left)
        # xx= ind_stance_right, xx2=ind_stance_left

        stance_r.append(100*len(ind_stance_right[0])/len(z_position_right))
        stance_l.append(100*len(ind_stance_left[0]) / len(z_position_left))

        # double support means both legs are in stance
        double.append((min([ind_stance_right[0][-1], ind_stance_left[0][-1]]) -\
                       max([ind_stance_right[0][0], ind_stance_left[0][0]]))\
                      / (max(ind_stance_right[0][-1],ind_stance_left[0][-1])-\
                         min(ind_stance_right[0][0],ind_stance_left[0][0])) * 100)
        single.append(100 -double[-1])



    gate_times = []
    gate_distance = []
    gate_angle = []
    heading = []
    length = []
    width = []
    dxx = []
    dyy = []
    ##############################right#########################################
    gaitReport['right']['single support-mean'] = np.mean(single)
    gaitReport['right']['single support-std'] = np.std(single)
    gaitReport['right']['double support-mean'] = np.mean(double)
    gaitReport['right']['double support-std'] = np.std(double)
    for i in range(len(gaitReport['right']['stepsArray'])):
        gate_times.append(ts[gaitReport['right']['stepsArray'][i][1]] - ts[gaitReport['right']['stepsArray'][i][0]])
        dx = df['Right Foot x'][gaitReport['right']['stepsArray'][i][1]] - df['Right Foot x'][gaitReport['right']['stepsArray'][i][0]]
        dy = df['Right Foot y'][gaitReport['right']['stepsArray'][i][1]] - df['Right Foot y'][gaitReport['right']['stepsArray'][i][0]]
        dxx.append(abs(dx))
        dyy.append(abs(dy))
        gate_distance.append(abs(dx))
        gate_angle.append(math.atan2(dy, dx) * 180 / 3.14)
        heading.append(df2['Right Foot z'][gaitReport['right']['stepsArray'][i][0]])
        dheading = df2['Right Foot z'][gaitReport['right']['stepsArray'][i][0]] - math.atan2(dy, dx) * 180 / 3.14
        length.append(np.sqrt(dx ** 2 + dy ** 2) * math.cos(dheading * 3.14 / 180))
        width.append(np.sqrt(dx ** 2 + dy ** 2) * math.sin(dheading * 3.14 / 180))

    gaitReport['right']['gait speed-mean'] = np.mean([i / j for i, j in zip(dxx, gate_times)])

    gaitReport['right']['gait speed-std'] = np.std([i / j for i, j in zip(gate_distance, gate_times)])
    gaitReport['right']['gait length-mean'] = np.mean(dxx)
    gaitReport['right']['gait width-mean'] = np.mean(dyy)
    gaitReport['right']['gait length-std'] = np.std(dxx)
    gaitReport['right']['gait width-std'] =  np.std(dyy)

    # gate frequency
    gaitReport['right']['frequency-mean'] = np.mean([1 / j for j in gate_times])
    gaitReport['right']['frequency-std'] =  np.std([1 / j for j in gate_times])

    # step height
    height = []
    for step in gaitReport['right']['stepsArray']:
        height.append(df['Right Foot z'][step[0]] - min(df['Right Foot z'][step[0]:step[1]]))
    gaitReport['right']['stepsHeight-mean'] = np.mean(height)
    gaitReport['right']['stepsHeight-std'] = np.std(height)

    # steps per minute
    gaitReport['right']['cadence-mean'] = 60 / np.mean(gate_times)
    gaitReport['right']['cadence-std'] = 60 / np.std(gate_times)

    # number of steps
    gaitReport['right']['number of steps'] = []
    gaitReport['right']['number of steps'] = len(gaitReport['right']['stepsArray'])


    # precentage of stance
    gaitReport['right']['stance precentage-mean'] = np.mean(stance_r)
    gaitReport['right']['stance precentage-std'] = np.std(stance_r)

    # precentage of swing
    swing = []
    for i in range(len(stance_r)):
        swing.append(100-stance_r[i])

    gaitReport['right']['swing precentage-mean'] = np.mean(swing)
    gaitReport['right']['swing precentage-std'] = np.std(swing)



    ############################left####################################
    gaitReport['left']['single support-mean'] = np.mean(single)
    gaitReport['left']['single support-std'] = np.std(single)
    gaitReport['left']['double support-mean'] = np.mean(double)
    gaitReport['left']['double support-std'] = np.std(double)

    for i in range(len(gaitReport['left']['stepsArray'])):
        gate_times.append(ts[gaitReport['left']['stepsArray'][i][1]] - ts[gaitReport['left']['stepsArray'][i][0]])
        dx = df['Left Foot x'][gaitReport['left']['stepsArray'][i][1]] - df['Left Foot x'][
            gaitReport['left']['stepsArray'][i][0]]
        dy = df['Left Foot y'][gaitReport['left']['stepsArray'][i][1]] - df['Left Foot y'][
            gaitReport['left']['stepsArray'][i][0]]
        dxx.append(abs(dx))
        dyy.append(abs(dy))
        gate_distance.append(abs(dx))
        gate_angle.append(math.atan2(dy, dx) * 180 / 3.14)
        heading.append(df2['Left Foot z'][gaitReport['left']['stepsArray'][i][0]])
        dheading = df2['Left Foot z'][gaitReport['left']['stepsArray'][i][0]] - math.atan2(dy, dx) * 180 / 3.14
        length.append(np.sqrt(dx ** 2 + dy ** 2) * math.cos(dheading * 3.14 / 180))
        width.append(np.sqrt(dx ** 2 + dy ** 2) * math.sin(dheading * 3.14 / 180))

    gaitReport['left']['gait speed-mean'] = np.mean([i / j for i, j in zip(dxx, gate_times)])

    gaitReport['left']['gait speed-std'] = np.std([i / j for i, j in zip(gate_distance, gate_times)])
    gaitReport['left']['gait length-mean'] = np.mean(dxx)
    gaitReport['left']['gait width-mean'] = np.mean(dyy)
    gaitReport['left']['gait length-std'] = np.std(dxx)
    gaitReport['left']['gait width-std'] = np.std(dyy)

    # gate frequency
    gaitReport['left']['frequency-mean'] = np.mean([1 / j for j in gate_times])
    gaitReport['left']['frequency-std'] = np.std([1 / j for j in gate_times])

    # step height
    height = []
    for step in gaitReport['left']['stepsArray']:
        height.append(df['Left Foot z'][step[0]] - min(df['Left Foot z'][step[0]:step[1]]))
    gaitReport['left']['stepsHeight-mean'] = np.mean(height)
    gaitReport['left']['stepsHeight-std'] = np.std(height)

    # steps per minute
    # gaitReport['left']['cadence'] = []
    gaitReport['left']['cadence-mean'] = 60 / np.mean(gate_times)
    gaitReport['left']['cadence-std'] = 60 / np.std(gate_times)

    # number of steps
    gaitReport['left']['number of steps'] = []
    gaitReport['left']['number of steps'] = len(gaitReport['left']['stepsArray'])

    # precentage of stance
    gaitReport['left']['stance precentage-mean'] = np.mean(stance_l)
    gaitReport['left']['stance precentage-std'] = np.std(stance_l)

    # precentage of swing
    swing_l = []
    for i in range(len(stance_l)):
        swing_l.append(100 - stance_l[i])

    gaitReport['left']['swing precentage-mean'] = np.mean(swing_l)
    gaitReport['left']['swing precentage-std'] = np.std(swing_l)


    # prepare to write to xlsx file
    gaitReport['left'].pop('stepsArray')
    gaitReport['right'].pop('stepsArray')
    user_dict=gaitReport
    df= pd.DataFrame.from_dict({i: user_dict[i] for i in user_dict.keys()}, orient='index')

    # write to xlsx file
    df.to_excel(path+'reports/'+file[:-4]+'rep.xlsx')
    return gaitReport

def correlate_steps(gait,show=False,file=''):
    gait_out=[]
    gait_out2 = []
    plt.close()
    for indd,gait1 in enumerate(gait[0:math.floor(len(gait)/3)]):
        if show:
            peakRef = np.argmax(gait1)
            gait_out.append([[x-peakRef for x in range(len(gait1))],gait1.values])

            ind_start=-15
            ind_end=8
        # for i in np.linspace(ind_start,ind_end,1-ind_start+ind_end):
            gait_out2.append([gait_out[indd][1][gait_out[indd][0].index(ind_start):gait_out[indd][0].index(ind_end)]])

    if show:
        for i in range(len(gait_out2)):
            plt.plot(gait_out2[i][0])
        plt.title(file)
        # plt.show()
    return np.array(gait_out2)
def showGR(gr):
    keys=list(gr['right'].keys())
    print('len is ',len(keys))
    i=0
    print('##############################################')
    for key in keys[2:]:
        if len(gr['right'][key])==2:
            print('on right foot')
            print('*'+key+'* mean is ',gr['right'][key][0],' and his std is ',\
                  gr['right'][key][1],'\n')
            print('on left foot')
            print('*'+key + '* mean is ', gr['left'][key][0], ' and his std is ', \
                  gr['right'][key][1], '\n')
        else:
            print('on right foot')
            print('*' + key + '* mean is ', gr['right'][key][0], '\n')
            print('on left foot')
            print('*' + key + '* mean is ', gr['left'][key][0], '\n')
    print('##############################################')


def get_subject_reports():

    os.chdir(r'C:\Users\User\Desktop\yocahi\master\thesis')
    path = os.getcwd() + '\\data\\iter\\'
    files = list(os.listdir(path))
    walk_file = 'WALK\WALK.csv'
    walkos_file = 'WALK ON SPOT\WALK ON SPOT.csv'
    resaults = {}
    resaults['WALK'] = {}
    resaults['WALK ON SPOT'] = {}
    for file in files:
        resaults['WALK'][file] = getReport(path + file + '\\' + walk_file)
        resaults['WALK ON SPOT'][file] = getReport(path + file + '\\' + walkos_file)
    return resaults


def visulizeGRFeatures(grDict, type):
    patients = list(grDict['WALK'].keys())
    features = list(grDict['WALK'][patients[0]]['left'].keys())
    # show walk features
    i = 0
    a = []
    line = i % 3
    column = int(math.floor(i / 4))
    l = 3
    c = math.ceil(int(len(features[2:]) / l))
    fig, a = plt.subplots(nrows=l, ncols=c, figsize=(12, 7), tight_layout=True)
    plt.suptitle('visulize GR Features ' + type)
    for feature in features[2:]:
        for patient in patients:
            a[line][column].plot(grDict[type][patient]['left'][feature][0], '*')
            a[line][column].text(0, grDict[type][patient]['left'][feature][0], patient)
            a[line][column].set_title(feature)
        i = i + 1
        line = i % 3
        column = int(math.floor(i / 4))


def visualizeWalkVSWalkOnSpot(grDict):
    rel_features = ['stepsHeight', 'cadence', 'leg angle max', 'foot angle max']
    patients = list(grDict['WALK'].keys())
    # show walk features
    i = 0
    a = []
    line = i % 2
    column = int(math.floor(i / 2))
    l = 2
    c = math.ceil(int(len(rel_features) / l))
    fig, a = plt.subplots(nrows=l, ncols=c, figsize=(12, 7), tight_layout=True)
    plt.suptitle('visualize Walk VS Walk On Spot')
    for feature in rel_features:
        for patient in patients:
            a[line][column].plot(grDict['WALK'][patient]['left'][feature][0],
                                 grDict['WALK ON SPOT'][patient]['left'][feature][0], '*')
            a[line][column].text(grDict['WALK'][patient]['left'][feature][0],
                                 grDict['WALK ON SPOT'][patient]['left'][feature][0], patient)
            #             plt.title(figure_title, y=1.08)
            a[line][column].set_title(rel_features[i], y=1.08)
            a[line][column].set_xlabel('WALK')
            a[line][column].set_ylabel('WALK ON SPOT')
        i = i + 1
        line = i % 2
        column = int(math.floor(i / 2))


def visualizeSymmwtry(grDict, type):
    rel_features = ['stepsHeight', 'cadence', 'leg angle max', 'foot angle max']
    patients = list(grDict['WALK'].keys())
    # show walk features
    i = 0
    a = []
    line = i % 2
    column = int(math.floor(i / 2))
    l = 2
    c = math.ceil(int(len(rel_features) / l))
    fig, a = plt.subplots(nrows=l, ncols=c, figsize=(12, 7), tight_layout=True)
    plt.suptitle('visualize right leg VS left leg ' + type)
    for feature in rel_features:
        for patient in patients:
            a[line][column].plot(
                np.abs(grDict[type][patient]['right'][feature][0] - grDict[type][patient]['left'][feature][0]), '*')
            a[line][column].text(0, np.abs(
                grDict[type][patient]['right'][feature][0] - grDict[type][patient]['left'][feature][0]), patient)
            a[line][column].set_title(rel_features[i], y=1.08)
            a[line][column].set_ylabel('abs(feat[right]-feat[left])')
        i = i + 1
        line = i % 2
        column = int(math.floor(i / 2))


def visualizeGR(subject_resaults, vis_regular, vis_walk_vs_on_spot, vis_symmetry):
    # show all the features alone
    if vis_regular:
        visulizeGRFeatures(subject_resaults, 'WALK')
    #         visulizeGRFeatures(subject_resaults,'WALK ON SPOT')
    # show the walk features vs the walk on spot features
    if vis_walk_vs_on_spot:
        visualizeWalkVSWalkOnSpot(subject_resaults)

    # show symmetry features
    if vis_symmetry:
        visualizeSymmwtry(subject_resaults, 'WALK')


