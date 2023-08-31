# Import libraries
import numpy
from array import array
# import cv2
import datetime
import dateutil.parser
import pandas as pd
import myFunctions as mf
import datetime
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import numpy as np
import xml.dom.minidom as md
import os.path
from os import path
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from scipy.signal import kaiserord, lfilter, firwin, freqz
from tqdm import tqdm
import scipy.signal


def readXsens(path2,xsens_file,cols=['Right Foot x', 'Right Foot y', 'Right Foot z'],sheet='Sensor Free Acceleration'):

    # assemble a "file+sheet.csv" str
    file_out = path2[:-6]+'files_out/'+xsens_file[0:-4] + '_' + sheet + '.csv'

    # check if exist
    if path.exists(file_out):
        df = pd.read_csv(file_out, usecols=cols)

    # if not create a file wit all coulomns
    else:
        df = pd.read_excel(path2+xsens_file.replace('.csv', '.xlsx'), sheet_name=None, usecols=None)

        # for sheet in df.
        for sheet in df.keys():
            file_out2 = xsens_file[0:-4] + '_' + sheet + '.csv'
            df[sheet].to_csv(path2[:-6]+'files_out/'+file_out2)

    # return appropriate df
    df = pd.read_csv(file_out, usecols=cols)
    return df

# def createDateTimeVec(start,frame_rate,length):
#     out=[start+datetime.timedelta(seconds=1/int(frame_rate)*x) for x in range(length)]
#     return np.array(out)
def parseXsensTime(time0,L,FR):
    start=datetime.datetime.strptime(time0,'%H:%M:%S:%f')
    fr=FR
    return mf.createDateTimeVec(start, fr,L)



def getMsFromMvnx(file):
    doc = md.parse(file)
    fc = doc.getElementsByTagName("subject").item(0)
    fc2=fc.getElementsByTagName("frames").item(0)
    index2=0
    index=0
    i=0
    t= {}
    while (index==0) or (index=='') or (index!=index2):
        index=index2
        fc3 = fc2.getElementsByTagName("frame").item(i)
        if fc3!=None:
            index2 = fc3.getAttribute("index")
            if index2!='':
                t[index2]=fc3.getAttribute("ms")
        i = i + 1
        print(index+'from')
    tt=np.array(list(t.values()), dtype='float')
    return numpy.divide(tt,1000)


# def plot3(x,y,z,xm=0,ym=0,zm=0):
#     plt.subplot(3,1,1)
#     plt.plot(x)
#     plt.plot(xm,x[xm],'*')
#     plt.subplot(3, 1, 2)
#     plt.plot(y)
#     plt.plot(ym, y[ym], '*')
#     plt.subplot(3, 1, 3)
#     plt.plot(z)
#     plt.plot(zm, z[zm], '*')
#     plt.show()
# def plot2(roll,pitch):
#     plt.subplot(3,1,1)
#     plt.plot(roll)
#     plt.subplot(3, 1, 2)
#     plt.plot(pitch)
#     plt.show()
# def filter(x,y,z,sr=12.5,width=1,rip=60.0,cut_off=0.005):
#     sample_rate = sr
#     nsamples = len(x)
#
#
#     # ------------------------------------------------
#     # Create a FIR filter and apply it to x.
#     # ------------------------------------------------
#
#     # The Nyquist rate of the signal.
#     nyq_rate = sample_rate / 2.0
#
#     # The desired width of the transition from pass to stop,
#     # relative to the Nyquist rate.  We'll design the filter
#     # with a 5 Hz transition width.
#     width = width / nyq_rate
#
#     # The desired attenuation in the stop band, in dB.
#     ripple_db = rip
#
#     # Compute the order and Kaiser parameter for the FIR filter.
#     N, beta = kaiserord(ripple_db, width)
#
#     # The cutoff frequency of the filter.
#     cutoff_hz = cut_off
#
#     # Use firwin with a Kaiser window to create a lowpass FIR filter.
#     taps = firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta))
#     # print(type(taps))
#     # Use lfilter to filter x with the FIR filter.
#     xf = lfilter(taps, 1.0, x)
#     xf2=np.ones(len(xf))*xf[-len(taps)]
#     xf2[0:-len(taps)]=xf[len(taps):]
#     yf = lfilter(taps, 1.0, y)
#     yf2 = np.ones(len(xf)) * yf[-len(taps)]
#     yf2[0:-len(taps)] = yf[len(taps):]
#     zf = lfilter(taps, 1.0, z)
#     zf2 = np.ones(len(xf)) * zf[-len(taps)]
#     zf2[0:-len(taps)] = zf[len(taps):]
#     # plot3(xf,yf,zf)
#     return xf2,yf2,zf2
#
# def scoreKobi(patient,kind='pos',df=1,showEvents=False):
#     # df=pd.read_csv(r"C:\Users\User\Desktop\yocahi\master\kobi\data\old"+"\p"+str(patient)+".csv")
#     axis = findingAxis(patient,df=df)
#     roll, pitch, ax, ay, az = angles(patient, axis=axis,df=df)
#     df2 = pd.read_csv(r"C:\Users\User\Desktop\yocahi\master\kobi\data\old\events.csv",
#                      usecols=[patient * 2 - 2, patient * 2 - 1])
#
#     event=len(df2)
#     # if event and showEvents:
#     indCalib, chair = mf.showEvents(patient, plot=False,df2=df)
#     calibAng = np.mean(pitch[indCalib:indCalib + 25 * 60])
#     hist, bin_edges = np.histogram(pitch * 180 / 3.14, density=False)
#     # histd, bin_edgesd = np.histogram(np.diff(pitch) * 180 / 3.14, density=False)
#     # ang = bin_edges[hist.argmax() - 1]
#     # calibAng = ang
#     stdAng = 30
#     indPos = np.where(
#         np.bitwise_and((pitch * 180 / 3.14) < calibAng + stdAng, (pitch * 180 / 3.14) > calibAng - stdAng))
#     stdMov = 1
#     dpitch = np.diff(pitch)
#     indMov = np.where(np.bitwise_and((dpitch * 180 / 3.14) < stdMov, (dpitch * 180 / 3.14) > -stdMov))
#
#     if kind == 'posRatio':
#         score = len(indPos[0]) / len(pitch)
#     elif kind == 'movRatio':
#         score = len(indMov[0]) / len(pitch)
#     elif kind == 'a_mag':
#         score=np.mean(np.sqrt(ax**2+ay**2+az**2))-1
#     elif kind == 'movStd':
#         score = np.std(np.diff(pitch))
#     return score
# def putTimeStampAndPlot2(video,data,data_time,vout_file):
#         #read timecode from video
#     TC=readVidTC(video)
#     TC=datetime.datetime.strptime(TC, '%H:%M:%S:%f')
#     print(TC)
#     #open video object
#     fig=plt.figure(1)
#     vidcap = cv2.VideoCapture(video)
#     fps=vidcap.get(cv2.CAP_PROP_FPS )
#     # decide about width and height for new vid
#     img1 = np.asarray(fig2img(fig))
#     image1 = np.ones((int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH )),int(3)),dtype=np.uint8)
#     #     #https: // note.nkmk.me / en / python - opencv - hconcat - vconcat - np - tile /
#     image1 = vconcat_resize_min([image1, img1])
#     out = cv2.VideoWriter(vout_file,cv2.VideoWriter_fourcc(*'XVID'), fps, (image1.shape[1],image1.shape[0]),True)
#
#     #iterate tru frames
#     for ii in range(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))):
#         for i in tqdm(range(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)))):
#
#             success, image = vidcap.read()
#             if success==False:
#                 if vidcap.get(cv2.CAP_PROP_POS_FRAMES)<vidcap.get(cv2.CAP_PROP_FRAME_COUNT):
#                     continue
#                 else:
#                     break
#             MS=vidcap.get(cv2.CAP_PROP_POS_MSEC )
#             # print(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
#             DT=datetime.timedelta(milliseconds=MS)
#             TCnow=TC+DT
#         #find frame rate and frame number
#
#         #calculate frame time
#
#         #find nearest time in data time
#             TCDatanow=nearest(data_time, TCnow)
#             indexDataNearest=np.where(data_time==TCDatanow)
#             indexDataNearest =indexDataNearest[0]
#             dataNow=data.values[indexDataNearest]
#             # dataNow=data_time[data_time==TCDatanow]
#             #print times
#             #plot on frame
#                 #prepere plot
#             time2=data_time[indexDataNearest[0]-10:indexDataNearest[0]+10]
#             data2 = data.values[indexDataNearest[0] - 10:indexDataNearest[0] + 10]
#             figure, ax = plt.subplots(2)
#             ax[0].plot_date(time2, data2, '-.')
#             ax[0].plot_date(data_time[indexDataNearest[0]], data[indexDataNearest[0]], '*')
#             # plt.show()
#             ax[1].plot_date(data_time, data.values, '-.')
#             ax[1].plot_date(data_time[indexDataNearest[0]], data.values[indexDataNearest[0]], '*')
#                 #convert plot to matrix
#             img = np.asarray(fig2img(figure))
#             plt.close(figure)
#             #     #https: // note.nkmk.me / en / python - opencv - hconcat - vconcat - np - tile /
#             image = vconcat_resize_min([image, img])
#             # cv2.imshow('frame', image)
#             # if cv2.waitKey(0):
#             #     break
#             # font
#             font = cv2.FONT_HERSHEY_SIMPLEX
#
#             # org
#             org = (40, 500)
#
#             # fontScale
#             fontScale = .5
#
#             # Blue color in BGR
#             color = (0, 0, 0)
#
#             # Line thickness of 2 px
#             thickness = 1
#
#             # Using cv2.putText() method
#             image = cv2.putText(image, 'vid '+str(TCnow)+'data '+str(TCDatanow), org, font,
#                                 fontScale, color, thickness, cv2.LINE_AA)
#             out.write(image)
#     vidcap.release()
#     out.release()
#     cv2.destroyAllWindows()


def getFrameXsens(mvnx_file):
    doc = md.parse(mvnx_file)
    fc = doc.getElementsByTagName("subject").item(0)
    fc2 = fc.getElementsByTagName("frames").item(0)
    index = ''
    i=0
    while not index == '0' :
        fc3 = fc2.getElementsByTagName("frame").item(i)
        index = fc3.getAttribute("index")
        i=i+1
    return fc3.getAttribute("tc")
def readVidTC(video):
    props = get_video_properties(video)
    return props['tags']['timecode']
def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

def readAndParseChair(file,show=False):
    f = open(file, "r")
    line='0'
    dateTime=[]
    data=[]
    time=[]
    while len(line)!=0:
        line=f.readline()
        line2=line.split(',')
        if len(line2) == 3:
            dateTime.append(line2[0])
            time.append(int(line2[1]))
            data.append(int(line2[2]))
        else:
            continue
    dateTime=mf.parseMMrTime(dateTime)

    if show:
        plt.plot_date(dateTime,data,'.-')
        plt.show()
    return data,dateTime,time

def findSitToStand(tnorm_pelz,norm_pelz,norm_pelzd,buf_len=30,silence_thresh=0.01,peak_height=0.3,show=True):
    #looking from when the silence ends untill the next big enough pick
    print(1)
    for index in range(0,len(tnorm_pelz)-buf_len):
        buf=norm_pelzd[index:index+buf_len]
        buf[buf<silence_thresh]=0
        if np.count_nonzero(buf)==buf_len:
            break
    peaks,_=scipy.signal.find_peaks(norm_pelz,height=peak_height)
    if show==True:
        plt.figure(0,figsize=(8,8))
        plt.plot(tnorm_pelz,norm_pelz)
        plt.plot(tnorm_pelz[peaks[0]],norm_pelz[40000+peaks[0]],'x')
        plt.plot(tnorm_pelz,norm_pelzd)
        plt.plot(tnorm_pelz[norm_pelzd<0.01],norm_pelzd[norm_pelzd<0.01],'*')
        plt.xticks([data_time[i] for i in range(40000,40500,30)],[[data_time[i].hour,data_time[i].minute,data_time[i].second,data_time[i].microsecond] for i in range(40000,40500,30)],rotation=90)
        plt.plot(tnorm_pelz[index],norm_pelzd[index],'o')
        plt.legend(['pelz','peak','diff_pelz','diff_pelz<'+str(silence_thresh),'end of silence'])
        plt.grid(True)
    return [index,peaks[0]]

