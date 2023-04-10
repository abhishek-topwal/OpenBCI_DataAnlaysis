#this file contains the funtions to get the features from the data
# raise SystemExit
import numpy as np
import pandas as pd
import scipy as scp
from spectrum import *
import pywt

#============================ Fractal Dimension  ====================================
def fractal_dimension(Z, threshold=0.9):

    # Only for 2d image
    assert(len(Z.shape) == 2)

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])


    # Transform Z into a binary array
    Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]
#====================================================================================



#============================  Coefficient of Variation  ========================================
def coeff_var(a):
    b = a #Extracting the data from the 8 channels
    output = np.zeros(len(b)) #Initializing the output array with zeros
    k = 0; #For counting the current row no.
    for i in b:
        mean_i = np.mean(i) #Saving the mean of array i
        std_i = np.std(i) #Saving the standard deviation of array i
        output[k] = std_i/mean_i #computing coefficient of variation
        k=k+1
    return np.sum(output)/8

#====================================================================================


#============================ Mean of Vertex to Vertex Slope  ========================================
import heapq
from scipy.signal import argrelextrema

def first_diff(i):
    b=i

    out = np.zeros(len(b))

    for j in range(len(i)):
        out[j] = b[j-1]-b[j]# Obtaining the 1st Diffs

        j=j+1
        c=out[1:len(out)]
    return c #returns first diff


def slope_mean(p):
    b = p #Extracting the data from the 8 channels
    output = np.zeros(len(b)) #Initializing the output array with zeros
    res = np.zeros(len(b)-1)

    k = 0; #For counting the current row no.
    for i in b:
        x=i
        amp_max = i[argrelextrema(x, np.greater)[0]]
        t_max = argrelextrema(x, np.greater)[0]
        amp_min = i[argrelextrema(x, np.less)[0]]
        t_min = argrelextrema(x, np.less)[0]
        t = np.concatenate((t_max,t_min),axis=0)
        t.sort()#sort on the basis of time

        h=0
        amp = np.zeros(len(t))
        res = np.zeros(len(t)-1)
        for l in range(len(t)):
            amp[l]=i[t[l]]


        amp_diff = first_diff(amp)

        t_diff = first_diff(t)

        for q in range(len(amp_diff)):
            res[q] = amp_diff[q]/t_diff[q]
        output[k] = np.mean(res)
        k=k+1
    return np.sum(output)/8

#====================================================================================


#============================  Variance of Vertex to Vertex Slope  ========================================
def slope_var(p):
    b = p #Extracting the data from the 8 channels
    output = np.zeros(len(b)) #Initializing the output array with zeros
    res = np.zeros(len(b)-1)

    k = 0; #For counting the current row no.
    for i in b:
        x=i
        amp_max = i[argrelextrema(x, np.greater)[0]]#storing maxima value
        t_max = argrelextrema(x, np.greater)[0]#storing time for maxima
        amp_min = i[argrelextrema(x, np.less)[0]]#storing minima value
        t_min = argrelextrema(x, np.less)[0]#storing time for minima value
        t = np.concatenate((t_max,t_min),axis=0) #making a single matrix of all matrix
        t.sort() #sorting according to time

        h=0
        amp = np.zeros(len(t))
        res = np.zeros(len(t)-1)
        for l in range(len(t)):
            amp[l]=i[t[l]]


        amp_diff = first_diff(amp)

        t_diff = first_diff(t)

        for q in range(len(amp_diff)):
            res[q] = amp_diff[q]/t_diff[q] #calculating slope

        output[k] = np.var(res)
        k=k+1#counting k
    return np.sum(output)/8

#====================================================================================


#============================  Hjorth  ========================================
def hjorth(input):
    realinput = input
    hjorth_activity = np.zeros(len(realinput))
    hjorth_mobility = np.zeros(len(realinput))
    hjorth_diffmobility = np.zeros(len(realinput))
    hjorth_complexity = np.zeros(len(realinput))
    diff_input = np.diff(realinput)
    diff_diffinput = np.diff(diff_input)
    k = 0
    for j in realinput:
        hjorth_activity[k] = np.var(j)
        hjorth_mobility[k] = np.sqrt(np.var(diff_input[k])/hjorth_activity[k])
        hjorth_diffmobility[k] = np.sqrt(np.var(diff_diffinput[k])/np.var(diff_input[k]))
        hjorth_complexity[k] = hjorth_diffmobility[k]/hjorth_mobility[k]
        k = k+1
    return np.sum(hjorth_activity)/8, np.sum(hjorth_mobility)/8,np.sum(hjorth_complexity)/8

#====================================================================================



#============================  Kurtosis  ========================================
def kurtosis(a):
    b = a # Extracting the data from the8 channels
    output = np.zeros(len(b)) # Initializing the output array with zeros (length = 8)
    k = 0; # For counting the current row no.
    for i in b:
        mean_i = np.mean(i) # Saving the mean of array i
        std_i = np.std(i) # Saving the standard deviation of array i
        t = 0.0
        for j in i:
            t += (pow((j-mean_i)/std_i,4)-3)
        kurtosis_i = t/len(i) # Formula: (1/N)*(summation(x_i-mean)/standard_deviation)^4-3
        output[k] = kurtosis_i # Saving the kurtosis in the array created
        k +=1 # Updating the current row no.
    return np.sum(output)/8

#====================================================================================


#============================  Second Difference Mean   ========================================
def secDiffMean(a):
    b = a # Extracting the data of the 8  channels
    output = np.zeros(len(b)) # Initializing the output array with zeros (length = 8)
    temp1 = np.zeros(len(b[0])-1) # To store the 1st Diffs
    k = 0; # For counting the current row no.
    for i in b:
        t = 0.0
        for j in range(len(i)-1):
            temp1[j] = abs(i[j+1]-i[j]) # Obtaining the 1st Diffs
        for j in range(len(i)-2):
            t += abs(temp1[j+1]-temp1[j]) # Summing the 2nd Diffs
        output[k] = t/(len(i)-2) # Calculating the mean of the 2nd Diffs
        k +=1 # Updating the current row no.
    return np.sum(output)/8

#====================================================================================


#============================  Second Difference Max   ========================================
def secDiffMax(a):
    b = a # Extracting the data from the 8 channels
    output = np.zeros(len(b)) # Initializing the output array with zeros (length = 8)
    temp1 = np.zeros(len(b[0])-1) # To store the 1st Diffs
    k = 0; # For counting the current row no.
    t = 0.0
    for i in b:
        for j in range(len(i)-1):
            temp1[j] = abs(i[j+1]-i[j]) # Obtaining the 1st Diffs
        t = temp1[1] - temp1[0]
        for j in range(len(i)-2):
            if abs(temp1[j+1]-temp1[j]) > t :
                t = temp1[j+1]-temp1[j] # Comparing current Diff with the last updated Diff Max

        output[k] = t # Storing the 2nd Diff Max for channel k
        k +=1 # Updating the current row no.
    return np.sum(output)/8

#====================================================================================


#============================  Skewness   ========================================
def skewness(arr):
    data = arr
    skew_array = np.zeros(len(data)) #Initializing the array as all 0s
    index = 0; #current cell position in the output array

    for i in data:
        skew_array[index]=scp.stats.skew(i,axis=0,bias=True)
        index+=1 #updating the cell position

    return np.sum(skew_array)/8

#====================================================================================


#============================  First Difference Mean  ========================================
def first_diff_mean(arr):
    data = arr
    diff_mean_array = np.zeros(len(data)) #Initialinling the array as all 0s
    index = 0; #current cell position in the output array

    for i in data:
        sum=0.0#initializing the sum at the start of each iteration
        for j in range(len(i)-1):
            sum += abs(i[j+1]-i[j]) # Obtaining the 1st Diffs

        diff_mean_array[index]=sum/(len(i)-1)
        index+=1 #updating the cell position
    return np.sum(diff_mean_array)/8

#====================================================================================


#============================  First Difference Max  ========================================
def first_diff_max(arr):
    data = arr
    diff_max_array = np.zeros(len(data)) #Initialinling the array as all 0s
    first_diff = np.zeros(len(data[0])-1)#Initialinling the array as all 0s
    index = 0; #current cell position in the output array

    for i in data:
        max=0.0#initializing at the start of each iteration
        for j in range(len(i)-1):
            first_diff[j] = abs(i[j+1]-i[j]) # Obtaining the 1st Diffs
            if first_diff[j]>max:
                max=first_diff[j] # finding the maximum of the first differences
        diff_max_array[index]=max
        index+=1 #updating the cell position
    return np.sum(diff_max_array)/8

#====================================================================================


#============================  Wavelet Features  ========================================

def wavelet_features(epoch,channels):
    cA_values = []
    cD_values = []
    cA_mean = []
    cA_std = []
    cA_Energy =[]
    cD_mean = []
    cD_std = []
    cD_Energy = []
    Entropy_D = []
    Entropy_A = []
    wfeatures = []
    for i in range(channels):
        cA,cD=pywt.dwt(epoch[i,:],'coif1')
        cA_values.append(cA)
        cD_values.append(cD)		#calculating the coefficients of wavelet transform.
    for x in range(channels):
        cA_mean.append(np.mean(cA_values[x]))
        wfeatures.append(np.mean(cA_values[x]))

        cA_std.append(abs(np.std(cA_values[x])))
        wfeatures.append(abs(np.std(cA_values[x])))

        cA_Energy.append(abs(np.sum(np.square(cA_values[x]))))
        wfeatures.append(abs(np.sum(np.square(cA_values[x]))))

        cD_mean.append(np.mean(cD_values[x]))		# mean and standard deviation values of coefficents of each channel is stored .
        wfeatures.append(np.mean(cD_values[x]))

        cD_std.append(abs(np.std(cD_values[x])))
        wfeatures.append(abs(np.std(cD_values[x])))

        cD_Energy.append(abs(np.sum(np.square(cD_values[x]))))
        wfeatures.append(abs(np.sum(np.square(cD_values[x]))))

        Entropy_D.append(abs(np.sum(np.square(cD_values[x]) * np.log(np.square(cD_values[x])))))
        wfeatures.append(abs(np.sum(np.square(cD_values[x]) * np.log(np.square(cD_values[x])))))

        Entropy_A.append(abs(np.sum(np.square(cA_values[x]) * np.log(np.square(cA_values[x])))))
        wfeatures.append(abs(np.sum(np.square(cA_values[x]) * np.log(np.square(cA_values[x])))))
    return wfeatures

#====================================================================================



#===============  FFT Max Poer - Delta, Theta, ALpha and Beta Band  =========================
from scipy import signal

def maxPwelch(data_win,Fs):


    BandF = [0.1, 3, 7, 12, 30]
    PMax = np.zeros([8,(len(BandF)-1)]);

    for j in range(8):
        f,Psd = signal.welch(data_win[j,:], Fs)

        for i in range(len(BandF)-1):
            fr = np.where((f>BandF[i]) & (f<=BandF[i+1]))
            PMax[j,i] = np.max(Psd[fr])

    return np.sum(PMax[:,0])/8,np.sum(PMax[:,1])/8,np.sum(PMax[:,2])/8,np.sum(PMax[:,3])/8

#====================================================================================

#============================  Shannon Entropy  ========================================
def shanon_entropy(labels): # Shanon Entropy
    """ Computes entropy of 0-1 vector. """
    n_labels = len(labels)
    counts = np.bincount(labels)
    probs = counts[np.nonzero(counts)] / n_labels

    n_classes = len(probs)

    if n_classes <= 1:
        return 0
    return - np.sum(probs * np.log(probs)) / np.log(n_classes)

#====================================================================================


#============================  Spectral Entropy  ========================================
from numpy.fft import fft
from numpy import zeros, floor, log10, log, mean, array, sqrt, vstack, cumsum, ones, log2, std
from numpy.linalg import svd, lstsq
import time

def bin_power(X,Band,Fs):
    C = fft(X)
    C = abs(C)
    Power =zeros(len(Band)-1);
    for Freq_Index in xrange(0,len(Band)-1):
        Freq = float(Band[Freq_Index])   ## Xin Liu
        Next_Freq = float(Band[Freq_Index+1])
        Power[Freq_Index] = sum(C[floor(Freq/Fs*len(X)):floor(Next_Freq/Fs*len(X))])
    Power_Ratio = Power/sum(Power)
    return Power, Power_Ratio


def spectral_entropy(X, Fs, Power_Ratio = None):

    Band = [0.1, 3, 7, 12, 30]
    if Power_Ratio is None:
        Power, Power_Ratio = bin_power(X, Band, Fs)

    Spectral_Entropy = 0
    for i in xrange(0, len(Power_Ratio) - 1):
        Spectral_Entropy += Power_Ratio[i] * log(Power_Ratio[i])
    Spectral_Entropy /= log(len(Power_Ratio))     # to save time, minus one is omitted
    print('Shape of Spectral Entropy = ',np.shape(Spectral_Entropy))
    return -1 * Spectral_Entropy

#====================================================================================


#==================  Autoregression - Burg Algorithm  ==============================
def autogressiveModelParametersBurg(labels):
    feature = []
    feature1 = []
    model_order = 3
    for i in range(8):
        AR, rho, ref = arburg(labels[i], model_order)
        # AR, rho, ref = arburg(labels[i], model_order)
        feature.append(AR)
    for j in range(8):
        for i in range(model_order):
            feature1.append(feature[j][i])

    return feature1

#====================================================================================





