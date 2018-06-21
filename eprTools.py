import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from scipy.signal import savgol_filter
from scipy.integrate import cumtrapz

#Load in specatra (Only Y data)
def eprload(file, preprocess = False, k=0):
    deer = False
    with open(file, 'rb') as f:
        
        if file[-3:] == 'DTA':
            DTA_data = np.frombuffer(f.read(), dtype='>d')
            
            #Look for Xdata from DSC file
            paramfile = file[:-3] + 'DSC'
            
            try:
                fin = open(paramfile,'r');
                for line in fin:
                    
                    p = line[0:4]
                    q = line[0:14]

                    if p == 'XPTS':
                        xpoints = int(line[5:len(line)]);
                        
                    if p == 'XMIN':
                        xmin = float(line[5:len(line)]);
                        
                    if p == 'XWID':
                        xwid = float(line[5:len(line)]);

                    if q == 'PlsSPELEXPSlct':
                        if line[14:len(line)].strip() == '4P-DEER':
                            deer = True
                        
                xmax = xmin + xwid
                xsampling = xwid / xpoints
                xdata = [];
                    
                for l in range(0,xpoints,1):
                    xdata.append(xmin + (xsampling * (l - 1)))
                
                xdata = np.array(xdata)

            except OSError:
                
                print("No X axis data, guessing = 3487g, sweepwidth 100g")
                xwid = 100.0
                xpoints = len(DTA_data)
                xmin = 3437
                xmax = xmin + xwid
                xsampling = xwid / xpoints
                xdata = [];
                    
                for l in range(0,xpoints,1):
                    xdata.append(xmin + (xsampling * (l - 1)))
                
                xdata = np.array(xdata)

        elif file[-3:] == 'csv':
            DTA_data = np.genfromtxt(file, delimiter=',')

    #If this is a deer experiment split real and imaginary portions
    if deer:
        data1 = DTA_data[::2]
        data2 = DTA_data[1::2]
        DTA_data = np.array([xdata, data1, data2])
    else:
        DTA_data = np.array([xdata, DTA_data])

    
    if preprocess:
        DTA_data = normalize(basecorr(DTA_data, k=k))
    
    return DTA_data
    
#Perform offset baseline correction using ALL data points
def basecorr(spectra, k=0, ends = 25):
    xdata = spectra[0]
    spectra = spectra[1]
    #Create an X axis vector for the spectra
    x = np.arange(len(spectra))
    
    #Conditional for higher order fit
    if k > 0:
        subX = np.concatenate([x[:ends], x[-ends:]])
        subY = spectra[subX]
        fit1 = np.polyfit(subX, subY, k)
        bground = np.polyval(fit1, x)
        spectra = spectra - bground
    
    #Perform polynomial fit (Should be the average off all points)
    pfit = np.polyfit(x, spectra, 0)
    yfit = np.polyval(pfit,x)
    
    #Subtract fit from data 
    specSubt = np.subtract(spectra, yfit)
    
    #return data
    out_data = np.array([xdata, specSubt])
    return out_data

#Center Spectrra arround midpoint
def center(spectra, midmax=False):
    xdata = spectra[0]
    spectra = spectra[1]
    
    #Find the min and max of the spectra
    Xmax = np.argmax(spectra)
    Xmin = np.argmin(spectra)
    
    #Take a subset of min and max to find midpoint
    myMidSub = spectra[Xmax:Xmin]
    subMidpoint = (np.abs(myMidSub)).argmin()
    midPoint = Xmax + subMidpoint
    LefRig = 475
    
    #Logic re midmoint is peak or 0 cross 
    if midmax:
        midPoint = Xmax
        LefRig = 450
    
    #Create 1001 length vector centered at midpoint
    newMin = midPoint - LefRig
    newMax = midPoint + LefRig
    centerSpec = spectra[newMin:newMax]
    
    #Return the centered spectra
    return centerSpec

def normalize(spc):
    xdata = spc[0]
    spc = spc[1]
    ydata = spc / np.trapz(cumtrapz(spc, xdata, initial=0), xdata)
    out = np.array([xdata, ydata])
    return out

def abs_half_moment(spc):
    xdata = np.arange(0,100,(100/len(spc[1]))) #spc[0]
    spc = spc[1]
    
    #Calculate the absorbance spectrum
    ispc = cumtrapz(spc, xdata, initial=0)
    
    #Calculate the half moment
    x_bar = xdata[np.argmax(ispc)]
    print(x_bar)
    M = 0
    for x, y in enumerate(ispc):
        if x == 0:
            continue
        else:
            M = M + (abs(xdata[x] - x_bar) * y) * (xdata[x] - xdata[x -1])
    return M

