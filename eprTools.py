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
        DTA_data[1] = data[1] / data[1].max()
        DTA_data[2] = (data[2] / data[2].max()) - 0.5
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

####DEER Tools

def fit_pase(data):
    
    from scipy.optimize import minimize
    
    time = data[0]
    
    #Make complex array for phase adjustment
    cData = data[1] + 1j*data[2]
    
    #Normalize real and imaginary data
    real = data[1] / data[1].max()
    imaginary = (data[2] / data[2].max()) - 0.5
    
    phi0 = 0.5 #np.arctan2(imaginary[-1], real[-1])
    
    fitset = cData[int(round(len(cData)/8)):]
    
    def get_iphase(phi):
        updated = np.imag(fitset * np.exp(1j* phi))
        return np.dot(updated, updated)
    
    phi = minimize(get_iphase, phi0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
    print(phi0)
    print(np.rad2deg(phi.x))
    
    
    cData = cData * np.exp(1j * phi.x)
    
    real = np.real(cData)
    imaginary = np.imag(cData)
    if real[0] < real[-1]:
        real = -1* real
        imaginary = -1* imaginary
    return np.array([time, real, imaginary])

def fit_zero_time(data):
    
    from scipy.interpolate import interp1d
    from scipy.stats import moment
    
    def moment_about_zero(data2):
        thing = int(len(data2)/2)
        
        thing = range(-thing,thing+1)
        
        summ = 0
        for index, value in enumerate(data2):
            summ = summ + thing[index] * value
        return summ
        
        
    time = data[0]
    real = data[1]
    
    #interpolate data to get a point for each ns
    f = interp1d(time,real, 'cubic')
    time_interp = np.arange(int(time.min()), int(time.max())) 
    real_interp =  f(time_interp)
    
    #Take first moment of all windows tx(tmax)/2  and find minimum
    tmax = real_interp.argmax()
    
    half_tmax = int(tmax/2)    
    low_moment = moment_about_zero(real_interp[tmax -half_tmax:tmax + half_tmax + 1 ])
    tmaxnew = tmax
    for i in range(half_tmax, 500):
        try_moment = moment_about_zero(real_interp[i - half_tmax: i + half_tmax + 1])
        if abs(try_moment) < abs(low_moment):
            lowest_moment = try_moment
            tmax = i
    
    time_interp = time_interp - time_interp[tmax]
    data = np.array([time_interp[tmax:],real_interp[tmax:]])
    
    return data

##Background adjustments
def subtract_background(data):
    from scipy.optimize import curve_fit
    
    time = data[0]
    real = data[1]
    
    fit_time = time[int(len(real)/4):] 
    fit_data = real[int(len(real)/4):] 
    
    def homogeneous_3d(t, a, k):
        k = k/1000000000
        return a * np.exp(-k * t)
    
    popt, pcov = curve_fit(homogeneous_3d, fit_time, fit_data)
    
    print(popt)
    plt.plot(time, homogeneous_3d(time, *popt))
    plt.plot(time, real)
    plt.show()
    
    #Adjust for background
    real = real - homogeneous_3d(time, *popt) + popt[0]
    
    return np.array([time, real])

def fit_Probability(data):
    VecDem = 512
    from scipy.special import fresnel
    t = np.arange(0.000001,len(data[0]), len(data[0])/VecDem)
    r = np.arange(1, 100, 99/VecDem)

    L = np.zeros((VecDem-1, VecDem))
    spots = np.arange(VecDem -1)
    L[spots, spots] = - 1
    L[spots, spots + 1] = 1

    omega_dd = (2 * np.pi * 52.04) / (r ** 3)
    z = np.sqrt((6 * np.outer(t, omega_dd)/np.pi))
    S_z, C_z = fresnel(z)
    SzNorm = S_z / z
    CzNorm = C_z/z

    trigterm = np.outer(t, omega_dd)
    costerm = np.cos(trigterm)
    sinterm = np.sin(trigterm)
    K = CzNorm * costerm + SzNorm * sinterm

    from scipy.interpolate import interp1d
    f = interp1d(data[0], data[1])
    newx = t 
    data_512 = f(newx)
    
    alpha = 25

    from sklearn.linear_model.base import LinearModel
    X, y, X_offset, y_offset, X_scale = LinearModel._preprocess_data(
        K, data_512, True, False, True,
        sample_weight=None)
    
    #NNLS Regression on Tikhonov equation
    C = np.concatenate([X, alpha * L])
    d = np.concatenate([y, np.zeros(shape=VecDem-1)])
    from scipy.optimize import nnls
    P = nnls(C, d)
    
    #Plot distribution and fit to V(t)
    fit = K.dot(P[0]) + (1 - K.dot(P[0]).max() )
    plt.plot(r, P[0])
    plt.show()

    plt.plot(data_512)
    plt.plot(fit)
    
    return np.array((r, P[0])), fit
 


##Utility functions
def change_phase(data, phi):
    cData = data[1] + 1j*data[2]
    phi = phi % (2*np.pi)
    updated = data * np.exp(1j * phi)
    rval = np.array([data[0], np.real(updated), np.imag(updated)])
    return updated