import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit, nnls
from scipy.interpolate import interp1d
from scipy.special import fresnel
from sklearn.linear_model.base import LinearModel

class DEER_Spec:
    
    def __init__(self, time, yreal, yimag):
        self.time = time
        self.real = yreal / yreal.max()
        self.imag = (yimag / yimag.max()) - 0.5
        self.background = None
        self.dipolar_evolution = None
        self.P = None
        self.kernel_len = 200
        self.fit_time = None 
        self.r = np.linspace(1, 100, self.kernel_len)
        self.fit = None
        self.alpha = None
    
    @classmethod
    def from_file(cls, filename):
        d = {}
        ydata = []
        with open(filename, 'rb') as f:
        
            if filename[-3:] == 'DTA':
                ydata = np.frombuffer(f.read(), dtype='>d')
                
                #Look for Xdata from DSC file
                paramfile = filename[:-3] + 'DSC'
                try:
                    with open(paramfile, 'r') as f:
                        for line in f:
                            #Skip blank lines and lines with comment chars
                            if line.startswith(("*", "#", "\n")):
                                continue
                            else:
                                line = line.split()
                                try: 
                                    key = line[0]
                                    val = list(map(str.strip, line[1:]))
                                except IndexError:
                                    key = line
                                    val = None
                                d[key] = val
                except OSError:
                    print("Error: No parameter file found")
                       
        xpoints = int(d['XPTS'][0])
        xmin = float(d['XMIN'][0]) 
        xwid = float(d['XWID'][0])
       
        xmax = xmin + xwid
        xdata = np.linspace(xmin, xmax, xpoints)
        
        yreal = ydata[::2]
        yimag = ydata[1::2]
        
        return cls(xdata, yreal, yimag)
        
        ##Processing methods
        
    def phase(self):
        #Make complex array for phase adjustment
        cData = self.real + 1j*self.imag
        
        #Initial guess for phase shift
        phi0 = np.arctan2(self.imag[-1], self.real[-1])
        
        #Use last 7/8ths of data to fit phase
        fit_set = cData[int(round(len(cData)/8)):]
        
        def get_imag_norm_squared(phi):
            temp = np.imag(fit_set * np.exp(1j * phi))
            return np.dot(temp, temp)
        
        #Find Phi that minimizes norm of imaginary data
        phi = minimize(get_imag_norm_squared, phi0)
        
        cData = cData * np.exp(1j * phi.x)
        
        #Test for 180 degree inversion of real data
        if np.real(cData[0]) < np.real(cData[0]):
            cData = -1 * cData
        
        self.real = np.real(cData)
        self.imag = np.imag(cData)
    
    def zero_time(self):
        
        def zero_moment(data):
            xSize = int(len(data) / 2)
            xData = np.arange(-xSize, xSize + 1)
            
            return np.dot(data, xData)
        
        #Interpolate data
        freal = interp1d(self.time, self.real, 'cubic')
        fimag = interp1d(self.time, self.imag, 'cubic')
        
        self.time = np.arange(self.time.min(), self.time.max())
        self.real = freal(self.time)
        self.imag = fimag(self.time)
        
        #Take zero_moment of all windows tx(tmax)/2 and find minimum
        tmax = self.real.argmax()
        half_tmax = int(tmax/2)
        
        lFrame = tmax - half_tmax
        uFrame = tmax + half_tmax + 1
        low_moment = zero_moment(self.real[lFrame : uFrame])
        
        #Only look in first 500ns of data
        for i in range(half_tmax, 500):
            lFrame = i - half_tmax
            uFrame = i + half_tmax + 1
            
            test_moment = zero_moment(self.real[lFrame : uFrame])
            
            if abs(test_moment) < abs(low_moment):
                low_moment = test_moment
                tmax = i
        
        #Adjust time to appropriate zero
        self.time = self.time - self.time[tmax]
        
        #Remomve time < 0
        self.time = self.time[tmax:]
        self.real = self.real[tmax:]
        self.imag = self.imag[tmax:]
        
        self.fit_time = np.linspace(1e-6, self.time.max(), self.kernel_len)
        
    def correct_background(self):
        
        #Use last 3/4 of data to fit background
        fit_time = self.time[(int(len(self.time)/4)):]
        fit_real = self.real[(int(len(self.real)/4)):]
                              
        def homogeneous_3d(t, a, k):
            k = k/1e9
            return a * np.exp(-k * t)
        
        popt, pcov = curve_fit(homogeneous_3d, fit_time, fit_real)
        
        self.background = homogeneous_3d(self.time, *popt)
        self.dipolar_evolution = self.real - homogeneous_3d(self.time, 1, popt[1]) + 1
            
    def get_fit(self):
        
        #Compute Kernel
        omega_dd = (2 * np.pi * 52.04) / (self.r ** 3)
        z = np.sqrt((6 * np.outer(self.fit_time, omega_dd)/np.pi))
        S_z, C_z = fresnel(z)
        SzNorm = S_z / z
        CzNorm = C_z/z

        trigterm = np.outer(self.fit_time, omega_dd)
        costerm = np.cos(trigterm)
        sinterm = np.sin(trigterm)
        K = CzNorm * costerm + SzNorm * sinterm
        
        #Define L matrix 
        L = np.zeros((self.kernel_len - 2, self.kernel_len))
        spots = np.arange(self.kernel_len - 2)
        L[spots, spots] = 1
        L[spots, spots + 1] = - 2
        L[spots, spots + 2] = 1
        
        #Compress Data to kernel dimensions
        f = interp1d(self.time, self.dipolar_evolution)
        kData = f(self.fit_time)
        
        #Preprocess kernel and data for nnls fitting
        X, y, X_offset, y_offset, X_scale = LinearModel._preprocess_data(
        K, kData, True, False, True, sample_weight=None)
        
        def get_P(X, y, alpha):
            C = np.concatenate([X, alpha * L])
            d = np.concatenate([y, np.zeros(shape = self.kernel_len - 2)])
            
            P = nnls(C,d)
            
            temp_fit = X.dot(P[0]) + y_offset
            return P, temp_fit
        
        def get_AIC_score(alpha):
            P, temp_fit = get_P(X, y, alpha)

            K_alpha = np.linalg.inv(X.T.dot(X) + (alpha**2)* L.T.dot(L)).dot(X.T)
            H_alpha = X.dot(K_alpha) 

            nt = self.kernel_len
            score = nt * np.log((np.linalg.norm((y + y_offset) - temp_fit)**2)/ nt) + (2 * np.trace(H_alpha))
            
            return score
        
        res = minimize(get_AIC_score, 1, bounds = ((1e-3,1e3),))
        self.alpha = res.x
        
        P, self.fit = get_P(X, y, self.alpha)
        self.P = P[0]
            
            
        