import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit, nnls
from scipy.interpolate import interp1d
from scipy.special import fresnel
from sklearn.linear_model.base import LinearModel
import numba
from eprTools.tntnn import tntnn
#from eprTools.nnlsbpp import nnlsm_blockpivot
from time import time


class DEER_spec:
    
    def __init__(self, time, yreal, yimag, rmin, rmax, do_phase):

        #Working values
        self.time = time
        self.real = yreal
        self.imag = yimag 

        #Tikhonov fit results
        self.fit = None
        self.alpha = None
        self.P = None
        self.fit_time = None 
        
        # Kernel perameteres 
        self.rmin = rmin
        self.rmax = rmax
        self.kernel_len = 200
        self.r = np.linspace(rmin, rmax, self.kernel_len)
        
        # Default phase and trimming parameters
        self.phi = None
        self.do_phase = do_phase
        self.trim_length = None
        self.zt = None

        # Default background correction parameters
        self.bkgrnd_kind ='3D'
        self.bkgrnd_k = 1
        self.bkgrnd_fit_t = None

        # Background correction results
        self.dipolar_evolution = None
        self.bkgrnd = None
        self.bkgrnd_param = None

        #Raw Data Untouched
        self.raw_time = time 
        self.raw_real = yreal
        self.raw_imag = yimag


        self.update()



    @classmethod
    def from_file(cls, filename, rmin = 1, rmax = 100):
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
                                    val = [arg.strip() for arg in line[1:]]
                                except IndexError:
                                    key = line
                                    val = None
                                d[key] = val
                except OSError:
                    print("Error: No parameter file found")
        
        do_phase = True
        if d['PlsSPELLISTSlct'][0] == 'none':
            do_phase = False

        xpoints = int(d['XPTS'][0])
        xmin = float(d['XMIN'][0]) 
        xwid = float(d['XWID'][0])
       
        xmax = xmin + xwid
        xdata = np.linspace(xmin, xmax, xpoints)
        
        yreal = ydata[::2]
        yimag = ydata[1::2]
        
        return cls(xdata, yreal, yimag, rmin, rmax, do_phase)

    def update(self):

        self.trim()
        self.zero_time()

        if self.do_phase:
            self.phase()

        self.correct_background()
        self.compute_kernel()


    def set_kernel_len(self, length):
        self.kernel_len = length
        self.r = np.linspace(self.rmin, self.rmax, self.kernel_len)
        self.fit_time = np.linspace(1e-6, self.time.max(), self.kernel_len)
        self.update()

    def set_kernel_r(self, rmin = 0, rmax = 100):
        self.r = np.linspace(rmin, rmax, self.kernel_len)
        self.update()

    def set_phase(self, phi = 0, degrees = True):
        
        if degrees == False:
            self.phi = phi
        elif degrees == True:
            self.phi = phi * np.pi / 180.0


        self.update()


    def set_trim(self, trim = None):
        self.trim_length = trim
        self.update()

    def set_zero_time(self, zt = None):
        self.zt = zt;
        self.update()

    def set_background_correction(self, kind='3d', k=1, fit_time = None):
        self.bkgrnd_kind =kind
        self.bkgrnd_k = k
        self.bkgrnd_fit_t = fit_time
        self.update()

    def compute_kernel(self):

        #Compute Kernel
        omega_dd = (2 * np.pi * 52.04) / (self.r ** 3)
        z = np.sqrt((6 * np.outer(self.fit_time, omega_dd)/np.pi))
        S_z, C_z = fresnel(z)
        SzNorm = S_z / z
        CzNorm = C_z / z

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
        self.K, self.y, X_offset, self.y_offset, X_scale = LinearModel._preprocess_data(
        K, kData, True, False, True, sample_weight=None)
        self.L = L

    def trim(self):
        self.real = self.raw_real
        self.imag = self.raw_imag
        self.time = self.raw_time

        #normalize working values
        if min(self.real < 0):
            self.real = self.real - 2 * min(self.real)

            self.imag = self.imag/max(self.real)
            self.real = self.real/max(self.real)


        if not self.trim_length:

            #take last quarter of data
            start = -int(len(self.real)/3)
            window = 11

            #get minimum std
            min_std = self.real[-window:].std()
            min_i = -window
            for i in range(start, -window):
                test_std = self.real[i:i+window].std()
                if test_std < min_std:
                    min_std = test_std
                    min_i = i

            max_std =  3 * min_std
            cutoff = len(self.real)
            
            for i in range(start, - window):
                test_std = self.real[i:i+window].std()
                if (test_std > max_std) & (i > min_i):
                    cutoff = i
                    break

        elif self.trim_length:
            cutoff = self.trim_length
            freal = interp1d(self.time, self.real, 'cubic')
            fimag = interp1d(self.time, self.imag, 'cubic')
        
            self.time = np.arange(self.time.min(), self.time.max())
            self.real = freal(self.time)
            self.imag = fimag(self.time)


        self.time = self.time[:cutoff] 
        self.real = self.real[:cutoff]
        self.imag = self.imag[:cutoff]

        
    def phase(self):


        #Make complex array for phase adjustment
        cData = self.real + 1j*self.imag
        
        if self.phi == None:
            #Initial guess for phase shift
            phi0 = np.arctan2(self.imag[-1], self.real[-1])
            
            #Use last 7/8ths of data to fit phase
            fit_set = cData[int(round(len(cData)/8)):]
            
            def get_imag_norm_squared(phi):
                temp = np.imag(fit_set * np.exp(1j * phi))
                return np.dot(temp, temp)
            
            #Find Phi that minimizes norm of imaginary data
            phi = minimize(get_imag_norm_squared, phi0)
            phi = phi.x
            temp = cData * np.exp(1j * phi)
            
            #Test for 180 degree inversion of real data
            if np.real(temp).sum() < 0:
                phi = phi + np.pi

            self.phi = phi

        cData = cData * np.exp(1j * self.phi)
        self.real = np.real(cData)/np.real(cData).max() 
        self.imag = np.imag(cData)/np.real(cData).max()
    
    def zero_time(self):
        
        def zero_moment(data):
            xSize = int(len(data) / 2)
            xData = np.arange(-xSize, xSize + 1)

            if len(xData) != len(data):
                xData = xData[:-1]
            
            return np.dot(data, xData)
        
        

        #Interpolate data
        freal = interp1d(self.time, self.real, 'cubic')
        fimag = interp1d(self.time, self.imag, 'cubic')
        
        self.time = np.arange(self.time.min(), self.time.max())
        self.real = freal(self.time)
        self.imag = fimag(self.time)
    
        if not self.zt: 
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
        
        else:
            tmax = self.zt


        #Adjust time to appropriate zero
        self.time = self.time - self.time[tmax]
        
        #Remomve time < 0
        self.time = self.time[tmax:]
        self.real = self.real[tmax:]
        self.imag = self.imag[tmax:]
        
        self.fit_time = np.linspace(1e-6, self.time.max(), self.kernel_len)
        
    def correct_background(self):        
        
        #calculate t0 for fit_t if none given
        if not self.bkgrnd_fit_t:
            self.bkgrnd_fit_t = (int(len(self.time)/4))


        #Use last 3/4 of data to fit background
        fit_time = self.time[self.bkgrnd_fit_t:]
        fit_real = self.real[self.bkgrnd_fit_t:]

        if self.bkgrnd_kind in ['3D', '2D'] :
            if self.bkgrnd_kind == '2D':
                d = 2
            elif self.bkgrnd_kind =='3D':
                d = 3

            def homogeneous_3d(t, a, k, j):
                return a * np.exp(-k * (t ** (d/3)) + j)

            popt, pcov = curve_fit(homogeneous_3d, fit_time, fit_real, p0 = (1, 1e-5, 1e-2) )
        
            self.bkgrnd = homogeneous_3d(self.time, *popt)
            self.bkgrnd_param = popt

            self.dipolar_evolution = self.real - homogeneous_3d(self.time, *popt) + (popt[0] * np.exp(popt[2]))

        elif self.bkgrnd_kind == 'poly':

            popt = np.polyfit(fit_time, fit_real, deg = self.bkgrnd_k)
            self.bkgrnd = np.polyval(popt, self.time)
            self.bkgrnd_param = popt
            self.dipolar_evolution = self.real - np.polyval(popt, self.time) + popt[-1]
    
    def get_fit(self, alpha=None):
        
        if alpha == None:
            
            res = minimize(self.get_AIC_score, 1, args = (self.K, self.y), bounds = ((1e-3,1e3),))
            self.alpha = res.x

        else:
            self.alpha = alpha

        P, self.fit = self.get_P(self.K, self.y, self.alpha)
        self.P = P[0] / np.sum(P[0])
            
    def get_P(self, X, y, alpha):
        C = np.concatenate([self.K, alpha * self.L])
        d = np.concatenate([self.y, np.zeros(shape = self.kernel_len - 2)])
        
        if self.kernel_len > 350:
            P = tntnn(C, d, use_AA = True)
        else:
            #start = time()
            #P = nnlsm_blockpivot(C,d)
            #print('nnls_bpp:', time() - start)
            #start = time()
            P = nnls(C,d)
            #print('nnls:', time() - start)

        temp_fit = X.dot(P[0]) + self.y_offset
        return P, temp_fit
    
    def get_AIC_score(self, alpha, X, y):
        P, temp_fit = self.get_P(X, y, alpha)
        
        K_alpha = np.linalg.inv(self.K.T.dot(self.K) + (alpha**2)* self.L.T.dot(self.L)).dot(self.K.T)
        H_alpha = self.K.dot(K_alpha) 

        nt = self.kernel_len
        score = nt * np.log((np.linalg.norm((y + self.y_offset) - temp_fit)**2)/ nt) + (2 * np.trace(H_alpha))
        
        return score
