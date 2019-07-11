import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks_cwt
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d


class CWSpec:

    def __init__(self, field, spec, preprocess=False, k=0, ends=50):
        self.field = field
        self.spec = spec
        if preprocess:
            self.prep(k=k, ends=ends)

    # import methods
    @classmethod
    def from_file(cls, filename, preprocess=False, k=0, ends=50):
        with open(filename, 'rb') as f:

            if filename[-3:] == 'DTA':
                y_data = np.frombuffer(f.read(), dtype='>d')

                # Look for x_data from DSC file
                param_file = filename[:-3] + 'DSC'
                try:
                    d = {}
                    with open(param_file, 'r') as f2:

                        for line in f2:
                            # Skip blank lines and lines with comment chars
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

                    xpoints = int(d['XPTS'][0]) - 1
                    xmin = float(d['XMIN'][0])
                    xwid = float(d['XWID'][0])

                    xmax = xmin + xwid
                    deltaX = xwid / xpoints
                    x_data = np.arange(xmin, xmax + deltaX, deltaX)

                except OSError:

                    print("Param file not present or incorrectly named")
                    print("Guessing centerfield = 3487g, sweepwidth 100g")
                    xwid = 100.0
                    xpoints = len(y_data) - 1
                    xmin = 3437
                    xmax = xmin + xwid
                    deltaX = xwid / xpoints
                    x_data = np.arange(xmin, xmax + deltaX, deltaX)

            elif filename[-3:] == 'spc':
                y_data = np.frombuffer(f.read(), dtype='f')

                # Look for x_data from DSC file
                param_file = filename[:-3] + 'par'

                try:
                    d = {}
                    with open(param_file, 'r') as f:

                        for line in f:

                            # Skip blank lines and lines with comment chars
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

                    xpoints = int(d['ANZ'][0]) - 1
                    xmin = float(d['GST'][0])
                    xwid = float(d['GSI'][0])

                    xmax = xmin + xwid
                    deltaX = xwid / xpoints
                    x_data = np.arange(xmin, xmax + deltaX, deltaX)

                except OSError:

                    print("Param file not present or incorrectly named")
                    print("Guessing centerfield = 3487g, sweepwidth 100g")
                    xwid = 100.0
                    xpoints = len(y_data) - 1
                    xmin = 3437
                    xmax = xmin + xwid
                    deltaX = xwid / xpoints
                    x_data = np.arange(xmin, xmax + deltaX, deltaX)

            else:
                if filename[-3:] == 'csv':
                    y_data = np.genfromtxt(f, delimiter=',')
                else:
                    y_data = np.genfromtxt(f)

                print("No X axis data, guessing = 3487g, sweepwidth 100g")
                xwid = 100.0
                xpoints = len(y_data) - 1
                xmin = 3437
                xmax = xmin + xwid
                deltaX = xwid / xpoints
                x_data = np.arange(xmin, xmax + deltaX, deltaX)
                x_data = x_data[:len(y_data)]

                if len(x_data) == 2048:
                    x_data = x_data[::2]
                    y_data = y_data[::2]

            CW_obj = cls(x_data, y_data, preprocess, k, ends)

            return CW_obj

    # Preparatory methods
    def prep(self, k=0, ends=50):
        self.basecorr(k, ends)
        self.normalize()

    def basecorr(self, k=0, ends=100):
        x = np.arange(len(self.spec))

        iSpec = cumtrapz(self.spec, self.field, initial=0)
        pk = find_peaks_cwt(iSpec, np.arange(7, 10))
        front = np.arange(0, ends, 5)
        back = np.arange(len(self.spec) - ends, len(self.spec), 5)

        myList = np.concatenate([front, pk, back, np.array([len(self.spec) - 1])])
        myList = np.sort(np.unique(myList))

        # If k > 0 perform polynomial fit on ends of data
        if k > 0:
            fit1 = np.polyfit(self.field[myList], self.spec[myList], k)
            bground = np.polyval(fit1, self.field)
            self.spec = self.spec - bground
        # If k == 0 subtract mean
        else:
            self.spec = self.spec - self.spec.mean()

    def normalize(self):
        first_integral = cumtrapz(self.spec, self.field,
                                  initial=0)
        first_integral = first_integral - min(first_integral)

        self.spec = self.spec / np.trapz(first_integral, self.field)

    def center(self, center_field = 0):
        #Find the min and max of the spectra
        Xmax = np.argmax(self.spec)
        Xmin = np.argmin(self.spec)
    
        #Take a subset of min and max to find midpoint
        myMidSub = self.spec[Xmax:Xmin]
        subMidpoint = (np.abs(myMidSub)).argmin()
        midIdx = Xmax + subMidpoint
        
        midPoint = self.field[midIdx]
        
        #Set field s.t. the spectral midpoint is at center_field
        self.cfield = np.arange(-500,500) #self.field - midPoint + center_field
        self.cspec = self.spec[midIdx - 500 : midIdx + 500]

    # Special Methods
    def __add__(self, a):

        fieldmin = min([self.field.min(), a.field.min()])
        fieldmax = max([self.field.max(), a.field.max()])
        step = (fieldmax - fieldmin) / 1024
        fieldrange = np.arange(fieldmin, fieldmax, step)

        f1 = interp1d(self.field, self.spec, kind='cubic')
        f2 = interp1d(a.field, a.spec, kind='cubic')

        spc1 = f1(fieldrange)
        spc2 = f2(fieldrange)

        newspc = spc1 + spc2

        return CW_spec(fieldrange, newspc)

    def __sub__(self, a):

        fieldmin = min([self.field.min(), a.field.min()])
        fieldmax = max([self.field.max(), a.field.max()])
        step = (fieldmax - fieldmin) / 1024
        fieldrange = np.arange(fieldmin, fieldmax, step)

        f1 = interp1d(self.field, self.spec, kind='cubic')
        f2 = interp1d(a.field, a.spec, kind='cubic')

        spc1 = f1(fieldrange)
        spc2 = f2(fieldrange)

        newspc = spc1 - spc2

        return CW_spec(fieldrange, newspc)

    def __mul__(self, a):
        return CW_spec(self.field, a * self.spec)

    # Analysis methods
    def abs_first_moment(self):
        iSpec = cumtrapz(self.spec, self.field, initial=0)
        cen_field_arg = iSpec.argmax()

        plt.plot(iSpec)
        plt.axhline(0, 0, 1024)
        plt.show()

        M = 0
        for i in range(cen_field_arg + 1):
            M = M + np.abs(self.field[i] - self.field[cen_field_arg]) * iSpec[i]

        M = M * 2
        return M
