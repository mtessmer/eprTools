import logging
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks_cwt
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from .utils import read_param_file

class CWSpec:
    """Continuous wave spectroscopy object

    Attributes:
        field (:obj:numpy.ndarray):
        spec (:obj:numpy.ndarray):

    """

    def __init__(self, field, spec, preprocess=False, k=0, ends=50, **kwargs):
        """
        Initialize the CW experiment object.

        :param field: (ndarray)
            Magnetic field of the continuous wave experiment.

        :param spec: (ndarray)
            First derivative of the absorbance spectrum at each magnetic field point.

        :param preprocess: (bool)
            When set to true the data will be baseline corrected using a polynomial fit of degree k. If k > 0
            the polynomial will be fit the the first and last 'ends' points of the spectrum normalized such that the
            second integral is equal to 1.

        :param k: (int)
            The order of the polynomial for baseline correction.

        :param ends:
            Number of points to use from each end of the spectrum for fitting polynomial for baseline correction.
        """

        self.field = field
        self.spec = spec
        self.param_dict = kwargs.get('param_dict', None)
        self.bg_spectrum = kwargs.get('bg_spectrum', None)
        self.k = k
        self.ends = ends

        if preprocess:
            self._prep()

    # import methods
    @classmethod
    def from_file(cls, file_name, preprocess=False, k=0, ends=50, **kwargs):
        """
        Import raw data from file.

        file types supported:
            Bruker DTA,DSC
            CSV
            Bruker winEPR spc

        :param file_name: (string)
            Name of file being imported.

        :param preprocess: (bool)
            When set to true the data will be baseline corrected using a polynomial fit of degree k. If k > 0
            the polynomial will be fit the the first and last 'ends' points of the spectrum normalized such that the
            second integral is equal to 1.

        :param k: (int)
            The order of the polynomial for baseline correction.

        :param ends:
            Number of points to use from each end of the spectrum for fitting polynomial for baseline correction.

        :return: CWobj
            An initialized CW experiment object
        """

        # Open the file
        with open(file_name, 'rb') as file:
            extension = file_name[-3:]
            # Determine file type
            if  extension == 'DTA':
                field, spec = load_bruker(file, file_name)
                parameter_file = file_name[:-3] + 'DSC'
                filetype = 'BES3T'
            elif extension == 'spc':
                field, spec = load_winepr(file, file_name)
                parameter_file = file_name[:-3] + 'par'
                filetype = 'ESP'
            else:
                field, spec = load_csv(file, file_name)
                parameter_file = None
                filetype='UNK'

            pardict = read_param_file(parameter_file)
            pardict['filetype'] = filetype
            CW_obj = cls(field, spec, preprocess, k, ends, param_dict=pardict, **kwargs)

            return CW_obj

    # Preparatory methods
    def _prep(self):
        self.basecorr()
        self.normalize()

    def basecorr(self):
        if self.bg_spectrum is not None:
            self.exp_basecorr()
        else:
            self.poly_basecorr()

    def exp_basecorr(self):

        parname = 'JSD' if self.param_dict['filetype'] == 'ESP' else 'AVGS'

        if isinstance(self.bg_spectrum, str):
            bg_spc = CWSpec.from_file(self.bg_spectrum)
        elif isinstance(self.bg_spectrum, CWSpec):
            bg_spc = self.bg_spectrum
        else:
            raise ValueError('`bg_spectrum` must be a CWSpec object or a file name containing a readable CW spectrum')

        bg_spc.spec = bg_spc.spec - np.mean(bg_spc.spec)
        norm_spec = bg_spc.spec / float(bg_spc.param_dict[parname][0])

        self.spec = self.spec / float(self.param_dict[parname][0])

        self.spec -= norm_spec




    def poly_basecorr(self):
        k = self.k
        ends = self.ends
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

        self.spec /= np.trapz(first_integral, self.field)

    def center(self, center_field=0):
        # Find the min and max of the spectra
        Xmax = np.argmax(self.spec)
        Xmin = np.argmin(self.spec)

        # Take a subset of min and max to find midpoint
        myMidSub = self.spec[Xmax:Xmin]
        subMidpoint = (np.abs(myMidSub)).argmin()
        midIdx = Xmax + subMidpoint
        midPoint = self.field[midIdx] - center_field
        self.center_field = self.field[midIdx]
        shape = np.minimum(midIdx, len(self.spec) - midIdx)

        # Set field s.t. the spectral midpoint is at center_field
        self.cfield = self.field[midIdx - shape: midIdx + shape] - midPoint  # self.field - midPoint + center_field
        self.cspec = self.spec[midIdx - shape: midIdx + shape]

    # Special Methods
    def __add__(self, a):

        field_min = min([self.field.min(), a.field.min()])
        field_max = max([self.field.max(), a.field.max()])
        step = (field_max - field_min) / 1024
        field_range = np.arange(field_min, field_max, step)

        f1 = interp1d(self.field, self.spec, kind='cubic')
        f2 = interp1d(a.field, a.spec, kind='cubic')

        spc1 = f1(field_range)
        spc2 = f2(field_range)

        new_spec = spc1 + spc2

        return CWSpec(field_range, new_spec)

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

    @property
    def abs_first_moment(self):
        H_hat = np.trapz(self.absorbance_spectrum * self.field, self.field)
        return np.trapz(np.abs(self.field - H_hat) * self.absorbance_spectrum, self.field)

    @property
    def second_moment(self):
        H_hat = np.trapz(self.absorbance_spectrum * self.field, self.field)
        integral = ((self.field - H_hat)**2) * self.absorbance_spectrum
        return np.trapz(integral, self.field)

    @property
    def absorbance_spectrum(self):
        return cumtrapz(self.spec, self.field, initial=0)


def guess_field(spec):
    logging.info("Param file not present or incorrectly named")
    logging.info("Guessing center field = 3487g, sweep width 100g")
    field_width = 100.0
    points = len(spec) - 1
    field_min = 3437
    field_max = field_min + field_width
    field_delta = field_width / points
    field = np.arange(field_min, field_max + field_delta, field_delta)

    return field

def load_csv(file, file_name):
    if file_name[-3:] == 'csv':
        spec = np.genfromtxt(file, delimiter=',')
    else:
        spec = np.genfromtxt(file)

    if len(spec.shape) > 1:
        if spec.shape[0] == 2:
            field, spec = spec
        else:
            field, spec = spec.T
    else:
        field = guess_field(spec)

    if len(field) == 2048:
        field = field[::2]
        spec = spec[::2]

    return field, spec


def load_winepr(file, file_name):
    spec = np.frombuffer(file.read(), dtype='f')

    # Look for x_data from DSC file
    param_file = file_name[:-3] + 'par'

    try:
        param_dict = {}
        with open(param_file, 'r', encoding='utf8', errors='ignore') as f:

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

                param_dict[key] = val

        points = int(param_dict['ANZ'][0])
        field_min = float(param_dict['GST'][0])
        field_width = float(param_dict['GSI'][0])
        field_max = field_min + field_width
        field = np.linspace(field_min, field_max, points)

    except OSError:

        field = guess_field(spec)

    if len(field) == 2048:
        field = field[::2]
        spec = spec[::2]

    return field, spec


def load_bruker(file, file_name):
    spec = np.frombuffer(file.read(), dtype='>d')

    # Look for x_data from DSC file
    param_file = file_name[:-3] + 'DSC'
    try:
        param_dict = {}
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

                param_dict[key] = val

        points = int(param_dict['XPTS'][0])
        field_min = float(param_dict['XMIN'][0])
        field_width = float(param_dict['XWID'][0])
        field_max = field_min + field_width
        field = np.linspace(field_min, field_max, points)

    except OSError:
        field = guess_field(spec)

    if len(field) == 2048:
        print('array size 2048. downsizing to 1024')
        field = field[::2]
        spec = spec[::2]

    return field, spec
