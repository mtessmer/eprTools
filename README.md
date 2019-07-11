# eprTools

eprTools is a package for interacting with and analyzing EPR data using numpy, scipy and sklearn.
The two main classes DEERspec and CWspec corresponding to Double Electron-Electron Resonance experiments and Continuous Wave experiments

## Installation
```bash
git clone https://gitlab.com/mtessmer/eprTools.git 
cd eprTools
python setup.py install
```

## Getting Started -- CWspec

```python
import matplotlib.pyplot as plt
from eprTools import CWSpec

mySpc1 = CWSpec.from_file('Example_Apo.DTA', preprocess = True)
mySpc2 = CWSpec.from_file('Example_Holo.DTA', preprocess = True)

plt.plot(mySpc1.field, mySpc1.spec)
plt.plot(mySpc2.field, mySpc2.spec)
plt.show()
```

## Getting Started -- DEERSpec

```python
import matplotlib.pyplot as plt
from eprTools import DEERSpec

# compute distance distribution
spc = DEERSpec.from_file('Example_DEER.DTA')
spc.set_kernel_r(rmin=15, rmax=60)
spc.set_kernel_len(250)

spc.get_fit()

# plot background corrected dipolar evolution, fit and distance distribution
fig, (ax1, ax2) = plt.subplots(1,2, figsize = [20, 10.5])
ax1.plot(spc.time, spc.dipolar_evolution)
ax1.plot(spc.fit_time, spc.fit)
ax2.plot(spc.r, spc.P)
plt.show()

# plot L-curve
rho, eta, alpha_idx = spc.get_L_curve()

fig2, ax = plt.subplots()
ax.scatter(rho, eta)
ax.scatter(rho[alpha_idx], eta[alpha_idx], c = 'r', facecolor=None)
plt.show()

print(spc.alpha)
```
