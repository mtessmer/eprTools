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

# import EPR data
mySpc1 = CWSpec.from_file('Example_Apo.DTA', preprocess = True)
mySpc2 = CWSpec.from_file('Example_Holo.DTA', preprocess = True)

# plot CW spectra
fig, ax = plt.subplots()
ax.plot(mySpc1.field, mySpc1.spec)
ax.plot(mySpc2.field, mySpc2.spec)
ax.set_xlabel('field (G)')
ax.set_yticks([])
plt.show()
fig.savefig('CW.png')
```
![CW](examples/CW.png)


## Getting Started -- DEERSpec

```python
import matplotlib.pyplot as plt
from eprTools import DeerExp

# import data
spc = DeerExp.from_file('Example_DEER.DTA', r=(15, 60))

# fit data
spc.get_fit()

# plot form factor, background correction, fit and distance distribution
fig, (ax1, ax2) = plt.subplots(2, constrained_layout=True)
ax1.plot(spc.t, spc.V)
ax1.plot(spc.t, spc.Vfit)
ax1.plot(spc.t, spc.B)
ax1.set_xlabel(r'time ($\rm\mu s$)')

ax2.plot(spc.r, spc.P)
ax2.set_yticks([])
ax2.set_xlabel(r'distance ($\rm\AA$)')
plt.show()
fig.savefig('DEER.png')
```
![DEER](examples/DEER.png)


```python
# plot L-curve
rho, eta, alpha_idx = spc.get_L_curve()

fig2, ax = plt.subplots()
ax.scatter(rho, eta)
ax.scatter(rho[alpha_idx], eta[alpha_idx], c='r', facecolor=None)
ax.set_xlabel(r'$\rm\rho$')
ax.set_ylabel(r'$\rm\eta$')

plt.show()
fig2.savefig('L_curve.png')
```
![L-curve](examples/L_curve.png)
