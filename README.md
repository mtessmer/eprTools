# eprTools

eprTools is a package for interacting with, analyzing and plotting EPR data.
The two main classes DEER_spec and CW_spec corresponding to Double Electron-Electron Resonance experiments and Continuous Wave experiments

## Installation
```bash
git clone https://gitlab.com/mtessmer/eprTools.git 
cd eprTools
python setup.py install
```

## Getting Started -- CW_spec

```python
import matplotlib.pyplot as plt
from eprTools import CW_spec

mySpc = CW_spec.from_file('Example_Apo.DTA', preprocess = True)
plt.plot(mySpc.field, mySpc.spec)
plt.show()
```

## Getting Started -- DEER_Spec

```python
import matplotlib.pyplot as plt
from eprTools import DEER_spec

data = DEER_spec.from_file('Example_DEER.DTA')
data.set_kernel_len(500)

data.get_fit()

fig, (ax1, ax2) = plt.subplots(1,2, figsize = [20, 10.5])
ax1.plot(data.time, data.dipolar_evolution)
ax1.plot(data.fit_time, data.fit)
ax2.plot(data.r, data.P)
plt.show()
```


