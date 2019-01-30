
import matplotlib.pyplot as plt
from eprTools import DEER_spec

spc = DEERSpec.from_file('Example_DEER.DTA')
spc.set_kernel_len(350)                                                 # todo add examples for all features

spc.get_fit()


fig, (ax1, ax2) = plt.subplots(1,2, figsize = [20, 10.5])
ax1.plot(spc.time, spc.dipolar_evolution)
ax1.plot(spc.fit_time, spc.fit)
ax2.plot(spc.r, spc.P)
plt.show()




