
import matplotlib.pyplot as plt
from eprTools import DEERSpec

spc = DEERSpec.from_file('examples/Example_DEER.DTA')
#spc.set_trim(3000)
#spc.set_background_correction(fit_time=700)
spc.set_L_criteria(mode="aic")
spc.set_kernel_r(rmin=15, rmax=60)
spc.set_kernel_len(250)                                                 # todo add examples for all features

spc.get_fit()


fig, (ax1, ax2) = plt.subplots(1,2, figsize = [20, 10.5])
ax1.plot(spc.time, spc.dipolar_evolution)
ax1.plot(spc.fit_time, spc.fit)
ax2.plot(spc.r, spc.P)
plt.show()



