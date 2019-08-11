#!/usr/bin/env python
import matplotlib.pyplot as plt
from eprTools import DEERSpec
from eprTools import do_it_for_me
from time import  time

#do_it_for_me('Example_DEER.DTA', method = None)
#do_it_for_me('Example_DEER.DTA', method = None)
spc = DEERSpec.from_file('Example_DEER.DTA')
print("background: ", spc.background_param)

spc.set_kernel_r(rmin=15, rmax=80)
spc.set_kernel_len(250)                                                 # todo add examples for all features

spc.get_fit()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[20, 10.5])
ax1.plot(spc.time, spc.dipolar_evolution)
ax1.plot(spc.fit_time, spc.fit)
ax2.plot(spc.r, spc.P)
plt.show()



#Get L-curve
rho, eta, alpha_idx = spc.get_L_curve()

fig2, ax = plt.subplots()
ax.scatter(rho, eta)
ax.scatter(rho[alpha_idx], eta[alpha_idx], c='r', facecolor=None)
plt.show()

print(spc.alpha)

