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
