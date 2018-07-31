import matplotlib.pyplot as plt
from eprTools import CW_spec

mySpc = CW_spec.from_file('Example_Apo.DTA', preprocess = True)
plt.plot(mySpc.field, mySpc.spec)
plt.show()

