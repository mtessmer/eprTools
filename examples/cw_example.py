import matplotlib.pyplot as plt
from eprTools import CWSpec

mySpc = CWSpec.from_file('Example_Apo.DTA', preprocess = True)
plt.plot(mySpc.field, mySpc.spec)
plt.show()

