#!/usr/bin/env python
import matplotlib.pyplot as plt
from eprTools import CWSpec

mySpc1 = CWSpec.from_file('Example_Apo.DTA', preprocess = True)
mySpc2 = CWSpec.from_file('Example_Holo.DTA', preprocess = True)

plt.plot(mySpc1.field, mySpc1.spec)
plt.plot(mySpc2.field, mySpc2.spec)
plt.show()

mySpc1 = CWSpec.from_file('20160630_VOTPP_10.spc', preprocess = True)

plt.plot(mySpc1.field, mySpc1.spec)
plt.show()

