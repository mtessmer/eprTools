import os
import numpy as np
from eprTools import CWSpec, DeerExp


with open('../README.md', 'r') as f:

    subscripts = []
    active = False
    for line in f:
        if line.startswith('```python'):
            active = True
            scrpt = []
            continue
        if active and line.startswith('```'):
            scrpt = "".join(scrpt)
            subscripts.append(scrpt)
            active = False
            continue

        elif active:
            scrpt.append(line)


def test_readme_script():
    os.chdir('../examples')
    for scrpt in subscripts:
        exec(scrpt)
    os.chdir('../tests')