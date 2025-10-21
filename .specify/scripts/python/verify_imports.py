import sys

mods = {}
try:
    import numpy as np; mods['numpy'] = np.__version__
    import pandas as pd; mods['pandas'] = pd.__version__
    import matplotlib; mods['matplotlib'] = matplotlib.__version__
    import yfinance; mods['yfinance'] = yfinance.__version__
    import yaml; mods['pyyaml'] = getattr(yaml, '__version__', 'n/a')
    import pptx; mods['python-pptx'] = getattr(pptx, '__version__', 'n/a')
    print('OK', mods)
except Exception as e:
    print('FAIL', e)
    sys.exit(1)

