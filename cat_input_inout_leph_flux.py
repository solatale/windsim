
"""
To produce an input catalog to Lephare from the "mag-err-zspec" catalog.
Usage: python cat_input_leph.py cosmos_allfilters.txt
"""

import numpy as np
import sys
from astropy.io import ascii
from astropy.io import fits
from astropy.table import Table, Column

# #424
cssbands = ['Nuv', 'u', 'g', 'r', 'i', 'z', 'y']
#222
# cssbands = ['Nuv', 'WNuv', 'u', 'g', 'r', 'i', 'z', 'Wg', 'Wi']
# #4262
# cssbands = ['Nuv', 'u', 'g', 'r', 'i', 'z']


cataorig = ascii.read(sys.argv[1])
catafilled = cataorig.filled(-99)

# for i in range(len(catafilled)):
for cssband in cssbands:
    catafilled['SimFnu_'+cssband] = 10**(-0.4*(catafilled['Simag_'+cssband]+48.6))

catafilled['Context'] = '127'

catafilled.keep_columns(['ID']+['SimFnu_'+aband for aband in cssbands]+['Errflux_'+aband for aband in cssbands]+['Context','Z_BEST'])

catafilled.rename_column('Z_BEST','Z-SPEC')
catafilled['ID'].format = '.0f'

outname = sys.argv[1].split('.')[0]+'_toLephare.dat'
ascii.write(catafilled, outname, format='commented_header',comment='#',overwrite=True)
