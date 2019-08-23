# import numpy as np
import sys
from astropy.io import ascii
# from astropy.table import Table, Column
import itertools

simcatname = sys.argv[1]
schemecode = sys.argv[2]

if schemecode == '424':
    cssbands = ['Nuv', 'u', 'g', 'r', 'i', 'z', 'y']
    filtnumb = [4, 2, 2, 2, 2, 2, 4]
elif schemecode == '222':
    cssbands = ['Nuv', 'u', 'g', 'r', 'i', 'z', 'WNuv', 'Wg', 'Wi']
    filtnumb = [2, 2, 2, 2, 2, 2, 2, 2, 2]
elif schemecode == '4262':
    cssbands = ['Nuv', 'u', 'g', 'r', 'i', 'z']
    filtnumb = [4, 2, 2, 2, 6, 2]
else:
    print("Please asign a scheme code, which should be '424', '222', or '4262'.")
    sys.exit()

simcat = ascii.read(simcatname)
simcat['Context'] = 0

namelists = map(lambda mag, err, aband:[mag+aband, err+aband],
                ['Magsim_']*len(cssbands), ['ErrMag_'] * len(cssbands), cssbands)

namelists = ['ID']+list(itertools.chain(*namelists))+['Context', 'Z_BEST']

lephcat = simcat[namelists]

for i,catline in enumerate(lephcat):
    for j,cssband in enumerate(cssbands):
        if catline['Magsim_'+cssband] >= 0:
            sign = 1
        else:
            sign = 0
        catline['Context'] = catline['Context']+2**j*sign

ascii.write(lephcat,sys.argv[1].split('.')[0]+'_toLephare.txt',format='commented_header', comment='#', overwrite=True)

print('Lephare Input Catalog Finished.')