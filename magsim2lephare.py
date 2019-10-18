# import numpy as np
import sys
from astropy.io import ascii
# from astropy.table import Table, Column
import itertools
import numpy as np

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

namelists = map(lambda mag, err, aband:[mag+aband, err+aband], ['MagSim_']*len(cssbands), ['ErrMag_'] * len(cssbands), cssbands)
# namelists = map(lambda mag, err, aband:[mag+aband, err+aband], ['MOD_']*len(cssbands), ['ErrMag_'] * len(cssbands), cssbands)

namelists = ['ID']+list(itertools.chain(*namelists))+['Context', 'Z_BEST']

snrthr = 3
magdiff = 1

for i,catline in enumerate(simcat):
    nbands = 0
    for j,cssband in enumerate(cssbands):
        if ((catline['MagSim_'+cssband]>=0) and (simcat[i]['SNR_'+cssband]>=snrthr)):
        # if ((catline['MOD_'+cssband]>=0) and (simcat[i]['SNR_'+cssband]>=snrthr)):
            if (np.abs(catline['MagSim_' + cssband] - catline['MOD_' + cssband]) < magdiff):
                sign = 1
                nbands = nbands + 1
        else:
            sign = 0
        catline['Context'] = catline['Context']+2**j*sign
    if nbands<=4:
        catline['Context'] = -99
        # for j,cssband in enumerate(cssbands):
        #     if simcat[i]['SNR_'+cssband]<=snrthr:
                # catline['MagSim_'+cssband] = -99
                # catline['ErrMag_'+cssband] = -99


lephcat = simcat[namelists]

ascii.write(lephcat,sys.argv[1].split('.')[0]+'_toLephare.txt',format='commented_header', comment='#', overwrite=True)

print(sys.argv[1].split('.')[0]+'_toLephare.txt')
print('Lephare Input Catalog Finished.')