"""
Usage: python3 fluxsim2lephare_SNRcomb_allin1.py $mergedsimmagfile 424
Select samples whose composite SNR>10
"""


# import numpy as np
import sys
from astropy.io import ascii
from astropy.table import Table, Column
import itertools
import numpy as np

simcatname = sys.argv[1]
schemecode = sys.argv[2]

if schemecode == '424':
    cssbands = ['NUV', 'u', 'g', 'r', 'i', 'z', 'y']
    # filtnumb = [4, 2, 2, 2, 2, 2, 4]
    totnbands = 7
elif schemecode == '222':
    cssbands = ['NUV2', 'u', 'g', 'r', 'i', 'z', 'WNUV', 'Wg', 'Wi']
    # filtnumb = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    totnbands = 9
elif schemecode == '4262':
    cssbands = ['NUV', 'u', 'g', 'r', 'i6', 'z']
    # filtnumb = [4, 2, 2, 2, 6, 2]
    totnbands = 6
else:
    print("Please asign a scheme code, which should be '424', '222', or '4262'.")
    sys.exit()

snrthr = 10
snrthr_single = 0

simcat0 = ascii.read(simcatname)

# Sample selection:
idx = np.where((simcat0['SNR_g']**2+simcat0['SNR_r']**2+simcat0['SNR_i']**2+simcat0['SNR_z']**2) >= snrthr**2)
# idx = np.where((simcat0['SNR_r']**2+simcat0['SNR_i']**2) >= snrthr**2)
# idx = np.where((simcat0['SNR_r']>=snrthr)|(simcat0['SNR_i']>=snrthr))

simcat = simcat0[idx]

simcat['Context'] = 0

namelists = map(lambda flux, err, aband:[flux+aband, err+aband], ['FluxSim_']*len(cssbands), ['ErrFlux_'] * len(cssbands), cssbands)
# namelists = map(lambda mag, err, aband:[mag+aband, err+aband], ['MOD_']*len(cssbands), ['ErrMag_'] * len(cssbands), cssbands)

namelists = ['ID']+list(itertools.chain(*namelists))+['Context', 'Z_BEST', 'SNR_u', 'SNR_g', 'SNR_r', 'SNR_i', 'SNR_z']

for i,catline in enumerate(simcat):
    nbands = 0
    for j,cssband in enumerate(cssbands):
        # if ((catline['MagSim_'+cssband]>0) and (simcat[i]['SNR_'+cssband]>=snrthr)):
        # # if ((catline['MOD_'+cssband]>=0) and (simcat[i]['SNR_'+cssband]>=snrthr)):
        #     if (np.abs(catline['MagSim_' + cssband] - catline['MOD_' + cssband]) < magdiff):
        #         sign = 1
        # nbands = nbands + 1
        # else:
        if simcat[i]['SNR_'+cssband]>=snrthr_single:
            sign = 1
            nbands = nbands + 1
        else:
            sign = 0

        catline['Context'] = catline['Context']+2**j
    if nbands < (totnbands):
        catline['Context'] = -99

lephcat = simcat[namelists]
lephcat = lephcat[lephcat['Context']>0]
print('')
print(str(len(lephcat))+'/'+str(len(simcat0)),' meet SNR criterian.')

ascii.write(lephcat,sys.argv[1].split('.')[0]+'_'+schemecode+'_flux_toLephare.txt',format='commented_header', comment='#', overwrite=True)

print(sys.argv[1].split('.')[0]+'_'+schemecode+'_flux_toLephare.txt')
print('Lephare Input Catalog Generated.\n')