"""
Usage: python3 fluxsim2lephare_SNR10_allin1.py $mergedsimmagfile 424
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
    cssbands = ['NUV_2', 'u', 'g', 'r', 'i', 'z', 'WNUV', 'Wg', 'Wi']
    # filtnumb = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    totnbands = 9
elif schemecode == '4262':
    cssbands = ['NUV', 'u', 'g', 'r', 'i', 'z']
    # filtnumb = [4, 2, 2, 2, 6, 2]
    totnbands = 6
else:
    print("Please asign a scheme code, which should be '424', '222', or '4262'.")
    sys.exit()

simcat0 = ascii.read(simcatname)

# Sample selection:

# simcat = simcat0[np.where((simcat0['SNR_g']>=5 & simcat0['SNR_r']>=5 & simcat0['SNR_i']>=5 & simcat0['SNR_z']>=5) | (simcat0['SNR_r']>=7 & simcat0['SNR_i']>=7) | simcat0['SNR_r']>=10 | simcat0['SNR_g']>=10 | (simcat0['SNR_g']**2+simcat0['SNR_r']**2+simcat0['SNR_i']**2+simcat0['SNR_z']**2)>=100)]


# simcat = Table(names=simcat0.colnames, dtype=list(simcat0.dtype))
# print(simcat)
simcat = simcat0.copy()
del simcat[:]

for i,aline in enumerate(simcat0):
    # if (((aline['SNR_g']>=5) and (aline['SNR_r']>=5) and (aline['SNR_i']>=5) and (aline['SNR_z']>=5))
    if (((aline['SNR_r']**2+aline['SNR_i']**2)>=100) \
            # or (aline['SNR_r']>=7 and (aline['SNR_i']>=7)) 
            or (aline['SNR_g']>=10) or (aline['SNR_r']>=10) \
            or (aline['SNR_i']>=10) or (aline['SNR_z']>=10) \
            or ((aline['SNR_g']**2+aline['SNR_r']**2+aline['SNR_i']**2+aline['SNR_z']**2)>=100)):
            # and (aline['SNR_NUV_2']>3)):
        simcat.add_row(aline)

print(len(simcat), len(simcat0))

simcat['Context'] = 0

namelists = map(lambda flux, err, aband:[flux+aband, err+aband], ['FluxSim_']*len(cssbands), ['ErrFlux_'] * len(cssbands), cssbands)
# namelists = map(lambda mag, err, aband:[mag+aband, err+aband], ['MOD_']*len(cssbands), ['ErrMag_'] * len(cssbands), cssbands)

namelists = ['ID']+list(itertools.chain(*namelists))+['Context', 'Z_BEST']

snrthr = 0

for i,catline in enumerate(simcat):
    nbands = 0
    for j,cssband in enumerate(cssbands):
        # if ((catline['MagSim_'+cssband]>0) and (simcat[i]['SNR_'+cssband]>=snrthr)):
        # # if ((catline['MOD_'+cssband]>=0) and (simcat[i]['SNR_'+cssband]>=snrthr)):
        #     if (np.abs(catline['MagSim_' + cssband] - catline['MOD_' + cssband]) < magdiff):
        #         sign = 1
        # nbands = nbands + 1
        # else:
        if simcat[i]['SNR_'+cssband]>=snrthr:
            sign = 1
            nbands = nbands + 1
        else:
            sign = 0

        catline['Context'] = catline['Context']+2**j
    if nbands < (totnbands):
        catline['Context'] = -99



lephcat = simcat[namelists]
lephcat = lephcat[lephcat['Context']>0]

ascii.write(lephcat,sys.argv[1].split('.')[0]+'_'+schemecode+'_flux_toLephare.txt',format='commented_header', comment='#', overwrite=True)

print(sys.argv[1].split('.')[0]+'_'+schemecode+'_flux_toLephare.txt')
print('Lephare Input Catalog Finished.')