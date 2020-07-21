"""
Usage: python3 fluxsim2lephare_SNRcomb_allin1.py $mergedsimmagfile 424
Select samples whose composite SNR>10
"""


# import numpy as np
import configparser
import sys
from astropy.io import ascii
from astropy.table import Table, Column
import itertools
import numpy as np

simcatname = sys.argv[1]
schemecode = sys.argv[2]

defaults = {'basedir': '/work/CSSOS/filter_improve/fromimg/windextract'}
config = configparser.ConfigParser(defaults)
config.read('cssos_config_uBgN.ini')
# allbands = config.get('Hst2Css', 'CssBands').split(',')
allbands =['NUV', 'NUV2', 'u', 'g', 'r', 'i', 'z', 'y', 'y2', 'WNUV', 'Wg', 'Wi', 'i4', 'uB', 'gN']
# allbands =['NUV', 'u', 'g', 'r', 'i', 'z', 'y', 'WNUV', 'Wg', 'Wi']

if schemecode == '424':
    # NUV and NUV2 both have 2 NUV filters, in together contain 4 ; y and y2 contain 4 in together, too;
    cssbands = ['NUV', 'NUV2', 'u', 'g', 'r', 'i', 'z', 'y', 'y2']
elif schemecode == '424uBgN':
    cssbands = ['NUV', 'NUV2', 'uB', 'gN', 'r', 'i', 'z', 'y', 'y2']
elif schemecode == '222':
    cssbands = ['NUV2', 'u', 'g', 'r', 'i', 'z', 'WNUV', 'Wg', 'Wi']
elif schemecode == '222uBgN':
    cssbands = ['NUV2', 'uB', 'gN', 'r', 'i', 'z', 'WNUV', 'Wg', 'Wi']
elif schemecode == '4262':
    cssbands = ['NUV', 'NUV2', 'u', 'g', 'r', 'i', 'z', 'i4']
elif schemecode == '4262uBgN':
    cssbands = ['NUV', 'NUV2', 'uB', 'gN', 'r', 'i', 'z', 'i4']
else:
    print("Please asign a scheme code, which should be '424', '222', '424uBgN', '222uBgN', '4262', or '4262uBgN'.")
    sys.exit()

snrthr = 10
snrthr_single = 0

simcat0 = ascii.read(simcatname)

# Sample selection:
if schemecode == '4262uBgN':
    idx = np.where((simcat0['SNR_gN']**2+simcat0['SNR_r']**2+simcat0['SNR_i6']**2+simcat0['SNR_z']**2) >= snrthr**2)
elif (schemecode=='424uBgN') or (schemecode=='222uBgN'):
    idx = np.where((simcat0['SNR_gN']**2+simcat0['SNR_r']**2+simcat0['SNR_i']**2+simcat0['SNR_z']**2) >= snrthr**2)
elif (schemecode=='424') or (schemecode=='222'):
    idx = np.where((simcat0['SNR_g']**2+simcat0['SNR_r']**2+simcat0['SNR_i']**2+simcat0['SNR_z']**2) >= snrthr**2)
# idx = np.where((simcat0['SNR_r']**2+simcat0['SNR_i']**2) >= snrthr**2)
# idx = np.where((simcat0['SNR_r']>=snrthr)|(simcat0['SNR_i']>=snrthr))

simcat = simcat0[idx]

simcat['Context'] = 0

#namelists = map(lambda flux, err, aband:[flux+aband, err+aband], ['FluxSim_']*len(cssbands), ['ErrFlux_'] * len(cssbands), cssbands)

# print(set(allbands)-set(cssbands))

#namelists = ['ID']+list(itertools.chain(*namelists))+['Context', 'Z_BEST', 'SNR_NUV', 'SNR_NUV2', 'SNR_u', 'SNR_g', 'SNR_r',  'SNR_i', 'SNR_z', 'SNR_y', 'SNR_y2', 'SNR_WNUV', 'SNR_Wg', 'SNR_Wi','SNR_i4', 'SNR_uB', 'SNR_gN', 'Drms_sec']
# if schemecode == '4262uBgN':
# elif (schemecode=='424uBgN') or (schemecode=='222uBgN'):
#     namelists = ['ID']+list(itertools.chain(*namelists))+['Context', 'Z_BEST', 'SNR_uB', 'SNR_gN', 'SNR_r', 'SNR_i', 'SNR_z', 'Drms_sec']

namelists = map(lambda flux, err, aband:[flux+aband, err+aband], ['FluxSim_']*len(allbands), ['ErrFlux_'] * len(allbands), allbands)
snrlists = map(lambda snr, aband:[snr+aband], ['SNR_']*len(allbands), allbands)
namelists = ['ID']+list(itertools.chain(*namelists))+['Context', 'Z_BEST'] +list(itertools.chain(*snrlists))+['Drms_sec']

# namelists = map(lambda flux, err, aband:[flux+aband, err+aband], ['FluxSim_']*len(allbands), ['ErrFlux_'] * len(allbands), allbands)
# namelists = ['ID']+list(itertools.chain(*namelists))+['Context', 'Z_BEST']

for i,catline in enumerate(simcat):
    # nbands = 0
    for j,aband in enumerate(allbands):
        # if ((catline['MagSim_'+cssband]>0) and (simcat[i]['SNR_'+cssband]>=snrthr)):
        # # if ((catline['MOD_'+cssband]>=0) and (simcat[i]['SNR_'+cssband]>=snrthr)):
        #     if (np.abs(catline['MagSim_' + cssband] - catline['MOD_' + cssband]) < magdiff):
        #         sign = 1
        # nbands = nbands + 1
        # else:
        if (aband in cssbands) and (simcat[i]['SNR_'+aband]>=snrthr_single):
            sign = 1
            # nbands = nbands + 1
        else:
            sign = 0

        catline['Context'] = catline['Context']+2**j*sign


lephcat = simcat[namelists]
# lephcat = lephcat[lephcat['Context']>0]
print('')
print(str(len(lephcat))+'/'+str(len(simcat0)),' meet SNR criterian.')

ascii.write(lephcat,sys.argv[1].split('.')[0]+'_'+schemecode+'_flux_toLephare.txt',format='commented_header', comment='#', overwrite=True)

print(sys.argv[1].split('.')[0]+'_'+schemecode+'_flux_toLephare.txt')
print('Lephare Input Catalog Generated.\n')
