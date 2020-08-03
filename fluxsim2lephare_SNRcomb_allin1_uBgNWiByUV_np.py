"""
Usage: python3 fluxsim2lephare_SNRcomb_allin1_uBgNWiByUV_np.py $mergedsimmagfile 424 4
n processors for lephare.
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
NLe = int(sys.argv[3])

defaults = {'basedir': '/work/CSSOS/filter_improve/fromimg/windextract'}
config = configparser.ConfigParser(defaults)
config.read('cssos_config_uBgN.ini')
# allbands = config.get('Hst2Css', 'CssBands').split(',')
allbands =['NUV', 'NUV2', 'u', 'g', 'r', 'i', 'z', 'y', 'y2', 'WNUV', 'WNUV2', 'Wg', 'Wi', 'i4', 'uB', 'gN', 'WiBy', 'zN', 'WgB', 'WiN', 'UV', 'uB410', 'gN410']

senarioavail = {'424': ['NUV', 'NUV2', 'u', 'g', 'r', 'i', 'z', 'y', 'y2'],
                '424uBgN':['NUV', 'NUV2', 'uB', 'gN', 'r', 'i', 'z', 'y', 'y2'],
                '424uBgN410':['NUV', 'NUV2', 'uB410', 'gN410', 'r', 'i', 'z', 'y', 'y2'],
                '222':['NUV2', 'u', 'g', 'r', 'i', 'z', 'WNUV2', 'Wg', 'Wi'],
                '222uBgN':['NUV2', 'uB', 'gN', 'r', 'i', 'z', 'WNUV2' 'Wg', 'Wi'],
                '222uBgN410':['NUV2', 'uB410', 'gN410', 'r', 'i', 'z', 'WNUV2', 'Wg', 'Wi'],
                '222WiBy':['NUV2', 'u', 'g', 'r', 'i', 'z', 'WNUV2', 'Wg', 'WiBy'],
                '222uBgNWiBy':['NUV2', 'uB', 'gN', 'r', 'i', 'z', 'WNUV2', 'Wg', 'WiBy'],
                '222uBgNWiByzN':['NUV2', 'uB', 'gN', 'r', 'i', 'zN', 'WNUV2', 'Wg', 'WiBy'],
                '222UVgN':['UV', 'gN', 'r', 'i', 'z', 'WNUV', 'WNUV2', 'Wg', 'Wi'],
                '4262':['NUV', 'NUV2', 'u', 'g', 'r', 'i', 'z', 'i4'],
                '4262uBgN':['NUV', 'NUV2', 'uB', 'gN', 'r', 'i', 'z', 'i4']}
# NUV and NUV2 both have 2 NUV filters, in together contain 4 ; 
# y and y2 contain 4 in together, too;
# WNUV and WNUV2 contain 4 in together, too;

try:
    cssbands = senarioavail[schemecode]
except Exception as error:
    print("Please asign an available scheme code")
    sys.exit()

snrthr = 10
snrthr_single = 0

simcat = ascii.read(simcatname)

simcat['Context'] = 0

namelists = map(lambda flux, err, aband:[flux+aband, err+aband], ['FluxSim_']*len(allbands), ['ErrFlux_'] * len(allbands), allbands)
snrlists = map(lambda snr, aband:[snr+aband], ['SNR_']*len(allbands), allbands)
namelists = ['ID']+list(itertools.chain(*namelists))+['Context', 'Z_BEST'] +list(itertools.chain(*snrlists))+['Drms_sec']

for i,catline in enumerate(simcat):
    for j,aband in enumerate(allbands):
        if (aband in cssbands) and (simcat[i]['SNR_'+aband]>=snrthr_single):
            sign = 1
        else:
            sign = 0
        catline['Context'] = catline['Context']+2**j*sign


lephcat = simcat[namelists]
del simcat

ascii.write(lephcat,sys.argv[1].split('.')[0]+'_'+schemecode+'_toLephare.txt',format='commented_header', comment='#', overwrite=True)

print(sys.argv[1].split('.')[0]+'_'+schemecode+'_toLephare.txt')

lengthcat = len(lephcat)
sublength = int(lengthcat/NLe)

for i in range(NLe):
    if i<NLe-1:
        ascii.write(lephcat[sublength*i:sublength*(i+1)],sys.argv[1].split('.')[0]+'_'+schemecode+'_toLephare_'+str(i+1)+'.txt',format='commented_header', comment='#', overwrite=True)
        print(sys.argv[1].split('.')[0]+'_'+schemecode+'_toLephare_'+str(i+1)+'.txt')
    elif i==(NLe-1):
        ascii.write(lephcat[sublength*i:],sys.argv[1].split('.')[0]+'_'+schemecode+'_toLephare_'+str(i+1)+'.txt',format='commented_header', comment='#', overwrite=True)
        print(sys.argv[1].split('.')[0]+'_'+schemecode+'_toLephare_'+str(i+1)+'.txt')

print('Lephare Input Catalog Generated.\n')
