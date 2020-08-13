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
from astropy import table


simcatname = sys.argv[1]
schemecode = sys.argv[2]
NLe = int(sys.argv[3])

defaults = {'basedir': '/work/CSSOS/filter_improve/fromimg/windextract'}
config = configparser.ConfigParser(defaults)
config.read('cssos_config_uBgN.ini')
# allbands = config.get('Hst2Css', 'CssBands').split(',')
allbands =['NUV', 'NUV2', 'u', 'g', 'r', 'i', 'z', 'z2', 'y', 'y2', 'WNUV', 'WNUV2', 'Wg', 'Wi', 'i4', 'uB', 'gN', 'WiBy',  'zN', 'WgB', 'WiN', 'WiNy','UV', 'uB410', 'gN410']

senarioavail = {'424': ['NUV', 'NUV2', 'u', 'g', 'r', 'i', 'z', 'y', 'y2'],
                '424FNUV2': ['NUV', 'u', 'g', 'r', 'i', 'z', 'y', 'y2'],
                '424Fu': ['NUV', 'NUV2', 'g', 'r', 'i', 'z', 'y', 'y2'],
                '424Fg': ['NUV', 'NUV2', 'u', 'r', 'i', 'z', 'y', 'y2'],
                '424Fr': ['NUV', 'NUV2', 'u', 'g', 'i', 'z', 'y', 'y2'],
                '424Fi': ['NUV', 'NUV2', 'u', 'g', 'r', 'z', 'y', 'y2'],
                '424Fz': ['NUV', 'NUV2', 'u', 'g', 'r', 'i', 'y', 'y2'],
                '424Fy2': ['NUV', 'NUV2', 'u', 'g', 'r', 'i', 'z', 'y'],
                '222':['NUV2', 'u', 'g', 'r', 'i', 'z', 'WNUV2', 'Wg', 'Wi'],
                '222FNUV2':['u', 'g', 'r', 'i', 'z', 'WNUV2', 'Wg', 'Wi'],
                '222Fu':['NUV2', 'g', 'r', 'i', 'z', 'WNUV2', 'Wg', 'Wi'],
                '222Fg':['NUV2', 'u', 'r', 'i', 'z', 'WNUV2', 'Wg', 'Wi'],
                '222Fr':['NUV2', 'u', 'g', 'i', 'z', 'WNUV2', 'Wg', 'Wi'],
                '222Fi':['NUV2', 'u', 'g', 'r', 'z', 'WNUV2', 'Wg', 'Wi'],
                '222Fz':['NUV2', 'u', 'g', 'r', 'i', 'WNUV2', 'Wg', 'Wi'],
                '222FWNUV2':['NUV2', 'u', 'g', 'r', 'i', 'z', 'Wg', 'Wi'],
                '222FWg':['NUV2', 'u', 'g', 'r', 'i', 'z', 'WNUV2', 'Wi'],
                '222Fwi':['NUV2', 'u', 'g', 'r', 'i', 'z', 'WNUV2', 'Wg'],
                '424uBgN':['NUV', 'NUV2', 'uB', 'gN', 'r', 'i', 'z', 'y', 'y2'],
                '424uBgN410':['NUV', 'NUV2', 'uB410', 'gN410', 'r', 'i', 'z', 'y', 'y2'],
				'424uBgNzN':['NUV', 'NUV2', 'uB', 'gN', 'r', 'i', 'zN', 'y', 'y2'],
                '222uBgN':['NUV2', 'uB', 'gN', 'r', 'i', 'z', 'WNUV2', 'Wg', 'Wi'],
                '222uBgN410':['NUV2', 'uB410', 'gN410', 'r', 'i', 'z', 'WNUV2', 'Wg', 'Wi'],
                '222WiBy':['NUV2', 'u', 'g', 'r', 'i', 'z', 'WNUV2', 'Wg', 'WiBy'],
                '222uBgNWiBy':['NUV2', 'uB', 'gN', 'r', 'i', 'z', 'WNUV2', 'Wg', 'WiBy'],
                '222uBgNWgBWiN':['NUV2', 'uB', 'gN', 'r', 'i', 'z', 'WNUV2', 'WgB', 'WiN'],
                '222uBgNWiByzN':['NUV2', 'uB', 'gN', 'r', 'i', 'zN', 'WNUV2', 'Wg', 'WiBy'],
                '4262':['NUV', 'NUV2', 'u', 'g', 'r', 'i', 'z', 'i4'],
                '4262Fz':['NUV', 'NUV2', 'u', 'g', 'r', 'i', 'i4'],
                '4262Fr':['NUV', 'NUV2', 'u', 'g', 'i', 'z', 'i4'],
                '4262Fg':['NUV', 'NUV2', 'u', 'r', 'i', 'z', 'i4'],
                '4262Fu':['NUV', 'NUV2', 'g', 'r', 'i', 'z', 'i4'],
                '4262FNUV2':['NUV', 'u', 'g', 'r', 'i', 'z', 'i4'],
                '4262Fi':['NUV', 'NUV2', 'u', 'g', 'r', 'z', 'i4'],
                '424uBgNFNUV2':['NUV', 'uB', 'gN', 'r', 'i', 'z', 'y', 'y2'],
                '222uBgNFNUV2':['uB', 'gN', 'r', 'i', 'z', 'WNUV2', 'Wg', 'Wi'],
                '222uBgNFWNUV2':['NUV2', 'uB', 'gN', 'r', 'i', 'z', 'Wg', 'Wi'],
                '424uBgNFuB':['NUV', 'NUV2', 'gN', 'r', 'i', 'z', 'y', 'y2'],
                '222uBgNFuB':['NUV2', 'gN', 'r', 'i', 'z', 'WNUV2', 'Wg', 'Wi'],
                '424uBgNFgN':['NUV', 'NUV2', 'uB', 'r', 'i', 'z', 'y', 'y2'],
                '222uBgNFgN':['NUV2', 'uB', 'r', 'i', 'z', 'WNUV2', 'Wg', 'Wi'],
                '424uBgNFr':['NUV', 'NUV2', 'uB', 'gN', 'i', 'z', 'y', 'y2'],
                '222uBgNFr':['NUV2', 'uB', 'gN', 'i', 'z', 'WNUV2', 'Wg', 'Wi'],
                '424uBgNFi':['NUV', 'NUV2', 'uB', 'gN', 'r', 'z', 'y', 'y2'],
                '222uBgNFi':['NUV2', 'uB', 'gN', 'r', 'z', 'WNUV2', 'Wg', 'Wi'],
                '424uBgNFz':['NUV', 'NUV2', 'uB', 'gN', 'r', 'i', 'y', 'y2'],
                '222uBgNFz':['NUV2', 'uB', 'gN', 'r', 'i', 'WNUV2', 'Wg', 'Wi'],
                '424uBgNFy2':['NUV', 'NUV2', 'uB', 'gN', 'r', 'i', 'z', 'y'],
                '222uBgNFWi':['NUV2', 'uB', 'gN', 'r', 'i', 'z', 'WNUV2', 'Wg'],
                '222uBgNFWg':['NUV2', 'uB', 'gN', 'r', 'i', 'z', 'WNUV2', 'Wi'],
                '242UVgN':['UV', 'gN', 'r', 'i', 'z', 'WNUV', 'WNUV2', 'Wg', 'Wi'],
                '242UVgNFUV':['gN', 'r', 'i', 'z', 'WNUV', 'WNUV2', 'Wg', 'Wi'],
                '242UVgNFWNUV2':['UV', 'gN', 'r', 'i', 'z', 'WNUV', 'Wg', 'Wi'],
                '242UVgNFgN':['UV', 'r', 'i', 'z', 'WNUV', 'WNUV2', 'Wg', 'Wi'],
                '242UVgNFr':['UV', 'gN', 'i', 'z', 'WNUV', 'WNUV2', 'Wg', 'Wi'],
                '242UVgNFi':['UV', 'gN', 'r', 'z', 'WNUV', 'WNUV2', 'Wg', 'Wi'],
                '242UVgNFz':['UV', 'gN', 'r', 'i', 'WNUV', 'WNUV2', 'Wg', 'Wi'],
                '242UVgNFWg':['UV', 'gN', 'r', 'i', 'z', 'WNUV', 'WNUV2', 'Wi'],
                '242UVgNFWi':['UV', 'gN', 'r', 'i', 'z', 'WNUV', 'WNUV2', 'Wg'],
                '224UVz4':   ['UV', 'g', 'r', 'i', 'z', 'z2', 'WNUV2', 'Wg', 'Wi'],
                '224UVz4Fz2':['UV', 'g', 'r', 'i', 'z', 'WNUV2', 'Wg', 'Wi'],
                '224UVz4FUV':['g', 'r', 'i', 'z', 'z2', 'WNUV2', 'Wg', 'Wi'],
                '224UVz4Fg': ['UV', 'r', 'i', 'z', 'z2', 'WNUV2', 'Wg', 'Wi'],
                '224UVz4Fr': ['UV', 'g', 'i', 'z', 'z2', 'WNUV2', 'Wg', 'Wi'],
                '224UVz4Fi': ['UV', 'g', 'r', 'z', 'z2', 'WNUV2', 'Wg', 'Wi'],
                '224UVz4FWNUV2':['UV', 'g', 'r', 'i', 'z', 'z2', 'Wg', 'Wi'],
                '224UVz4FWg':['UV', 'g', 'r', 'i', 'z', 'z2', 'WNUV2', 'Wi'],
                '224UVz4FWi':['UV', 'g', 'r', 'i', 'z', 'z2', 'WNUV2', 'Wg']
                }
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
# simcat = table.unique(simcat, keys='ID', silent=True)

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
