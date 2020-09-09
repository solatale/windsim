"""
Usage: python3 fluxsim2lephare_SNRcomb_allin1_uBgNWIByUV_np.py $mergedsimmagfile 424 4
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
allbands =['NUV', 'NUV2', 'u', 'g', 'r', 'i', 'z', 'z2', 'y', 'y2', 'WU', 'WU2', 'WV', 'WI', 'i4', 'uB', 'gN', 'WIBy',  'zN', 'WVB', 'WIN', 'WINy','WUv', 'uB410', 'gN410']

senarioavail = {'424': ['NUV', 'NUV2', 'u', 'g', 'r', 'i', 'z', 'y', 'y2'],
                '424FNUV2': ['NUV', 'u', 'g', 'r', 'i', 'z', 'y', 'y2'],
                '424Fu': ['NUV', 'NUV2', 'g', 'r', 'i', 'z', 'y', 'y2'],
                '424Fg': ['NUV', 'NUV2', 'u', 'r', 'i', 'z', 'y', 'y2'],
                '424Fr': ['NUV', 'NUV2', 'u', 'g', 'i', 'z', 'y', 'y2'],
                '424Fi': ['NUV', 'NUV2', 'u', 'g', 'r', 'z', 'y', 'y2'],
                '424Fz': ['NUV', 'NUV2', 'u', 'g', 'r', 'i', 'y', 'y2'],
                '424Fy2': ['NUV', 'NUV2', 'u', 'g', 'r', 'i', 'z', 'y'],
                '222':['NUV2', 'u', 'g', 'r', 'i', 'z', 'WU2', 'WV', 'WI'],
                '222FNUV2':['u', 'g', 'r', 'i', 'z', 'WU2', 'WV', 'WI'],
                '222Fu':['NUV2', 'g', 'r', 'i', 'z', 'WU2', 'WV', 'WI'],
                '222Fg':['NUV2', 'u', 'r', 'i', 'z', 'WU2', 'WV', 'WI'],
                '222Fr':['NUV2', 'u', 'g', 'i', 'z', 'WU2', 'WV', 'WI'],
                '222Fi':['NUV2', 'u', 'g', 'r', 'z', 'WU2', 'WV', 'WI'],
                '222Fz':['NUV2', 'u', 'g', 'r', 'i', 'WU2', 'WV', 'WI'],
                '222FWU2':['NUV2', 'u', 'g', 'r', 'i', 'z', 'WV', 'WI'],
                '222FWV':['NUV2', 'u', 'g', 'r', 'i', 'z', 'WU2', 'WI'],
                '222Fwi':['NUV2', 'u', 'g', 'r', 'i', 'z', 'WU2', 'WV'],
                '424uBgN':['NUV', 'NUV2', 'uB', 'gN', 'r', 'i', 'z', 'y', 'y2'],
                '424uBgN410':['NUV', 'NUV2', 'uB410', 'gN410', 'r', 'i', 'z', 'y', 'y2'],
				'424uBgNzN':['NUV', 'NUV2', 'uB', 'gN', 'r', 'i', 'zN', 'y', 'y2'],
                '222uBgN':['NUV2', 'uB', 'gN', 'r', 'i', 'z', 'WU2', 'WV', 'WI'],
                '222uBgN410':['NUV2', 'uB410', 'gN410', 'r', 'i', 'z', 'WU2', 'WV', 'WI'],
                '222WIBy':['NUV2', 'u', 'g', 'r', 'i', 'z', 'WU2', 'WV', 'WIBy'],
                '222uBgNWIBy':['NUV2', 'uB', 'gN', 'r', 'i', 'z', 'WU2', 'WV', 'WIBy'],
                '222uBgNWVBWIN':['NUV2', 'uB', 'gN', 'r', 'i', 'z', 'WU2', 'WVB', 'WIN'],
                '222uBgNWIByzN':['NUV2', 'uB', 'gN', 'r', 'i', 'zN', 'WU2', 'WV', 'WIBy'],
                '4262':['NUV', 'NUV2', 'u', 'g', 'r', 'i', 'z', 'i4'],
                '4262Fz':['NUV', 'NUV2', 'u', 'g', 'r', 'i', 'i4'],
                '4262Fr':['NUV', 'NUV2', 'u', 'g', 'i', 'z', 'i4'],
                '4262Fg':['NUV', 'NUV2', 'u', 'r', 'i', 'z', 'i4'],
                '4262Fu':['NUV', 'NUV2', 'g', 'r', 'i', 'z', 'i4'],
                '4262FNUV2':['NUV', 'u', 'g', 'r', 'i', 'z', 'i4'],
                '4262Fi':['NUV', 'NUV2', 'u', 'g', 'r', 'z', 'i4'],
                '424uBgNFNUV2':['NUV', 'uB', 'gN', 'r', 'i', 'z', 'y', 'y2'],
                '222uBgNFNUV2':['uB', 'gN', 'r', 'i', 'z', 'WU2', 'WV', 'WI'],
                '222uBgNFWU2':['NUV2', 'uB', 'gN', 'r', 'i', 'z', 'WV', 'WI'],
                '424uBgNFuB':['NUV', 'NUV2', 'gN', 'r', 'i', 'z', 'y', 'y2'],
                '222uBgNFuB':['NUV2', 'gN', 'r', 'i', 'z', 'WU2', 'WV', 'WI'],
                '424uBgNFgN':['NUV', 'NUV2', 'uB', 'r', 'i', 'z', 'y', 'y2'],
                '222uBgNFgN':['NUV2', 'uB', 'r', 'i', 'z', 'WU2', 'WV', 'WI'],
                '424uBgNFr':['NUV', 'NUV2', 'uB', 'gN', 'i', 'z', 'y', 'y2'],
                '222uBgNFr':['NUV2', 'uB', 'gN', 'i', 'z', 'WU2', 'WV', 'WI'],
                '424uBgNFi':['NUV', 'NUV2', 'uB', 'gN', 'r', 'z', 'y', 'y2'],
                '222uBgNFi':['NUV2', 'uB', 'gN', 'r', 'z', 'WU2', 'WV', 'WI'],
                '424uBgNFz':['NUV', 'NUV2', 'uB', 'gN', 'r', 'i', 'y', 'y2'],
                '222uBgNFz':['NUV2', 'uB', 'gN', 'r', 'i', 'WU2', 'WV', 'WI'],
                '424uBgNFy2':['NUV', 'NUV2', 'uB', 'gN', 'r', 'i', 'z', 'y'],
                '222uBgNFWI':['NUV2', 'uB', 'gN', 'r', 'i', 'z', 'WU2', 'WV'],
                '222uBgNFWV':['NUV2', 'uB', 'gN', 'r', 'i', 'z', 'WU2', 'WI'],
                '242WUvgN':['WUv', 'gN', 'r', 'i', 'z', 'WU', 'WU2', 'WV', 'WI'],
                '242WUvgNFWUv':['gN', 'r', 'i', 'z', 'WU', 'WU2', 'WV', 'WI'],
                '242WUvgNFWU2':['WUv', 'gN', 'r', 'i', 'z', 'WU', 'WV', 'WI'],
                '242WUvgNFgN':['WUv', 'r', 'i', 'z', 'WU', 'WU2', 'WV', 'WI'],
                '242WUvgNFr':['WUv', 'gN', 'i', 'z', 'WU', 'WU2', 'WV', 'WI'],
                '242WUvgNFi':['WUv', 'gN', 'r', 'z', 'WU', 'WU2', 'WV', 'WI'],
                '242WUvgNFz':['WUv', 'gN', 'r', 'i', 'WU', 'WU2', 'WV', 'WI'],
                '242WUvgNFWV':['WUv', 'gN', 'r', 'i', 'z', 'WU', 'WU2', 'WI'],
                '242WUvgNFWI':['WUv', 'gN', 'r', 'i', 'z', 'WU', 'WU2', 'WV'],
                '224WUvz4':   ['WUv', 'g', 'r', 'i', 'z', 'z2', 'WU2', 'WV', 'WI'],
                '224WUvz4Fz2':['WUv', 'g', 'r', 'i', 'z', 'WU2', 'WV', 'WI'],
                '224WUvz4FWUv':['g', 'r', 'i', 'z', 'z2', 'WU2', 'WV', 'WI'],
                '224WUvz4Fg': ['WUv', 'r', 'i', 'z', 'z2', 'WU2', 'WV', 'WI'],
                '224WUvz4Fr': ['WUv', 'g', 'i', 'z', 'z2', 'WU2', 'WV', 'WI'],
                '224WUvz4Fi': ['WUv', 'g', 'r', 'z', 'z2', 'WU2', 'WV', 'WI'],
                '224WUvz4FWU2':['WUv', 'g', 'r', 'i', 'z', 'z2', 'WV', 'WI'],
                '224WUvz4FWV':['WUv', 'g', 'r', 'i', 'z', 'z2', 'WU2', 'WI'],
                '224WUvz4FWI':['WUv', 'g', 'r', 'i', 'z', 'z2', 'WU2', 'WV'],
                '224WUvuz4':   ['WUv', 'u', 'g', 'r', 'i', 'z', 'z2', 'WV', 'WI'],
                '224WUvuz4FWUv':   ['u', 'g', 'r', 'i', 'z', 'z2', 'WV', 'WI'],
                '224WUvuz4Fu':   ['WUv', 'g', 'r', 'i', 'z', 'z2', 'WV', 'WI'],
                '224WUvNUVz4':     ['WUv', 'NUV', 'g', 'r', 'i', 'z', 'z2', 'WV', 'WI'],
                '224WUvNUVz4FWUv': ['NUV', 'g', 'r', 'i', 'z', 'z2', 'WV', 'WI'],
                '224WUvNUVz4FNUV': ['WUv', 'g', 'r', 'i', 'z', 'z2', 'WV', 'WI'],
                '224WUvNUVz4Fg':   ['WUv', 'NUV', 'r', 'i', 'z', 'z2', 'WV', 'WI'],
                '224WUvNUVz4Fr':   ['WUv', 'NUV', 'g', 'i', 'z', 'z2', 'WV', 'WI'],
                '224WUvNUVz4Fi':   ['WUv', 'NUV', 'g', 'r', 'z', 'z2', 'WV', 'WI'],
                '224WUvNUVz4Fz2':  ['WUv', 'NUV', 'g', 'r', 'i', 'z', 'WV', 'WI'],
                '224WUvNUVz4FWV':  ['WUv', 'NUV', 'g', 'r', 'i', 'z', 'z2', 'WI'],
                '224WUvNUVz4FWI':  ['WUv', 'NUV', 'g', 'r', 'i', 'z', 'z2', 'WV']
                }
# NUV and NUV2 both have 2 NUV filters, in together contain 4 ; 
# y and y2 contain 4 in together, too;
# WU and WU2 contain 4 in together, too;

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
