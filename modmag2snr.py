
"""
To produce an input catalog to Lephare from the "mag-err-zspec" catalog.
Usage: python modmag2lephfnu_222.py OutCssos_tile065_222_0430.txt
"""

import numpy as np
import sys,os
from astropy.io import ascii
from astropy.io import fits
from astropy.table import Table, Column
import csstpkg

# #424
# cssbands = ['Nuv', 'u', 'g', 'r', 'i', 'z', 'y']
# filtnumb = [4, 2, 2, 2, 2, 2, 4]
#222
cssbands = ['Nuv', 'u', 'g', 'r', 'i', 'z', 'WNuv', 'Wg', 'Wi']
filtnumb = [2, 2, 2, 2, 2, 2, 2, 2, 2]
# #4262
# cssbands = ['Nuv', 'u', 'g', 'r', 'i', 'z']

datafile = open(sys.argv[1],'r').readlines()

newdatafile = []

for aline in datafile:
    newdatafile.append(aline.replace('-99','99'))

dataout = open(sys.argv[1].split('.')[0]+'.tmp','w')
dataout.writelines(newdatafile)

cataorig = ascii.read(sys.argv[1].split('.')[0]+'.tmp')
print(len(cataorig))

catafilled = cataorig.filled('99')

os.remove(sys.argv[1].split('.')[0]+'.tmp')

lencatfill = len(catafilled)
catafilled['Context'] = 0


for cssband,numb in zip(cssbands,filtnumb):
    # print(cssband)

    catafilled['SimCnt_'+cssband] = csstpkg.mag2cr(catafilled['Simag_'+cssband], band=cssband) * 150 * numb
    #catafilled['CntPois_'+cssband] = np.random.poisson(catafilled['SimCnt_'+cssband])
    # catafilled['SimFnu_'+cssband] = 10**(-0.4*(catafilled['Simag_'+cssband]+48.6))
    catafilled['Fnu_'+cssband] = csstpkg.cr2fnu(catafilled['SimCnt_'+cssband]/150./numb,band=cssband)
    for i in range(lencatfill):
        if catafilled[i]['Fnu_'+cssband] < 1e-33:
            catafilled[i]['Simag_'+cssband] = -99
            catafilled[i]['Errmag_'+cssband] = -99
            catafilled[i]['Fnu_'+cssband] = 0
            catafilled[i]['Errflux_'+cssband] = -99
            continue
    catafilled['SNR_'+cssband] = catafilled['Fnu_'+cssband]/catafilled[i]['Errflux_'+cssband]
    
    
    
cataorig['SNRrms'] = np.sqrt(cataorig['SNR_g']**2+cataorig['SNR_r']**2+cataorig['SNR_i']**2+cataorig['SNR_z']**2)
snr10idx = np.where(cataorig['SNRrms']>=10)
print('SNRgriz>10:',len(cataorig[snr10idx]))
print('SNRi>10:',len(np.where(cataorig['SNR_i']>10)[0]))
print('SNRg>10:',len(np.where(cataorig['SNR_g']>10)[0]))
print('SNRwi>10:',len(np.where(cataorig['SNR_Wi']>10)[0]))
print('SNRwvwi>10:',len(np.where((cataorig['SNR_Wg']**2+cataorig['SNR_Wi']**2)**0.5>10)[0]))
