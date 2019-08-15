
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

#424
cssbands = ['Nuv', 'u', 'g', 'r', 'i', 'z', 'y']
filtnumb = [4, 2, 2, 2, 2, 2, 4]
#222
#cssbands = ['Nuv', 'u', 'g', 'r', 'i', 'z', 'WNuv', 'Wg', 'Wi']
#filtnumb = [2, 2, 2, 2, 2, 2, 2, 2, 2]
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
cataorig['SNRrms'] = np.sqrt(cataorig['SNR_g']**2+cataorig['SNR_r']**2+cataorig['SNR_i']**2+cataorig['SNR_z']**2)
snr10idx = np.where(cataorig['SNRrms']>=10)
print(len(cataorig[snr10idx]))
catafilled = cataorig[snr10idx].filled('99')

os.remove(sys.argv[1].split('.')[0]+'.tmp')

lencatfill = len(catafilled)
catafilled['Context'] = 0

k=0
for cssband,numb in zip(cssbands,filtnumb):
    # print(cssband)
    k+=1
    catafilled['SimCnt_'+cssband] = csstpkg.mag2cr(catafilled['Simag_'+cssband], band=cssband) * 150 * numb
    catafilled['CntPois_'+cssband] = np.random.poisson(catafilled['SimCnt_'+cssband])
    # catafilled['SimFnu_'+cssband] = 10**(-0.4*(catafilled['Simag_'+cssband]+48.6))
    catafilled['Fnu_'+cssband] = csstpkg.cr2fnu(catafilled['CntPois_'+cssband]/150./numb,band=cssband)
    for i in range(lencatfill):
        if catafilled[i]['Fnu_'+cssband] < 1e-33:
            catafilled[i]['Simag_'+cssband] = -99
            catafilled[i]['Errmag_'+cssband] = -99
            catafilled[i]['Fnu_'+cssband] = -99
            catafilled[i]['Errflux_'+cssband] = -99
            continue

        catafilled[i]['Context'] = catafilled[i]['Context']+2**(k-1)
    
        
        
        
catafilled.rename_column('Z_BEST','Z-SPEC')
catafilled['ID'].format = '.0f'

#ascii.write(catafilled, sys.argv[1].split('.')[0]+'_forcheck.txt', format='commented_header',comment='#',overwrite=True)

ColNameList = ['ID']+['Fnu_'+aband for aband in cssbands]+['Errflux_'+aband for aband in cssbands]+['Context','Z-SPEC']

Cat2Wrt = Table(catafilled[ColNameList], names=ColNameList)

outname = sys.argv[1].split('.')[0]+'_toLeph.dat'
ascii.write(Cat2Wrt, outname, format='commented_header',comment='#',overwrite=True)

