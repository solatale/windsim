
# Usage: python3 SimuCataMerge.py toLephare_toMerge.lst Cssos_magsim_SNRd_toLephare.txt


import numpy as np
# import datetime
import time, sys, os
from astropy.io import ascii
from astropy.stats import SigmaClip
import matplotlib.pyplot as plt


def clip_outlier(tableobj, band):
    FnuBand = 10**(-0.4*(tableobj['MOD_'+band]+48.6))

    Fnu_resid = np.abs(tableobj['FluxSim_'+band] - FnuBand)

    # plt.figure()
    # plt.scatter(tableobj['MOD_'+band], FnuBand, s=2)
    # plt.scatter(tableobj['MOD_'+band], tableobj['FluxSim_'+band], s=2)
    # plt.scatter(tableobj['MOD_'+band], Fnu_resid, s=2)
    # plt.ylim(np.min(FnuBand), np.max(FnuBand))

    sigclip = SigmaClip(sigma=5, maxiters=5)
    resid_clipped = sigclip(Fnu_resid)
    table_clipped = tableobj[resid_clipped.mask==False]

    nclipped = len(tableobj)-len(table_clipped)
    print(str(nclipped) + '/' + str(len(tableobj)) + ' clipped')
    # plt.scatter(table_clipped['MOD_'+band], table_clipped['FluxSim_'+band], s=2, c='red')
    # plt.show()
    # time.sleep(10)

    return table_clipped



with open(sys.argv[1],'r') as listfile:
    listread = listfile.readlines()

files = [afile.strip() for afile in listread]

if os.path.exists(sys.argv[2]):
    os.remove(sys.argv[2])

mergecata = open(sys.argv[2],'w')


for i,afile in enumerate(files):
    if i == 0:
        # os.system("cp "+afile+" "+sys.argv[2])
        print(afile)
        with open(afile,'r') as readafile:
            content=readafile.read()
        if content is not '':
            mergecata.write(content)
    else:
        with open(afile,'r') as readafile:
            print(afile)
            readafile.readline()
            content = readafile.read()
        if content is not '':
            mergecata.write(content)

mergecata.close()

print('\n'+sys.argv[2])
