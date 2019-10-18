
# Usage: python3 SimuCataMerge.py toLephare_toMerge.lst Cssos_magsim_SNRd_toLephare.txt


import numpy as np
# import datetime
import time, sys, os

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

print(sys.argv[2])
