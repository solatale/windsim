"""
Usage python zstat_.py inoutfit_.txt
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import sys, math, time
from astropy.io import ascii
from pylab import gca
import datetime as dt


def signmad(tab):
    deltz = tab['zfit']-tab['zinput']
    # print np.median(deltz)
    # print (deltz-np.median(deltz))/(1+tab['zinput'])
    sigmanmad = 1.48*np.median(np.abs((deltz-np.median(deltz))/(1+tab['zinput'])))
    # sigmanmad = 1.48 * np.median(np.abs(deltz / (1 + tab['zinput'])))
    return sigmanmad


def keyfilter(astropytable, key, keythr):
    idx = np.where(np.array(astropytable[key])>=keythr)
    filtered = astropytable[idx]
    return filtered



snrthrsh = 10
taborig = ascii.read(sys.argv[1])

taborig = keyfilter(taborig, 'col2', 0.0)
print(taborig)
taborig['snr_i'] = 1/(10**(0.4*taborig['col29'])-1)
tabsnr = keyfilter(taborig, 'snr_i', snrthrsh)
# print(tabsnr['snr_i'])

colnames = tabsnr.colnames
ncols = len(colnames)

nameskeep = [colnames[0],colnames[1],colnames[-2],colnames[-1]]
tabsnr.keep_columns(nameskeep)
#print(tabsnr)

tabsnr.rename_column(tabsnr.colnames[0], 'ID')
tabsnr.rename_column(tabsnr.colnames[1], 'zfit')
tabsnr.rename_column(tabsnr.colnames[-2], 'zinput')
#print(tabsnr)
ascii.write(tabsnr, 'zstatfile.txt', overwrite=True)

# Clip catastrophic fittings:
deltaz = np.abs(tabsnr['zfit']-tabsnr['zinput'])
tab = tabsnr[deltaz/(1+tabsnr['zinput'])<=0.15]
print(len(tab),'/',len(deltaz))
fc = 1.-float(len(tab))/len(deltaz)


sigma=signmad(tab)

print('SNR >=', snrthrsh)
print('Catastrophic fraction =',fc)
print('sigma_NMAD =',sigma)

corcoef = np.corrcoef(tab['zfit'],tab['zinput'])

print('correlation coefficient =', corcoef[0,1])

# slope, intercept, r, p, stderr = stats.linregress(tab['zinput'],tab['zfit'])
# resid = tab['zfit']-(tab['zinput']*slope+intercept)
# print(np.std(stats.sigmaclip(resid,4,4)[0]),r)

x=np.arange(6)
y=x#*slope+intercept

plt.figure()
plt.scatter(tabsnr['zinput'],tabsnr['zfit'],s=2,c='black')
plt.scatter(tab['zinput'],tab['zfit'],s=2,c='red')
plt.plot(x,y,'--')
ax=gca()
ax.set_xlim(0,5)
ax.set_ylim(0,5)
ax.set_title('Bands Added Scheme')
ax.set_xlabel('$z_{\\rm input}$')
ax.set_ylabel('$z_{\\rm fit}$')
ax.annotate('$\sigma_{\\rm NMAD} = $'+'{0:7.4f}'.format(sigma),
            xy=(0.9,0.1), xycoords='axes fraction', horizontalalignment='right')
ax.annotate('$f_c = $'+'{0:7.4%}'.format(fc),
            xy=(0.9,0.05), xycoords='axes fraction', horizontalalignment='right')
datetag=dt.date.today().strftime('%m%d')
plt.savefig(sys.argv[1].split('.')[0]+'.png', format='png', dpi=300)
