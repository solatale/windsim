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


def inffilter(astropytable):
    # length = len(astropytable)
    i = 0
    for i,line in enumerate(astropytable):
        if ((np.inf in astropytable[i]) or ('*********' in astropytable[i])):
            del astropytable[i]
        else:
            i = i+1
    return astropytable



taborig0 = ascii.read(sys.argv[1])

taborig = inffilter(taborig0)
# print(taborig[0])

taborig = keyfilter(taborig, 'col2', 0.0)
colnames = taborig.colnames
nameskeep = [colnames[0],colnames[1],colnames[-1]]
taborig.keep_columns(nameskeep)
taborig.rename_column(taborig.colnames[0], 'ID')
taborig.rename_column(taborig.colnames[1], 'zfit')
taborig.rename_column(taborig.colnames[-1], 'zinput')

# print(taborig)

deltaz = np.abs(taborig['zfit']-taborig['zinput'])
tab = taborig[deltaz/(1+taborig['zinput'])<=0.15]
print(len(tab),'/',len(deltaz))
fc = 1.-float(len(tab))/len(deltaz)
sigma=signmad(tab)
sigmaorig=signmad(taborig)


# print(taborig['col1','col2','col41'])
# taborig['snr_i'] = 1/(10**(0.4*taborig['col29'])-1)
# tabsnr = keyfilter(taborig, 'snr_i', snrthrsh)
# print(tabsnr['snr_i'])

# colnames = tabsnr.colnames
# ncols = len(colnames)
#
# nameskeep = [colnames[0],colnames[1],colnames[-2],colnames[-1]]
# tabsnr.keep_columns(nameskeep)
# #print(tabsnr)
#
# tabsnr.rename_column(tabsnr.colnames[0], 'ID')
# tabsnr.rename_column(tabsnr.colnames[1], 'zfit')
# tabsnr.rename_column(tabsnr.colnames[-2], 'zinput')
# print(tabsnr)
# ascii.write(tabsnr, 'zstatfile.txt', overwrite=True)
#
# # Clip catastrophic fittings:
# deltaz = np.abs(tabsnr['zfit']-tabsnr['zinput'])
# tab = tabsnr[deltaz/(1+tabsnr['zinput'])<=0.15]
# # print(tab)
# print(len(tab),'/',len(deltaz))
# fc = 1.-float(len(tab))/len(deltaz)
#
#
# sigma=signmad(tab)
# print('SNR_i >=', snrthrsh)

print('Catastrophic fraction =',fc)
print('sigma_NMAD =',sigma)
print('sigma_NMAD All =',sigmaorig)

# corcoef = np.corrcoef(tab['zfit'],tab['zinput'])
#
# print('correlation coefficient =', corcoef[0,1])

# slope, intercept, r, p, stderr = stats.linregress(tab['zinput'],tab['zfit'])
# resid = tab['zfit']-(tab['zinput']*slope+intercept)
# print(np.std(stats.sigmaclip(resid,4,4)[0]),r)


plt.figure(figsize=(5,5))
plt.plot([0,10],[0,10],'-', color='dodgerblue', linewidth=0.5)
plt.plot(np.array([0,10]), 0.15+1.15*np.array([0,10]), ls='--',  color='dodgerblue', linewidth=0.5)
plt.plot(np.array([0,10]), -0.15+0.85*np.array([0,10]), '--', color='dodgerblue', linewidth=0.5)
plt.scatter(taborig['zinput'],taborig['zfit'],s=0.5,c='black')
plt.scatter(tab['zinput'],tab['zfit'],s=0.5,c='red')
ax=gca()
ax.set_xlim(0,3)
ax.set_ylim(0,3)
ax.set_title('Bands Added Scheme')
ax.set_xlabel('$z_{\\rm input}$')
ax.set_ylabel('$z_{\\rm fit}$')
ax.annotate('$\sigma_{\\rm NMAD} = $'+'{0:7.4f}'.format(sigma),
            xy=(0.95,0.2), xycoords='axes fraction', horizontalalignment='right')
ax.annotate('$\sigma_{\\rm NMAD\ All} = $'+'{0:7.4f}'.format(sigmaorig),
            xy=(0.95,0.15), xycoords='axes fraction', horizontalalignment='right')
ax.annotate('$f_c = $'+'{0:7.4%}'.format(fc),
            xy=(0.95,0.1), xycoords='axes fraction', horizontalalignment='right')
datetag=dt.date.today().strftime('%m%d')
plt.savefig(sys.argv[1].split('.')[0]+'.png', format='png', dpi=300)
print('Figure '+sys.argv[1].split('.')[0]+'.png'+' saved.')
# plt.show()
