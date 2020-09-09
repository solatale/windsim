"""
To format the header of lephare's .out file to standard header .txt file, and do statistics for photo-z.
For NLe processes output merging.
Usage python3 zstat_lephout_uBgNWiBy_np.py 3 Cssos_FluxSim_SNR_tilemrg_065_allin1_uBgNWiBy_20200801_424_flux_OutFit 424 i20 i10 gi10 griz10 
"""


import sys
import configparser
import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import sys, math, time
from astropy.io import ascii
from pylab import gca
import datetime as dt
from astropy import table


snrthr = 10
Ntiles = 16
fovtile = 105  # arcmin^2
dstar = 0.1677/1.7941


def signmad(tab):
    deltz = tab['Z_BEST']-tab['ZSPEC']
    # print np.median(deltz)
    # print (deltz-np.median(deltz))/(1+tab['ZSPEC'])
    sigmanmad = 1.48*np.median(np.abs((deltz-np.median(deltz))/(1+tab['ZSPEC'])))
    # sigmanmad = 1.48 * np.median(np.abs(deltz / (1 + tab['ZSPEC'])))
    return sigmanmad


def keyfilter(astropytable, key, keythr, gtlt='gt'):
    if gtlt=='gt':
        idx = np.where(astropytable[key] >= keythr)[0]
    elif gtlt=='lt':
        # print(astropytable[key].dtype)
        idx = np.where(np.float32(astropytable[key]) <= keythr)[0]
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


def snrfilter(astropytable, snrcri, band='i'):
    # astropytable should contain columns named SNR_g,SNR_r,SNR_i,SNR_z
    # catsnr = astropytable.copy()
    # catsnr['SNR_g'] = 1.0857/catsnr['ERR_MAG_OBS_g']
    # catsnr['SNR_r'] = 1.0857/catsnr['ERR_MAG_OBS_r']
    # catsnr['SNR_i'] = 1.0857/catsnr['ERR_MAG_OBS_i']
    # catsnr['SNR_z'] = 1.0857/catsnr['ERR_MAG_OBS_z']

    # del catsnr[:]
    if snrcri=='i20':
        idx = np.where((astropytable['SNR_g']>=(snrthr*2))|(astropytable['SNR_i']>=(snrthr*2)))
    elif snrcri=='i10':
        idx = np.where((astropytable['SNR_g']>=snrthr)|(astropytable['SNR_i']>=snrthr))
    elif snrcri=='gi10':
        idx = np.where((astropytable['SNR_g']**2+astropytable['SNR_i']**2)>=snrthr**2)
    elif snrcri=='WI10':
        idx = np.where((astropytable['SNR_WV']>=snrthr)|(astropytable['SNR_WI']>=snrthr))    
    elif snrcri=='WVI10':
        idx = np.where((astropytable['SNR_WV']**2+astropytable['SNR_WI']**2)>=snrthr**2)
    elif snrcri=='griz10':
        # if senario[0:3]=='424':
        idx = np.where((astropytable['SNR_g']**2+astropytable['SNR_r']**2+astropytable['SNR_i']**2+astropytable['SNR_z']**2)>=snrthr**2)
        # elif senario=='424uBgN':
        #     idx = np.where((astropytable['SNR_g']**2+astropytable['SNR_r']**2+astropytable['SNR_i']**2+astropytable['SNR_z']**2)>=snrthr**2)
    else:
        idx = np.where(astropytable['SNR_'+band]>=snrthr)
        return astropytable[idx]
    return astropytable[idx]


def GalDens(astpytable):
    densities = {}
    for cssband in allbands:
        bandtable = snrfilter(astpytable, cssband, band=cssband)
        densities[cssband] = len(bandtable)/Ntiles/fovtile
        # print(cssband, '{0:6.2f}'.format(len(bandtable)/Ntiles/fovtile), 'gal/arcmin^2')
    return densities



if __name__ == '__main__':

    NLe = int(sys.argv[1])
    outfilerootnm = sys.argv[2]
    senario = str(sys.argv[3])
    snrcri = sys.argv[4:]

    allbands =['NUV', 'NUV2', 'u', 'g', 'r', 'i', 'z', 'z2', 'y', 'y2', 'WU', 'WU2', 'WV', 'WI', 'i4', 'uB', 'gN', 'WIBy', 'zN', 'WVB', 'WIN', 'WINy', 'WUv', 'uB410', 'gN410']
    snrlists = map(lambda snr, aband:[snr+aband], ['SNR_']*len(allbands), allbands)

    header = '# IDENT  Z_BEST  ZSPEC  Z_ML  CONTEXT  ' + ' '.join(list(itertools.chain(*snrlists))) + ' Drms_sec\n'

    datafilenm = outfilerootnm+'.txt'
    datafile = open(datafilenm, mode='w')
    datafile.write(header)
    for i in range(NLe):
        outfilecont = open(outfilerootnm+'_'+str(i+1)+'.out','r').readlines()
        del outfilecont[0:55]
        datafile.writelines(outfilecont)
    datafile.close()


    # Do statistics for photo-z.
    taborig = ascii.read(datafilenm)
    print('\n', len(taborig), 'outputs.')
    taborig = keyfilter(taborig, 'Drms_sec', dstar*1.6, gtlt='gt')
    densities = GalDens(taborig)
    print('Galaxy Densities (SNR>10, dgal>1.6dstar):\n', densities)

    taborig = table.unique(taborig, keys='IDENT', silent=True)
    lenorig = len(taborig)
    print(lenorig,'unique outputs.\n\n')
    # taborig = inffilter(taborig0)
    taborig = keyfilter(taborig, 'Z_BEST', 0.1, gtlt='gt')
    taborig = keyfilter(taborig, 'ZSPEC', 0.1, gtlt='gt')    # May have effect of different galaxy density

    tabzlt3 = keyfilter(taborig, 'Z_BEST', 3, gtlt='lt')
    tabzlt3 = keyfilter(tabzlt3, 'ZSPEC', 3, gtlt='lt')

    print('Senario '+senario+'\n\n')
    for snr in snrcri:
        tabsnr = snrfilter(taborig, snr)


        deltaz = np.abs(tabsnr['Z_BEST']-tabsnr['ZSPEC'])
        coretab = tabsnr[deltaz/(1+tabsnr['ZSPEC'])<=0.15]
        fc = 1.-float(len(coretab))/len(deltaz)
        sigma=signmad(coretab)
        sigmaorig=signmad(tabsnr)

        print('SNR =',snr)
        print(len(tabsnr),'/',lenorig,'meet SNR & Drms criterian.')

        print('{0:6.2f}'.format(len(tabsnr)/Ntiles/fovtile), "gal/□′")
        print('sigma_NMAD =','{0:6.4f}'.format(sigma))
        print('sigma_NMAD All =','{0:6.4f}'.format(sigmaorig))
        print('Catastrophic fraction =','{0:5.2%}'.format(fc))
        # print('')


        tabsnrpz3 = snrfilter(tabzlt3, snr)


        deltazpz3 = np.abs(tabsnrpz3['Z_BEST']-tabsnrpz3['ZSPEC'])
        coretabpz3 = tabsnrpz3[deltazpz3/(1+tabsnrpz3['ZSPEC'])<=0.15]
        fcpz3 = 1-float(len(coretabpz3))/len(deltazpz3)
        sigmapz3 = signmad(coretabpz3)
        sigmaorigpz3 = signmad(tabsnrpz3)

        # print(len(tabsnrpz3),'/',lenorig,'meet SNR & Drms criterian.')
        # print('{0:6.2f}'.format(len(tabsnrpz3)/Ntiles/fovtile), "gal/□′")
        print('sigma_NMAD_z<3 =','{0:6.4f}'.format(sigmapz3))
        print('sigma_NMAD_All_z<3 =','{0:6.4f}'.format(sigmaorigpz3))
        print('Catastrophic fraction z<3 =','{0:5.2%}'.format(fcpz3))
        # print('')


        plt.figure(figsize=(6,6))
        plt.plot([0,10],[0,10],'-', color='dodgerblue', linewidth=0.5)
        plt.plot(np.array([0,10]), 0.15+1.15*np.array([0,10]), ls='--',  color='dodgerblue', linewidth=0.5)
        plt.plot(np.array([0,10]), -0.15+0.85*np.array([0,10]), '--', color='dodgerblue', linewidth=0.5)
        plt.scatter(tabsnr['ZSPEC'],tabsnr['Z_BEST'], s=2, c='black', alpha=0.2, edgecolors='none')
        #plt.scatter(tab['ZSPEC'],tab['Z_BEST'],s=0.2,c='red')
        ax=gca()
        ax.set_xlim(0,6)
        ax.set_ylim(0,6)
        if snr=='i20':
            ax.set_title('Band Senario '+str(sys.argv[3])+'; SNR$_{g\,\mathrm{or}\,i}\geq 20$')
        elif snr=='i10':
            ax.set_title('Band Senario '+str(sys.argv[3])+'; SNR$_{g\,\mathrm{or}\,i}\geq 10$')
        elif snr=='WI10':
            ax.set_title('Band Senario '+str(sys.argv[3])+'; SNR$_{WV\,\mathrm{or}\,WI}\geq 10$')
        elif snr=='WVI10':
            ax.set_title('Band Senario '+str(sys.argv[3])+'; SNR$_{WU+WV+WI}\geq 10$')    
        elif snr=='gi10':
            ax.set_title('Band Senario '+str(sys.argv[3])+'; SNR$_{g+i}\geq 10$')
        elif snr=='griz10':
            ax.set_title('Band Senario '+str(sys.argv[3])+'; SNR$_{g+r+i+z}\geq 10$')
        ax.set_xlabel('$z_{\\rm input}$', fontSize='large')
        ax.set_ylabel('$z_{\\rm fit}$', fontSize='large')
        ax.annotate('{0:5.1f}'.format(len(tabsnr)/Ntiles/fovtile)+' gal/□′', xy=(0.95,0.35), xycoords='axes fraction', horizontalalignment='right')
        ax.annotate('$\sigma_{\\rm NMAD} = $'+'{0:6.4f}'.format(sigma),
                    xy=(0.95,0.3), xycoords='axes fraction', horizontalalignment='right')
        ax.annotate('$\sigma_{\\rm NMAD\ All} = $'+'{0:6.4f}'.format(sigmaorig),
                    xy=(0.95,0.25), xycoords='axes fraction', horizontalalignment='right')
        ax.annotate('$f_c = $'+'{0:5.2%}'.format(fc),
                    xy=(0.95,0.2), xycoords='axes fraction', horizontalalignment='right')
        ax.annotate('$\sigma_{\\rm NMAD\,(z<3)} = $'+'{0:6.4f}'.format(sigmapz3),
                    xy=(0.95,0.15), xycoords='axes fraction', horizontalalignment='right')
        ax.annotate('$\sigma_{\\rm NMAD\,All\,(z<3)} = $'+'{0:6.4f}'.format(sigmaorigpz3),
                    xy=(0.95,0.1), xycoords='axes fraction', horizontalalignment='right')
        ax.annotate('$f_{\\rm c\,(z<3)} = $'+'{0:5.2%}'.format(fcpz3),
                    xy=(0.95,0.05), xycoords='axes fraction', horizontalalignment='right')
        datetag=dt.date.today().strftime('%m%d')
        plt.savefig(sys.argv[2].split('.')[0]+'_'+snr+'.png', format='png', dpi=100)
        print('Figure '+sys.argv[2].split('.')[0]+'_'+snr+'.png'+' saved.\n')
        plt.close()
