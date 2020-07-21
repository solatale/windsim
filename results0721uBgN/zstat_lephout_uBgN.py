"""
To format the header of lephare's .out file to standard header .txt file, and do statistics for photo-z.
Usage python3 zstat_lephout.py inoutfit.out 424 r10
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


snrthr = 10
Ntiles = 16
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
        print(astropytable[key].dtype)
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


def snrfilter(astropytable, snrcri):
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
    elif snrcri=='griz10':
        if senario=='424':
            idx = np.where((astropytable['SNR_g']**2+astropytable['SNR_r']**2+astropytable['SNR_i']**2+astropytable['SNR_z']**2)>=snrthr**2)
        elif senario=='424uBgN':
            idx = np.where((astropytable['SNR_g']**2+astropytable['SNR_r']**2+astropytable['SNR_i']**2+astropytable['SNR_z']**2)>=snrthr**2)
    else:
        return astropytable
    return astropytable[idx]


if __name__ == '__main__':

    senario = str(sys.argv[2])
    snrcri = sys.argv[3:]

    if senario == '424':
        cssbands = ['NUV', 'u', 'g', 'r', 'i', 'z', 'y']
        filtnumb = [4, 2, 2, 2, 2, 2, 4]
    elif senario == '424uBgN':
        cssbands = ['NUV', 'uB', 'gN', 'r', 'i', 'z', 'y']
        filtnumb = [4, 2, 2, 2, 2, 2, 4]
    elif senario == '222':
        cssbands = ['NUV', 'u', 'g', 'r', 'i', 'z', 'WNUV', 'Wg', 'Wi']
        filtnumb = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    elif senario == '222uBgN':
        cssbands = ['NUV', 'uB', 'gN', 'r', 'i', 'z', 'WNUV', 'Wg', 'Wi']
        filtnumb = [2, 2, 2, 2, 2, 2, 2, 2, 2]    
    elif senario == '4262':
        cssbands = ['NUV', 'u', 'g', 'r', 'i', 'z']
        filtnumb = [4, 2, 2, 2, 6, 2]
    else:
        print('SenarioCode is Error.')

    allbands =['NUV', 'NUV2', 'u', 'g', 'r', 'i', 'z', 'y', 'y2', 'WNUV', 'Wg', 'Wi', 'i4', 'uB', 'gN']
    snrlists = map(lambda snr, aband:[snr+aband], ['SNR_']*len(allbands), allbands)

    # header0 = '# IDENT  Z_BEST  Z_ML  CHI_BEST  MOD_BEST  EXTLAW_BEST  EBV_BEST  PDZ_BEST  SCALE_BEST  NBAND_USED  Z_SEC  CHI_SEC  MOD_SEC  Z_QSO  CHI_QSO  MOD_QSO  MOD_STAR  CHI_STAR '

    # magobser = map(lambda magobs, aband: [magobs+aband], ['MAG_OBS_']*len(allbands), allbands)
    # magobser = ' '.join(list(itertools.chain(*magobser)))+'  '
    # # print(magobser)

    # errmagobser = map(lambda errmagobs, aband: [errmagobs+aband], ['ERR_MAG_OBS_']*len(allbands), allbands)
    # errmagobser = ' '.join(list(itertools.chain(*errmagobser)))+'  '

    # magmod = map(lambda magmod, aband: [magmod+aband], ['MAG_MOD_']*len(allbands), allbands)
    # magmod = ' '.join(list(itertools.chain(*magmod)))+'  '

    #header = header0 + magobser + errmagobser + magmod + ' Context  ZSPEC  SNR_u SNR_g SNR_r SNR_i SNR_z Drms_sec\n'
    # header = header0 + magobser + errmagobser + magmod + ' Context  ZSPEC ' + ' '.join(list(itertools.chain(*snrlists))) + ' Drms_sec\n'

    header = '# IDENT  Z_BEST  ZSPEC  Z_ML  CONTEXT  ' + ' '.join(list(itertools.chain(*snrlists))) + ' Drms_sec\n'

    outfile = open(sys.argv[1],'r').readlines()
    del outfile[0:55]
    # print(outfile)

    outfilecont = list(header)+outfile

    datafilenm = sys.argv[1].split('.')[0]+'.txt'
    datafile = open(datafilenm,mode='w')
    for aline in outfilecont:
        datafile.write(aline)
    datafile.close()


    # Do statistics for photo-z.
    taborig0 = ascii.read(datafilenm)
    # taborig = inffilter(taborig0)
    taborig = keyfilter(taborig0, 'Z_BEST', 0.041, gtlt='gt')
    taborig = keyfilter(taborig, 'Drms_sec', dstar*1.6, gtlt='gt')

    for snr in snrcri:
        tabsnr = snrfilter(taborig, snr)

        print(len(tabsnr),'/',len(taborig),'meet SNR & Drms criterian.')

        # colnames = tabsnr.colnames

        # nameskeep = [colnames[0],colnames[1],colnames[-1]]
        # tabsnr.keep_columns(nameskeep)
        # tabsnr.rename_column(tabsnr.colnames[0], 'ID')
        # tabsnr.rename_column(tabsnr.colnames[1], 'zfit')
        # tabsnr.rename_column('ZSPEC', 'zinput')

        deltaz = np.abs(tabsnr['Z_BEST']-tabsnr['ZSPEC'])
        tab = tabsnr[deltaz/(1+tabsnr['ZSPEC'])<=0.15]
        fc = 1.-float(len(tab))/len(deltaz)
        sigma=signmad(tab)
        sigmaorig=signmad(tabsnr)

        print('\nSNR =',snr)
        print('{0:5.1f}'.format(len(tabsnr)/Ntiles/120.), 'gal/arcmin^2')
        print('sigma_NMAD =','{0:6.4f}'.format(sigma))
        print('sigma_NMAD All =','{0:6.4f}'.format(sigmaorig))
        print('Catastrophic fraction =','{0:5.2%}'.format(fc))
        print('')


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
            ax.set_title('Band Senario '+str(sys.argv[2])+'; SNR$_{g\,or\,i}\geq 20$')
        elif snr=='i10':
            ax.set_title('Band Senario '+str(sys.argv[2])+'; SNR$_{g\,or\,i}\geq 10$')
        elif snr=='gi10':
            ax.set_title('Band Senario '+str(sys.argv[2])+'; SNR$_{g+i}\geq 10$')
        elif snr=='griz10':
            ax.set_title('Band Senario '+str(sys.argv[2])+'; SNR$_{g+r+i+z}\geq 10$')
        ax.set_xlabel('$z_{\\rm input}$', fontSize='large')
        ax.set_ylabel('$z_{\\rm fit}$', fontSize='large')
        ax.annotate('{0:5.1f}'.format(len(tabsnr)/Ntiles/120.)+' gal/arcmin$^2$', xy=(0.95,0.25), xycoords='axes fraction', horizontalalignment='right')
        ax.annotate('$\sigma_{\\rm NMAD} = $'+'{0:6.4f}'.format(sigma),
                    xy=(0.95,0.2), xycoords='axes fraction', horizontalalignment='right')
        ax.annotate('$\sigma_{\\rm NMAD\ All} = $'+'{0:6.4f}'.format(sigmaorig),
                    xy=(0.95,0.15), xycoords='axes fraction', horizontalalignment='right')
        ax.annotate('$f_c = $'+'{0:5.2%}'.format(fc),
                    xy=(0.95,0.1), xycoords='axes fraction', horizontalalignment='right')
        datetag=dt.date.today().strftime('%m%d')
        plt.savefig(sys.argv[1].split('.')[0]+'_'+snr+'.png', format='png', dpi=300)
        print('Figure '+sys.argv[1].split('.')[0]+'_'+snr+'.png'+' saved.')
        plt.close()