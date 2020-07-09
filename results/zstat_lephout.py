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


def snrfilter(astropytable, snrcri):

    catsnr = astropytable.copy()
    catsnr['SNR_g'] = 1.0857/catsnr['ERR_MAG_OBS_g']
    catsnr['SNR_r'] = 1.0857/catsnr['ERR_MAG_OBS_r']
    catsnr['SNR_i'] = 1.0857/catsnr['ERR_MAG_OBS_i']
    catsnr['SNR_z'] = 1.0857/catsnr['ERR_MAG_OBS_z']

    # del catsnr[:]
    if snrcri=='r10':
        idx = np.where((catsnr['SNR_r']>=snrthr)|(catsnr['SNR_i']>=snrthr))
        # for i,aline in enumerate(astropytable):
        #     if (aline['SNR_r']>=snrthr) or (aline['SNR_i']>=snrthr):
        #         catsnr.add_row(aline)
        #     else:
        #         pass
    elif snrcri=='ri10':
        idx = np.where((catsnr['SNR_r']**2+catsnr['SNR_i']**2)>=snrthr**2)
        # for i,aline in enumerate(astropytable):
        #     if (aline['SNR_r']**2+aline['SNR_i']**2)>=snrthr**2:
        #         catsnr.add_row(aline)
        #     else:
        #         pass
    elif snrcri=='griz10':
        idx = np.where((catsnr['SNR_g']**2+catsnr['SNR_r']**2+catsnr['SNR_i']**2+catsnr['SNR_z']**2)>=snrthr**2)
        # for i,aline in enumerate(astropytable):
        #     if (aline['SNR_g']**2+aline['SNR_r']**2+aline['SNR_i']**2+aline['SNR_z']**2)>=snrthr**2:
        #         catsnr.add_row(aline)
        #     else:
        #         pass
    else:
        return astropytable
    return astropytable[idx]


if __name__ == '__main__':

    # defaults = {'basedir': '/work/CSSOS/filter_improve/fromimg/windextract'}
    # config = configparser.ConfigParser(defaults)
    # config.read('cssos_config.ini')

    # Format the header
    senario = str(sys.argv[2])
    snrcri = str(sys.argv[3])

    if senario == '424':
        cssbands = ['NUV', 'u', 'g', 'r', 'i', 'z', 'y']
        filtnumb = [4, 2, 2, 2, 2, 2, 4]
    elif senario == '222':
        cssbands = ['NUV', 'u', 'g', 'r', 'i', 'z', 'WNUV', 'Wg', 'Wi']
        filtnumb = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    elif senario == '4262':
        cssbands = ['NUV', 'u', 'g', 'r', 'i', 'z']
        filtnumb = [4, 2, 2, 2, 6, 2]
    else:
        print('SenarioCode is Error.')

    header0 = '# IDENT  Z_BEST  Z_ML  CHI_BEST  MOD_BEST  EXTLAW_BEST  EBV_BEST  PDZ_BEST  SCALE_BEST  NBAND_USED  Z_SEC  CHI_SEC  MOD_SEC  Z_QSO  CHI_QSO  MOD_QSO  MOD_STAR  CHI_STAR '

    magobser = map(lambda magobs, aband: [magobs+aband], ['MAG_OBS_']*len(cssbands), cssbands)
    magobser = ' '.join(list(itertools.chain(*magobser)))+'  '
    # print(magobser)

    errmagobser = map(lambda errmagobs, aband: [errmagobs+aband], ['ERR_MAG_OBS_']*len(cssbands), cssbands)
    errmagobser = ' '.join(list(itertools.chain(*errmagobser)))+'  '

    magmod = map(lambda magmod, aband: [magmod+aband], ['MAG_MOD_']*len(cssbands), cssbands)
    magmod = ' '.join(list(itertools.chain(*magmod)))+'  '

    header = header0 + magobser + errmagobser + magmod + ' Context  ZSPEC\n'

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
    taborig = keyfilter(taborig0, 'Z_BEST', 0.0)

    tabsnr = snrfilter(taborig, snrcri)

    colnames = tabsnr.colnames
    nameskeep = [colnames[0],colnames[1],colnames[-1]]
    tabsnr.keep_columns(nameskeep)
    tabsnr.rename_column(tabsnr.colnames[0], 'ID')
    tabsnr.rename_column(tabsnr.colnames[1], 'zfit')
    tabsnr.rename_column(tabsnr.colnames[-1], 'zinput')

    # print(tabsnr)

    deltaz = np.abs(tabsnr['zfit']-tabsnr['zinput'])
    tab = tabsnr[deltaz/(1+tabsnr['zinput'])<=0.15]
    print(len(tab),'/',len(deltaz))
    fc = 1.-float(len(tab))/len(deltaz)
    sigma=signmad(tab)
    sigmaorig=signmad(tabsnr)

    print('sigma_NMAD =','{0:6.4f}'.format(sigma))
    print('sigma_NMAD All =','{0:6.4f}'.format(sigmaorig))
    print('Catastrophic fraction =','{0:5.2%}'.format(fc))


    plt.figure(figsize=(5,5))
    plt.plot([0,10],[0,10],'-', color='dodgerblue', linewidth=0.5)
    plt.plot(np.array([0,10]), 0.15+1.15*np.array([0,10]), ls='--',  color='dodgerblue', linewidth=0.5)
    plt.plot(np.array([0,10]), -0.15+0.85*np.array([0,10]), '--', color='dodgerblue', linewidth=0.5)
    plt.scatter(tabsnr['zinput'],tabsnr['zfit'],s=0.5,c='black')
    plt.scatter(tab['zinput'],tab['zfit'],s=0.5,c='red')
    ax=gca()
    ax.set_xlim(0,4)
    ax.set_ylim(0,4)
    ax.set_title('Band Senario '+str(sys.argv[2]))
    ax.set_xlabel('$z_{\\rm input}$')
    ax.set_ylabel('$z_{\\rm fit}$')
    ax.annotate('$\sigma_{\\rm NMAD} = $'+'{0:6.4f}'.format(sigma),
                xy=(0.95,0.2), xycoords='axes fraction', horizontalalignment='right')
    ax.annotate('$\sigma_{\\rm NMAD\ All} = $'+'{0:6.4f}'.format(sigmaorig),
                xy=(0.95,0.15), xycoords='axes fraction', horizontalalignment='right')
    ax.annotate('$f_c = $'+'{0:5.2%}'.format(fc),
                xy=(0.95,0.1), xycoords='axes fraction', horizontalalignment='right')
    datetag=dt.date.today().strftime('%m%d')
    plt.savefig(sys.argv[1].split('.')[0]+'.png', format='png', dpi=300)
    print('Figure '+sys.argv[1].split('.')[0]+'.png'+' saved.')
    plt.show()
