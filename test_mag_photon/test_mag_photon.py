
"""
Usage: python test_magphoton.py
"""

import configparser
import test_csstpkg as csstpkg
# from astropy.io import fits
import astropy.io.ascii as ascii
# import matplotlib.pyplot as plt
# from pylab import gca
# from mpl_toolkits.mplot3d import axes3d
# from astropy import wcs
import sys,math,time,os,io
import numpy as np
# import scipy.ndimage as spimg
# from scipy.stats import poisson
# from astropy.table import Table
# from astropy.modeling import models, fitting
# import numpy.lib.recfunctions as rft
from progressive.bar import Bar
import datetime as dt
# import sep
# from matplotlib.patches import Ellipse
import itertools
import multiprocessing as mp
import gc

gc.enable()




def simul_css(CataSect, cssbands, filtnumb, npi):

    # print('Process'+str(npi)

    OutSecStr = ''

    LenCatSec = len(CataSect)
    procedi = 0

    if IfProgBarOn == True:
        bar = Bar(max_value=LenCatSec, empty_color=7, filled_color=18+npi*6, title='Process-'+str(npi))
        bar.cursor.clear_lines(1)
        bar.cursor.save()

    for procedi,cataline in enumerate(CataSect, 1):

        np.random.seed()

        ident = str(cataline['IDENT'])

        outcatrowi = [ident, cataline['Z_BEST']]

        if DebugTF == True:
            print(' '.join([ident, '\nRA DEC:', str(cataline['RA']), str(cataline['DEC'])]))

        sedname = seddir + 'Id' + '{:0>9}'.format(ident) + '.spec'
        modsed = csstpkg.readsed(sedname)
        modsed[:, 1] = csstpkg.mag2flam(modsed[:, 1], modsed[:, 0])  # to convert model SED from magnitude to f_lambda(/A)


        for bandi, cssband in enumerate(cssbands):

            mag_en = csstpkg.mag_ener(modsed, cssband, magab_ener_zeros[bandi])
            mag_ph = csstpkg.mag_phot(modsed, cssband, magab_phot_zeros[bandi])
            if DebugTF == True:
                # print(' '.join([cssband, 'band model electrons = ', str(NeABand), 'e-']))
                print('MOD_' + cssband + '_css =', cataline['MOD_' + cssband + '_css'])
                print('Magsim_' + cssband + '_photon =', mag_ph)
                print('Magsim_' + cssband + '_energy =', mag_en)

            outcatrowi = outcatrowi + [cataline['MOD_'+cssband+'_css'],mag_en,mag_ph]
            # bandi = bandi + 1

        # del WinImgBands
        colnumb = len(outcatrowi)

        OutRowStr = ('{} '+(colnumb-1)*'{:8.3f}').format(*outcatrowi)+'\n'
        OutSecStr = OutSecStr + OutRowStr

        if IfProgBarOn == True:
            bar.cursor.restore()  # Return cursor to start
            bar.draw(value=procedi)

    if IfProgBarOn == True:
        bar.cursor.restore()  # Return cursor to start
        bar.draw(value=bar.max_value)  # Draw the bar!


    with write_lock:
        OutCssCat.write(OutSecStr)
        OutCssCat.flush()
    # write_lock.release()

    print('\n')



def kill_zombies():

    while any(mp.active_children()):
        time.sleep(2)
        print(mp.active_children())
        for p in mp.active_children():
            p.terminate()




if __name__ == '__main__':

    defaults = {'basedir': '/work/CSSOS/filter_improve/fromimg/windextract'}
    config = configparser.ConfigParser(defaults)
    config.read('test_mag_config.ini')

    NProcesses = config.getint('Hst2Css','NProcesses')

    DebugTF = config.getboolean('Hst2Css','DebugTF')
    thrghdir = config['Hst2Css']['thrghdir']
    seddir = config['Hst2Css']['seddir']

    IfProgBarOn = config.getboolean('Hst2Css','IfProgBarOn')
    IfPlotImgArr = config.getboolean('Hst2Css','IfPlotImgArr')  # whether to plot image iteractively or not
    IfPlotObjWin = config.getboolean('Hst2Css','IfPlotObjWin')  # whether to plot object image or not

    begintime = time.time()

    datestr = dt.date.today().strftime("%Y%m%d")

    ExpCssFrm = config.getfloat('Hst2Css','ExpCssFrm')
    ExpHst = config.getfloat('Hst2Css','ExpHst')
    TelArea = math.pi*100**2
    Gain = config.getfloat('Hst2Css', 'Gain')

    CatOfTile = ascii.read(config['Hst2Css']['CssCatIn'])

    for scheme_i in range(1):

        if scheme_i == 0:
            cssbands = ['Nuv', 'u', 'g', 'r', 'i', 'z', 'y']
            filtnumb = [4, 2, 2, 2, 2, 2, 4]
            schemecode = '424'
        elif scheme_i == 1:
            cssbands = ['Nuv', 'u', 'g', 'r', 'i', 'z', 'WNuv', 'Wg', 'Wi']
            filtnumb = [2, 2, 2, 2, 2, 2, 2, 2, 2]
            # cssbands = ['g','r','i','z','WNuv', 'Wg', 'Wi']
            # filtnumb = [2,2,2,2,2,2,2]
            schemecode = '222'
        elif scheme_i == 2:
            cssbands = ['Nuv', 'u', 'g', 'r', 'i', 'z']
            filtnumb = [4, 2, 2, 2, 6, 2]
            schemecode = '4262'

        print('Scheme '+schemecode)

        magab_ener_zeros = []
        magab_phot_zeros = []
        for cssband in cssbands:
        #     expcss = 150. * numb  # s
            magab_ener_zeros.append(csstpkg.MagAB_Ener_Zero(cssband))
            magab_phot_zeros.append(csstpkg.MagAB_Phot_Zero(cssband))

        namelists = map(lambda modmag, mag_enr, mag_pht, aband: \
                            [modmag+aband, mag_enr+aband, mag_pht+aband], ['MOD_']*len(cssbands), \
                        ['Mag_enr_'] * len(cssbands), ['Mag_pht_'] * len(cssbands), cssbands)
        colnames = ['ID','Z_BEST']+list(itertools.chain(*namelists))

        LenCatTile = len(CatOfTile)

        # Output catalog for one tile
        OutCssCatName = 'test_mag_'+config['Hst2Css']['hst814file'][-12:-9]+'_'+schemecode+'.txt'
        if os.path.isfile(OutCssCatName) is True:
            os.remove(OutCssCatName)
        OutCssCat = open(OutCssCatName, mode='w')
        OutCssCat.write('# '+' '.join(colnames)+'\n')
        OutCssCat.flush()

        write_lock = mp.Lock()

        Nbat = int(LenCatTile / NProcesses)
        Nleft = LenCatTile % NProcesses

        OutCssCatQueue = mp.Queue(20000)
        FinishQueue = mp.Queue(NProcesses*2)
        finishstat = []

        if NProcesses == 1:
            simul_css(CatOfTile, cssbands, filtnumb, 0)

        elif Nbat > 0:
            jobs=[]
            for npi in range(NProcesses):
                i_low, i_high = npi*Nbat, (npi+1)*Nbat
                jobs.append(mp.Process(target=simul_css, name='Process'+str(npi), args=(CatOfTile[i_low:i_high], cssbands, filtnumb, npi)))

            for sti in range(NProcesses):
                jobs[sti].start()
            for jni in range(NProcesses):
                jobs[jni].join()

            if Nleft > 0:
                print('Processing the rest')
                simul_css(CatOfTile[int(Nbat * NProcesses):], cssbands, filtnumb, 0)

        else:
            if Nleft > 0:
                simul_css(CatOfTile, cssbands, filtnumb, 0)
            else:
                print('The catalog is empty. Please check it.')

        OutCssCat.close()

    finishtime = time.time()
    print('Time Consumption:', finishtime - begintime, 's')
    print('\nFinished.\n')
