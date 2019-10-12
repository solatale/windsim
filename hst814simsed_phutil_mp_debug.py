
"""
Usage: python hst814simsed_phutil_mp.py 065
"""

import configparser
import csstpkg_phutil_mp as csstpkg
from astropy.io import fits
import astropy.io.ascii as ascii
import matplotlib.pyplot as plt
# from pylab import gca
# from mpl_toolkits.mplot3d import axes3d
from astropy import wcs
import sys,math,time,os,io
import numpy as np
# import scipy.ndimage as spimg
# from scipy.stats import poisson
from astropy.table import Table
# from astropy.modeling import models, fitting
# import numpy.lib.recfunctions as rft
from progressive.bar import Bar
import datetime as dt
import sep
from matplotlib.patches import Ellipse
import itertools
import multiprocessing as mp
import gc

gc.enable()




def simul_css(CataSect, _CssImg, cssbands, filtnumb, npi):

    # print('Process'+str(npi)

    CssHei, CssWid = _CssImg.shape

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

        objwind = csstpkg.windcut(_CssImg, cataline)
        if objwind is None:
            continue
        # DataArr2Fits(objwind, ident+'_convwin.fits')

        objwinshape = objwind.shape

        WinImgBands = np.zeros((len(cssbands), objwinshape[0], objwinshape[1]))  # 3-D array contains images of all the cssbands

        if IfPlotObjWin == True:
            csstpkg.PlotObjWin(objwind, cataline)

        outcatrowi = [ident, cataline['Z_BEST']]

        if DebugTF == True:
            print(' '.join([ident, '\nRA DEC:', str(cataline['RA']), str(cataline['DEC'])]))

        # Photometry for the central object on the convolved window
        ObjWinPhot = csstpkg.CentrlPhot(objwind.data, id=str(outcatrowi[0]) + " ConvWdW")
        ObjWinPhot.Bkg(idb=str(outcatrowi[0]) + " ConvWdW", debug=DebugTF, thresh=2, minarea=10, deblend_nthresh=32, deblend_cont=0.01)
        ObjWinPhot.Centract(idt=str(outcatrowi[0]) + " ConvWdW", thresh=2.5, deblend_nthresh=32, deblend_cont=0.1, minarea=10, debug=DebugTF)
        if ObjWinPhot.centobj is np.nan:
            continue
        else:
            ObjWinPhot.KronR(idk=str(outcatrowi[0]) + " ConvWdW", debug=DebugTF, mask_bool=True)

        NeConv, ErrNeConv = ObjWinPhot.EllPhot(ObjWinPhot.kronr, mask_bool=True)


        if DebugTF == True:
            print('self.bkg Flux & ErrFlux =', ObjWinPhot.bkg.background_median, ObjWinPhot.bkg.background_rms_median)
            print('Class processed Neconv & ErrNeConv:', NeConv, ErrNeConv)

        if ((NeConv <= 0) or (NeConv is np.nan)):
            if DebugTF == True:
                print('NeConv for a winimg <= 0 or NeConv is np.nan')
            continue


        # Read model SED to NDArray
        # modsednum = cataline['MOD_BEST']
        sedname = seddir + 'Id' + '{:0>9}'.format(ident) + '.spec'
        modsed = csstpkg.readsed(sedname)
        modsed[:, 1] = csstpkg.mag2flam(modsed[:, 1], modsed[:, 0])  # to convert model SED from magnitude to f_lambda(/A)

        bandi = 0
        for cssband, numb in zip(cssbands, filtnumb):

            expcss = 150. * numb  # s
            # cssbandpath = thrghdir+cssband+'.txt'

            NeABand = csstpkg.NeObser(modsed, cssband, expcss, TelArea)  # *cataline['SCALE_BEST']
            if DebugTF == True:
                print(' '.join([cssband, 'band model electrons = ', str(NeABand), 'e-']))
                print('MOD_' + cssband + '_css =', cataline['MOD_' + cssband + '_css'])
                print('Magsim_' + cssband + ' =', csstpkg.Ne2MagAB(NeABand, cssband, expcss, TelArea))

            Scl2Sed = NeABand / NeConv
            if DebugTF == True:
                print(ident, Scl2Sed)

            ZeroLevel = config.getfloat('Hst2Css', 'BZero')
            SkyLevel = csstpkg.backsky[cssband] * expcss
            DarkLevel = config.getfloat('Hst2Css', 'BDark') * expcss
            IdealImg = objwind.data * Scl2Sed + SkyLevel + DarkLevel  # e-
            IdealImg[IdealImg < 0] = 0
            csstpkg.DataArr2Fits((IdealImg+ZeroLevel)/Gain, 'Ideal_Zero_Gain_check_'+ident+'_'+cssband+'.fits')
            # if DebugTF == True:
            #     print(cssband, ' band Sum of IdealImg =', np.sum(IdealImg))
            # ImgPoiss = np.random.poisson(lam=IdealImg, size=objwinshape)
            #
            ImgPoiss = IdealImg
            NoisNorm = csstpkg.NoiseArr(objwinshape, loc=0, scale=config.getfloat('Hst2Css', 'RNCss') * (numb) ** 0.5, func='normal')
            DigitizeImg = (ImgPoiss + NoisNorm + ZeroLevel) / Gain
            # DigitizeImg = (IdealImg + ZeroLevel)/Gain
            csstpkg.DataArr2Fits(DigitizeImg, 'Ideal_Zero_Gain_RN_check_'+ident+'_'+cssband+'.fits')

            WinImgBands[bandi, ::] = DigitizeImg

            bandi = bandi + 1

        if DebugTF == True:
            print('Stack all bands and detect objects:')

        WinImgStack = WinImgBands.sum(0)
        # print(WinImgStack.shape)
        # AduStack, ErrAduStack, ObjectStack, KronRStack, MaskStack = septract(WinImgStack, id=str(outcatrowi[0])+" Stack", debug=DebugTF, thresh=1.2, minarea=10)
        StackPhot = csstpkg.CentrlPhot(WinImgStack, id=str(outcatrowi[0]) + " Stack")
        StackPhot.Bkg(idb=str(outcatrowi[0]) + " Stack", debug=DebugTF, thresh=1.5, minarea=10)
        StackPhot.Centract(idt=str(outcatrowi[0]) + " Stack", thresh=1.5, minarea=10, deblend_nthresh=32, deblend_cont=0.1, debug=DebugTF)
        if StackPhot.centobj is np.nan:
            if DebugTF == True:
                print('No central object on STACK image.')
            continue
        else:
            StackPhot.KronR(idk=str(outcatrowi[0]) + " Stack", debug=DebugTF, mask_bool=True)
        AduStack, ErrAduStack = StackPhot.EllPhot(StackPhot.kronr, mask_bool=True)
        if AduStack is np.nan:
            if DebugTF == True:
                print('RSS error for STACK image.')
            continue

        if DebugTF == True:
            csstpkg.PlotKronrs(WinImgStack, StackPhot)

        bandi = 0
        for cssband, numb in zip(cssbands, filtnumb):
            expcss = 150. * numb  # s
            # if DebugTF == True:
            #     print(cssband, ' band Array Slice Sum =', np.sum(WinImgBands[bandi, ::]), 'e-')
            #     print(cssband, ' band Array Slice MagAB =', csstpkg.Ne2MagAB(np.sum(WinImgBands[bandi, ::]), cssband, expcss, TelArea))
            AduObser, ErrAduObs = csstpkg.septractSameAp(WinImgBands[bandi, ::], StackPhot.centobj, StackPhot.kronr, mask_det=StackPhot.mask_other, debug=DebugTF, annot=cssband+'_cssos', thresh=1.2, minarea=10)

            if DebugTF == True:
                print(''.join([cssband, ' band simu ADU=', str(AduObser), ' ErrNe=', str(ErrAduObs)]))

            if AduObser > 0:
                SNR = AduObser / ErrAduObs
                # MagObser = Ne2MagAB(AduObser*Gain,cssband,expcss,TelArea)
                MagObser = -2.5 * math.log10(AduObser) + magab_zeros[bandi]
                ErrMagObs = 2.5 * math.log10(1 + 1 / SNR)
                if DebugTF == True:
                    if ((cssband == 'r') & (np.abs(MagObser - cataline['MOD_' + cssband + '_css']) > 1)):
                        csstpkg.DataArr2Fits(objwind.data, ident + '_convwin_r.fits')
                        csstpkg.DataArr2Fits(WinImgStack, ident + '_stack.fits')
            else:
                SNR = -99
                MagObser = -99
                ErrMagObs = -99
            if DebugTF == True:
                print(' '.join([cssband, 'band mag_model = ', str(cataline['MOD_' + cssband + '_css']), '(AB mag)']))
                print(' '.join([cssband, 'band mag_simul = ', str(MagObser), '(AB mag)']))
                print(' '.join([cssband, 'band magerr_simul = ', str(ErrMagObs), '(AB mag)']))

            outcatrowi = outcatrowi + [cataline['MOD_' + cssband + '_css'], MagObser, ErrMagObs, SNR]
            bandi = bandi + 1

        del WinImgBands
        colnumb = len(outcatrowi)

        OutRowStr = ('{} '+(colnumb-1)*'{:8.3f}').format(*outcatrowi)+'\n'
        OutSecStr = OutSecStr + OutRowStr

        if IfProgBarOn == True:
            bar.cursor.restore()  # Return cursor to start
            bar.draw(value=procedi)

    if IfProgBarOn == True:
        bar.cursor.restore()  # Return cursor to start
        bar.draw(value=bar.max_value)  # Draw the bar!

    # OutCatSecQueue.put(OutSecStr)
    # _FinishQueue.put(1)

    # write_lock.acquire()
    with write_lock:
        OutCssCat.write(OutSecStr)
        OutCssCat.flush()
    # write_lock.release()

    print('\n')
    # procname = mp.current_process()._name
    # print(procname+' is finished.\n')



def kill_zombies():

    while any(mp.active_children()):
        time.sleep(2)
        print(mp.active_children())
        for p in mp.active_children():
            p.terminate()




if __name__ == '__main__':

    defaults = {'basedir': '/work/CSSOS/filter_improve/fromimg/windextract'}
    config = configparser.ConfigParser(defaults)
    config.read('cssos_config.ini')

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

    HstFileName = config['Hst2Css']['Hst814File']
    HstAsCssFile = config['Hst2Css']['HstAsCssFile']
    HstAsCssFileTest = config['Hst2Css']['HstAsCssFileTest']

    if len(sys.argv)>1:
        HstFileName = HstFileName.replace(HstFileName[-12:-9], sys.argv[1])
        HstAsCssFile = HstAsCssFile.replace(HstAsCssFile[-8:-5], sys.argv[1])
        HstAsCssFileTest = HstAsCssFileTest.replace(HstAsCssFileTest[-8:-5], sys.argv[1])

    if DebugTF == False:
        if os.system('ls *[0-9]_convwin_r.fits'):
            os.system('rm *[0-9]_convwin_r.fits')
        if os.system('ls *[0-9]_stack.fits'):
            os.system('rm *[0-9]_stack.fits')

    IfDoConv = config.getboolean('Hst2Css','IfDoConv')
    if IfDoConv==True:
        # Formal work doing HST814 image convolve to CSSOS image.
        print(HstFileName+' --> '+HstAsCssFile)
        HstHdu = fits.open(HstFileName)

        HstImgArr = HstHdu[0].data
        HstHdr = HstHdu[0].header

        # HST image convolve PSF to make CSS image
        # HstWidth = HstHdr['NAXIS1']
        # HstHeight = HstHdr['NAXIS2']
        HstHeight, HstWidth = HstImgArr.shape

        ndivide = config.getint('Hst2Css','NDivide')
        nzoomin = config.getint('Hst2Css','NZoomIn')
        nzoomout = config.getint('Hst2Css','NZoomOut')

        R80Cssz = config.getfloat('Hst2Css','R80Cssz')
        FwhmCssz = R80Cssz * 2 / 1.7941 * 1.1774  # "
        HstPS = config.getfloat('Hst2Css','PixScaleHst')
        CssPS = config.getfloat('Hst2Css','PixScaleCss')
        ConvKernelNormal = csstpkg.ImgConvKnl(config.getfloat('Hst2Css','FwhmHst'), FwhmCssz, HstPS/nzoomin, widthinfwhm=4)
        ConvHst2Css = csstpkg.ImgConv(HstImgArr, ConvKernelNormal, NDivide=ndivide, NZoomIn=nzoomin, NZoomOut=nzoomout)

        CssHdr = csstpkg.CRValTrans(HstHdr, HstPS, CssPS)

        ConvHst2Css32 = np.array(ConvHst2Css, dtype='float32')
        del ConvHst2Css, ConvKernelNormal
        csstpkg.DataArr2Fits(ConvHst2Css32, HstAsCssFile, headerobj=CssHdr)
        # csstpkg.DataArr2Fits(ConvHst2Css32[0:int(CssHei/8),0:int(CssWid/8)], HstAsCssFileTest, headerobj=CssHdr)
        del ConvHst2Css32, CssHdr


    CssHdu = fits.open(HstAsCssFile)
    CssCat = ascii.read(config['Hst2Css']['CssCatIn'])
    CssImg = CssHdu[0].data
    CssHdr = CssHdu[0].header
    CssHei, CssWid = CssImg.shape

    w = wcs.WCS(CssHdr)
    pixcorner = np.array([[0,0],[CssHei,0],[CssHei,CssWid],[0,CssWid]])
    worldcorner = w.wcs_pix2world(pixcorner,1)
    RaMin = min(worldcorner[:,0])
    RaMax = max(worldcorner[:,0])
    DecMin = min(worldcorner[:,1])
    DecMax = max(worldcorner[:,1])

    try:
        CssCat.rename_column('RA07','RA')
        CssCat.rename_column('DEC07','DEC')
    except:
        print('Already using RA,DEC of Leauthaud2007.')
    CatCutIdx = np.where((CssCat['RA']>RaMin) & (CssCat['RA']<RaMax) & (CssCat['DEC']>DecMin) & (CssCat['DEC']<DecMax))
    CatOfTile = CssCat[CatCutIdx]

    radec = np.asarray([CatOfTile['RA'], CatOfTile['DEC']]).transpose()
    xyarr = w.wcs_world2pix(radec,1)-1  # start from (0,0)

    CatOfTile['ximage'] = xyarr[:,0]
    CatOfTile['yimage'] = xyarr[:,1]

    CssCatTileNm = config['Hst2Css']['CssCatTile']
    ascii.write(CatOfTile, CssCatTileNm.replace(CssCatTileNm[-7:-4], str(sys.argv[1])), format='commented_header', comment='#', overwrite=True)

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
        # cssbands = config.get('Hst2Css', 'CssBands').split(',')
        # filtnumb_str = config.get('Hst2Css', 'FiltNumb').split(',')
        # filtnumb = [int(numb) for numb in filtnumb_str]

        magab_zeros = []
        for cssband, numb in zip(cssbands, filtnumb):
            expcss = 150. * numb  # s
            magab_zeros.append(csstpkg.MagAB_Zero(Gain,cssband, expcss, TelArea))

        namelists = map(lambda modmag, magsim, magerr, snr, aband: \
                            [modmag+aband, magsim+aband, magerr+aband, snr+aband], \
                        ['MOD_']*len(cssbands), ['MagSim_'] * len(cssbands), ['ErrMag_'] * len(cssbands), ['SNR_'] * len(cssbands), cssbands)
        colnames = ['ID','Z_BEST']+list(itertools.chain(*namelists))

        LenCatTile = len(CatOfTile)

        # Output catalog for one tile
        OutCssCatName = 'Cssos_magsim_SNR_tile_'+str(sys.argv[1])+'_'+schemecode+'.txt'
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
            simul_css(CatOfTile, CssImg, cssbands, filtnumb, 0)

        elif Nbat > 0:
            jobs=[]
            for npi in range(NProcesses):
                i_low, i_high = npi*Nbat, (npi+1)*Nbat
                jobs.append(mp.Process(target=simul_css, name='Process'+str(npi), args=(CatOfTile[i_low:i_high], CssImg, cssbands, filtnumb, npi)))

            for sti in range(NProcesses):
                jobs[sti].start()
            for jni in range(NProcesses):
                jobs[jni].join()

            if Nleft > 0:
                print('Processing the rest')
                simul_css(CatOfTile[int(Nbat * NProcesses):], CssImg, cssbands, filtnumb, 0)

        else:
            if Nleft > 0:
                simul_css(CatOfTile, CssImg, cssbands, filtnumb, 0)
            else:
                print('The catalog is empty. Please check it.')

        # for nj in range(OutCssCatQueue.qsize()):
        #     try:
        #         OutCssCat.write(OutCssCatQueue.get_nowait())
        #     except Exception as queuerr:
        #         pass
        # print('write catalog finished.')
        # OutCssCatQueue.close()

        OutCssCat.close()

    finishtime = time.time()
    print('Time Consumption:', finishtime - begintime, 's')
    print('\nFinished.\n')