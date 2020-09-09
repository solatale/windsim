
"""
Usage: python hst814simsed_phutil_mp_flux_allin1.py 065
"""

# WinImgStack = WinImgBands[1:8,::].sum(0) should be checked.

import configparser
import csstpkg_phutil_mp_uBgNWiBy as csstpkg
from astropy.io import fits
import astropy.io.ascii as ascii
import matplotlib.pyplot as plt
# from pylab import gca
# from mpl_toolkits.mplot3d import axes3d
from astropy import wcs
import sys,math,time,os,io,glob,copy
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

        # if ((float(cataline['MOD_NUV_css'])<0) or (float(cataline['MOD_WNUV_css'])<0) or (float(cataline['MOD_NUV_css'])>50)):
        #     continue

        np.random.seed()

        ident = str(cataline['IDENT'])
        if DebugTF == True:
            print('\n', ident, '\n--------------------------------------------')

        objwind = csstpkg.windcut(_CssImg, cataline, StampSize)
        # print(objwind)

        if objwind is None:
            if DebugTF == True:
                print('\033[31mError: '+'Object stamp cutting error.\033[0m')
            continue
        # csstpkg.DataArr2Fits(objwind.data, ident+'_convwin.fits')

        objwinshape = objwind.shape
        # objwind.data = objwind.data * ExpCssFrm

        WinImgBands = np.zeros((len(cssbands), objwinshape[0], objwinshape[1]))  # 3-D array contains images of all the cssbands

        if DebugTF == True:
            if IfPlotObjWin == True:
                csstpkg.PlotObjWin(objwind, cataline)
            print(' '.join(['RA DEC:', str(cataline['RA']), str(cataline['DEC'])]))

        outcatrowi = [ident, cataline['Z_BEST']]

        # Photometry for the central object on the convolved window
        ObjWinPhot_DeBkg = csstpkg.CentrlPhot(objwind.data, id=str(outcatrowi[0]) + " ConvWdW DeBkg")
        ObjWinPhot_DeBkg.Bkg(idb=str(outcatrowi[0]) + " ConvWdW DeBkg", debug=DebugTF, thresh=1.5, minarea=10, deblend_nthresh=32, deblend_cont=0.01)
        ObjWinPhot_DeBkg.Centract(idt=str(outcatrowi[0]) + " ConvWdW DeBkg", thresh=2.5, minarea=10, deblend_nthresh=32, deblend_cont=0.1, debug=DebugTF, sub_backgrd_bool=True)
        if ObjWinPhot_DeBkg.centobj is np.nan:
            if DebugTF == True:
                print('--- No central object detected in convolved image ---')
            continue
        else:
            ObjWinPhot_DeBkg.KronR(idk=str(outcatrowi[0]) + " ConvWdW", debug=DebugTF, mask_bool=True)

        NeConv_DeBkg, ErrNeConv_DeBkg = ObjWinPhot_DeBkg.EllPhot(ObjWinPhot_DeBkg.kronr, mask_bool=True)


        if ((NeConv_DeBkg <= 0) or (NeConv_DeBkg is np.nan)):
            if DebugTF == True:
                print('NeConv_DeBkg for a winimg <= 0 or NeConv_DeBkg is np.nan')
            continue

        noisebkg_conv = ObjWinPhot_DeBkg.bkgstd

        if DebugTF == True:
            print('self.bkg Flux & ErrFlux =', ObjWinPhot_DeBkg.bkgmean, ObjWinPhot_DeBkg.bkgstd)
            print('Class processed NeConv_DeBkg & ErrNeConv_DeBkg:', NeConv_DeBkg, ErrNeConv_DeBkg)



        # Read model SED to NDArray
        # modsednum = cataline['MOD_BEST']
        sedname = seddir + 'Id' + '{:0>9}'.format(ident) + '.spec'
        modsed, readflag = csstpkg.readsed(sedname)
        if readflag == 1:
            modsed[:, 1] = csstpkg.magab2flam(modsed[:, 1], modsed[:, 0])  # to convert model SED from magnitude to f_lambda(/A)
        else:
            print('model sed not found.')
            continue
        bandi = 0
        NeBands = []
        magsimorigs = {}
        magsims = {}
        scalings = []
        ObjWinPhot_DeBkg_Errs = []

        for cssband, numb in zip(cssbands, filtnumb):

            expcss = 150. * numb  # s
            # cssbandpath = thrghdir+cssband+'.txt'
            magsim = csstpkg.Sed2Mag(modsed, cssband, MagSim_Zero[cssband])
            magsims[cssband] = magsim
            lambpivot = csstpkg.pivot(cssband)
            if DebugTF == False:
                flambandarr = 1
            elif DebugTF == True:
                if cssband=='i4':
                    cssband='i'
                elif cssband=='uB':
                    cssband='u'
                elif cssband=='gN':
                    cssband='g'
                flambandmod = csstpkg.magab2flam(float(cataline['MOD_' + cssband + '_css']), lambpivot)
                flambandsim = csstpkg.magab2flam(magsim, lambpivot)
                flambandarr=np.array([[lambpivot, flambandmod],[lambpivot, flambandsim]])
            NeABandObs = csstpkg.NeFromSED(modsed, cssband, expcss, TelArea, flambandarr, debug=DebugTF)

            # # magaband0 = csstpkg.Ne2MagAB(NeABandObs, cssband, expcss, TelArea)
            # delmag = float(cataline['MOD_' + cssband + '_css']) - magsim
            # NeABand = NeABandObs*10**(-0.4*delmag)  # in e-/band/exptime/telarea
            # NeBands.append(NeABand)

            NeBands.append(NeABandObs)
            NeABand = NeABandObs
            NeABand = np.random.poisson(lam=NeABandObs)   # Do poisson randomize


            if DebugTF == True:
                print(' Mag from Sim for '+cssband+' band =', magsim)
                # print(' Mag from Ne Calculation =', magaband0)
                # print('  DeltaMag_'+cssband+' = ', float(cataline['MOD_' + cssband + '_css'])-magsim, delmag)
                print(' '.join(['Counts on ConvImg:', str(NeConv_DeBkg/ExpCssFrm), 'e-']))
                print(' '.join([cssband, 'band model electrons = ', str(NeABand), 'e-']))
                print('MOD_' + cssband + '_css =', cataline['MOD_' + cssband + '_css'])
                if NeABand>0:
                    magsimorigs[cssband] = csstpkg.Ne2MagAB(NeABand, cssband, expcss, TelArea)
                else:
                    magsimorigs[cssband] = 99
                print('Magsim_' + cssband + ' =', magsimorigs[cssband])

            Scl2Sed = NeABand / NeConv_DeBkg  # To scale stamp HST detection/s to SED in e-/band/exptime/telarea.
            scalings.append(Scl2Sed)

            if DebugTF == True:
                print('Scaling Factor: ', Scl2Sed)


            # ZeroLevel = config.getfloat('Hst2Css', 'BZero')
            SkyLevel = csstpkg.backsky[cssband] * expcss
            DarkLevel = config.getfloat('Hst2Css', 'BDark') * expcss
            RNCssFrm = config.getfloat('Hst2Css', 'RNCss')

            # IdealImg = objwind.data * Scl2Sed + SkyLevel + DarkLevel  # e-
            IdealImg = ObjWinPhot_DeBkg.data_bkg * Scl2Sed # + SkyLevel + DarkLevel  # e-
            ObjWinPhot_DeBkg_Errs.append(ObjWinPhot_DeBkg.bkgstd * Scl2Sed)

            BkgNoiseTot = np.sqrt(SkyLevel + DarkLevel + RNCssFrm**2*numb)
            if BkgNoiseTot > noisebkg_conv*Scl2Sed:
                Noise2Add = np.sqrt(BkgNoiseTot**2 - (noisebkg_conv*Scl2Sed)**2)
            else:
                Noise2Add = 0

            if DebugTF == True:
                print('Noise Total '+cssband+' band: ', BkgNoiseTot)
                print('Noise Stamp '+cssband+' band: ', noisebkg_conv*Scl2Sed)
                print('Noise Added '+cssband+' band: ', Noise2Add)


            # ImgPoiss = copy.deepcopy(IdealImg)
            # ImgPoiss[ImgPoiss>0] = np.random.poisson(lam=IdealImg[IdealImg>0]*ExpCssFrm, size=IdealImg[IdealImg>0].shape)/ExpCssFrm
            NoisNormImg = csstpkg.NoiseArr(objwinshape, loc=0, scale=Noise2Add, func='normal')

            # DigitizeImg = IdealImg/Gain
            DigitizeImg = np.round((IdealImg + NoisNormImg) / Gain)  # IdealImg have already been poissonized
            # DigitizeImg = np.round((ImgPoiss + NoisNormImg + ZeroLevel) / Gain)

            # if DebugTF == True:
            #     csstpkg.DataArr2Fits(DigitizeImg, 'ImgWinSim_Gain_RN_'+ident+'_'+cssband+'.fits')

            WinImgBands[bandi, ::] = DigitizeImg

            bandi = bandi + 1

        if DebugTF == True:
            print('Stack all bands and detect objects:')

        WinImgStack = WinImgBands[2:7,::].sum(0)
        # WinImgStack = WinImgBands[0:7,::].sum(0)
        # print(WinImgStack.shape)
        # AduStack, ErrAduStack, ObjectStack, KronRStack, MaskStack = septract(WinImgStack, id=str(outcatrowi[0])+" Stack", debug=DebugTF, thresh=1.2, minarea=10)
        StackPhot = csstpkg.CentrlPhot(WinImgStack, id=ident + " Stack")
        StackPhot.Bkg(idb=ident + " Stack", debug=DebugTF, thresh=1.5, minarea=10)
        StackPhot.Centract(idt=ident + " Stack", thresh=1.5, minarea=10, deblend_nthresh=32, deblend_cont=0.1, debug=DebugTF)
        if StackPhot.centobj is np.nan:
            if DebugTF == True:
                print('No central object on STACK image.')
            continue
        else:
            A_stack = StackPhot.centobj['a']
            B_stack = StackPhot.centobj['b']
            Drms_stack = np.sqrt((A_stack**2+B_stack**2)/2)*pixscale  # RMS size in arcsec
            StackPhot.KronR(idk=ident + " Stack", debug=DebugTF, mask_bool=True)
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
            if DebugTF == True:
                plt.hist(WinImgBands[bandi, ::].flatten(), bins=np.arange(30) - 15, )
                plt.title(' '.join([cssband, 'simul image']))
                plt.show()

            AduObser, ErrAduObs, npix, bkgrms = csstpkg.septractSameAp(WinImgBands[bandi, ::], StackPhot, StackPhot.centobj, StackPhot.kronr, mask_det=StackPhot.mask_other, debug=DebugTF, annot=cssband+'_cssos', thresh=1.2, minarea=10, sub_backgrd_bool=False)
            # print(scalings)
            ErrAduTot = np.sqrt(ErrAduObs ** 2 + npix*(noisebkg_conv * scalings[bandi]) ** 2)
            # ErrAduTot = np.sqrt(ErrAduObs**2+npix*ObjWinPhot_DeBkg_Errs[bandi]**2)

            if AduObser > 0:
                SNR = AduObser / ErrAduTot
                # FluxMsr = csstpkg.Ne2Fnu(AduObser*Gain,cssband,expcss,TelArea)
                FluxMsr = AduObser * fluxadu_zeros[bandi]
                FLuxErr = ErrAduTot * fluxadu_zeros[bandi]  # FluxMsr/SNR

            else:
                # FluxMsr = 0
                # FLuxErr = csstpkg.Ne2Fnu(ErrAduTot*Gain,cssband,expcss,TelArea)
                FluxMsr = AduObser * fluxadu_zeros[bandi]
                FLuxErr = ErrAduTot*fluxadu_zeros[bandi]
                SNR = 0


            if DebugTF == True:
                npixel = math.pi*(ObjWinPhot_DeBkg.centobj['a']*csstpkg.kphotpar*ObjWinPhot_DeBkg.kronr)*(ObjWinPhot_DeBkg.centobj['b']*csstpkg.kphotpar*ObjWinPhot_DeBkg.kronr)
                print(' '.join([cssband, 'band model e- =', str(NeBands[bandi]), 'e-']))
                print(' '.join([cssband, 'band simul e- =', str(AduObser*Gain), 'e-', ' ErrNe=', str(ErrAduTot*Gain)]))
                # print(AduObser, Gain, NeBands[bandi], -2.5*math.log10(AduObser*Gain/NeBands[bandi]))
                print('SNR =', AduObser/ErrAduTot)
                print('Npixel =', npixel)
                # print(' '.join([cssband, 'band mag_model = ', str(cataline['MOD_' + cssband + '_css']), '(AB mag)']))
                # print(' '.join([cssband, 'band Magsim_orig = ', str(magsimorigs[bandi]), '(AB mag)']))
                # print(' '.join([cssband, 'band Mag_simul = ', str(MagObser), '(AB mag)']))
                # print(' '.join([cssband, 'band magerr_simul = ', str(ErrMagObs), '(AB mag)']))
                # print(' '.join(['Magsim - Magsimorig =', str(MagObser-magsimorigs[bandi])]))

            # if cssband=='uB':
            #     modmag = cataline['MOD_u_css']
            # elif cssband == 'gN':
            #     modmag = cataline['MOD_g_css']
            # elif cssband == 'i4':
            #     modmag = cataline['MOD_i_css']
            # else:
            #     modmag = cataline['MOD_' + cssband + '_css']
            outcatrowi = outcatrowi + [magsims[cssband], FluxMsr, FLuxErr, SNR]
            bandi = bandi + 1

        del WinImgBands

        outcatrowi = outcatrowi + [npix, Drms_stack]
        # outcatrowi = outcatrowi + [Drms_stack]
        colnumb = len(outcatrowi)

        OutRowStr = ('{} '+(colnumb-1)*'{:15.6E}').format(*outcatrowi)+'\n'
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
    config.read('cssos_config_uBgNWiBy.ini')

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
    pixscale = config.getfloat('Hst2Css', 'PixScaleCss')

    HstFileName = config['Hst2Css']['Hst814File']
    HstAsCssFile = config['Hst2Css']['HstAsCssFile']
    HstAsCssFileTest = config['Hst2Css']['HstAsCssFileTest']

    StampSize = config.getfloat('Hst2Css','StampSize')

    if len(sys.argv)>1:
        HstFileName = HstFileName.replace(HstFileName[-12:-9], sys.argv[1])
        HstAsCssFile = HstAsCssFile.replace(HstAsCssFile[-8:-5], sys.argv[1])
        HstAsCssFileTest = HstAsCssFileTest.replace(HstAsCssFileTest[-8:-5], sys.argv[1])

    remlst = glob.glob("Ideal_Zero_Gain_RN_check_*.fits")
    for arem in remlst:
        if os.path.exists(arem):
            os.remove(arem)

    # if DebugTF == False:
    #     if os.system('ls *[0-9]_convwin_r.fits'):
    #         os.system('rm *[0-9]_convwin_r.fits')
    #     if os.system('ls *[0-9]_stack.fits'):
    #         os.system('rm *[0-9]_stack.fits')

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
        ConvHst2Css = csstpkg.ImgConv(HstImgArr, ConvKernelNormal.image, NDivide=ndivide, NZoomIn=nzoomin, NZoomOut=nzoomout)

        CssHdr = csstpkg.CRValTrans(HstHdr, HstPS, CssPS)

        ConvHst2Css32 = np.array(ConvHst2Css, dtype='float32')
        del ConvHst2Css, ConvKernelNormal
        csstpkg.DataArr2Fits(ConvHst2Css32, HstAsCssFile, headerobj=CssHdr)
        # csstpkg.DataArr2Fits(ConvHst2Css32[0:int(CssHei/8),0:int(CssWid/8)], HstAsCssFileTest, headerobj=CssHdr)
        del ConvHst2Css32, CssHdr
    else:
    	pass

    IfBandSim = config.getboolean('Hst2Css','IfBandSim')
    IfTileCata = config.getboolean('Hst2Css','IfTileCata')

    if IfTileCata == True:
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
            print('Using RA,DEC of Leauthaud2007.')
        CatCutIdx = np.where((CssCat['RA']>RaMin) & (CssCat['RA']<RaMax) & (CssCat['DEC']>DecMin) & (CssCat['DEC']<DecMax))
        CatOfTile = CssCat[CatCutIdx]

        radec = np.asarray([CatOfTile['RA'], CatOfTile['DEC']]).transpose()
        xyarr = w.wcs_world2pix(radec,1)-1  # start from (0,0)

        CatOfTile['ximage'] = xyarr[:,0]
        CatOfTile['yimage'] = xyarr[:,1]

        # CssCatTileNm = config['Hst2Css']['CssCatTile']
        # ascii.write(CatOfTile, CssCatTileNm.replace(CssCatTileNm[-7:-4], str(sys.argv[1])), format='commented_header', comment='#', overwrite=True)

        if IfBandSim == True:
            cssbands = config.get('Hst2Css', 'CssBands').split(',')
            filtnumb_str = config.get('Hst2Css', 'FiltNumb').split(',')
            filtnumb = [int(numb) for numb in filtnumb_str]

            fluxadu_zeros = []
            MagSim_Zero = {}
            for cssband, numb in zip(cssbands, filtnumb):
                expcss = 150. * numb  # s
                # magab_zeros.append(csstpkg.MagAB_Zero(Gain, cssband, expcss, TelArea))
                fluxadu_zeros.append(csstpkg.FluxAdu_Zero(Gain, cssband, expcss, TelArea))
                MagSim_Zero[cssband] = csstpkg.SedMag0(cssband)


            namelists = map(lambda modmag, fluxsim, fluxsimerr, snrsim, aband: \
                                [modmag+aband, fluxsim+aband, fluxsimerr+aband, snrsim+aband], \
                            ['MOD_']*len(cssbands), ['FluxSim_'] * len(cssbands), ['ErrFlux_'] * len(cssbands), ['SNR_'] * len(cssbands), cssbands)
            colnames = ['ID','Z_BEST']+list(itertools.chain(*namelists))+['Npix','Drms_sec']

            LenCatTile = len(CatOfTile)
            print('Catalog Length: ',LenCatTile)

            # Output catalog for one tile
            OutCssCatName = 'Cssos_FluxSim_SNR_tile_'+str(sys.argv[1])+'_allin1_uBgNWiBy.txt'
            if os.path.isfile(OutCssCatName) is True:
                os.remove(OutCssCatName)
            OutCssCat = open(OutCssCatName, mode='w')
            headcomment = '# '+' '.join(colnames)+'\n'
            # headcomment = '# ID Z_BEST MOD_NUV FluxSim_NUV ErrFlux_NUV SNR_NUV MOD_NUV2 FluxSim_NUV2 ErrFlux_NUV2 SNR_NUV2 MOD_u FluxSim_u ErrFlux_u SNR_u MOD_g FluxSim_g ErrFlux_g SNR_g MOD_r FluxSim_r ErrFlux_r SNR_r MOD_i FluxSim_i ErrFlux_i SNR_i MOD_z FluxSim_z ErrFlux_z SNR_z MOD_y FluxSim_y ErrFlux_y SNR_y MOD_y2 FluxSim_y2 ErrFlux_y2 SNR_y2 MOD_WNUV  FluxSim_WNUV ErrFlux_WNUV SNR_WNUV MOD_WV FluxSim_WV ErrFlux_WV SNR_WV MOD_WI FluxSim_WI ErrFlux_WI SNR_WI MOD_i4 FluxSim_i4 ErrFlux_i4 SNR_i4 MOD_uB FluxSim_uB ErrFlux_uB SNR_uB MOD_gN FluxSim_gN ErrFlux_gN SNR_gN Npix Drms_sec\n'

            OutCssCat.write(headcomment)

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
        else:
            pass
    else:
        pass

    finishtime = time.time()
    print('Time Consumption:', finishtime - begintime, 's')
    print('\nFinished.\n')
