
"""
Usage: python hst814simsed.py 065
"""

import configparser
import csstpkg_phutil as csstpkg
from astropy.io import fits
import astropy.io.ascii as ascii
import matplotlib.pyplot as plt
# from pylab import gca
# from mpl_toolkits.mplot3d import axes3d
from astropy import wcs
import sys,math,time,os
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
import gc

gc.enable()




if __name__ == '__main__':


    defaults = {'basedir': '/work/CSSOS/filter_improve/fromimg/windextract'}
    config = configparser.ConfigParser(defaults)
    config.read('cssos_config.ini')

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
        CssHdu = fits.open(HstAsCssFile)
        csstpkg.DataArr2Fits(ConvHst2Css32[0:int(CssHei/8),0:int(CssWid/8)], HstAsCssFileTest, headerobj=CssHdr)
        CssCat = ascii.read(config['Hst2Css']['CssCatIn'])
    else:
        ConvHst2CssHdu = fits.open(HstAsCssFile)
        ConvHst2Css32 = ConvHst2CssHdu[0].data
        CssHdr = ConvHst2CssHdu[0].header
        CssHei, CssWid = ConvHst2Css32.shape

        # CssHdu = fits.open(HstAsCssFileTest)
        CssHdu = fits.open(HstAsCssFile)
        CssCat = ascii.read(config['Hst2Css']['CssCatTile'])
        # CssHdu = fits.open(HstAsCssFile)
        # CssCat = ascii.read(config['Hst2Css']['CssCatIn'])



    CssImg = CssHdu[0].data
    CssHdr = CssHdu[0].header
    CssHei, CssWid = CssImg.shape

    # RNnoise = NoiseArr(ConvHst2Css.shape, loc=0, scale=config.getfloat('Hst2Css','RNCss')*FrameCss**0.5, func='normal')
    # BkgLevel = (csstpkg.backsky['wfc_F814W'])*ExpCssFrm*FrameCss
    # BkgNoise = NoiseArr(ConvHst2Css.shape, loc=BkgLevel, func='poisson')-BkgLevel

    # nobjscl = ExpCssFrm*FrameCss*config.getfloat('Hst2Css','NobjPCss')/config.getfloat('Hst2Css','NobjPHst')
    # HstAsCss = ConvHst2Css*nobjscl + RNnoise + BkgNoise
    # HstAsCss32 = np.array(HstAsCss, dtype='float32')

    w = wcs.WCS(CssHdr)
    pixcorner = np.array([[0,0],[CssHei,0],[CssHei,CssWid],[0,CssWid]])
    worldcorner = w.wcs_pix2world(pixcorner,1)
    RaMin = min(worldcorner[:,0])
    RaMax = max(worldcorner[:,0])
    DecMin = min(worldcorner[:,1])
    DecMax = max(worldcorner[:,1])

    # print(worldcorner)

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

    # sys.exit()

    for scheme_i in [1,2]:#range(3):

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


        magab_zeros = []
        for cssband, numb in zip(cssbands, filtnumb):
            expcss = 150. * numb  # s
            magab_zeros.append(csstpkg.MagAB_Zero(Gain,cssband, expcss, TelArea))

        print('Filter scheme flag:',schemecode)

        namelists = map(lambda mag, magerr, snr, aband: \
                            [mag + aband, magerr+aband, snr+aband], \
                        ['Magsim_'] * len(cssbands), ['ErrMag_'] * len(cssbands), ['SNR_'] * len(cssbands), cssbands)
        colnames = ['ID','Z_BEST']+list(itertools.chain(*namelists))
        OutCat = Table(names=colnames)
        # print(OutCat)
        NColOut = len(OutCat.colnames)

        if IfProgBarOn == True:
            bar = Bar(max_value=len(CatOfTile), empty_color=7, filled_color=4)
            bar.cursor.clear_lines(2)  # Make some room
            bar.cursor.save()  # Mark starting line

        for i in range(len(CatOfTile)):

            # if DebugTF == True:
            # print(CatOfTile[i]['IDENT'])
            a = CatOfTile[i]['a_image_css']
            b = CatOfTile[i]['b_image_css']
            theta = CatOfTile[i]['theta_image']
            ident = str(CatOfTile[i]['IDENT'])

            # Cut a window of the object as objwind
            # cutwidrad = int((a*math.cos(theta/180.*math.pi)+b*abs(math.sin(theta/180.*math.pi)))*5)
            # cutheirad = int((a*abs(math.sin(theta/180.*math.pi))+b*math.cos(theta/180.*math.pi))*5)
            cutwidrad = int(max([a,b])*10)
            cutheirad = cutwidrad

            # if min(cutwidrad, cutheirad)<16:
            #     continue

            windleft = int(CatOfTile[i]['ximage'])-cutwidrad
            windright = int(CatOfTile[i]['ximage'])+cutwidrad+1
            windbott = int(CatOfTile[i]['yimage'])-cutheirad
            windtop = int(CatOfTile[i]['yimage'])+cutheirad+1


            if int(CatOfTile[i]['ximage'])-cutwidrad < 0:
                windleft = 0
            if int(CatOfTile[i]['ximage']) + cutwidrad + 1 > CssWid:
                windright = CssWid-1
            if int(CatOfTile[i]['yimage'])-cutheirad < 0:
                windbott = 0
            if int(CatOfTile[i]['yimage'])+cutheirad+1 > CssHei:
                windtop = CssHei-1

            objwind = ConvHst2Css32[windbott:windtop, windleft:windright]


            sidewid_l, sidewid_r, sidewid_b, sidewid_t = 0,0,0,0

            if windleft < 0:
                sidewid_l = abs(windleft)
                windleft = 0
            if windright > CssWid:
                windright = CssWid-1
            if windbott < 0:
                sidewid_b = abs(windbott)
                windbott = 0
            if windtop > CssHei:
                windtop = CssHei-1

            objwind0 = np.array(ConvHst2Css32[windbott:windtop, windleft:windright], dtype='float32')
            objwin0shape = objwind0.shape
            windback = sep.Background(objwind0, bw=16, bh=16)

            objwind = np.full((cutwidrad*2+1,cutwidrad*2+1), windback.globalback)
            objwinshape = objwind.shape

            objwind[sidewid_b:(objwin0shape[0]+sidewid_b), sidewid_l:(objwin0shape[1]+sidewid_l)] = objwind0

            # if (windleft < 0) or (windright > CssWid) or (windbott < 0) or (windtop > CssHei):
            #     continue
            # else:
            objwind = np.array(ConvHst2Css32[windbott:windtop, windleft:windright], dtype='float32')
            objwinshape = objwind.shape

            # csstpkg.DataArr2Fits(objwind, ident+'_convwin.fits')

            WinImgBands = np.zeros((len(cssbands), objwinshape[0],objwinshape[1]))  # 3-D array contains images of all the cssbands

            if IfPlotObjWin == True:
                # Plot each object's 3D surface
                x = np.linspace(0, objwinshape[1] - 1, objwinshape[1])
                y = np.linspace(0, objwinshape[0] - 1, objwinshape[0])
                xx, yy = np.meshgrid(x, y)

                mean=np.median(objwind)
                stddev = np.std(objwind)

                fig = plt.figure()
                ax=fig.gca()
                # ax = fig.gca(projection='3d')
                # ax.set_aspect(1.0)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                plt.imshow(objwind, vmin=0-stddev, vmax=mean+3*stddev, cmap=plt.cm.cividis, origin='lower', )
                # ax.plot_surface(xx, yy, objwind, cmap=plt.cm.cividis, alpha=0.7)
                titlewords = 'ID:'+str(CatOfTile[i]['IDENT'])+ \
                             ' RA:'+str(CatOfTile[i]['RA'])+ \
                             ' DEC:'+str(CatOfTile[i]['DEC'])+ \
                             ' FWHM:'+str(CatOfTile[i]['fwhm_image_css'])+'\n'+ \
                             'A:'+str(CatOfTile[i]['a_image_css'])+ \
                             ' B:'+str(CatOfTile[i]['b_image_css'])+ \
                             ' Theta:'+str(CatOfTile[i]['theta_image'])
                plt.title(titlewords)
                plt.show()

            # Read model SED to NDArray
            modsedtag = CatOfTile[i]['MOD_BEST']
            sedname = seddir+'Id'+'{:0>9}'.format(ident)+'.spec'
            modsed = csstpkg.readsed(sedname)
            modsed[:,1] = csstpkg.mag2flam(modsed[:,1], modsed[:,0])  # to convert model SED from magnitude to f_lambda(/A)

            outcatrowi = [ident, CatOfTile[i]['Z_BEST']]

            if DebugTF == True:
                print(' '.join([ident, '\nRA DEC:', str(CatOfTile[i]['RA']), str(CatOfTile[i]['DEC'])]))

            ObjWinPhot = csstpkg.CentrlPhot(objwind, id=str(outcatrowi[0])+" ConvWdW")
            ObjWinPhot.Bkg(idb=str(outcatrowi[0])+" ConvWdW", debug=DebugTF, thresh=2, minarea=10, deblend_nthresh=32, deblend_cont=0.01)
            ObjWinPhot.Centract(idt=str(outcatrowi[0])+" ConvWdW", thresh=2.5, deblend_nthresh=32, deblend_cont=0.1, minarea=10, debug=DebugTF)
            if ObjWinPhot.centobj is np.nan:
                continue
            else:
                ObjWinPhot.KronR(idk=str(outcatrowi[0])+" ConvWdW", debug=DebugTF)

            NeConv, ErrNeConv = ObjWinPhot.EllPhot(ObjWinPhot.kronr, mask_bool=True)
            if DebugTF == True:
                # print('data-bkg_masked array:', ObjWinPhot.data_bkg_masked)
                print('self.bkg Flux & ErrFlux =', ObjWinPhot.bkg.background_median, ObjWinPhot.bkg.background_rms_median)
                # print('bkgrms:', ObjWinPhot.bkgrms)
                # print('Object Central:', ObjWinPhot.centobj)
                # print('data-bkg array:', ObjWinPhot.data_bkg)
                print('Class processed Neconv & ErrNeConv:', NeConv, ErrNeConv)

            # NeConv, ErrNeConv, ObjectConv, KronRConv, MaskConv = csstpkg.septract(objwind, id=str(outcatrowi[0])+" ConvWdW", debug=DebugTF, thresh=2, minarea=10)
            # if DebugTF == True:
            #     print(' '.join(["Stamp's central obj electrons = ", str(NeConv), 'e-/s']))

            if ((NeConv<=0) or (NeConv is np.nan)):
                if DebugTF == True:
                    print('NeConv for a winimg <= 0 or NeConv is np.nan')
                    continue
                # for cssband in cssbands:
                #     outcatrowi = outcatrowi + [-99] + [-99] + [-99]
                # OutCat.add_row(outcatrowi)


            bandi = 0
            for cssband,numb in zip(cssbands,filtnumb):

                expcss = 150. * numb  # s
                # cssbandpath = thrghdir+cssband+'.txt'
                NeABand = csstpkg.NeObser(modsed, cssband, expcss, TelArea)#*CatOfTile[i]['SCALE_BEST']
                if DebugTF == True:
                    print(' '.join([cssband, 'band model electrons = ', str(NeABand), 'e-']))
                    print('MOD_'+cssband+'_css =', CatOfTile[i]['MOD_'+cssband+'_css'])
                    print('Magsim_'+cssband+' =', csstpkg.Ne2MagAB(NeABand,cssband,expcss,TelArea))

                Scl2Sed = NeABand/NeConv
                if DebugTF == True:
                    print(ident,Scl2Sed)

                ZeroLevel = config.getfloat('Hst2Css', 'BZero')
                skylevel = csstpkg.backsky[cssband]*expcss
                DarkImg = config.getfloat('Hst2Css', 'BDark') * expcss
                IdealImg = objwind*Scl2Sed + skylevel + DarkImg  # e-
                IdealImg[IdealImg<0] = 0
                if DebugTF == True:
                    print(cssband,' band Sum of IdealImg =', np.sum(IdealImg))
                ImgPoiss = np.random.poisson(lam=IdealImg, size=objwinshape)


                NoisNorm = csstpkg.NoiseArr(objwinshape, loc=0, scale=config.getfloat('Hst2Css', 'RNCss')*(numb)**0.5, func='normal')

                # DigitizeImg = ImgPoiss + NoisNorm + ZeroLevel
                DigitizeImg = np.floor((ImgPoiss+NoisNorm+ZeroLevel)/Gain)

                WinImgBands[bandi,::] = DigitizeImg

                bandi = bandi+1


            if DebugTF == True:
                print('Stack all bands and detect objects:')

            WinImgStack = WinImgBands.sum(0)
            # print(WinImgStack.shape)
            # AduStack, ErrAduStack, ObjectStack, KronRStack, MaskStack = csstpkg.septract(WinImgStack, id=str(outcatrowi[0])+" Stack", debug=DebugTF, thresh=1.2, minarea=10)
            StackPhot = csstpkg.CentrlPhot(WinImgStack, id=str(outcatrowi[0]) + " Stack")
            StackPhot.Bkg(idb=str(outcatrowi[0]) + " Stack", debug=DebugTF, thresh=1.2, minarea=10)
            StackPhot.Centract(idt=str(outcatrowi[0]) + " Stack", thresh=1.2, minarea=10, deblend_nthresh=24, deblend_cont=0.1)
            if StackPhot.centobj is np.nan:
                if DebugTF == True:
                    print('No central object on STACK image.')
                continue
            else:
                StackPhot.KronR(idk=str(outcatrowi[0]) + " Stack", debug=DebugTF)


            AduStack, ErrAduStack = StackPhot.EllPhot(StackPhot.kronr, mask_bool=True)
            if AduStack is np.nan:
                if DebugTF == True:
                    print('RSS error for STACK image.')
                continue
                # for cssband in cssbands:
                #     outcatrowi = outcatrowi + [-99] + [-99] + [-99]
            else:
                # if DebugTF == True:
                #     print('KronRStack:', KronRStack)
                #     print('ADU STACK =', AduStack, 'e-')
                #     fig, ax = plt.subplots()
                #     ax.imshow(MaskStack, interpolation='nearest', cmap='gray', origin='lower')
                #     # plot an ellipse for each object
                #     e = Ellipse(xy=(ObjectStack['x'], ObjectStack['y']), width=2 * KronRStack * ObjectStack['a'] * 2,
                #                 height=2 * KronRStack * ObjectStack['b'] * 2, angle=ObjectStack['theta'] * 180. / np.pi)
                #     e.set_facecolor('none')
                #     e.set_edgecolor('blue')
                #     ax.add_artist(e)
                #     plt.title(ident+' Mask')
                #     plt.show()
                bandi = 0
                for cssband, numb in zip(cssbands, filtnumb):
                    expcss = 150. * numb  # s
                    if DebugTF == True:
                        print(cssband, ' band Array Slice Sum =', np.sum(WinImgBands[bandi,::]), 'e-')
                        print(cssband, ' band Array Slice MagAB =', csstpkg.Ne2MagAB(np.sum(WinImgBands[bandi,::]), cssband, expcss, TelArea))
                    AduObser, ErrAduObs = csstpkg.septractSameAp(WinImgBands[bandi,::], StackPhot.centobj, StackPhot.kronr, mask_det=StackPhot.mask_other, debug=DebugTF, annot=cssband, thresh=1.2, minarea=10)
                    # AduObser, ErrAduObs = csstpkg.septract(WinImgBands[bandi,::], ident+' septract', debug=DebugTF)[0:2]
                    if DebugTF == True:
                        print(''.join([cssband, ' band simu ADU=', str(AduObser), ' ErrNe=', str(ErrAduObs)]))
                    if AduObser > 0:
                        SNR = AduObser/ErrAduObs
                        # MagObser = csstpkg.Ne2MagAB(AduObser*Gain,cssband,expcss,TelArea)
                        MagObser = -2.5*math.log10(AduObser)+magab_zeros[bandi]
                        ErrMagObs = 2.5*math.log10(1+1/SNR)
                        if DebugTF == True:
                            if ((cssband == 'r') & (np.abs(MagObser - CatOfTile[i]['MOD_' + cssband + '_css']) > 1)):
                                csstpkg.DataArr2Fits(objwind, ident + '_convwin_r.fits')
                                csstpkg.DataArr2Fits(WinImgStack, ident + '_stack.fits')
                    else:
                        SNR = '-99'
                        MagObser = '-99'
                        ErrMagObs = '-99'
                    if DebugTF == True:
                        print(' '.join([cssband, 'band mag_simul = ', str(MagObser), '(AB mag)']))
                        print(' '.join([cssband, 'band mag_model = ', str(CatOfTile[i]['MOD_'+cssband+'_css']), '(AB mag)']))

                    outcatrowi = outcatrowi + [MagObser] + [ErrMagObs] + [SNR]

                    bandi = bandi+1
                # if len(outcatrowi) == NColOut:
            OutCat.add_row(outcatrowi)
            del outcatrowi, bandi


            if IfProgBarOn == True:
                bar.cursor.restore()  # Return cursor to start
                bar.draw(value=i+1)  # Draw the bar!

            del WinImgBands

        if IfProgBarOn == True:
            bar.cursor.restore()  # Return cursor to start
            bar.draw(value=bar.max_value)  # Draw the bar!

        ascii.write(OutCat, 'Cssos_magsim_SNR_tile_'+config['Hst2Css']['hst814file'][-12:-9]+'_'+schemecode+'.txt', format='commented_header',comment='#',overwrite=True)


    finishtime = time.time()
    print('Time Consumption:', finishtime - begintime, 's')
    print('\nFinished.\n')
