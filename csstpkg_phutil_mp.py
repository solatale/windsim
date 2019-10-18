"""
using poisson distribution to yield a random count rate; in the snr function;
"""

import os,sys
import numpy as np
import math
import copy
from sympy import solve, nsolve, Eq
from sympy import Symbol
from sympy import lowergamma,gamma
from scipy.interpolate import interp1d
from astropy.modeling import models, fitting
from astropy.nddata import Cutout2D
from astropy.io import ascii
from scipy.stats import poisson
# from scipy.special import gamma as scigamma
from astropy.modeling.models import Sersic2D, Gaussian2D
from photutils import EllipticalAperture as ellipaptr
from photutils import aperture_photometry as aptrphoto
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
import sep
import configparser
from astropy.stats import SigmaClip
from photutils import Background2D, MedianBackground


# imagewidth = 100.
psfimgwidth = 10.
dx = 1.

snrlim = 10
texp0 = 150.  # s
tread0 = 40.  # s
# npix85 = 4.*4. # pixel
npix80 = 13.  # pixel
bdet0 = 0.02  #"e s-1 pixel-1"
scatlght = 0.  # e- s-1 pixel-1
nread0 = 2    # exposures, or readouts
rn0 = 5.  # "e pixel-1"

hplk = 6.626e-27  # erg.s
cvlcm = 2.998e10  # cm/s
pixscale = 0.074  # "/pix
pixsize = 10*1e-4  # cm
aeff = math.pi*100**2  # cm^2
averfact = 0.5
# ee = 0.85
ee = 0.8

fnu0w = 3.63e-23  # W/m^2/Hz
fnu0 = 3.63e-20   # erg/cm^2/s/Hz

psffwhm = 0.15*2/1.794*1.1774 # " for gaussian
psffwhmpix = psffwhm/pixscale # csst pix, 2.66 pix
psfrrms = 0.16722/0.074/2 # Rrms of psf in pix, =1.13

Bsky_f814w = 0.0279  # e-/s/pixel
# Bsky (e-/s/pixel):
Bsky_nuv = 0.002 #0.0032
Bsky_u = 0.0182
Bsky_g = 0.1451
Bsky_r = 0.1856
Bsky_i = 0.1903
Bsky_z = 0.0970
Bsky_y = 0.0233
Bsky_w1 = 0.0091
Bsky_w2 = 0.2515
Bsky_w3 = 0.3416

defaults = {'basedir': '/work/CSSOS/filter_improve/fromimg/windextract'}
config = configparser.ConfigParser(defaults)
config.read('cssos_config.ini')

thrghdir = config['Hst2Css']['thrghdir']
# seddir = config['Hst2Css']['seddir']

filt = {'Nuv': './throughput/Nuv.txt',
        'u': './throughput/u.txt',
        'g': './throughput/g.txt',
        'r': './throughput/r.txt',
        'i': './throughput/i.txt',
        'z': './throughput/z.txt',
        'y': './throughput/y.txt',
        'WNuv': './throughput/WNuv.txt',
        'Wg': './throughput/Wg.txt',
        'Wi': './throughput/Wi.txt',
        'wfc_F814W': './throughput/wfc_F814W.txt'}

bandpos = {'Nuv': [2480., 2877., 3260.],
            'v': [3510., 3825., 4170.],
            'u': [3130., 3595., 4080.],
            'g': [3910., 4798., 5610.],
            'r': [5380., 6186., 7020.],
            'i': [6770., 7642., 8540.],
            'z': [8250., 9046., 11000.],
            'y': [9140., 9654., 11000.],
            'WNuv': [2480., 3090., 3700.],
            'Wg': [3500., 4950., 6400.],
            'Wi': [6100., 7750., 9400.],
            'wfc_F814W': [6890., 7985., 9640.],
            'skmp_v': [3500., 3870, 4180]}

backsky = {'Nuv': 0.0032,
        'u': 0.017,
        'g': 0.1403,
        'r': 0.1797,
        'i': 0.1851,
        'z': 0.1,
        'y': 0.0271,
        'WNuv': 0.0077,
        'Wg': 0.2408,
        'Wi': 0.325,
        'wfc_F814W': 0.1364,
        'lssti': 35.759163, # in one LSST pixel per second
        'skmp_v': 0.0094}

lambd = np.linspace(1000, 12000, 11001, endpoint=True)
lambdarr = np.transpose(np.vstack((lambd,lambd)))

# solute snr--count rate equation, crs is in unit of e- s-1:
def crs_solut(texp=texp0*nread0, tread=tread0*nread0, snr=snrlim, npix=npix80, bsky=0.1, bdet=bdet0, bscat=scatlght, nread=nread0,
              rn=rn0):
    crs = Symbol('x', positive=True)
    
    solut = solve(Eq((crs ** 2) * (texp ** 2) - (snr ** 2) * (crs * texp) - (snr ** 2) * npix * (
                (bsky + bdet + bscat) * texp + bdet * tread + nread * rn ** 2), 0), crs)
    # solut = solve(Eq((crs ** 2) * (texp ** 2) - (snr ** 2) * ((crs * texp) +  npix *
    #               ((bsky + bdet + bscat) * texp + npix * bdet * tread + npix * nread * rn ** 2)), 0), crs)
    # print 'CRs solution:', solut
    return solut[0]

def cr2snr(crs, texp=texp0*nread0, tread=tread0*nread0, npix=npix80, bsky=0.1, bdet=bdet0, bscat=scatlght, nread=nread0,
           rn=rn0, poiss=False):
    # print 'Input Count Rate:', crs
    if poiss==True:
        cr300 = poisson.rvs(crs*texp,size=1)
        # print 'crt 300s & sample: ', crs * texp, cr300poiss
    else:
        cr300 = crs*texp
    snrcal = cr300/(cr300+npix*(bsky+bdet+bscat)*texp+bdet*tread*npix+npix*nread*rn**2)**0.5
    # print 'area:',npix
    return snrcal



def bskycalc(isky, filtcurv, lambdarr, xa=1000, xb=12000, dx=1., aeff=aeff, hplk=hplk,\
             cvl=cvlcm, pixscale=pixscale, averfact=averfact):
    bskycurv = curvemultiply(curvemultiply(isky, filtcurv), lambdarr)
    bsky = quadrat(bskycurv, xa, xb)*1e-8*aeff/hplk/cvl*pixscale**2*averfact
    return bsky

def cr2mag(crs, hplk=hplk, cvl=cvlcm, band='g', mirrarea=aeff, ee=1.):
    xa = bandpos[band][0]
    xb = bandpos[band][2]
    essctel = ecsscntel(band, xa, xb)
    fnu = crs/ee/mirrarea*hplk/essctel/math.log(bandpos[band][2]/bandpos[band][0])
    mag = -2.5*math.log10(fnu)-48.6
    # mag = -2.5*math.log10(fnu/1e-32)+31.4
    return mag


def mag2cr(mag, band='g'):
    # mag is measured within certain aperture;
    # filtertran = np.loadtxt(filt[band])
    xa = bandpos[band][0]
    xb = bandpos[band][2]
    essctel = ecsscntel(band, xa, xb)
    fnu = 10**(-0.4*(mag+48.6))
    crs = fnu/hplk*essctel*aeff*math.log(bandpos[band][2]/bandpos[band][0])
    return crs


def fnu2cr(fnu, hplk=hplk, band='g', mirrarea=aeff):
    xa = bandpos[band][0]
    xb = bandpos[band][2]
    essctel = ecsscntel(band, xa, xb)
    crs = fnu / hplk * essctel * mirrarea * math.log(bandpos[band][2] / bandpos[band][0])
    return crs


def cr2fnu(crs, hplk=hplk, band='g', mirrarea=aeff):
    xa = bandpos[band][0]
    xb = bandpos[band][2]
    essctel = ecsscntel(band, xa, xb)
    fnu = crs / mirrarea * hplk / essctel / math.log(bandpos[band][2] / bandpos[band][0])
    return fnu


def mag2snr(mag, band='g',area=100.):
    # mag is the magnitude inside the measurement aperture;
    cr = mag2cr(mag, band=band)
    # if cr<1:
    #     print 'Count Rate < 1 ! (',cr,')'
    #     raise SystemExit
    # crpoi = np.random.poisson(cr,1)
    snri = cr2snr(cr, texp=texp0*nread0, tread=tread0*nread0, npix=area, bsky=backsky[band], bdet=bdet0, nread=nread0, rn=rn0)
    # print snri
    return snri

def magpnt(band, bsky=0.1, texp=texp0*nread0, tread=tread0*2, nread=nread0):
    xa = bandpos[band][0]
    xb = bandpos[band][2]
    essctel = ecsscntel(band, xa, xb)
    # efflamb_1 = 0.2094
    # print essctel
    crs = crs_solut(texp=texp, tread=tread, bsky=bsky, nread=nread)
    fnu = crs / ee / aeff * hplk / essctel / 0.95 / math.log(bandpos[band][2]/bandpos[band][0])
    # print 'fnu =', fnu
    # mag = -2.5 * math.log10(fnu) - 48.6
    mag = -2.5*math.log10(fnu/fnu0)
    return mag

def magext(band, bsky=0.1, npixext=217, snr=10., lumfrac=0.9):
    xa = bandpos[band][0]
    xb = bandpos[band][2]
    essctel = ecsscntel(band, xa, xb)
    # crs = crs_solut(snr=10, npix = npixR90ser, bsky=bsky) # Sersic
    # fnu = crs / 0.9 / aeff * hplk / essctel / 0.95      # Sersic
    crs = crs_solut(snr=10, npix=npixext, bsky=bsky)  # Gaussian
    fnu = crs / lumfrac / aeff * hplk / essctel / 0.95 / math.log(bandpos[band][2]/bandpos[band][0])     # Gaussian
    mag = -2.5 * math.log10(fnu) - 48.6
    return mag


def extend(curve, xmin, xmax):
    if (xmin < curve[0,0]):
        curve = np.insert(curve,[0],[xmin,0],axis=0)
    if (xmax > curve[-1,0]):
        curve = np.append(curve,[[xmax, 0]], axis=0)
    # print curve
    return curve


def interp(curve,xmin=1000.,xmax=12000.,dx=1.):
    curve = extend(curve, xmin, xmax)
    f = interp1d(curve[:,0], curve[:,1])
    xnew = np.linspace(xmin, xmax, num=(xmax-xmin)/dx+1, endpoint=True)
    ynew = f(xnew)
    newarr = np.hstack((xnew.reshape(xnew.shape[0],1),ynew.reshape(ynew.shape[0],1)))
    return newarr


def curvemultiply(curve1,curve2):
    xmin1 = curve1[0,0]
    xmax1 = curve1[-1,0]

    xmin2 = curve2[0,0]
    xmax2 = curve2[-1,0]

    xmin12 = min(xmin1,xmin2)
    xmax12 = max(xmax1,xmax2)

    newcurve1 = interp(curve1, xmin12, xmax12, dx=1.)
    newcurve2 = interp(curve2, xmin12, xmax12, dx=1.)

    newcurve12y = newcurve1[:,1]*newcurve2[:,1]
    newcurve12x = newcurve1[:,0]

    newcurve12 = np.hstack((newcurve12x.reshape(newcurve12x.shape[0],1),\
                            newcurve12y.reshape(newcurve12y.shape[0],1)))
    return newcurve12

def curvedevision(curve1, curve2):
    xmin1 = curve1[0, 0]
    xmax1 = curve1[-1, 0]

    xmin2 = curve2[0, 0]
    xmax2 = curve2[-1, 0]

    xmin12 = min(xmin1, xmin2)
    xmax12 = max(xmax1, xmax2)

    newcurve1 = interp(curve1, xmin12, xmax12, dx=1.)
    newcurve2 = interp(curve2, xmin12, xmax12, dx=1.)

    newcurve2[newcurve2[:,1]==0,1] = 1

    newcurve12y = newcurve1[:, 1] / newcurve2[:, 1]
    newcurve12x = newcurve1[:, 0]

    newcurve12 = np.hstack((newcurve12x.reshape(newcurve12x.shape[0],1),\
                            newcurve12y.reshape(newcurve12y.shape[0],1)))
    return newcurve12

def quadrat(curve, a=1000., b=12000., dx=1.):
    xab = np.where((curve[:,0]>=a) & (curve[:,0]<=b))
    integr = np.trapz(curve[xab,1], curve[xab,0], dx=1)
    # integr = np.sum(curve[xab,1])*dx
    return integr

def equiveff(curve, xa, xb, dx=1.):
    sum = quadrat(curve, a=xa, b=xb, dx=dx)
    return sum/(xb-xa)

def ecsscntel(band, xa, xb, dx=1.):
    curve = np.loadtxt('./throughput/'+band+'.txt')
    newcurve = curve[np.where(curve[:,0]==xa)[0][0]:np.where(curve[:,0]==xb)[0][0]+1,:]
    # print newcurve
    lambdas = np.linspace(xa, xb, num=(xb-xa)/dx+1, endpoint=True)
    lambdarr = np.array([lambdas,lambdas]).transpose()
    tefflamb_1 = curvedevision(newcurve,lambdarr)
    lambinv = np.array([lambdarr[:,0], 1./lambdarr[:,1]]).transpose()
    lambinvinteg = quadrat(lambinv, a=xa, b=xb, dx=1.)
    q = quadrat(tefflamb_1, a=xa, b=xb, dx=1.)/lambinvinteg
    # effmodfact = {'nuv': 0.5/0.54,
    #               'u': 0.63/0.68,
    #               'g': 0.78/0.8,
    #               'r': 0.78/0.8,
    #               'i': 0.78/0.8,
    #               'z': 0.78/0.8,
    #               'y': 0.78/0.8,
    #               'w1': 0.5/0.55,
    #               'w2': 0.78/0.8,
    #               'w3': 0.78/0.8}
    # q = q*effmodfact[band]
    # print '\nEcssc_tel =', q
    return q

class galser():

    def __init__(self, lumtot=None, amplitude=1., reff = 10., nser=1.5, ellip=0.5, theta=0.):
        self.dx = dx
        self.reff = reff # CSST pixel
        self.reffpix = self.reff / self.dx # sampling pixel
        self.radii = np.array([1]) * self.reff
        self.radiipix = self.radii / dx
        if int(self.reffpix*20) % 2 == 0:
            imagewidth = int(self.reff * 20)
        else:
            imagewidth = int(self.reff * 20) - 1
        # imagewidth = 100
        # self.radii = np.array((1, 1.18, 1.8, 2.15, 5)) * self.reff
        # print self.radii
        # radii are in pixels (indexes);
        self.nser = nser
        self.bn = 2*self.nser - 1/3. + 4/405./self.nser
        if lumtot is not None:
            self.amplitude = lumtot*(self.bn**(2*self.nser))/(
                    math.e**self.bn)/self.reff**2/2./math.pi/self.nser/gamma(
                2*self.nser)
        else:
            self.amplitude = amplitude
        self.ellip = ellip
        self.a, self.b, self.theta = self.reffpix, self.reffpix*(1-self.ellip), theta # reff=(ab)**0.5
        
        self.xpixels = (imagewidth+self.dx)/self.dx # pixel number
        self.ypixels = (imagewidth+self.dx)/self.dx # pixel number

        self.x, self.y = np.meshgrid(np.arange(self.xpixels) * self.dx,\
                                     np.arange(self.ypixels) * self.dx) # actual value
        self.image = self.sermod()[2]
        self.aperture0 = self.apers()
        
    def sermod(self):
        self.orig0 = np.array([(self.xpixels + 1) / 2. * self.dx, (self.ypixels + 1) / 2. * self.dx])
        # actual value
        self.model = Sersic2D(amplitude=self.amplitude, r_eff=self.reff,\
                              n=self.nser, x_0=self.orig0[0], y_0=self.orig0[1],\
                              ellip=self.ellip, theta=self.theta)
        # r_eff is actual value;
        # ellip = 1-b/a;
        # x_0, y_0 are actual values;
        self.image = self.model(self.x, self.y) # actual value
        return self.orig0, self.model, self.image
    
    def apers(self):
        orig0, model, image = self.sermod()
        apertures = [ellipaptr(orig0 / self.dx, r, r * (1-self.ellip), 0) for r in
                     self.radiipix] # sampling pixel
        return apertures[0]
    
    def aperphot(self):
        orig0, model, image = self.sermod()
        # self.pos0 = [((self.xpixels-1)/2., (self.ypixels-1)/2.)] # in pixel (index)
        
        # print image[int(orig0[0]/self.dx), int(orig0[1]/self.dx)]
        # image[int(orig0[0] / self.dx), int(orig0[1] / self.dx)] = image[int(orig0[0]/self.dx), int(orig0[1]/self.dx)+1]
        apertures = [ellipaptr(orig0/self.dx, r, r*(1-self.ellip), 0) for r in
                     self.radiipix]
            # Elliptical aperture(s), defined in pixel coordinates.
        result = aptrphoto(image, apertures)
        return result

    def plotmodel(self):
        from matplotlib.patches import Ellipse
        import matplotlib.gridspec as gridspec
        orig0, model, image = self.sermod()
        plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

        ax1 = plt.subplot(gs[0])
        # ax = plt.figure()
        x1 = int((self.xpixels-1)/2.-3*self.reffpix)
        x2 = int((self.xpixels-1)/2.+3*self.reffpix)
        y1 = int((self.ypixels-1)/2.-3*self.reffpix)
        y2 = int((self.ypixels-1)/2.+3*self.reffpix)
        # plt.imshow(self.image[y1:y2,x1:x2], origin='lower', interpolation='nearest',
        #            cmap='nipy_spectral')
        plt.imshow(np.log(self.image), origin='lower', interpolation='nearest', cmap='nipy_spectral')

        # ellipses = [Ellipse((3*self.reffpix,3*self.reffpix), r*2, (1-self.ellip)*r*2, self.theta,
        #                     fill=False, ls='--', lw='0.5') for r in self.radii]
        ellipses = [Ellipse(self.orig0/self.dx, r, r*(1-self.ellip),\
                            self.theta, fill=False, ls='--', lw='0.5') for r in self.radiipix]
            # ellipses' position arrays and radii are in unit of pixels
        for e in ellipses:
            # e.set_label(str(e.width/2.))
            e.set_color('darkgray')
            ax1.add_artist(e)

        # plt.xlabel('x')
        # plt.ylabel('y')
        # cbar = plt.colorbar()
        # cbar.set_label('Intensity')
        # cbar.set_ticks([-1, 0, 1, 2], update_ticks=True)
        # print self.image[int((self.xpixels-1)/2.),x1:x2]

        ax2 = plt.subplot(gs[1])
        # scalfact = (y2-y1)/self.image[int((self.xpixels-1)/2.),int((self.ypixels-1)/2.)]
        plt.plot(np.arange(self.xpixels)[x1:x2], self.image[int((self.xpixels-1)/2.),x1:x2])
        plt.show()


    def plotmodel3d(self):
        from matplotlib import cm
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x1 = int((self.xpixels-1)/2.-3*self.reff)
        x2 = int((self.xpixels-1)/2.+3*self.reff)
        y1 = int((self.ypixels-1)/2.-3*self.reff)
        y2 = int((self.ypixels-1)/2.+3*self.reff)
        # ax.plot_surface(self.x[y1:y2,x1:x2], self.y[y1:y2,x1:x2], self.image[y1:y2,x1:x2], rstride=int(self.reff/5.), cstride=int(self.reff/5.), cmap=cm.jet)
        ax.plot_surface(self.x[y1:y2, x1:x2], self.y[y1:y2, x1:x2], self.sermod()[2][y1:y2, x1:x2],\
                        cmap=cm.jet)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Flux')
        plt.show()
    
    @property
    def input_units(self):
        if self.model.x_0.unit is None:
            return None
        else:
            return {
                'x': self.model.x_0.unit,
                'y': self.model.y_0.unit}
        
class psfgauss():
    def __init__(self, fwhm=psffwhmpix, ampli=1., imgwidth = psfimgwidth):
        self.dx = dx
        self.ampli = ampli
        self.fwhm = fwhm # CSST pixel
        # self.fwhmpix = self.fwhm/dx # sampling pixel
        self.xstddev = self.fwhm/2.355
        self.ystddev = self.fwhm/2.355
        
        self.xpixels = (imgwidth+self.dx)/self.dx # sampling pixel
        self.ypixels = (imgwidth+self.dx)/self.dx # sampling pixel
        self.x, self.y = np.meshgrid(np.arange(self.xpixels) * self.dx,\
                                     np.arange(self.ypixels) * self.dx) # actual value
        # print self.x, '\n', self.y
        
    def gauss(self):
        self.orig0 = np.array([(self.xpixels + 1) / 2. * self.dx, (self.ypixels + 1) / 2. * self.dx])
        self.model = Gaussian2D(amplitude=self.ampli, x_mean=self.orig0[0], y_mean=self.orig0[1],
                                x_stddev=self.xstddev, y_stddev=self.ystddev)
        self.image = self.model(self.x, self.y) # x,y are actual values
        return self.orig0, self.model, self.image
    
    def psfphot(self):
        orig0, model, image = self.gauss()
        # print model
        self.radii = self.xpixels/2.
        # print self.radii; radii are in pixels (indexes);
        aperture = ellipaptr(orig0/self.dx, self.radii, self.radii, 0)
        # Elliptical aperture(s), defined in pixel coordinates.
        # print apertures
        result = aptrphoto(image, aperture)
        return result

def mod2data(model,imagewidth):
    # self.dx = dx
    # self.xpixels = (imagewidth + self.dx) / self.dx  # pixel number
    # self.ypixels = (imagewidth + self.dx) / self.dx  # pixel number
    # self.x, self.y = np.meshgrid(np.arange(self.xpixels) * self.dx, np.arange(self.ypixels) * self.dx)  # actual value
    xpixels = (imagewidth + dx) / dx  # pixel number
    ypixels = (imagewidth + dx) / dx  # pixel number
    x, y = np.meshgrid(np.arange(xpixels) * dx, np.arange(ypixels) * dx)  # actual value
    data = model(x, y)
    return data

# for e=(1-b/a) ellipticity:
# def aptrphot(ndarray, origcen=np.array([0,0]), reff=10, ellip=0.5):
#     # origcen = agalser.orig0
#     aperradii = np.array([1]) * reff
#     apertures = [ellipaptr(origcen / dx, r/((1-ellip)**0.5), r*((1-ellip)**0.5), 0) for r in aperradii / dx]
#     result = aptrphoto(ndarray, apertures)
#     return result

# for e=b/a ellipticity:
def aptrphot(ndarray, origcen=np.array([0,0]), reff=10, ellip=0.5):
    # origcen = agalser.orig0
    aperradii = np.array([1]) * reff
    apertures = [ellipaptr(origcen / dx, r/(ellip**0.5), r*(ellip**0.5), 0) for r in aperradii / dx]
    result = aptrphoto(ndarray, apertures)
    return result

def aptrphotfrac(galmodel, ndarray, origcen=np.array([0,0]), reff=10, ellip=0.5, frac=0.5):
    # Ie = galmodel.amplitude
    nser = galmodel.nser
    # b = 2 * nser - 1 / 3.
    Reff = galmodel.reff
    # print Ie,n,b,Reff
    #
    # rsub = Symbol('x', positive=True)
    # # rsub = b * (r / Reff) ** (1 / n)
    # LrR = 2 * math.pi * nser * Ie * math.e ** b * Reff ** 2 / b ** (2 * nser) * lowergamma(float(2 * nser), rsub)
    #
    # Ltot = 2 * math.pi * nser * Ie * math.e ** b * Reff ** 2 / b ** (2 * nser) * gamma(2 * nser)
    # # print LrR
    # # print Ltot*frac
    # rsubsolv = nsolve(LrR - frac*Ltot, rsub, 1.5)
    # print rsubsolv
    # radiusfrac = (rsubsolv/b)**nser*Reff
    # print radiusfrac
    
    if frac==0.2:
        fracr = np.array([[0.3,0.635375],[0.5,0.578546],[1,0.494633],[1.5,0.436745],[2,0.392372],[2.5,0.35663],[3,0.326915],[3.5,0.301655],[4,0.279823],[4.5,0.260707],[5,0.243793],[5.5,0.2287],[7,0.191705],[8,0.172165]])
    elif frac==0.4:
        fracr = np.array([[0.3, 0.92254], [0.5, 0.87535], [1, 0.825853], [1.5, 0.793228], [2, 0.767052], [2.5, 0.74475], [3, 0.725141],
         [3.5, 0.707551], [4, 0.691548], [4.5, 0.676835], [5, 0.663198], [5.5, 0.650474], [6, 0.638537], [7,0.616646], [8, 0.596929]])
    elif frac==0.5:
        fracr = np.array([[0.3,1.],[0.6,1.],[1.0,1.],[1.5,1.],[2.,1.],[2.5,1.],[3.,1.],[3.5,1.],[4.,1.],[4.5,1.],[5.,1.],[6.0,1.],[7.0,1.],[8.,1.]])
    elif frac==0.6:
        fracr = np.array([[0.3,1.18222],[0.6,1.1788],[1.0,1.21339],[1.5,1.25666],[2.,1.29666],[2.5,1.33386],[3.,1.36888],[3.5,1.40218],[4.,1.43408],[4.5,1.46483],[5.,1.49461],[6.0,1.55182],[7.0,1.60653],[8.,1.65931]])
    elif frac==0.7:
        fracr = np.array([[0.3,1.32115],[0.5,1.34386],[1.,1.46353],[1.5,1.57874],[2.,1.68685],[2.5,1.78994],[3.,1.88949],[3.5,1.9865],[4.,2.08168],[4.5,2.17551],[5.,2.26838],[6,2.45226],[7.,2.63497],[8.,2.81759]])
    elif frac==0.8:
        fracr = np.array([[0.3,1.48172],[0.5,1.55376],[1.,1.79659],[1.5,2.03266],[2.,2.26233],[2.5,2.48923],[3.,2.71573],[3.5,2.94339],[4.,3.17327],[4.5,3.40617],[5.,3.64268],[5.5,3.88328],[6.,4.12835],[6.5,4.37822],[7.,4.63317],[8,5.15931]])
    elif frac==0.9:
        fracr = np.array([[0.3,1.69898],[0.5,1.85846],[1.,2.33383],[1.5,2.81967],[2.,3.3198],[2.5,3.84006],[3.,4.38443],[3.5,4.95585],[4.,5.55667],[4.5,6.18893],[5.,6.85447],[5.5,7.55502],[6.,8.29226],[7.,9.88333],[8,11.6407]])
        
    fracract = np.interp(nser, fracr[:, 0], fracr[:, 1])
    radiusfrac = fracract*Reff
    
    # aperradii = np.array([1]) * radiusfrac
    
    aperture = ellipaptr(origcen / dx, radiusfrac / (ellip ** 0.5), radiusfrac * (ellip ** 0.5), 0)
    result = aptrphoto(ndarray, aperture)
    return radiusfrac, result

def sqdeg(width, height=0.):
    """
    :param width: in degree; if give only width, it is apex angle of a cone;
    :param height: in degree; if not given, this function calculates square degree of a cone;
    :return: square degree.
    """
    sr2sqdeg = (360. / 2 / math.pi) ** 2
    if height != 0.:
        sqdeg = 4*math.asin(math.sin(width/2.)*math.sin(height/2.))
    else:
        theta = width/2.
        sqdeg = 2 * math.pi * (1 - math.cos(theta/180.*math.pi)) * sr2sqdeg
    return sqdeg

def round_mask(datarr, radius, maskval, loc=(0,0)):
    # loc values are natural numbers
    arrhei,arrwid = datarr.shape
    newarr = datarr.copy()
    y, x = np.ogrid[0-loc[0]:arrhei-loc[0], 0-loc[1]:arrwid-loc[1]]
    mask = x*x+y*y > radius*radius
    newarr[mask] = maskval
    return newarr


def DataArr2Fits(datarr, outfilename, headerobj=fits.Header()):
    # hdr = fits.Header()
    hdu = fits.PrimaryHDU(data=datarr, header=headerobj)
    hdu.writeto(outfilename, overwrite=True)



class CentrlPhot:
    """
    A window image class contains properties extracted through the "sep" program.
    :parameters:
        window image; ndarray
        ID; str
    :methods:
        Bkg();
        Centract();
        KronR();
        Phot();
    :properties:
        id;
        data; ndarray
        bkg; object
        bkgrms;
        centobj; object
        data_bkg; ndarray
        data_bkgmask; ndarray
        kphotpar;
        kronr;
        mark_centr;
        centflux;
        rsserr;
    """

    def __init__(self, dataorig, id='NA'):
        self.data = np.array(dataorig, dtype='float32')  #.byteswap().newbyteorder()
        self.id = id

    def Bkg(self, idb='NA', debug=False, thresh=1.5, minarea=10, deblend_nthresh=32, deblend_cont=0.005, clean_param=1.0):
        datahei, datawid = self.data.shape
        if max(datahei,datawid) > 32*2:
            back_size = 32
        else:
            back_size = 16

        # Fit background using photutil.Background2D
        sigma_clip = SigmaClip(sigma=3.)
        bkg_estimator = MedianBackground()
        if debug==True:
            bkg_value = bkg_estimator.calc_background(self.data)
            print(bkg_value)
        bkg0 = Background2D(self.data, back_size, filter_size=3, sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, edge_method='pad', exclude_percentile=100)
        objimg = self.data - bkg0.background
        obj0, seg0 = sep.extract(objimg, thresh, err=bkg0.background_rms_median, minarea=minarea, deblend_nthresh=deblend_nthresh, deblend_cont=deblend_cont, clean_param=clean_param, segmentation_map=True)
        seg0[seg0 > 0] = 1
        objs = objimg * seg0
        bkgimg = self.data - objs
        self.bkg = Background2D(bkgimg, (back_size, back_size), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, exclude_percentile=100)

        # if debug==True:
        #     print('bkg rms =', self.bkg.background_rms_median)
        #     vmin = np.min(bkgimg)
        #     vmax = np.max(bkgimg)
        #     plt.imshow(self.bkg.background, interpolation='nearest', cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        #     plt.title(idb+' Refined Background')
        #     plt.show()

        return self.bkg

    kphotpar = 2.5

    def Centract(self, idt='NA', debug=False, thresh=1.5, err=None, minarea=10, deblend_nthresh=32, deblend_cont=0.005, clean_param=1.0):
        """
        Generate data-background image and extract Object of sources at the center.
        :param idt: ID or annotation words.
        :param debug:
        :param thresh: Threshold pixel value for detection. If an err array is not given, this is interpreted as an absolute threshold. If err is given, this is interpreted as a relative threshold.
        :param err: float or ndarray, optional.
        :param minarea:
        :param deblend_nthresh:
        :param deblend_cont:
        :param clean_param:
        :return:
        """

        self.data_bkg = self.data - self.bkg.background
        if debug==True:
            print('Bkg.globalrms =', self.bkg.background_rms_median)


        objects, segarr = sep.extract(self.data_bkg, thresh, err=self.bkg.background_rms_median, minarea=minarea, deblend_nthresh=deblend_nthresh, deblend_cont=deblend_cont, clean_param=clean_param, segmentation_map=True)
        if debug == True:
            plt.imshow(segarr, interpolation='nearest', cmap='gray', origin='lower')
            plt.title(idt+' extracted segments')
            plt.show()

        if debug == True:
            # plot background-subtracted image
            fig, ax = plt.subplots()
            m, s = np.mean(self.data_bkg), np.std(self.data_bkg)
            ax.imshow(self.data_bkg, interpolation='nearest', cmap='gray', vmin=m - s, vmax=m + 2*s, origin='lower')
            # plot an ellipse for each object
            for objecti in objects:
                kronrdets= sep.kron_radius(self.data_bkg, objecti['x'], objecti['y'], objecti['a'], objecti['b'], objecti['theta'], 4.0)[0]
                e = Ellipse(xy=(objecti['x'], objecti['y']), width=objecti['a']*kronrdets*CentrlPhot.kphotpar*2, height=objecti['b']*kronrdets*CentrlPhot.kphotpar*2, angle=objecti['theta'] * 180. / np.pi)
                e.set_facecolor('none')
                e.set_edgecolor('red')
                ax.add_artist(e)
            plt.title(str(idt)+"'s detected objects")
            plt.show()


        # print(len(objects))
        if len(objects)>=1:
            objxy = np.array([objects['y'], objects['x']]).transpose()
            # print(objxy)
            tocenter = np.abs(objxy-(np.array(self.data_bkg.shape)/2.+0.5))
            distances = (tocenter[:,0]**2+tocenter[:,1]**2)**0.5
            if debug == True:
                print('Distances to center:\n',distances)
            if np.min(distances) > 4:
                self.centobj = np.nan
                if debug == True:
                    print('objects all deviate center')
            else:
                idx = np.argmin(distances)
                self.centobj=objects[idx]
        else:
            if debug == True:
                print('no object detected')
            self.centobj = np.nan

        #     idx = np.argmin(tocenter[:,0]**2+tocenter[:,1]**2)
        #     # idx = np.argmin(tocenter[:,0]*tocenter[:,1])
        #     self.centobj=objects[idx]
        #     # if ((abs(self.centobj['x'] - self.data_bkg.shape[1] / 2. + 1) > 4) & (abs(self.centobj['y'] - self.data_bkg.shape[0] / 2. + 1) > 4)):
        #     if (((self.centobj['x']-(self.data_bkg.shape[1]/2.+0.5))**2+(self.centobj['y']-(self.data_bkg.shape[0]/2.+0.5))**2)**0.5 > 4):
        #         if debug == True:
        #             print('multiple objects all deviate center')
        #             self.centobj = np.nan
        #     # print('An object extracted')
        # elif len(objects)==1:
        #     self.centobj = objects[0]
        #     # print(abs(self.centobj['x']-self.data_bkg.shape[1]/2.+1))
        #     # print(abs(self.centobj['y']-self.data_bkg.shape[0]/2.+1))
        #     # if ((abs(self.centobj['x']-self.data_bkg.shape[1]/2.+1)>4) & (abs(self.centobj['y']-self.data_bkg.shape[0]/2.+1)>4)):
        #     if (((self.centobj['x']-((self.data_bkg.shape[1]-1)/2.+1))**2+(self.centobj['y']-((self.data_bkg.shape[0]-1)/2.+1))**2)**0.5 > 4):
        #         if debug == True:
        #             print('single object deviates center')
        #         self.centobj = np.nan
        #     # print('An object extracted')
        # else:
        #     if debug == True:
        #         print('no object detected')
        #     self.centobj = np.nan

        if self.centobj is not np.nan:
            censeg = segarr[int(self.centobj['y']),int(self.centobj['x'])]
            self.mark_centr = copy.deepcopy(segarr)
            # convert mark_centr setting central object with 1, other with 0
            self.mark_centr[self.mark_centr != censeg] = 0
            self.mark_centr[self.mark_centr == censeg] = 1

            self.mask_other = copy.deepcopy(segarr)
            # convert mask_other, setting central object and background with 1, other with 0
            self.mask_other[self.mask_other==0] = np.max(segarr)+1
            self.mask_other[self.mask_other==censeg] = np.max(segarr)+1
            # self.mask_other[self.mask_other==0] = max(segarr)+1
            # self.mask_other[self.mask_other==censeg] = max(segarr)+1
            self.mask_other[self.mask_other<=np.max(segarr)] = 0
            self.mask_other[self.mask_other>0] = 1

            self.maskall = copy.deepcopy(segarr)
            # maskall is for background estimation, objects are set to 0, bkg is set to 1
            self.maskall[self.maskall==0] = np.max(segarr)+1
            self.maskall[self.maskall<(np.max(segarr)+1)] = 0
            self.maskall[self.maskall>0] = 1

            self.data_bkg_masked = self.data_bkg * self.mask_other # self.mark_centr

            if debug==True:
                plt.imshow(self.mask_other, interpolation='nearest', cmap='gray', origin='lower')
                plt.title(idt+"'s Mask")
                plt.show()


    def KronR(self, idk='NA', debug=False, mask_bool=False):

        if mask_bool is False:
            data = self.data_bkg
        elif mask_bool is True:
            data = self.data_bkg_masked
        else:
            print("'mask_bool' type error:\nmask_bool parameter is boolean.")

        try:
            self.kronr, krflag = sep.kron_radius(data, self.centobj['x'], self.centobj['y'], \
                                             self.centobj['a'], self.centobj['b'], self.centobj['theta'], 4) #, mask=self.mark_centr,maskthresh=0)
        except Exception as e:
            print(self.centobj)
            print(e)

        if debug==True:
            print('a b kronri:', self.centobj['a'], self.centobj['b'], self.kronr)
            # print(' '.join(['Kron Radius: ', str(self.kronri), '(pix)']))

            # Plot Cleaned object
            fig, ax = plt.subplots()
            m, s = np.mean(data), np.std(data)
            ax.imshow(data, interpolation='nearest', cmap='gray', origin='lower', vmin=m-2*s, vmax=m+3*s)
            # plot an ellipse for each object
            e = Ellipse(xy=(self.centobj['x'], self.centobj['y']), width=CentrlPhot.kphotpar*self.kronr*self.centobj['a']*2, height=CentrlPhot.kphotpar*self.kronr*self.centobj['b']*2, angle=self.centobj['theta'] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('blue')
            ax.add_artist(e)
            plt.title(idk+"'s central object & aperture photometry")
            plt.show()

        return self.kronr

    def EllPhot(self, kronr, debug=False, mask_bool=False):

        if mask_bool == False:
            data = self.data_bkg
        elif mask_bool == True:
            data = self.data_bkg_masked
        else:
            print("'mask_bool' type error:\nmask_bool parameter is boolean.")

        self.centflux, centfluxerr, flag = sep.sum_ellipse(data, self.centobj['x'], self.centobj['y'], self.centobj['a'], self.centobj['b'], self.centobj['theta'], CentrlPhot.kphotpar * kronr, subpix=1) #, mask=self.mark_centr,maskthresh=0.0)
        # Here, self.centflux is electron counts, not fnu
        if self.centflux <= 0:
            return np.nan, np.nan
        npix = math.pi*(self.centobj['a']*CentrlPhot.kphotpar*kronr)*(self.centobj['b']*CentrlPhot.kphotpar*kronr)
        self.rsserr = np.sqrt(self.centflux+npix*self.bkg.background_rms_median**2)
        if debug==True:
            print('Flux:',self.centflux,'RSSErr:', self.rsserr)

        return self.centflux, self.rsserr



def septractSameAp(dataorig, stackphot, object_det, kronr_det, mask_det=0, debug=False, annot='', thresh=2., minarea=5, deblend_nthresh=32, deblend_cont=0.005, clean_param=1.0):
    # extract objects using "sep" program.
    # if np.sum(mask_det) < 0.1:
    #     mask_det = np.zeros(dataorig.shape)

    data = np.array(dataorig, dtype='float32')  #.byteswap().newbyteorder()
    datahei, datawid = data.shape

    if max(datahei,datawid) > 32*2:
        back_size = 32
    else:
        back_size = 16

    # Background fitted by photutils.Background2D
    sigma_clip = SigmaClip(sigma=3.)
    bkg_estimator = MedianBackground()
    bkg0 = Background2D(data, (back_size,back_size), filter_size=(3, 3),
                        sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, exclude_percentile=100)
    objimg = data - bkg0.background
    obj0, seg0 = sep.extract(objimg, thresh, err=bkg0.background_rms_median, minarea=minarea, deblend_nthresh=deblend_nthresh, deblend_cont=deblend_cont, clean_param=clean_param, segmentation_map=True)
    seg0[seg0>0] = 1
    objs = objimg * seg0
    bkgimg = data - objs
    bkg = Background2D(bkgimg, (back_size,back_size), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, exclude_percentile=100)

    # if debug==True:
    #     print('bkg mean =', bkg.background_median)
    #     print('bkg rms =', np.median(bkg.background_rms_median))
    #     vmin = np.min(bkg.background)
    #     vmax = np.max(bkg.background)
    #     plt.imshow(bkg.background, interpolation='nearest', cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    #     plt.title(annot+' Refined Background')
    #     plt.show()

    data_sub = data - bkg.background
    backstatarr = np.ma.array(data_sub, mask=(stackphot.maskall<1))
    data_sub = data_sub - np.mean(backstatarr)

    if np.sum(mask_det)>1:
        data_sub = data_sub*mask_det


    # centmasked = copy.deepcopy(mask_det)
    # print(backstatarr)
    if debug==True:
        backstatarr2 = np.ma.array(data_sub, mask=(mask_det+stackphot.mark_centr)>1.01)
        print('Mean data_sub background:', np.mean(backstatarr2))
        backstatarr[backstatarr.mask]=0
        m, s = np.mean(backstatarr), np.std(backstatarr)
        plt.imshow(backstatarr, interpolation='nearest', cmap='gray', vmin=m - s, vmax=m + 2*s, origin='lower')
        plt.show()

    kphotpar = 2.5

    if debug==True:
        # Plot Cleaned object
        fig, ax = plt.subplots()
        m, s = np.mean(data_sub), np.std(data_sub)
        im = ax.imshow(data_sub, interpolation='nearest', cmap='gray', vmin=m - s, vmax=m + 3*s, origin='lower')
        # plot an ellipse for each object
        e = Ellipse(xy=(object_det['x'], object_det['y']), width=kphotpar*kronr_det * object_det['a']*2, height=kphotpar*kronr_det * object_det['b']*2, angle=object_det['theta'] * 180. / np.pi)
        e.set_facecolor('none')
        e.set_edgecolor('blue')
        ax.add_artist(e)
        plt.title(annot+"-band photometry")
        plt.show()

    # kphot_autopar = np.array([kphotpar])
    flux, fluxerr, flag = sep.sum_ellipse(data_sub, object_det['x'], object_det['y'], object_det['a'], object_det['b'], object_det['theta'], kphotpar * kronr_det, subpix=1) #, mask=mask_det ,maskthresh=0)
    # flux = np.sum(data_sub)
    if flux < 0:
        flux=0
    # Here, flux is ADU counts, not fnu
    npix = math.pi*(object_det['a']*kphotpar*kronr_det)*(object_det['b']*kphotpar*kronr_det)
    rsserr = np.sqrt(flux+npix*bkg.background_rms_median**2)
    # rsserr = np.sqrt(flux+npix*bkg.globalrms**2)
    if debug == True:
        # print(annot,'Flux:',flux,'FluxErr:', rsserr)
        print(annot, 'bkg rms:', bkg.background_rms_median)
    if rsserr == 0:
        return np.nan, np.nan
    else:
        return flux, rsserr


def CRRatio(cssosband='i', nread=2, hstband='wfc_F814W', mag814=25, magcss=25):
    Ecssci = ecsscntel(cssosband, bandpos[cssosband][0], bandpos[cssosband][2])
    Ef814tel = ecsscntel(hstband, bandpos[hstband][0], bandpos[hstband][2])
    areahst = math.pi * (1.2 ** 2 - 0.5016 ** 2)
    areacsst = math.pi * 1.
    expcss = 150*nread
    ExpHst = 2028

    crratio = areahst * Ef814tel * ExpHst * 10**(-0.4*(mag814-magcss)) * \
              math.log(bandpos['wfc_F814W'][2]/bandpos['wfc_F814W'][0]) / \
              areacsst / Ecssci / expcss / \
              math.log(bandpos[cssosband][2] / bandpos[cssosband][0])

    return crratio[0]


def simag(modmag, band, texp=300):
    if modmag==np.inf:
        return -99
    cr = mag2cr(modmag, band=band)
    count = np.random.poisson(lam=cr * texp, size=1)  #poisson.rvs(cr * texp, size=1)
    if count<=0:
        return -99
    # print(cr*texp,count)
    magsim = cr2mag(count/texp,band=band)
    # print(modmag,magsim)
    return magsim


def cnt2fnu(count, band, texp=300):
    # convert electron number counts to flux (fnu)
    # also appliable for flux error
    aeff = math.pi * 100 ** 2  # cm^2
    hplk = 6.626e-27  # erg.s
    cr = count/texp
    xa = bandpos[band][0]
    xb = bandpos[band][2]
    essctel = ecsscntel(band, xa, xb)
    fnu = cr/aeff/essctel/math.log(xb/xa)*hplk
    return fnu


def moments(objarr, gaussmod, order=2):
    ftot = np.sum(objarr)

    xsize = objarr.shape[1]
    ysize = objarr.shape[0]
    y, x = np.mgrid[:ysize, :xsize]

    xmom1 = np.sum(objarr * x) / ftot  # 1st order moment
    ymom1 = np.sum(objarr * y) / ftot

    if order==1:
        return xmom1, ymom1

    if order==2:

        initarr = gaussmod(x, y)
        suminit = np.sum(initarr)
        gaussarr = gaussmod(x, y)
        # print gaussarr

        ftotweit = np.sum(objarr * gaussarr)
        xmom2 = np.sum(objarr * gaussarr * ((x-xmom1)**2)) / ftotweit  # 2nd order moment
        ymom2 = np.sum(objarr * gaussarr * ((y-ymom1)**2)) / ftotweit
        xymom = np.sum(objarr * gaussarr * ((x-xmom1)*(y-ymom1))) / ftotweit
        e1 = (xmom2-ymom2)/(xmom2+ymom2)
        e2 = 2*xymom/(xmom2+ymom2)
        return xmom2, ymom2, e1, e2, gaussarr


def modelgauss(objarr, wx_sig, wy_sig, x_0=0, y_0=0):
    xsize = objarr.shape[1]
    ysize = objarr.shape[0]
    y, x = np.mgrid[:ysize, :xsize]

    gaussini = models.Gaussian2D(amplitude=np.max(objarr)*0.5, x_mean=x_0, y_mean=y_0, x_stddev=wx_sig, y_stddev=wy_sig)
    gaussini.amplitude.fixed = False
    gaussini.x_stddev.fixed = True
    gaussini.y_stddev.fixed = True
    fit_g = fitting.LevMarLSQFitter()
    gaussmod = fit_g(gaussini, x, y, objarr)

    # return gaussmod
    return gaussmod


def err2snr(magerr):
    snr = 1./(10**(0.4*magerr)-1)
    return snr


class psfgauss():
    def __init__(self, fwhm=1, ampli=1., imgwidth=10):
        self.ampli = ampli
        self.fwhm = fwhm  # CSST pixel
        # self.fwhmpix = self.fwhm/dx # sampling pixel
        self.xstddev = self.fwhm / 2.355
        self.ystddev = self.fwhm / 2.355

        self.xpixels = imgwidth  # sampling pixel
        self.ypixels = imgwidth  # sampling pixel
        self.x, self.y = np.meshgrid(np.arange(self.xpixels), np.arange(self.ypixels))  # actual value

    def gauss(self):
        self.orig0 = np.array([(self.xpixels - 1) / 2., (self.ypixels - 1) / 2.])
        self.model = models.Gaussian2D(amplitude=self.ampli, x_mean=self.orig0[0], y_mean=self.orig0[1], x_stddev=self.xstddev, y_stddev=self.ystddev)
        self.image = self.model(self.x, self.y)  # x,y are actual values
        normimg = self.image/np.sum(self.image)
        return self.orig0, self.model, normimg


def ImgConvKnl(fwhmorig, fwhmgoal, pixscale, widthinfwhm=4):
    """
    Generating convolving PSF NDarray image to match FWHM from original to goal.
    The outcoming kernel image width is <fwhm * widthinfwhm + 1>
    :param fwhmorig: FWHM (in arcsec) of the original image
    :param fwhmgoal: FWHM (in arcsec) of the goal to achieve
    :param pixscale: pixel scale in unit of arcsec/pixel, exp. 0.03
    :param widthinfwhm: multiple factor to FWHM of kernel, used to determine image width of PSF kernel (fwhm * widthinfwhm + 1)
    :return: NDarray of the additional PSF kernel
    """

    FwhmKnl = int((fwhmgoal ** 2 - fwhmorig ** 2) ** 0.5 / pixscale)  # in pixel
    PsfImgWidth = FwhmKnl * widthinfwhm + 1
    ConvKernelNormal = psfgauss(fwhm=FwhmKnl, ampli=1, imgwidth=PsfImgWidth).gauss()[2]
    DataArr2Fits(ConvKernelNormal, 'ExtraKernel.fits')

    return ConvKernelNormal


def ImgMosaic(Infileprefix, NDivide=1):

    """
    Read sub-images and tile them into a Mosaiced image.
    :param Infileprefix: file name prefixes for the subimages
    :param NDivide: number of dividing of the large image for all the axis. Exp, 4
    :return: Mosaiced image ndarray
    """

    xsubwths = []; ysubwths = []

    for i in range(NDivide):
        ijidx = '0' + str(i)
        tilehdr = fits.open(Infileprefix + ijidx + '.fits')[0].header
        xsubwths.append(tilehdr['NAXIS1'])
    for j in range(NDivide):
        ijidx = str(j) + '0'
        tilehdr = fits.open(Infileprefix + ijidx + '.fits')[0].header
        ysubwths.append(tilehdr['NAXIS2'])

    Imgwidth=sum(xsubwths); Imgheight=sum(ysubwths)

    MosaicImg = np.zeros((int(Imgheight), int(Imgwidth)))

    xidx = [0]+list(np.cumsum(xsubwths)); yidx = [0]+list(np.cumsum(ysubwths))

    for j in range(NDivide):  # j is Y-axis
        for i in range(NDivide):  # i is X-axis
            ijidx = str(j) + str(i)
            tiledata = fits.open(Infileprefix + ijidx + '.fits')[0].data
            MosaicImg[yidx[j]:yidx[j+1], xidx[i]:xidx[i+1]] = tiledata
            del tiledata

    os.system('rm ' + Infileprefix + '*.fits')
    return MosaicImg


def ImgConv(imgdata, kernldata, NDivide=1, NZoomIn=1, NZoomOut=1, SubPrefix='conv_sub_'):

    """
    Convolve two images, including capability of zooming in before convolution.
    (Zoom image and PSF kernel both)
    :param imgdata: NDarray of image
    :param kernldata: NDarray of the convolving kernel; have already zoomed in;
    :param NDivide: number of dividing of the large image for both axis. Exp, 2
    :param NZoomIn: zoom in factor, only apply to image
    :param NZoomOut: zoom out factor, yielding the output convolved image
    :param SubPrefix: prefix of filenames of sub-images
    :return: True
    """

    from scipy import signal
    import scipy.ndimage as spimg

    PsfImgWidth = kernldata.shape[0]
    ImgHeight, ImgWidth = imgdata.shape

    xn = [int(ImgWidth/NDivide*i) for i in range(int(NDivide)) ] + [ImgWidth]
    yn = [int(ImgHeight/NDivide*i) for i in range(int(NDivide))] + [ImgHeight]
    print(xn,yn)

    for j in range(len(yn) - 1):  # j is Y-axis
        for i in range(len(xn) - 1):  # i is X-axis

            ijidx = str(j) + str(i)

            xlft = max(0, xn[i] - int((PsfImgWidth-1)/NZoomIn))
            xrgt = min(xn[i + 1] + int((PsfImgWidth-1)/NZoomIn), ImgWidth)
            ydn = max(0, yn[j] - int((PsfImgWidth-1)/NZoomIn))
            yup = min(yn[j + 1] + int((PsfImgWidth-1)/NZoomIn), ImgHeight)

            # print('\nsubimg_', ijidx, ': [', ydn, ':', yup, ',', xlft, ':', xrgt, ']', sep='')
            SubArr = imgdata[ydn:yup, xlft:xrgt]

            # substddev = np.std(SubArr)
            # submean = np.median(SubArr)
            # plt.figure()
            # plt.imshow(SubArr, vmin=submean-0.2*substddev, vmax=submean+substddev)
            # plt.show()

            print('Subimage zooming '+ijidx)
            SubZoomArr = spimg.zoom(SubArr, NZoomIn, order=0) / (NZoomIn ** 2)
            # DataArr2Fits(subSclZoomArr, 'sub_scale_zoom_'+ijidx+'.fits')
            # del SubArr
            # KernlZoomArr = spimg.zoom(kernldata, NZoomIn, order=0) / (NZoomIn ** 2)

            print('Zoomed subimage Convolving Gaussian kernel')
            # SubZoomConvArr = signal.fftconvolve(SubZoomArr, KernlZoomArr, mode='same')
            SubZoomConvArr = signal.fftconvolve(SubZoomArr, kernldata, mode='same')
            # del SubZoomArr


            print('Cutting PsfImgWidth edges')
            subwid = SubZoomConvArr.shape[1]
            subhei = SubZoomConvArr.shape[0]
            cutlft = PsfImgWidth-1 #* NZoomIn
            cutrgt = subwid - (PsfImgWidth-1) #* NZoomIn
            cutdn = PsfImgWidth-1 #* NZoomIn
            cutup = subhei - (PsfImgWidth-1) #* NZoomIn
            if i == 0:
                cutlft = 0
            elif i == int(NDivide) - 1:
                cutrgt = subwid
            if j == 0:
                cutdn = 0
            elif j == int(NDivide) - 1:
                cutup = subhei
            print('Clip subimage_%s: [%d:%d,%d:%d]' % (ijidx, cutdn, cutup, cutlft, cutrgt))
            CutSubZoomConvArr = SubZoomConvArr[cutdn:cutup, cutlft:cutrgt]
            # DataArr2Fits(subSclZoomConvArr, 'sub_scale_zoom_conv_'+ijidx+'.fits')
            SubArrHei, SubArrWid = int(CutSubZoomConvArr.shape[0] / NZoomOut), int(CutSubZoomConvArr.shape[1] / NZoomOut)
            ConvSubZoomOutArr = CutSubZoomConvArr.reshape((SubArrHei, NZoomOut, SubArrWid, NZoomOut)).mean(3).mean(1) * NZoomOut ** 2
            DataArr2Fits(ConvSubZoomOutArr, SubPrefix + ijidx + '.fits')
            # del SubZoomConvArr, CutSubZoomConvArr, ConvSubZoomOutArr

    convimg = ImgMosaic(SubPrefix, NDivide=NDivide)


    return convimg


def DataArr2Fits(datarr, outfilename, headerobj=fits.Header()):
    # hdr = fits.Header()
    hdu = fits.PrimaryHDU(data=datarr, header=headerobj)
    hdu.writeto(outfilename, overwrite=True)


def NoiseArr(shape, loc=0, scale=1, func='normal'):
    """
    Genarate noise NDarray
    :param shape: NDarray shape puple
    :param loc: mean value
    :param scale: noise scale
    :param func: 'normal' or 'poisson', right now; distribution function of noises
    :return: NDarray of noise image
    """
    if func=='poisson':
        noisearr = np.random.poisson(lam=loc,size=shape)
    elif func=='normal':
        noisearr = np.random.normal(loc=loc, scale=scale, size=shape)
    return noisearr

def magab2fnu(magab):
    fnu = (3.63078e-20)*10**(-0.4*magab)  # in ergs/cm^2/s/Hz
    return fnu

def mag2flam(magab, lambda0):
    """
    To convert AB magnitude to flux in f_lambda
    :param magab: AB magnitude, scalar or an array
    :param lambda0: wavelength in angstrom, scalar or an array
    :return: f_lambda
    """
    fnu = (3.63078e-20)*10**(-0.4*magab)  # in ergs/cm^2/s/Hz
    flam = fnu*3e18/(lambda0**2)  # in ergs/cm^2/s/A
    return flam


def readsed(filename):
    """
    read a galaxy SED from a .spec file output from LePhare.
    nfilters is the number of filters in the configure file of LePhare.
    return an ndarray.
    """
    specfile = open(filename,'r')
    header = []
    for i in range(13):
        header.append(specfile.readline())
    nfilters = int(header[3].split()[1])
    galsednrow = int(header[7].split()[1])
    skiprows = 13+nfilters
    specfile.close()
    fitsed=np.loadtxt(filename, skiprows=skiprows, max_rows=galsednrow)
    return fitsed


def NeObser(sedarr, cssband, exptime, telarea):
    """
    calculate number of electrons by multiplying SED and throughput curves.
    sedarr and thrarr are not necessory to be sampled evenly.
    sedarr is in f_lambda(/A);
    throughput array is in T_lambda(/A).
    """
    # thrputdir = '/work/CSSOS/filter_improve/fromimg/windextract/throughput/'
    thrput = np.loadtxt(thrghdir+cssband+'.txt')
    sedfine = interp(sedarr, xmin=bandpos[cssband][0], xmax=bandpos[cssband][2], dx=1)
    thrputfine = interp(thrput, xmin=bandpos[cssband][0], xmax=bandpos[cssband][2], dx=1)
    obserspec = sedfine
    obserspec[:,1] = np.multiply(sedfine[:,1], thrputfine[:,1])
    ToBeInteg = obserspec[:,1]*obserspec[:,0]*(1e-8)  # to be integrated
    Integ = np.trapz(ToBeInteg, x=obserspec[:,0], dx=1.)  # integration per second*cm^2
    Ne_obs = Integ*exptime*telarea/hplk/cvlcm  # total electrons recieved
    return Ne_obs


def Ne2MagAB(NeDet, cssband, exptime, telarea):
    """
    calculate AB magnitude from detected electron number, assuming f_nu is constantly.
    thrarr is not necessory to be sampled evenly.
    throughput array is in T_lambda(/A).
    """
    # thrputdir = '/work/CSSOS/filter_improve/fromimg/windextract/throughput/'
    thrput = np.loadtxt(thrghdir+cssband+'.txt')
    thrfine = interp(thrput, xmin=bandpos[cssband][0], xmax=bandpos[cssband][2], dx=1)
    ToBeInteg = thrfine[:,1]/thrfine[:,0]  # to be integrated
    Integ = np.trapz(ToBeInteg, x=thrfine[:,0], dx=1.)
    fnu = NeDet*hplk/exptime/telarea/Integ  # fnu in ergs/Hz/s/cm^2
    magab = -2.5*math.log10(fnu)-48.6
    return magab


def MagAB_Zero(Gain, cssband, exptime, telarea):
    """
    calculate AB magnitude zero point from 1 ADU, assuming f_nu is constantly as 1 ergs/Hz/s/cm^2.
    thrarr is not necessory to be sampled evenly.
    throughput array is in T_lambda(/A).
    """
    # thrputdir = '/work/CSSOS/filter_improve/fromimg/windextract/throughput/'
    thrput = np.loadtxt(thrghdir+cssband+'.txt')
    thrfine = interp(thrput, xmin=bandpos[cssband][0], xmax=bandpos[cssband][2], dx=1)
    ToBeInteg = thrfine[:,1]/thrfine[:,0]  # to be integrated
    Integ = np.trapz(ToBeInteg, x=thrfine[:,0], dx=1.)
    fnu0 = 1.0*Gain*hplk/exptime/telarea/Integ  # fnu in ergs/Hz/s/cm^2
    magab0 = -2.5*math.log10(fnu0)-48.6
    return magab0


def CRValTrans(fitsheader,iniPS,finPS):
    """
    To transform WCS frame for a image of different pixel scale.
    Condition: images before and after convolution/cutting are all relative to the origion(0,0) point.
    """

    crpix1 = fitsheader['CRPIX1']
    crpix2 = fitsheader['CRPIX2']
    # crval1 = fitsheader['CRVAL1']
    # crval2 = fitsheader['CRVAL2']
    cd1_1 = fitsheader['CD1_1']
    cd1_2 = fitsheader['CD1_2']
    cd2_1 = fitsheader['CD2_1']
    cd2_2 = fitsheader['CD2_2']

    crpix1_fin = (crpix1-0.5)*iniPS/finPS+0.5
    crpix2_fin = (crpix2-0.5)*iniPS/finPS+0.5
    cd1_1_fin = cd1_1/iniPS*finPS
    cd1_2_fin = cd1_2/iniPS*finPS
    cd2_1_fin = cd2_1/iniPS*finPS
    cd2_2_fin = cd2_2/iniPS*finPS

    fitsheader_fin = fitsheader
    fitsheader_fin['CRPIX1'] = crpix1_fin
    fitsheader_fin['CRPIX2'] = crpix2_fin
    fitsheader_fin['CD1_1'] = cd1_1_fin
    fitsheader_fin['CD1_2'] = cd1_2_fin
    fitsheader_fin['CD2_1'] = cd2_1_fin
    fitsheader_fin['CD2_2'] = cd2_2_fin

    return fitsheader_fin


def windcut(cssimg, cataline):
    """
    Cut a window of the object as objwind
    :param cssimg: the image from which the window should be cutted from
    :param posits: a tuple of the central position of the window
    :param absizes: a tuple of the A,B size in pixel of the central source
    :return: a window Object for the source
    """
    # theta = cataline['theta_image']
    xyposits = tuple(cataline['ximage', 'yimage'])
    absizes = tuple(cataline['a_image_css', 'b_image_css'])
    # cutwidrad = int((a*math.cos(theta/180.*math.pi)+b*abs(math.sin(theta/180.*math.pi)))*5)
    # cutheirad = int((a*abs(math.sin(theta/180.*math.pi))+b*math.cos(theta/180.*math.pi))*5)
    cutwid = int(max(absizes)*20)+1
    cuthei = cutwid

    if min(cutwid, cuthei) < 16:
    # size too small
        return None

    objwind0 = Cutout2D(cssimg, xyposits, (cuthei, cutwid), mode='trim')
    # objwind0.data = objwind0.data.copy(order='C')
    objwind0.data = objwind0.data.byteswap().newbyteorder()
    windback = sep.Background(objwind0.data, bw=16, bh=16)
    objwind = Cutout2D(cssimg, xyposits, (cuthei, cutwid), mode='partial', fill_value=windback.globalback)

    return objwind


def PlotObjWin(objwind, catline):
    # Plot each object's image
    objwinshape = objwind.shape
    x = np.linspace(0, objwinshape[1] - 1, objwinshape[1])
    y = np.linspace(0, objwinshape[0] - 1, objwinshape[0])
    xx, yy = np.meshgrid(x, y)

    mean = np.median(objwind.data)
    stddev = np.std(objwind.data)

    fig = plt.figure()
    ax = fig.gca()
    # ax = fig.gca(projection='3d')
    # ax.set_aspect(1.0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.imshow(objwind.data, vmin=0 - stddev, vmax=mean + 3 * stddev, cmap=plt.cm.cividis, origin='lower', )
    # ax.plot_surface(xx, yy, objwind, cmap=plt.cm.cividis, alpha=0.7)
    titlewords = 'ID:' + str(catline['IDENT']) + \
                 ' RA:' + str(catline['RA']) + \
                 ' DEC:' + str(catline['DEC']) + \
                 ' FWHM:' + str(catline['fwhm_image_css']) + '\n' + \
                 'A:' + str(catline['a_image_css']) + \
                 ' B:' + str(catline['b_image_css']) + \
                 ' Theta:' + str(catline['theta_image'])
    plt.title(titlewords)
    plt.show()


def PlotKronrs(image, SourceObj):
    print('KronR =', SourceObj.kronr)
    print('ADU =', SourceObj.centflux, 'e-')
    fig, ax = plt.subplots()
    ax.imshow(image, interpolation='nearest', cmap='gray', origin='lower')
    # plot an ellipse for each object
    e = Ellipse(xy=(SourceObj.centobj['x'], SourceObj.centobj['y']), width=2 * SourceObj.kronr * SourceObj.centobj['a'] * 2,
                height=2 * SourceObj.kronr * SourceObj.centobj['b'] * 2, angle=SourceObj.centobj['theta'] * 180. / np.pi)
    e.set_facecolor('none')
    e.set_edgecolor('blue')
    ax.add_artist(e)
    plt.title(SourceObj.id+' Mask')
    plt.show()

def sed2magab_energy_zero(cssband):
    """
    calculate AB magnitude zero point with a throughput curve corresponds to energy, assuming f_nu is constantly as 1.
    mag_AB = -2.5*lg(Integ(f_lambda*T_lambda*d_lambda)) + mag_AB_zero
    mag_AB_zero = -2.5*lg(Integ(1*c/lambda^2*T_lambda*d_lambda))-48.6
    mag_AB = -2.5*lg(Integ(f_lambda*T_lambda*d_lambda)/Integ(1*c/lambda^2*T_lambda*d_lambda))-48.6
    thrarr is not necessory to be sampled evenly.
    throughput array is in T_lambda(/A).
    """
    # thrputdir = '/work/CSSOS/filter_improve/fromimg/windextract/throughput/'
    thrput = np.loadtxt(thrghdir + cssband + '.txt')
    thrfine = interp(thrput, xmin=bandpos[cssband][0], xmax=bandpos[cssband][2], dx=1)
    ToBeInteg = thrfine[:, 1] / (thrfine[:, 0] ** 2)  # to be integrated
    Integ = np.trapz(ToBeInteg, x=thrfine[:, 0], dx=1.)
    area = 1.0 * 3e18 * Integ  # fnu in ergs/Hz/s/cm^2
    magab0 = 2.5 * math.log10(area) - 48.6
    return magab0

def sed2magab_energy(modsed, cssband, magzero):
    magzb0 = MagAB_Ener_Zero(cssband)
    thrput = np.loadtxt(thrghdir + cssband + '.txt')
    thrfine = interp(thrput, xmin=bandpos[cssband][0], xmax=bandpos[cssband][2], dx=1)
    sedfine = interp(modsed, xmin=bandpos[cssband][0], xmax=bandpos[cssband][2], dx=1)
    ToBeInteg = sedfine[:, 1] * thrfine[:, 1]
    flux = np.trapz(ToBeInteg, x=thrfine[:, 0], dx=1.)
    magab = -2.5 * math.log10(flux) + magzero
    return magab

def sed2magab_photon_zero(cssband):
    """
    calculate AB magnitude zero point with a throughput curve corresponds to number of photons, assuming f_nu is constantly as 1.
    mag_AB = -2.5*lg(Integ(f_lambda*T_lambda*d_lambda)) + mag_AB_zero
    mag_AB_zero = -2.5*lg(Integ(1*c/lambda^2*T_lambda*d_lambda))-48.6
    mag_AB = -2.5*lg(Integ(f_lambda*T_lambda*d_lambda)/Integ(1*c/lambda^2*T_lambda*d_lambda))-48.6
    thrarr is not necessory to be sampled evenly.
    throughput array is in T_lambda(/A).
    """
    # thrputdir = '/work/CSSOS/filter_improve/fromimg/windextract/throughput/'
    thrput = np.loadtxt(thrghdir + cssband + '.txt')
    thrfine = interp(thrput, xmin=bandpos[cssband][0], xmax=bandpos[cssband][2], dx=1)
    ToBeInteg = thrfine[:, 1] / thrfine[:, 0]  # to be integrated
    Integ = np.trapz(ToBeInteg, x=thrfine[:, 0], dx=1.)
    area = 1.0 * 3e18 * Integ  # fnu in ergs/Hz/s/cm^2
    magab0 = 2.5 * math.log10(area) - 48.6

    return magab0

def sed2magab_photon(modsed, cssband, magzero):
    magzb0 = MagAB_Ener_Zero(cssband)
    thrput = np.loadtxt(thrghdir + cssband + '.txt')
    thrfine = interp(thrput, xmin=bandpos[cssband][0], xmax=bandpos[cssband][2], dx=1)
    sedfine = interp(modsed, xmin=bandpos[cssband][0], xmax=bandpos[cssband][2], dx=1)
    ToBeInteg = sedfine[:, 1] * thrfine[:, 1] * thrfine[:, 0]
    flux = np.trapz(ToBeInteg, x=thrfine[:, 0], dx=1.)
    magab = -2.5 * math.log10(flux) + magzero
    return magab