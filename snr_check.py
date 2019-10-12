
# Usage: python3 snr_check.py Cssos_magsim_SNR_tile_077_424.txt

import astropy.io.ascii as ascii
import numpy as np
from scipy import stats
import sys
import matplotlib.pyplot as plt
from scipy.stats import sigmaclip


def sigmaclip(ndarray, nsigma):
    mean = np.median(ndarray)
    std = np.std(ndarray)
    idx = np.where((ndarray>(mean-std*nsigma)) & (ndarray<(mean+std*nsigma)))
    return idx

def clip(ndarray, min, max):
    idx = np.where((ndarray>=min) & (ndarray<=max))
    return idx


snr = 10

data = ascii.read(sys.argv[1])
# data = ascii.read('Cssos_magsim_SNR_tile_077_424.txt')


slope, intercept, r_value, p_value, std_err = stats.linregress(data['MOD_r'],data['MagSim_r'])

print(slope,intercept)

plt.scatter(data['MOD_r'], data['MagSim_r'], s=1)
plt.plot(np.array([18,30]), slope*np.array([18,30])+intercept, color='red')
plt.xlim(18,28)
plt.ylim(18,28)
plt.show()


data['Resid_r'] = data['MagSim_r'] - (slope*data['MOD_r']+intercept)

data = data[sigmaclip(data['Resid_r'], 3)]
data = data[sigmaclip(data['Resid_r'], 3)]

dataclipsnr = data[clip(data['SNR_r'], snr-0.3, snr+0.3)]

plt.scatter(dataclipsnr['MOD_r'], dataclipsnr['MagSim_r'], s=1)
plt.plot(np.array([18,30]), slope*np.array([18,30])+intercept, color='red')
plt.xlim(min(dataclipsnr['MOD_r']),max(dataclipsnr['MOD_r']))
plt.ylim(min(dataclipsnr['MOD_r']),max(dataclipsnr['MOD_r']))
plt.show()

plt.scatter(dataclipsnr['MOD_r'], dataclipsnr['Resid_r'], s=1)
plt.show()

std_resid = np.std(dataclipsnr['Resid_r'])

print('SNR =', snr)
print('STD =', std_resid)
print(snr*std_resid)