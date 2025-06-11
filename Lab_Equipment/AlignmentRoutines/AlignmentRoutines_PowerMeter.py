from Lab_Equipment.Config import config

# import tomography.standard as standard
# import tomography.masks as masks

# Python Libs
import cv2
import numpy as np
import matplotlib.pyplot as plt
import ctypes
import copy
from IPython.display import display, clear_output
import ipywidgets
import multiprocessing
import time
import scipy.io

from scipy import io, integrate, linalg, signal
from scipy.io import savemat, loadmat
from scipy.fft import fft, fftfreq, fftshift,ifftshift, fft2,ifft2,rfft2,irfft2
# Defult Pploting properties 
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = [5,5]

# from script_functions import start_worker
# import CameraWindowForm as CamForm
#SLM Libs
import Lab_Equipment.SLM.pyLCOS as pyLCOS
import Lab_Equipment.ZernikeModule.ZernikeModule as zernMod

#Camera Libs
import Lab_Equipment.Camera.CameraObject as CamForm

# Power Meter Libs
import  Lab_Equipment.PowerMeter.PowerMeterObject as PMLib


# Alginment Functions
import  Lab_Equipment.AlignmentRoutines.AlignmentFunctions as AlignFunc


def CourseSweepAcrossSLMPowerMeter(slm:pyLCOS.LCOS,channel,power_meter,flipCount=25):

    slm.LCOS_Clean()
    # flipMin=//2-flipCount//2
    flipMin=0
    flipMax=slm.slmHeigth//2+flipCount//2
    flipMax=slm.slmWidth//2+flipCount//2
    print(power_meter.read)
    powerReadingX=np.empty(0)
    powerReadingY=np.empty(0)

    #Left to right sweep
    for iflip in range(0,slm.slmWidth,flipCount):

        powerReadingX=np.append(powerReadingX,power_meter.read)
        # PiFlip_cmplx =np.ones((slm.slmHeigth,slm.slmWidth),dtype=complex)
        PiFlip_cmplx =np.zeros((slm.slmHeigth,slm.slmWidth),dtype=np.float32)
        # PiFlip_cmplx =np.ones((slm.slmHeigth,slm.slmWidth),dtype=np.float32)*(-1*np.pi)

        # PiFlip_cmplx[0:flipMin+iflip,:]=np.exp(1j*np.pi)
        # PiFlip_cmplx[:,0:flipMin+iflip]=np.exp(1j*np.pi)
        PiFlip_cmplx[:,0:flipMin+iflip]=(np.pi)


        # np.angle( np.random.rand(1200,1920) + np.random.rand(1200,1920) * 1j)
        ArryForSLM=slm.phaseTolevel(np.angle(PiFlip_cmplx), aperture = 1)
        # slm.LCOS_Display(ArryForSLM, slm.GLobProps[channel].rgbChannelIdx)
        slm.LCOS_Display(ArryForSLM, channel)
        
        
        time.sleep(slm.GLobProps[channel].RefreshTime)
        
    # top to bottom sweep    
    for iflip in range(0,slm.slmHeigth,flipCount):
        powerReadingY=np.append(powerReadingY,power_meter.read)

        # PiFlip_cmplx =np.ones((slm.slmHeigth,slm.slmWidth),dtype=complex)
        PiFlip_cmplx =np.zeros((slm.slmHeigth,slm.slmWidth),dtype=np.float32)

        # PiFlip_cmplx[0:flipMin+iflip,:]=np.exp(1j*np.pi)
        PiFlip_cmplx[0:flipMin+iflip,:]=(np.pi)

        # PiFlip_cmplx[:,0:flipMin+iflip]=np.exp(1j*np.pi)

        # np.angle( np.random.rand(1200,1920) + np.random.rand(1200,1920) * 1j)
        ArryForSLM=slm.phaseTolevel(np.angle(PiFlip_cmplx), aperture = 1)
        # slm.LCOS_Display(ArryForSLM, slm.GLobProps[channel].rgbChannelIdx)
        slm.LCOS_Display(ArryForSLM, channel)
        
        time.sleep(slm.GLobProps[channel].RefreshTime)
    
    slm.LCOS_Clean(channel)
    return powerReadingX,powerReadingY

